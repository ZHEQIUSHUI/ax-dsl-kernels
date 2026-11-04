"""DeepSeek-V4 HCA attention kernels and helpers.

The reusable mHC stream operations live in :mod:`.mhc`; this module owns the
C128 compressor and paged HCA attention pipeline.
"""

from __future__ import annotations

import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from sgl_kernel_axera.attention.parallel_utils import attention_core_num, linear_program_slice

from .attn_utils.common import (
    KernelMeta,
    attention_prefix,
    as_phase,
    as_tp_layout,
    load_attention_prefix,
    project_attention_local_output,
    project_attention_output,
    project_attention_query,
    kernel_cache_enabled,
    kernel_cache_path,
    env_flag,
    rmsnorm as canonical_rmsnorm,
    tail_rope as canonical_tail_rope,
    token_slice as slice_tokens,
)
from .attn_utils.hca_helper import (
    hca_attention_paged,
    hca_compressor,
    hca_decode_compressor_update,
    write_hca_short_state,
)
from .attn_utils.swa_helper import swa_attention_paged
from .swa import (
    SWA_TP1_LAYOUT,
    validate_swa_launch,
    write_swa_cache,
)

# Backward-compatible aliases for callers that imported mHC helpers from HCA.
# The implementations themselves are defined in ``mhc.py``.
from .mhc import (
    check_pre_inputs,
    hca_head,
    hca_head_kernel,
    hca_head_value,
    hca_post,
    hca_post_kernel,
    hca_post_value,
    hca_pre,
    hca_pre_kernel,
    hca_pre_value,
    kernel_weight,
    rms_denominator,
    run_mode,
    scalar,
    sinkhorn,
)


def hca_compress_value(x, wkv, wgate, ape, norm_weight, cos, sin, *, eps: float, rotary_dim: int, output_dtype):
    """Record canonical C128 compression with four-core channel ownership."""
    batch, seq_len, _ = x.shape
    core_id = ax.program_id(0)
    width = wkv.shape[0]
    channel_slice = linear_program_slice(width)
    local_width = channel_slice.stop - channel_slice.start
    if local_width <= 0:
        raise ValueError("HCA compressor channel slice is empty")
    content = ax.dsl.matmul(x, wkv[channel_slice, :], transposed=True, output_dtype=x.dtype).reshape(
        batch, 1, seq_len, local_width
    )
    gate = ax.dsl.matmul(x, wgate[channel_slice, :], transposed=True, output_dtype=x.dtype).reshape(
        batch, 1, seq_len, local_width
    )
    gate = ax.dsl.add(gate, ape.reshape(1, 1, seq_len, width)[:, :, :, channel_slice], output_dtype=ax.fp32)
    probability = ax.dsl.softmax(gate, dim=2, output_dtype=ax.fp32)
    pooled_shared = ax.dsl.create_tensor((batch, 1, width), ax.fp32, ax.MemoryType.SHARED_OCM, name="hca_pooled")
    ax.dsl.sum(
        ax.dsl.mul(content, probability, output_dtype=ax.fp32),
        dim=2,
        output_dtype=ax.fp32,
        r=pooled_shared[:, :, channel_slice],
    )
    axd.sync_all()
    compressed_shared = ax.dsl.create_tensor(
        (batch, 1, width), output_dtype, ax.MemoryType.SHARED_OCM, name="hca_compressed"
    )
    if core_id == 0:
        normalized = canonical_rmsnorm(pooled_shared, norm_weight, eps=eps, output_dtype=output_dtype)
        canonical_tail_rope(
            normalized,
            cos.reshape(batch, 1, rotary_dim // 2),
            sin.reshape(batch, 1, rotary_dim // 2),
            rotary_dim,
            r=compressed_shared,
        )
    axd.sync_all()
    return compressed_shared


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_hca_short_attention(
    residual,
    cos,
    sin,
    positions_casual,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
    compress_state_cache,
    compress_state_out_loc,
    branch_partial,
    residual_path,
    kv_out,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    input_norm_weight,
    q_a_weight,
    q_a_weight_scale,
    q_a_norm_weight,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    kv_weight,
    kv_weight_scale,
    kv_norm_weight,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    compress_kv_weight,
    compress_kv_weight_scale,
    compress_gate_weight,
    compress_gate_weight_scale,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Run four-core HCA before the first complete C128 block exists."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)
    prefix = attention_prefix(
        residual,
        hc_fn,
        hc_fn_scale,
        hc_base,
        hc_scale,
        input_norm_weight,
        q_a_weight,
        q_a_weight_scale,
        q_a_norm_weight,
        kv_weight,
        kv_weight_scale,
        kv_norm_weight,
        cos,
        sin,
        kv_out,
        meta,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        residual_path_out=residual_path,
    )
    write_hca_short_state(
        prefix.normalized,
        compress_state_cache,
        compress_state_out_loc,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        meta,
        name_prefix="hca_short",
    )
    query = project_attention_query(
        prefix.q_residual,
        prefix.cos,
        prefix.sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=prefix.kv.shape[-1],
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    context = swa_attention_paged(
        prefix,
        query,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
        source_chunk_size=source_chunk_size,
    )
    project_attention_output(
        prefix,
        context,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        branch_partial,
        meta,
        rotary_dim=rotary_dim,
        residual_path_out=residual_path,
    )
    write_swa_cache(prefix, swa_kv_cache, swa_out_loc, core_id)
    return branch_partial, residual_path, kv_out


def launch_hca_short_attention(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
    compress_state_cache,
    compress_state_out_loc,
    branch_partial,
    residual_path,
    kv_out,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    input_norm_weight,
    q_a_weight,
    q_a_weight_scale,
    q_a_norm_weight,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    kv_weight,
    kv_weight_scale,
    kv_norm_weight,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    compress_kv_weight,
    compress_kv_weight_scale,
    compress_gate_weight,
    compress_gate_weight_scale,
    *,
    positions,
    phase,
    tp_layout,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    core_num=None,
):
    """Launch incomplete-C128 prefill or continuation on four physical cores."""
    seq_len = residual.shape[1] if residual.ndim == 4 else 0
    phase = as_phase(phase)
    tp_layout = as_tp_layout(tp_layout)
    if tp_layout is not SWA_TP1_LAYOUT:
        raise ValueError(f"HCA short attention requires tp_layout={SWA_TP1_LAYOUT!r}")
    if positions is None or positions.ndim != 1 or positions.numel() != seq_len:
        raise ValueError("HCA short attention requires one position per token")
    if not 0 < seq_len < window == 128:
        raise ValueError("HCA short attention requires prefill/decode before the first C128 block")
    first_position = int(positions[0].item())
    expected_positions = torch.arange(
        first_position, first_position + seq_len, dtype=positions.dtype, device=positions.device
    )
    if first_position < 0 or first_position + seq_len >= window or not torch.equal(positions, expected_positions):
        raise ValueError("HCA short attention requires consecutive positions before C128 closure")
    if source_chunk_size <= 0:
        raise ValueError("HCA source_chunk_size must be positive")
    core_num = attention_core_num(core_num)
    validate_swa_launch(
        residual,
        cos,
        sin,
        positions,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        swa_out_loc,
        branch_partial,
        residual_path,
        kv_out,
        q_a_weight,
        q_b_weight_chip,
        kv_weight,
        sinks_chip,
        o_a_weight_chip,
        o_b_weight_chip,
        phase=phase,
        tp_layout=tp_layout,
        window=window,
        rotary_dim=rotary_dim,
        core_num=core_num,
    )
    if compress_state_cache.ndim != 3 or compress_state_cache.shape[1] != window:
        raise ValueError("HCA state cache must have shape [blocks, 128, 2 * head_dim]")
    if compress_state_cache.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("HCA state cache must be FP32 or BF16")
    if compress_kv_weight.ndim != 2 or tuple(compress_gate_weight.shape) != tuple(compress_kv_weight.shape):
        raise ValueError("HCA content and gate weights must have identical shapes")
    if (
        tuple(compress_kv_weight.shape) != (kv_out.shape[-1], residual.shape[-1])
        or compress_state_cache.shape[-1] != 2 * kv_out.shape[-1]
    ):
        raise ValueError("HCA state cache and compressor weights must match the head and hidden dimensions")
    if (
        compress_state_out_loc.ndim != 1
        or compress_state_out_loc.numel() != seq_len
        or compress_state_out_loc.dtype != torch.int32
    ):
        raise ValueError("HCA state write locations must contain one INT32 slot per token")
    state_rows = compress_state_cache.shape[0] * compress_state_cache.shape[1]
    if int(compress_state_out_loc.min()) < 0 or int(compress_state_out_loc.max()) >= state_rows:
        raise ValueError("HCA state write locations must index valid cache rows")
    ax.init(chip_num=1, core_num=core_num)
    attn_hca_short_attention[core_num](
        residual,
        cos,
        sin,
        positions,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        swa_out_loc,
        compress_state_cache,
        compress_state_out_loc,
        branch_partial,
        residual_path,
        kv_out,
        hc_fn,
        hc_fn_scale,
        hc_base,
        hc_scale,
        input_norm_weight,
        q_a_weight,
        q_a_weight_scale,
        q_a_norm_weight,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        kv_weight,
        kv_weight_scale,
        kv_norm_weight,
        sinks_chip,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        window,
        source_chunk_size,
        rotary_dim,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    return branch_partial, residual_path, kv_out


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_hca_prefill(
    residual,
    cos,
    sin,
    positions_casual,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_cache_loc,
    c128_kv_cache,
    c128_page_indices,
    c128_topk_lengths_clamp1,
    c128_out_loc,
    branch_partial,
    residual_path,
    kv_out,
    compressed_out,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    input_norm_weight,
    q_a_weight,
    q_a_weight_scale,
    q_a_norm_weight,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    kv_weight,
    kv_weight_scale,
    kv_norm_weight,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    compress_kv_weight,
    compress_kv_weight_scale,
    compress_gate_weight,
    compress_gate_weight_scale,
    compress_position_bias,
    compress_norm_weight,
    compress_rate=128,
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record one chip's HCA contribution from SGLang paged SWA/C128 metadata."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    head_dim = kv_weight.shape[0]
    compressed_len = seq_len // compress_rate
    assert core_num in (1, 4)
    assert batch == 1 and streams == 4
    assert seq_len == compress_rate
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)

    # prefix.normalized/q_residual/kv: [B,S,D], [B,S,Rq], and [B,S,Dh], BF16.
    prefix = attention_prefix(
        residual,
        hc_fn,
        hc_fn_scale,
        hc_base,
        hc_scale,
        input_norm_weight,
        q_a_weight,
        q_a_weight_scale,
        q_a_norm_weight,
        kv_weight,
        kv_weight_scale,
        kv_norm_weight,
        cos,
        sin,
        kv_out,
        meta,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        residual_path_out=residual_path,
    )
    # query: [B,S,Hq_core,Dh], BF16 private OCM.
    query = project_attention_query(
        prefix.q_residual,
        prefix.cos,
        prefix.sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=prefix.kv.shape[-1],
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    # compressed: [B,1,Dh], BF16.
    compressed = hca_compressor(
        prefix.normalized,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        compress_position_bias,
        compress_norm_weight,
        prefix.cos,
        prefix.sin,
        meta,
        compress_rate=compress_rate,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    # C128 page metadata includes the current compressed entry. Core 0 is its
    # single writer, matching CSA's paged-cache ownership.
    if core_id == 0:
        # c128_write_indices: [Nc], INT32.
        c128_write_indices = ax.dsl.load(c128_out_loc)
        # c128_kv_cache selected rows: [Nc,Dh], compressed.dtype.
        ax.dsl.select_scatter(
            compressed.reshape(compressed_len, head_dim),
            c128_write_indices,
            dim=0,
            r=c128_kv_cache.reshape(-1, head_dim),
        )
    # SWA attention consumes paged history plus ``prefix.kv`` directly, so the
    # current raw-KV cache update remains the caller's persistence boundary.
    # Synchronization only; no tensor output.
    axd.sync_all()
    # context: [B,S,Hq_core,Dh], BF16.
    context = hca_attention_paged(
        prefix,
        query,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c128_kv_cache,
        c128_page_indices,
        c128_topk_lengths_clamp1,
        positions_casual,
        meta,
        compress_rate=compress_rate,
        window=window,
    )
    # branch_partial: [B,S,4,D], BF16.
    project_attention_output(
        prefix,
        context,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        branch_partial,
        meta,
        rotary_dim=rotary_dim,
    )
    if core_id == 0:
        # compressed_out: [B,1,Dh], compressed.dtype; production [1,1,512] BF16.
        ax.dsl.store(compressed, r=compressed_out)
    return branch_partial, residual_path, kv_out, compressed_out


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_hca_decode(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c128_kv_cache,
    c128_page_indices,
    c128_topk_lengths_clamp1,
    positions_casual,
    c128_out_loc,
    compress_content_state,
    compress_gate_state,
    compress_cos,
    compress_sin,
    branch_partial,
    residual_path,
    kv_out,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    input_norm_weight,
    q_a_weight,
    q_a_weight_scale,
    q_a_norm_weight,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    kv_weight,
    kv_weight_scale,
    kv_norm_weight,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    compress_kv_weight,
    compress_kv_weight_scale,
    compress_gate_weight,
    compress_gate_weight_scale,
    compress_position_bias,
    compress_norm_weight,
    window=128,
    compress_rate=128,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record one HCA decode contribution for runtime C128/SWA lengths."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    assert window == compress_rate == 128
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)

    prefix = attention_prefix(
        residual,
        hc_fn,
        hc_fn_scale,
        hc_base,
        hc_scale,
        input_norm_weight,
        q_a_weight,
        q_a_weight_scale,
        q_a_norm_weight,
        kv_weight,
        kv_weight_scale,
        kv_norm_weight,
        cos,
        sin,
        kv_out,
        meta,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        residual_path_out=residual_path,
    )
    # query: [1,1,Hq_core,Dh], BF16 private OCM.
    query = project_attention_query(
        prefix.q_residual,
        prefix.cos,
        prefix.sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=prefix.kv.shape[-1],
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    compressed = hca_decode_compressor_update(
        prefix.normalized,
        positions_casual,
        compress_content_state,
        compress_gate_state,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        compress_position_bias,
        compress_norm_weight,
        compress_cos,
        compress_sin,
        meta,
        compress_rate=compress_rate,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    if core_id == 0:
        # c128_write_indices: [1], INT32.
        c128_write_indices = ax.dsl.load(c128_out_loc)
        # c128_kv_cache selected row: [1,Dh], compressed.dtype.
        ax.dsl.select_scatter(
            compressed.reshape(1, compressed.shape[-1]),
            c128_write_indices,
            dim=0,
            r=c128_kv_cache.reshape(-1, compressed.shape[-1]),
        )
    # Core 0 owns the S=1 KV row. The dynamic HCA path then iterates the raw
    # C128 prefix in 128-row blocks and appends one private masked SWA tile.
    # Synchronization only; no tensor output.
    axd.sync_all()
    context = hca_attention_paged(
        prefix,
        query,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c128_kv_cache,
        c128_page_indices,
        c128_topk_lengths_clamp1,
        positions_casual,
        meta,
        compress_rate=compress_rate,
        window=window,
        history_source_chunk_size=128,
    )
    project_attention_output(
        prefix,
        context,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        branch_partial,
        meta,
        rotary_dim=rotary_dim,
    )
    return branch_partial, residual_path, kv_out


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_hca_projection_prefill(
    residual,
    cos,
    sin,
    c128_kv_cache,
    c128_out_loc,
    q_residual_out,
    post_out,
    residual_path_out,
    kv_out,
    compressed_out,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    input_norm_weight,
    q_a_weight,
    q_a_weight_scale,
    q_a_norm_weight,
    kv_weight,
    kv_weight_scale,
    kv_norm_weight,
    compress_kv_weight,
    compress_kv_weight_scale,
    compress_gate_weight,
    compress_gate_weight_scale,
    compress_position_bias,
    compress_norm_weight,
    compress_rate=128,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Produce common projection tensors and persist the current C128 row."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    assert core_num in (1, 4) and streams == 4 and batch == 1 and seq_len >= core_num
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)
    prefix = attention_prefix(
        residual,
        hc_fn,
        hc_fn_scale,
        hc_base,
        hc_scale,
        input_norm_weight,
        q_a_weight,
        q_a_weight_scale,
        q_a_norm_weight,
        kv_weight,
        kv_weight_scale,
        kv_norm_weight,
        cos,
        sin,
        kv_out,
        meta,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        residual_path_out=residual_path_out,
    )
    token_slice = slice_tokens(meta.core_id, meta.core_num, meta.seq_len)
    # q_residual_out token slice: [B,T_core,Rq], residual.dtype.
    ax.dsl.store(prefix.q_residual[:, token_slice, :], r=q_residual_out[:, token_slice, :])
    # post_out token slice: [B,T_core,4], FP32.
    ax.dsl.store(prefix.state.post[:, token_slice, :], r=post_out[:, token_slice, :])
    compressed = hca_compressor(
        prefix.normalized,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        compress_position_bias,
        compress_norm_weight,
        prefix.cos,
        prefix.sin,
        meta,
        compress_rate=compress_rate,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    if meta.core_id == 0:
        compressed_len = compressed.shape[1]
        head_dim = compressed.shape[-1]
        # c128_write_indices: [Nc], INT32.
        c128_write_indices = ax.dsl.load(c128_out_loc)
        # c128_kv_cache selected rows: [Nc,Dh], compressed.dtype.
        ax.dsl.select_scatter(
            compressed.reshape(compressed_len, head_dim),
            c128_write_indices,
            dim=0,
            r=c128_kv_cache.reshape(-1, head_dim),
        )
        # compressed_out: [B,Nc,Dh], compressed.dtype.
        ax.dsl.store(compressed, r=compressed_out)
    return q_residual_out, post_out, residual_path_out, kv_out, compressed_out


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_hca_core_prefill(
    q_residual,
    kv,
    cos,
    sin,
    positions_casual,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c128_kv_cache,
    c128_page_indices,
    c128_topk_lengths_clamp1,
    local_output,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    compress_rate=128,
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
):
    """Run HCA and store the pre-Post TP contribution `[B,S,D]`."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, hidden = local_output.shape
    assert core_num in (1, 4) and batch == 1 and seq_len >= core_num
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)
    prefix = load_attention_prefix(q_residual, kv, cos, sin, meta)
    query = project_attention_query(
        prefix.q_residual,
        prefix.cos,
        prefix.sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=kv.shape[-1],
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    context = hca_attention_paged(
        prefix,
        query,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c128_kv_cache,
        c128_page_indices,
        c128_topk_lengths_clamp1,
        positions_casual,
        meta,
        compress_rate=compress_rate,
        window=window,
    )
    project_attention_local_output(
        prefix,
        context,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        local_output,
        meta,
        rotary_dim=rotary_dim,
    )
    return local_output
