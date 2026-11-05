"""DeepSeek-V4 compressed sparse attention (CSA) kernels.

This module owns the ratio-4 compressor, Lightning Indexer boundary, and the
local four-core prefill/decode entries. Persistent cache/state tensors remain
owned by the caller.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from sgl_kernel_axera.attention.parallel_utils import (
    attention_core_num,
    linear_program_slice,
    validate_core_num,
)

from .attn_utils.common import (
    AttentionPhase,
    AttentionPrefix,
    KernelMeta,
    attention_prefix,
    as_tp_layout,
    configured_run_mode,
    env_flag,
    load_attention_prefix,
    paged_swa_source,
    positive_i32,
    project_attention_local_output,
    project_attention_output,
    project_attention_query,
    rmsnorm as common_rmsnorm,
    kernel_cache_enabled,
    kernel_cache_path,
    runtime_linear,
    runtime_weight_scale_slice,
    token_slice as slice_tokens,
)
from .attn_utils.csa_helper import (
    csa_attention_from_topk,
    csa_attention_paged,
    csa_lightning_topk,
    normalized_hadamard,
    quantize_index_int8_rows,
    stride4_compressor,
    stride4_state_closure,
)
from .attn_utils.decode import decode_static_attention
from .attn_utils.swa_helper import swa_attention_paged
from .swa import (
    SWA_TP1_LAYOUT,
    validate_swa_launch,
    write_swa_cache,
)


def run_mode() -> ax.RunMode:
    return configured_run_mode()


def kernel_weight(value: torch.Tensor) -> torch.Tensor:
    """Use BF16 projection weights for AX8860 CRUN MatMul."""
    if run_mode() is ax.RunMode.CRUN and value.dtype == torch.float32:
        return value.to(torch.bfloat16)
    return value


def constant(value, name=None):
    """Create a contiguous DSL constant, preserving integer metadata dtypes."""
    array = np.asarray(value)
    if array.dtype.fields is None and np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    array = np.ascontiguousarray(array)
    return ax.dsl.from_numpy(array, prefix=name) if name is not None else ax.dsl.from_numpy(array)


def csa_compress_value(x, wkv, wgate, ape, *, ratio: int, eps: float, output_dtype, norm_weight=None):
    """Record gated ratio-group pooling with ratio-4 overlap."""

    def project(weight):
        # Compression projections are too large to materialize as one OCM load.
        # Keep the checkpoint tensor in DDR and load bounded output-channel tiles.
        if weight.memory_type != ax.MemoryType.DDR:
            return ax.dsl.matmul(x, weight, transposed=True, output_dtype=x.dtype)
        rows = int(np.prod(x.shape[:-1]))
        flat = x.reshape(rows, x.shape[-1])
        parts = []
        for start in range(0, weight.shape[0], 64):
            stop = min(start + 64, weight.shape[0])
            tile = ax.dsl.load(weight[start:stop, :])
            parts.append(ax.dsl.matmul(flat, tile, transposed=True, output_dtype=x.dtype))
        value = parts[0] if len(parts) == 1 else ax.dsl.concat(*parts, dim=-1)
        return value.reshape(*x.shape[:-1], weight.shape[0])

    batch, seq_len, _ = x.shape
    ratio = int(ratio)
    if ratio <= 0 or seq_len % ratio:
        raise ValueError("CSA compression requires ratio to divide seq_len")
    coff = 2 if ratio == 4 else 1
    head_dim = wkv.shape[0] // coff
    groups = seq_len // ratio
    kv = project(wkv).reshape(batch, groups, ratio, coff, head_dim)
    score = project(wgate).reshape(batch, groups, ratio, coff, head_dim)
    score = ax.dsl.add(score, ape.reshape(1, 1, ratio, coff, head_dim), output_dtype=ax.fp32)
    if coff == 2:
        current_kv = kv[:, :, :, 1:2, :].reshape(batch, groups, ratio, head_dim)
        current_score = score[:, :, :, 1:2, :].reshape(batch, groups, ratio, head_dim)
        zero_kv = ax.dsl.zeros((batch, 1, ratio, head_dim), dtype=x.dtype)
        zero_score = ax.dsl.full((batch, 1, ratio, head_dim), ax.fp32, fill_value=-float("inf"))
        if groups == 1:
            previous_kv, previous_score = zero_kv, zero_score
        else:
            previous_kv = ax.dsl.concat(
                zero_kv, kv[:, :-1, :, :1, :].reshape(batch, groups - 1, ratio, head_dim), dim=1
            )
            previous_score = ax.dsl.concat(
                zero_score, score[:, :-1, :, :1, :].reshape(batch, groups - 1, ratio, head_dim), dim=1
            )
        pooled_kv = ax.dsl.concat(previous_kv, current_kv, dim=2)
        pooled_score = ax.dsl.concat(previous_score, current_score, dim=2)
    else:
        pooled_kv, pooled_score = kv.reshape(batch, groups, ratio, head_dim), score.reshape(
            batch, groups, ratio, head_dim
        )
    weights = ax.dsl.softmax(pooled_score, dim=2, output_dtype=ax.fp32)
    pooled = ax.dsl.sum(ax.dsl.mul(pooled_kv, weights, output_dtype=ax.fp32), dim=2, output_dtype=ax.fp32)
    return common_rmsnorm(pooled, weight=norm_weight, eps=eps, output_dtype=output_dtype)


def csa_indexer_value(query, compressed_cache, gate, score_mask, *, index_topk: int, softmax_scale: float):
    """Record relu dot-product Indexer scores and Top-K."""
    batch, seq_len, head_num, head_dim = query.shape
    token_count = compressed_cache.shape[1]
    if batch != 1:
        raise ValueError("CSA Indexer currently supports batch size 1")
    logits = ax.dsl.matmul(
        query.reshape(seq_len * head_num, head_dim),
        compressed_cache.reshape(token_count, head_dim),
        transposed=True,
        output_dtype=ax.fp32,
    ).reshape(seq_len, head_num, token_count)
    logits = ax.dsl.maximum(logits, ax.dsl.zeros(logits.shape, ax.fp32), output_dtype=ax.fp32)
    scores = ax.dsl.sum(
        ax.dsl.mul(logits, gate.reshape(seq_len, head_num, 1), output_dtype=ax.fp32), dim=1, output_dtype=ax.fp32
    )
    scores = ax.dsl.mul(scores, constant(softmax_scale), output_dtype=ax.fp32)
    scores = ax.dsl.add(scores.reshape(batch, seq_len, token_count), score_mask, output_dtype=ax.fp32)
    _, positions = ax.dsl.topk(scores, k=int(index_topk), dim=-1)
    return scores, positions


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
)
def csa_compress_kernel(x, wkv, wgate, ape, compressed, ratio, eps=1e-6):
    """Record ratio-grouped gated pooling, including ratio-4 overlap rows."""
    batch, seq_len, hidden = x.shape
    group_count = seq_len // ratio
    output_width = compressed.shape[-1]
    core_num = ax.num_programs(0)
    if core_num == 1:
        pooled = csa_compress_value(
            ax.dsl.load(x),
            wkv,
            wgate,
            ax.dsl.load(ape),
            ratio=ratio,
            eps=eps,
            output_dtype=compressed.dtype,
        )
        ax.dsl.store(pooled, r=compressed)
        return compressed
    assert batch == 1
    group_slice = linear_program_slice(group_count)
    local_groups = group_slice.stop - group_slice.start
    halo_groups = int(ratio == 4 and group_slice.start > 0)
    load_start = group_slice.start - halo_groups
    x_groups = x.reshape(group_count, ratio, hidden)
    compressed_groups = compressed.reshape(group_count, output_width)
    pooled = csa_compress_value(
        ax.dsl.load(x_groups[load_start : group_slice.stop, :, :]).reshape(
            1, (local_groups + halo_groups) * ratio, hidden
        ),
        wkv,
        wgate,
        ax.dsl.load(ape),
        ratio=ratio,
        eps=eps,
        output_dtype=compressed.dtype,
    )
    pooled = pooled[:, halo_groups:, :].reshape(local_groups, output_width)
    ax.dsl.store(pooled, r=compressed_groups[group_slice, :])
    return compressed


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
)
def csa_indexer_kernel(
    query, compressed_cache, gate, score_mask, output_scores, topk_indices, index_topk, softmax_scale
):
    """Record Indexer scores and top-k positions for the owned query rows."""
    batch, seq_len, head_num, head_dim = query.shape
    token_count = compressed_cache.shape[1]
    row_slice = linear_program_slice(seq_len)
    local_rows = row_slice.stop - row_slice.start
    scores, positions = csa_indexer_value(
        ax.dsl.load(query[:, row_slice, :, :]).reshape(batch, local_rows, head_num, head_dim),
        ax.dsl.load(compressed_cache),
        ax.dsl.load(gate[:, row_slice, :]).reshape(batch, local_rows, head_num),
        ax.dsl.load(score_mask[:, row_slice, :]).reshape(batch, local_rows, token_count),
        index_topk=index_topk,
        softmax_scale=softmax_scale,
    )
    ax.dsl.store(scores, r=output_scores[:, row_slice, :])
    ax.dsl.store(positions, r=topk_indices[:, row_slice, :])
    return output_scores, topk_indices


def csa_compress(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    *,
    output: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    core_num: int = 1,
) -> torch.Tensor:
    """Compress a complete prefill prefix into one entry per ratio tokens."""
    if x.ndim != 3 or wkv.ndim != 2 or wgate.shape != wkv.shape:
        raise ValueError("x must be [B,S,D], and wkv/wgate must have identical [C,D] shapes")
    if ratio <= 0 or x.shape[1] % ratio:
        raise ValueError("ratio must be positive and divide the current sequence length")
    coff = 2 if ratio == 4 else 1
    if wkv.shape[0] % coff or ape.shape != (ratio, wkv.shape[0]):
        raise ValueError("CSA projection and APE shapes are incompatible")
    groups = x.shape[1] // ratio
    output = (
        output
        if output is not None
        else torch.empty((x.shape[0], groups, wkv.shape[0] // coff), dtype=x.dtype, device=x.device)
    )
    if tuple(output.shape) != (x.shape[0], groups, wkv.shape[0] // coff):
        raise ValueError("output has an incompatible compressed-cache shape")
    if core_num > 1 and x.shape[0] != 1:
        raise ValueError("multi-core CSA compression currently requires batch size 1")
    validate_core_num(core_num, groups, "CSA compression groups")
    ax.init(chip_num=1, core_num=core_num)
    csa_compress_kernel[core_num](x, kernel_weight(wkv), kernel_weight(wgate), ape, output, ratio, eps)
    return output


def csa_indexer(
    query: torch.Tensor,
    compressed_cache: torch.Tensor,
    gate: torch.Tensor,
    *,
    index_topk: int,
    softmax_scale: Optional[float] = None,
    score_mask: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    core_num: int = 1,
):
    """Score compressed entries and return (scores, topk_indices)."""
    if query.ndim != 4 or compressed_cache.ndim != 3 or gate.shape != query.shape[:3]:
        raise ValueError("query must be [B,S,Hi,Di], cache [B,T,Di], gate [B,S,Hi]")
    batch, seq_len, head_num, head_dim = query.shape
    if batch != 1:
        raise ValueError("CSA Indexer currently supports batch size 1")
    if compressed_cache.shape[0] != batch or index_topk <= 0 or index_topk > compressed_cache.shape[1]:
        raise ValueError("CSA Indexer dimensions or index_topk are invalid")
    scores = (
        scores
        if scores is not None
        else torch.empty((batch, seq_len, compressed_cache.shape[1]), dtype=torch.float32, device=query.device)
    )
    topk_indices = (
        topk_indices
        if topk_indices is not None
        else torch.empty((batch, seq_len, index_topk), dtype=torch.int32, device=query.device)
    )
    score_mask = (
        score_mask
        if score_mask is not None
        else torch.zeros((batch, seq_len, compressed_cache.shape[1]), dtype=torch.float32, device=query.device)
    )
    if (
        tuple(scores.shape) != (batch, seq_len, compressed_cache.shape[1])
        or tuple(topk_indices.shape) != (batch, seq_len, index_topk)
        or tuple(score_mask.shape) != tuple(scores.shape)
    ):
        raise ValueError("CSA Indexer outputs have incompatible shapes")
    scale = float(softmax_scale) if softmax_scale is not None else head_dim ** -0.5 * head_num**-0.5
    validate_core_num(core_num, seq_len, "CSA query rows")
    ax.init(chip_num=1, core_num=core_num)
    csa_indexer_kernel[core_num](query, compressed_cache, gate, score_mask, scores, topk_indices, index_topk, scale)
    return scores, topk_indices


def write_csa_short_state(
    normalized,
    state_cache,
    state_out_loc,
    kv_weight,
    kv_weight_scale,
    gate_weight,
    gate_weight_scale,
    meta,
    *,
    name_prefix,
):
    """Persist raw ratio-4 compressor projections for an incomplete group."""
    projection_width = kv_weight.shape[0]
    hidden = normalized.shape[-1]
    channel_slice = slice_tokens(meta.core_id, meta.core_num, projection_width)
    packed = ax.dsl.create_tensor(
        (meta.batch, meta.seq_len, 2 * projection_width),
        state_cache.dtype,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_raw_state",
    )
    runtime_linear(
        normalized,
        kv_weight[channel_slice, :],
        runtime_weight_scale_slice(
            kv_weight,
            kv_weight_scale,
            channel_slice,
            slice(0, hidden),
        ),
        output_dtype=state_cache.dtype,
        r=packed[:, :, channel_slice],
    )
    gate_slice = slice(
        projection_width + channel_slice.start,
        projection_width + channel_slice.stop,
    )
    runtime_linear(
        normalized,
        gate_weight[channel_slice, :],
        runtime_weight_scale_slice(
            gate_weight,
            gate_weight_scale,
            channel_slice,
            slice(0, hidden),
        ),
        output_dtype=state_cache.dtype,
        r=packed[:, :, gate_slice],
    )
    axd.sync_all()
    if meta.core_id == 0:
        ax.dsl.select_scatter(
            packed.reshape(meta.seq_len, 2 * projection_width),
            ax.dsl.load(state_out_loc),
            dim=0,
            r=state_cache.reshape(-1, 2 * projection_width),
        )
    axd.sync_all()


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_csa_short_prefill(
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
    index_state_cache,
    index_state_out_loc,
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
    index_kv_weight,
    index_kv_weight_scale,
    index_gate_weight,
    index_gate_weight_scale,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    combine_residual=False,
):
    """Run cold short CSA as SWA while advancing both raw state streams."""
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
    write_csa_short_state(
        prefix.normalized,
        compress_state_cache,
        compress_state_out_loc,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        meta,
        name_prefix="csa_short",
    )
    write_csa_short_state(
        prefix.normalized,
        index_state_cache,
        index_state_out_loc,
        index_kv_weight,
        index_kv_weight_scale,
        index_gate_weight,
        index_gate_weight_scale,
        meta,
        name_prefix="csa_short_index",
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
        residual_path_out=residual_path if combine_residual else None,
    )
    write_swa_cache(prefix, swa_kv_cache, swa_out_loc, core_id)
    return branch_partial, residual_path, kv_out


def validate_csa_short_state(
    state_cache,
    state_out_loc,
    kv_weight,
    gate_weight,
    hidden,
    seq_len,
    name,
):
    """Validate one raw compressor state stream before initializing AXDSL."""
    if state_cache.ndim != 3 or state_cache.shape[1] != 8:
        raise ValueError(f"{name} state cache must have shape [blocks, 8, width]")
    if state_cache.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"{name} state cache must be FP32 or BF16")
    if kv_weight.ndim != 2 or tuple(gate_weight.shape) != tuple(kv_weight.shape):
        raise ValueError(f"{name} content/gate weights must have identical [2D, hidden] shapes")
    if kv_weight.shape[1] != hidden or state_cache.shape[-1] != 2 * kv_weight.shape[0]:
        raise ValueError(f"{name} weights do not match hidden size or packed state width")
    if state_out_loc.ndim != 1 or state_out_loc.numel() != seq_len:
        raise ValueError(f"{name} state locations must contain one slot per token")
    if state_out_loc.dtype != torch.int32:
        raise ValueError(f"{name} state locations must be INT32")
    state_rows = state_cache.shape[0] * state_cache.shape[1]
    if state_out_loc.numel() and (int(state_out_loc.min()) < 0 or int(state_out_loc.max()) >= state_rows):
        raise ValueError(f"{name} state locations must index valid cache rows")


def launch_csa_short_prefill(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
    compress_state_cache,
    compress_state_out_loc,
    index_state_cache,
    index_state_out_loc,
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
    index_kv_weight,
    index_kv_weight_scale,
    index_gate_weight,
    index_gate_weight_scale,
    *,
    positions,
    tp_layout,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    core_num=None,
):
    """Launch the cold incomplete-C4 prefill path on the configured grid."""
    seq_len = residual.shape[1] if residual.ndim == 4 else 0
    if not 0 < seq_len < 4:
        raise ValueError(f"CSA short prefill requires S in [1, 3], got S={seq_len}")
    tp_layout = as_tp_layout(tp_layout)
    if tp_layout is not SWA_TP1_LAYOUT:
        raise ValueError(f"CSA short prefill requires tp_layout={SWA_TP1_LAYOUT!r}")
    if positions is None or positions.ndim != 1 or positions.numel() != seq_len:
        raise ValueError("CSA short prefill positions must contain one row per token")
    if int(positions[-1].item()) >= 3:
        raise ValueError("CSA short prefill covers only the first incomplete C4 group, positions < 3")
    if source_chunk_size <= 0:
        raise ValueError("CSA short prefill source_chunk_size must be positive")
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
        phase=AttentionPhase.PREFILL,
        tp_layout=tp_layout,
        window=window,
        rotary_dim=rotary_dim,
        core_num=core_num,
    )
    validate_csa_short_state(
        compress_state_cache,
        compress_state_out_loc,
        compress_kv_weight,
        compress_gate_weight,
        residual.shape[-1],
        seq_len,
        "CSA main",
    )
    validate_csa_short_state(
        index_state_cache,
        index_state_out_loc,
        index_kv_weight,
        index_gate_weight,
        residual.shape[-1],
        seq_len,
        "CSA indexer",
    )
    ax.init(chip_num=1, core_num=core_num)
    attn_csa_short_prefill[core_num](
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
        index_state_cache,
        index_state_out_loc,
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
        index_kv_weight,
        index_kv_weight_scale,
        index_gate_weight,
        index_gate_weight_scale,
        window,
        source_chunk_size,
        rotary_dim,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        True,
    )
    return branch_partial, residual_path, kv_out


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_csa_short_continuation(
    residual,
    cos,
    sin,
    positions_casual,
    compress_cos,
    compress_sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
    compress_state_cache,
    compress_state_out_loc,
    compress_state_read_loc,
    index_state_cache,
    index_state_out_loc,
    index_state_read_loc,
    c4_kv_cache,
    index_k_cache,
    index_scale_cache,
    c4_page_table,
    c4_out_loc,
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
    index_kv_weight,
    index_kv_weight_scale,
    index_gate_weight,
    index_gate_weight_scale,
    index_position_bias,
    index_norm_weight,
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    index_topk=512,
    close_group=False,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Advance one ratio-4 token and optionally publish its completed C4 row."""
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
    write_csa_short_state(
        prefix.normalized,
        compress_state_cache,
        compress_state_out_loc,
        compress_kv_weight,
        compress_kv_weight_scale,
        compress_gate_weight,
        compress_gate_weight_scale,
        meta,
        name_prefix="csa_continuation",
    )
    write_csa_short_state(
        prefix.normalized,
        index_state_cache,
        index_state_out_loc,
        index_kv_weight,
        index_kv_weight_scale,
        index_gate_weight,
        index_gate_weight_scale,
        meta,
        name_prefix="csa_continuation_index",
    )
    if close_group:
        compressed = stride4_state_closure(
            compress_state_cache,
            compress_state_read_loc,
            compress_position_bias,
            compress_norm_weight,
            compress_cos,
            compress_sin,
            rotary_dim=rotary_dim,
            rms_eps=rms_eps,
            name_prefix="csa_continuation",
        )
        index_k = stride4_state_closure(
            index_state_cache,
            index_state_read_loc,
            index_position_bias,
            index_norm_weight,
            compress_cos,
            compress_sin,
            rotary_dim=min(rotary_dim, index_norm_weight.shape[0]),
            rms_eps=rms_eps,
            name_prefix="csa_continuation_index",
        )
        index_k = normalized_hadamard(index_k, "csa_continuation_index_hadamard")
        index_k_int8, index_scale = quantize_index_int8_rows(
            index_k,
            "csa_continuation_index",
        )
        axd.sync_all()
        if core_id == 0:
            write_loc = ax.dsl.load(c4_out_loc)
            compressed_cache_value = ax.dsl.cast(
                compressed,
                dtype=c4_kv_cache.dtype,
            )
            ax.dsl.select_scatter(
                compressed_cache_value,
                write_loc,
                dim=0,
                r=c4_kv_cache.reshape(-1, compressed_cache_value.shape[-1]),
            )
            ax.dsl.select_scatter(
                index_k_int8,
                write_loc,
                dim=0,
                r=index_k_cache.reshape(-1, index_k_int8.shape[-1]),
            )
            ax.dsl.select_scatter(
                index_scale,
                write_loc,
                dim=0,
                r=index_scale_cache.reshape(-1, 1),
            )
        axd.sync_all()

    # Each physical core owns a disjoint main-attention head shard. The
    # Lightning Indexer is query-owned for S=1, so it runs as a private logical
    # one-core scan while output projection keeps the real four-core metadata.
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
    decode_meta = KernelMeta(0, 1, batch, seq_len, hidden)
    physical_top_ids, topk_lengths = csa_lightning_topk(
        prefix,
        index_k_cache,
        c4_page_table,
        positions_casual,
        index_q_weight,
        index_q_weight_scale,
        index_weight,
        index_weight_scale,
        decode_meta,
        index_topk=index_topk,
        rotary_dim=rotary_dim,
        index_scale_cache=index_scale_cache,
    )

    axd.sync_all()
    swa_source, swa_mask = paged_swa_source(
        prefix,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
        name_prefix="csa_continuation_swa",
        mask_query_slice=slice(0, seq_len),
        source_alignment=window,
        cooperative=False,
    )
    head_dim = prefix.kv.shape[-1]
    source = ax.dsl.create_tensor(
        (index_topk + window, head_dim),
        prefix.kv.dtype,
        ax.MemoryType.OCM,
        name="csa_continuation_source",
    )
    ax.dsl.index_select(
        c4_kv_cache.reshape(-1, head_dim),
        physical_top_ids.reshape(index_topk),
        dim=0,
        r=source[:index_topk, :],
    )
    ax.dsl.move(
        swa_source.reshape(window, head_dim),
        r=source[index_topk:, :],
    )
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    one_i32 = constant(np.array([1], dtype=np.int32), "csa_continuation_one")
    c4_offsets = constant(
        np.arange(index_topk, dtype=np.int32),
        "csa_continuation_c4_offsets",
    )
    remaining = ax.dsl.sub(
        topk_lengths.reshape(1),
        c4_offsets,
        output_dtype=ax.int32,
    )
    active = positive_i32(remaining, zero_i32, one_i32)
    inactive = ax.dsl.sub(one_i32, active, output_dtype=ax.int32)
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)
    c4_mask = ax.dsl.mul(inactive, negative_bias, output_dtype=ax.fp32)
    source_mask = ax.dsl.concat(c4_mask, swa_mask.reshape(window), dim=0)
    heads_per_core = query.shape[2]
    sink_slice = sinks_chip[core_id * heads_per_core : (core_id + 1) * heads_per_core]
    context = decode_static_attention(
        query,
        source,
        sink_slice,
        source_mask,
        name_prefix="csa_continuation",
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


def validate_csa_short_continuation_cache(
    c4_kv_cache,
    index_k_cache,
    index_scale_cache,
    c4_out_loc,
    close_group,
):
    """Validate the real ratio-4 cache ABI before AXDSL initialization."""
    if c4_kv_cache.ndim < 2 or index_k_cache.ndim < 2:
        raise ValueError("CSA C4 caches must expose token and channel axes")
    if c4_kv_cache.shape[:-1] != index_k_cache.shape[:-1]:
        raise ValueError("CSA main and Indexer caches must share page and row axes")
    if index_k_cache.dtype != torch.int8:
        raise ValueError("CSA Indexer key cache must be INT8")
    if index_scale_cache.dtype != torch.float16:
        raise ValueError("CSA Indexer scale cache must be FP16")
    if tuple(index_scale_cache.shape) != tuple(index_k_cache.shape[:-1]) + (1,):
        raise ValueError("CSA Indexer scale cache must contain one scale per key row")
    if c4_out_loc is None:
        raise ValueError("CSA continuation requires a C4 output-location tensor")
    allowed_rows = (1,) if close_group else (0, 1)
    if c4_out_loc.ndim != 1 or c4_out_loc.numel() not in allowed_rows:
        expected_rows = "one" if close_group else "zero or one"
        raise ValueError(f"CSA continuation requires {expected_rows} C4 output locations")
    cache_rows = c4_kv_cache.numel() // c4_kv_cache.shape[-1]
    index_rows = index_k_cache.numel() // index_k_cache.shape[-1]
    if cache_rows != index_rows:
        raise ValueError("CSA main and Indexer caches must expose the same row count")
    if close_group and (int(c4_out_loc.min()) < 0 or int(c4_out_loc.max()) >= cache_rows):
        raise ValueError("CSA C4 output location must index a valid cache row")


def nonempty_csa_c4_out_loc(c4_out_loc, c4_kv_cache, close_group):
    """Keep non-closure continuation inputs non-empty for PyRun graph checks."""
    if not close_group and c4_out_loc.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=c4_kv_cache.device)
    return c4_out_loc


def launch_csa_short_continuation(
    residual,
    cos,
    sin,
    compress_cos,
    compress_sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
    compress_state_cache,
    compress_state_out_loc,
    compress_state_read_loc,
    index_state_cache,
    index_state_out_loc,
    index_state_read_loc,
    c4_kv_cache,
    index_k_cache,
    index_scale_cache,
    c4_page_table,
    c4_out_loc,
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
    index_kv_weight,
    index_kv_weight_scale,
    index_gate_weight,
    index_gate_weight_scale,
    index_position_bias,
    index_norm_weight,
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    *,
    positions,
    tp_layout,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    core_num=None,
):
    """Launch one ratio-4 continuation token on the configured grid."""
    seq_len = residual.shape[1] if residual.ndim == 4 else 0
    if seq_len != 1:
        raise ValueError(f"CSA continuation requires S=1, got S={seq_len}")
    tp_layout = as_tp_layout(tp_layout)
    if tp_layout is not SWA_TP1_LAYOUT:
        raise ValueError(f"CSA continuation requires tp_layout={SWA_TP1_LAYOUT!r}")
    if positions is None or positions.ndim != 1 or positions.numel() != 1:
        raise ValueError("CSA continuation requires one absolute position")
    position = int(positions[0].item())
    if position < 3:
        raise ValueError("CSA continuation requires position >= 3")
    if source_chunk_size <= 0:
        raise ValueError("CSA continuation source_chunk_size must be positive")
    close_group = position % 4 == 3
    if compress_position_bias.ndim != 2 or tuple(compress_position_bias.shape) != (
        4,
        compress_kv_weight.shape[0],
    ):
        raise ValueError("CSA main APE must have shape [4, 2D]")
    if index_position_bias.ndim != 2 or tuple(index_position_bias.shape) != (
        4,
        index_kv_weight.shape[0],
    ):
        raise ValueError("CSA Indexer APE must have shape [4, 2D]")
    if tuple(compress_norm_weight.shape) != (compress_kv_weight.shape[0] // 2,):
        raise ValueError("CSA main compressor norm must have shape [D]")
    if tuple(index_norm_weight.shape) != (index_kv_weight.shape[0] // 2,):
        raise ValueError("CSA Indexer compressor norm must have shape [D]")
    expected_compress_angles = (1, 1, min(rotary_dim, index_norm_weight.shape[0]) // 2)
    if tuple(compress_cos.shape) != expected_compress_angles or tuple(compress_sin.shape) != expected_compress_angles:
        raise ValueError("CSA compressed RoPE tables must contain the completed group's start position")
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
        phase=AttentionPhase.DECODE,
        tp_layout=tp_layout,
        window=window,
        rotary_dim=rotary_dim,
        core_num=core_num,
    )
    validate_csa_short_state(
        compress_state_cache,
        compress_state_out_loc,
        compress_kv_weight,
        compress_gate_weight,
        residual.shape[-1],
        seq_len,
        "CSA main",
    )
    validate_csa_short_state(
        index_state_cache,
        index_state_out_loc,
        index_kv_weight,
        index_gate_weight,
        residual.shape[-1],
        seq_len,
        "CSA indexer",
    )
    for name, locations, cache in (
        ("CSA main", compress_state_read_loc, compress_state_cache),
        ("CSA indexer", index_state_read_loc, index_state_cache),
    ):
        if locations.ndim != 1 or locations.numel() != 8 or locations.dtype != torch.int32:
            raise ValueError(f"{name} closure locations must contain 8 INT32 rows")
        cache_rows = cache.shape[0] * cache.shape[1]
        if int(locations.min()) < 0 or int(locations.max()) >= cache_rows:
            raise ValueError(f"{name} closure locations must index valid state rows")
    validate_csa_short_continuation_cache(
        c4_kv_cache,
        index_k_cache,
        index_scale_cache,
        c4_out_loc,
        close_group,
    )
    # The kernel never reads this inert slot unless the C4 group closes.
    c4_out_loc = nonempty_csa_c4_out_loc(c4_out_loc, c4_kv_cache, close_group)
    if c4_page_table.ndim != 1 or not c4_page_table.numel() or c4_page_table.dtype != torch.int32:
        raise ValueError("CSA continuation C4 page table must be a non-empty INT32 vector")
    index_capacity = c4_page_table.numel() * index_k_cache.shape[1]
    index_topk = min(512, index_capacity)
    index_dim = index_k_cache.shape[-1]
    if index_q_weight.ndim != 2 or index_q_weight.shape[0] % index_dim:
        raise ValueError("CSA continuation Indexer query weight has an incompatible shape")
    index_heads = index_q_weight.shape[0] // index_dim
    if tuple(index_weight.shape) != (index_heads, residual.shape[-1]):
        raise ValueError("CSA continuation Indexer gate weight has an incompatible shape")
    ax.init(chip_num=1, core_num=core_num)
    attn_csa_short_continuation[core_num](
        residual,
        cos,
        sin,
        positions,
        compress_cos,
        compress_sin,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        swa_out_loc,
        compress_state_cache,
        compress_state_out_loc,
        compress_state_read_loc,
        index_state_cache,
        index_state_out_loc,
        index_state_read_loc,
        c4_kv_cache,
        index_k_cache,
        index_scale_cache,
        c4_page_table,
        c4_out_loc,
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
        index_kv_weight,
        index_kv_weight_scale,
        index_gate_weight,
        index_gate_weight_scale,
        index_position_bias,
        index_norm_weight,
        index_q_weight,
        index_q_weight_scale,
        index_weight,
        index_weight_scale,
        index_topk,
        close_group,
        window,
        source_chunk_size,
        rotary_dim,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    return branch_partial, residual_path, kv_out


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_csa_prefill(
    residual,
    cos,
    sin,
    positions_casual,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c4_kv_cache,
    index_k_cache,
    c4_page_table,
    c4_out_loc,
    compress_previous_content,
    compress_previous_gate,
    index_previous_content,
    index_previous_gate,
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
    index_kv_weight,
    index_kv_weight_scale,
    index_gate_weight,
    index_gate_weight_scale,
    index_position_bias,
    index_norm_weight,
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    index_topk=512,
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record one chip's fused Lightning Indexer and CSA contribution."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    assert core_num in (1, 4)
    assert batch == 1 and streams == 4
    assert seq_len >= 8 * core_num and seq_len % (4 * core_num) == 0
    assert window > core_num
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
    compressed_blocks = seq_len // 4
    # The compressors are query-parallel, but the paged caches retain a
    # single-writer contract. Publish disjoint local results through SHOCM so
    # every current C4 row is visible to core 0.
    # compressed_shared: [B,N4,Dh], prefix.normalized.dtype SHARED_OCM.
    compressed_shared = ax.dsl.create_tensor(
        (batch, compressed_blocks, compress_norm_weight.shape[0]),
        prefix.normalized.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_compressed",
    )
    # index_k_shared: [B,N4,Di], prefix.normalized.dtype SHARED_OCM.
    index_k_shared = ax.dsl.create_tensor(
        (batch, compressed_blocks, index_norm_weight.shape[0]),
        prefix.normalized.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_index_k",
    )
    # compress_next_content: compress_previous_content.shape/dtype SHARED_OCM.
    compress_next_content = ax.dsl.create_tensor(
        compress_previous_content.shape,
        compress_previous_content.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_compress_next_content",
    )
    # compress_next_gate: compress_previous_gate.shape/dtype SHARED_OCM.
    compress_next_gate = ax.dsl.create_tensor(
        compress_previous_gate.shape,
        compress_previous_gate.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_compress_next_gate",
    )
    # index_next_content: index_previous_content.shape/dtype SHARED_OCM.
    index_next_content = ax.dsl.create_tensor(
        index_previous_content.shape,
        index_previous_content.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_index_next_content",
    )
    # index_next_gate: index_previous_gate.shape/dtype SHARED_OCM.
    index_next_gate = ax.dsl.create_tensor(
        index_previous_gate.shape,
        index_previous_gate.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_index_next_gate",
    )
    static_index_source = None
    if c4_page_table.shape[0] == 1:
        # static_index_source: index_k_cache.shape/dtype SHARED_OCM.
        static_index_source = ax.dsl.create_tensor(
            index_k_cache.shape,
            index_k_cache.dtype,
            ax.MemoryType.SHARED_OCM,
            name="csa_static_index_source",
        )
    # compressed: [B,N4_core,Dh], BF16.
    compressed, compressed_block_slice = stride4_compressor(
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
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        name_prefix="csa",
        previous_content_state=compress_previous_content,
        previous_gate_state=compress_previous_gate,
        next_content_state=compress_next_content,
        next_gate_state=compress_next_gate,
    )
    # index_k: [B,N4_core,Di], BF16.
    index_k, index_block_slice = stride4_compressor(
        prefix.normalized,
        index_kv_weight,
        index_kv_weight_scale,
        index_gate_weight,
        index_gate_weight_scale,
        index_position_bias,
        index_norm_weight,
        prefix.cos,
        prefix.sin,
        meta,
        rotary_dim=min(rotary_dim, index_norm_weight.shape[0]),
        rms_eps=rms_eps,
        name_prefix="csa_index",
        previous_content_state=index_previous_content,
        previous_gate_state=index_previous_gate,
        next_content_state=index_next_content,
        next_gate_state=index_next_gate,
    )
    assert compressed_block_slice == index_block_slice
    # compressed_shared local slice: [B,N4_core,Dh], compressed.dtype.
    ax.dsl.move(compressed, r=compressed_shared[:, compressed_block_slice, :])
    # index_k_shared local slice: [B,N4_core,Di], index_k.dtype.
    ax.dsl.move(index_k, r=index_k_shared[:, index_block_slice, :])
    # Synchronization only; no tensor output.
    axd.sync_all()
    if core_id == 0:
        # c4_write_indices: [N4], INT32 physical cache entries.
        c4_write_indices = ax.dsl.load(c4_out_loc)
        if static_index_source is not None:
            # Preserve any earlier rows, then overlay this chunk before the
            # other cores consume the one-page static Indexer source.
            # static_index_source: index_k_cache.shape/dtype.
            ax.dsl.load(index_k_cache, r=static_index_source)
            # static_index_source selected rows: [N4,Di], index_k_shared.dtype.
            ax.dsl.select_scatter(
                index_k_shared.reshape(-1, index_k_shared.shape[-1]),
                c4_write_indices,
                dim=0,
                r=static_index_source.reshape(-1, index_k_shared.shape[-1]),
            )
        # c4_kv_cache: [P4,64,1,Dh], BF16 with all current C4 rows persisted.
        ax.dsl.select_scatter(
            compressed_shared.reshape(-1, compressed_shared.shape[-1]),
            c4_write_indices,
            dim=0,
            r=c4_kv_cache.reshape(-1, compressed_shared.shape[-1]),
        )
        # index_k_cache: [P4,64,1,Di], BF16 with all current index rows persisted.
        ax.dsl.select_scatter(
            index_k_shared.reshape(-1, index_k_shared.shape[-1]),
            c4_write_indices,
            dim=0,
            r=index_k_cache.reshape(-1, index_k_shared.shape[-1]),
        )
        # compress_previous_content: persistent state shape/dtype.
        ax.dsl.store(compress_next_content, r=compress_previous_content)
        # compress_previous_gate: persistent state shape/dtype.
        ax.dsl.store(compress_next_gate, r=compress_previous_gate)
        # index_previous_content: persistent state shape/dtype.
        ax.dsl.store(index_next_content, r=index_previous_content)
        # index_previous_gate: persistent state shape/dtype.
        ax.dsl.store(index_next_gate, r=index_previous_gate)
    # Synchronization only; no tensor output.
    axd.sync_all()
    # context: [B,S,Hq_chip,Dh] SHARED_OCM, BF16.
    context = csa_attention_paged(
        prefix,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c4_kv_cache,
        static_index_source if static_index_source is not None else index_k_cache,
        c4_page_table,
        positions_casual,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        index_q_weight,
        index_q_weight_scale,
        index_weight,
        index_weight_scale,
        meta,
        index_topk=index_topk,
        window=window,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    heads_per_core = context.shape[2] // core_num
    head_start = core_id * heads_per_core
    local_context = context[:, :, head_start : head_start + heads_per_core, :]
    # branch_partial: [B,S,4,D], BF16.
    project_attention_output(
        prefix,
        local_context,
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
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_csa_decode(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c4_kv_cache,
    index_k_cache,
    c4_page_table,
    positions_casual,
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
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    index_topk=512,
    window=128,
    compress_rate=4,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record one CSA decode contribution for a runtime C4 context length."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    assert window == 128 and compress_rate == 4 and index_topk == 512
    c4_page_size = c4_kv_cache.shape[1]
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

    # With one query token there is no query-axis work to split. Each head
    # owner runs the bounded Indexer locally, avoiding collectives inside the
    # runtime source-block range. The four copies are identical and consume one
    # bounded C4 source block at a time.
    decode_meta = KernelMeta(0, 1, batch, seq_len, hidden)
    physical_top_ids, topk_lengths = csa_lightning_topk(
        prefix,
        index_k_cache,
        c4_page_table,
        positions_casual,
        index_q_weight,
        index_q_weight_scale,
        index_weight,
        index_weight_scale,
        decode_meta,
        index_topk=index_topk,
        rotary_dim=rotary_dim,
    )

    # Core 0 owns the S=1 KV row. Publish it before each head owner builds a
    # private right-aligned SWA tile and padding mask.
    # Synchronization only; no tensor output.
    axd.sync_all()
    swa_source, swa_mask = paged_swa_source(
        prefix,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
        name_prefix="csa_decode_swa",
        mask_query_slice=slice(0, seq_len),
        source_alignment=window,
        cooperative=False,
    )
    head_dim = prefix.kv.shape[-1]
    source_len = index_topk + window
    # source: [K_top+W,Dh], prefix.kv.dtype private OCM.
    source = ax.dsl.create_tensor(
        (source_len, head_dim),
        prefix.kv.dtype,
        ax.MemoryType.OCM,
        name="csa_decode_source",
    )
    ax.dsl.index_select(
        c4_kv_cache.reshape(-1, head_dim),
        physical_top_ids.reshape(index_topk),
        dim=0,
        r=source[:index_topk, :],
    )
    # source SWA slice: [W,Dh], swa_source.dtype.
    ax.dsl.move(swa_source.reshape(window, head_dim), r=source[index_topk:, :])

    # zero_i32: [1], INT32.
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    one_i32 = constant(np.array([1], dtype=np.int32), "csa_decode_one")
    c4_offsets = constant(np.arange(index_topk, dtype=np.int32), "csa_decode_c4_offsets")
    # remaining: [K_top], INT32.
    remaining = ax.dsl.sub(topk_lengths.reshape(1), c4_offsets, output_dtype=ax.int32)
    active = positive_i32(remaining, zero_i32, one_i32)
    # inactive: [K_top], INT32.
    inactive = ax.dsl.sub(one_i32, active, output_dtype=ax.int32)
    # negative_bias: [1], FP32.
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)
    # c4_mask: [K_top], FP32.
    c4_mask = ax.dsl.mul(inactive, negative_bias, output_dtype=ax.fp32)
    # source_mask: [K_top+W], FP32.
    source_mask = ax.dsl.concat(c4_mask, swa_mask.reshape(window), dim=0)
    heads_per_core = query.shape[2]
    sink_slice = sinks_chip[core_id * heads_per_core : (core_id + 1) * heads_per_core]
    context = decode_static_attention(
        query,
        source,
        sink_slice,
        source_mask,
        name_prefix="csa_decode",
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
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_csa_projection_prefill(
    residual,
    cos,
    sin,
    c4_kv_cache,
    index_k_cache,
    c4_out_loc,
    compress_previous_content,
    compress_previous_gate,
    index_previous_content,
    index_previous_gate,
    normalized_out,
    q_residual_out,
    post_out,
    residual_path_out,
    kv_out,
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
    index_kv_weight,
    index_kv_weight_scale,
    index_gate_weight,
    index_gate_weight_scale,
    index_position_bias,
    index_norm_weight,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Produce common tensors and persist both current C4 projection streams."""
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
    # normalized_out: [B,S,D], residual.dtype. Only CSA exports it for the Indexer.
    ax.dsl.store(prefix.normalized[:, token_slice, :], r=normalized_out[:, token_slice, :])
    compressed_blocks = seq_len // 4
    # compressed_shared: [B,S/4,Dh], prefix.normalized.dtype SHOCM.
    compressed_shared = ax.dsl.create_tensor(
        (batch, compressed_blocks, compress_norm_weight.shape[0]),
        prefix.normalized.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_csa_compressed",
    )
    # index_k_shared: [B,S/4,Di], prefix.normalized.dtype SHOCM.
    index_k_shared = ax.dsl.create_tensor(
        (batch, compressed_blocks, index_norm_weight.shape[0]),
        prefix.normalized.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_csa_index_k",
    )
    # compress_next_content: compress_previous_content.shape/dtype SHOCM.
    compress_next_content = ax.dsl.create_tensor(
        compress_previous_content.shape,
        compress_previous_content.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_csa_compress_next_content",
    )
    # compress_next_gate: compress_previous_gate.shape/dtype SHOCM.
    compress_next_gate = ax.dsl.create_tensor(
        compress_previous_gate.shape,
        compress_previous_gate.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_csa_compress_next_gate",
    )
    # index_next_content: index_previous_content.shape/dtype SHOCM.
    index_next_content = ax.dsl.create_tensor(
        index_previous_content.shape,
        index_previous_content.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_csa_index_next_content",
    )
    # index_next_gate: index_previous_gate.shape/dtype SHOCM.
    index_next_gate = ax.dsl.create_tensor(
        index_previous_gate.shape,
        index_previous_gate.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_csa_index_next_gate",
    )
    compressed, block_slice = stride4_compressor(
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
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        name_prefix="split_csa",
        previous_content_state=compress_previous_content,
        previous_gate_state=compress_previous_gate,
        next_content_state=compress_next_content,
        next_gate_state=compress_next_gate,
    )
    index_k, _ = stride4_compressor(
        prefix.normalized,
        index_kv_weight,
        index_kv_weight_scale,
        index_gate_weight,
        index_gate_weight_scale,
        index_position_bias,
        index_norm_weight,
        prefix.cos,
        prefix.sin,
        meta,
        rotary_dim=min(rotary_dim, index_norm_weight.shape[0]),
        rms_eps=rms_eps,
        name_prefix="split_csa_index",
        previous_content_state=index_previous_content,
        previous_gate_state=index_previous_gate,
        next_content_state=index_next_content,
        next_gate_state=index_next_gate,
    )
    # Both stride-4 projections use the same token partition from ``meta``.
    # compressed_shared slice: [B,N4_core,Dh], compressed.dtype.
    ax.dsl.move(compressed, r=compressed_shared[:, block_slice, :])
    # index_k_shared slice: [B,N4_core,Di], index_k.dtype.
    ax.dsl.move(index_k, r=index_k_shared[:, block_slice, :])
    # Synchronization only; publishes all main/index C4 row shards.
    axd.sync_all()
    if meta.core_id == 0:
        # c4_write_indices: [S/4], INT32.
        c4_write_indices = ax.dsl.load(c4_out_loc)
        # c4_kv_cache selected rows: [S/4,Dh], compressed_shared.dtype.
        ax.dsl.select_scatter(
            compressed_shared.reshape(-1, compressed_shared.shape[-1]),
            c4_write_indices,
            dim=0,
            r=c4_kv_cache.reshape(-1, compressed_shared.shape[-1]),
        )
        # index_k_cache selected rows: [S/4,Di], index_k_shared.dtype.
        ax.dsl.select_scatter(
            index_k_shared.reshape(-1, index_k_shared.shape[-1]),
            c4_write_indices,
            dim=0,
            r=index_k_cache.reshape(-1, index_k_shared.shape[-1]),
        )
        # compress_previous_content: [B,4,Dh], compress_next_content.dtype.
        ax.dsl.store(compress_next_content, r=compress_previous_content)
        # compress_previous_gate: [B,4,Dh], compress_next_gate.dtype.
        ax.dsl.store(compress_next_gate, r=compress_previous_gate)
        # index_previous_content: [B,4,Di], index_next_content.dtype.
        ax.dsl.store(index_next_content, r=index_previous_content)
        # index_previous_gate: [B,4,Di], index_next_gate.dtype.
        ax.dsl.store(index_next_gate, r=index_previous_gate)
    return (
        normalized_out,
        q_residual_out,
        post_out,
        residual_path_out,
        kv_out,
    )


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_csa_core_prefill(
    q_residual,
    kv,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c4_kv_cache,
    physical_top_ids,
    topk_lengths,
    local_output,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    index_topk=512,
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
):
    """Run CSA from standalone Lightning Indexer output and store `[B,S,D]`."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, hidden = local_output.shape
    assert core_num in (1, 4) and batch == 1 and seq_len >= core_num
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)
    prefix = load_attention_prefix(q_residual, kv, cos, sin, meta)
    context = csa_attention_from_topk(
        prefix,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c4_kv_cache,
        physical_top_ids,
        topk_lengths,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        index_topk=index_topk,
        window=window,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    heads_per_core = context.shape[2] // meta.core_num
    head_start = meta.core_id * heads_per_core
    project_attention_local_output(
        prefix,
        context[:, :, head_start : head_start + heads_per_core, :],
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        local_output,
        meta,
        rotary_dim=rotary_dim,
    )
    return local_output


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def lightning_indexer(
    normalized,
    q_residual,
    cos,
    sin,
    positions_casual,
    index_k_cache,
    c4_page_table,
    physical_top_ids_out,
    topk_lengths_out,
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    index_topk=512,
    rotary_dim=64,
):
    """Run only the four-core paged Lightning Indexer used by CSA prefill."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, hidden = normalized.shape
    index_dim = index_k_cache.shape[-1]
    index_heads = index_q_weight.shape[0] // index_dim
    assert core_num in (1, 4)
    assert batch == 1 and seq_len >= core_num
    assert q_residual.shape[:2] == (batch, seq_len)
    assert cos.shape[:2] == sin.shape[:2] == (batch, seq_len)
    assert positions_casual.shape == (seq_len,)
    assert index_k_cache.shape[1:3] == (64, 1)
    assert c4_page_table.shape == (index_k_cache.shape[0],)
    assert index_q_weight.shape == (index_heads * index_dim, q_residual.shape[-1])
    assert index_weight.shape == (index_heads, hidden)
    assert physical_top_ids_out.shape == (batch, seq_len, index_topk)
    assert topk_lengths_out.shape == (batch, seq_len)
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)
    token_slice = slice_tokens(core_id, core_num, seq_len)
    token_count = token_slice.stop - token_slice.start

    # Match the CSA boundary: the prepared prefix tensors are chip-visible,
    # while every core projects and scans only its disjoint query slice.
    # normalized_shared: [B,S,D], normalized.dtype SHARED_OCM.
    normalized_shared = ax.dsl.create_tensor(
        normalized.shape,
        normalized.dtype,
        ax.MemoryType.SHARED_OCM,
        name="indexer_normalized",
    )
    # q_residual_shared: [B,S,Rq], q_residual.dtype SHARED_OCM.
    q_residual_shared = ax.dsl.create_tensor(
        q_residual.shape,
        q_residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="indexer_q_residual",
    )
    # cos_shared/sin_shared: [B,S,Dr/2], cos/sin.dtype SHARED_OCM.
    cos_shared = ax.dsl.create_tensor(
        cos.shape,
        cos.dtype,
        ax.MemoryType.SHARED_OCM,
        name="indexer_cos",
    )
    sin_shared = ax.dsl.create_tensor(
        sin.shape,
        sin.dtype,
        ax.MemoryType.SHARED_OCM,
        name="indexer_sin",
    )
    # Prepared prefix query slice: [B,T_core,*], matching input dtypes.
    ax.dsl.load(normalized[:, token_slice, :], r=normalized_shared[:, token_slice, :])
    ax.dsl.load(q_residual[:, token_slice, :], r=q_residual_shared[:, token_slice, :])
    ax.dsl.load(cos[:, token_slice, :], r=cos_shared[:, token_slice, :])
    ax.dsl.load(sin[:, token_slice, :], r=sin_shared[:, token_slice, :])
    # Synchronization only; publishes the disjoint prepared-prefix slices.
    axd.sync_all()

    prefix = AttentionPrefix(
        state=None,
        normalized=normalized_shared,
        q_residual=q_residual_shared,
        kv=None,
        cos=cos_shared,
        sin=sin_shared,
    )
    physical_top_ids, topk_lengths = csa_lightning_topk(
        prefix,
        index_k_cache,
        c4_page_table,
        positions_casual,
        index_q_weight,
        index_q_weight_scale,
        index_weight,
        index_weight_scale,
        meta,
        index_topk=index_topk,
        rotary_dim=rotary_dim,
    )
    # physical_top_ids_out slice: [B,T_core,K_top], INT32 DDR.
    ax.dsl.store(
        physical_top_ids.reshape(batch, token_count, index_topk),
        r=physical_top_ids_out[:, token_slice, :],
    )
    # topk_lengths_out slice: [B,T_core], INT32 DDR.
    ax.dsl.store(
        topk_lengths.reshape(batch, token_count),
        r=topk_lengths_out[:, token_slice],
    )
    return physical_top_ids_out, topk_lengths_out
