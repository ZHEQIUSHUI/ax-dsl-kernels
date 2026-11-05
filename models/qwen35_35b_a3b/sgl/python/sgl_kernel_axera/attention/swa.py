"""DeepSeek-V4 sliding-window attention (SWA) kernels and public helpers.

The public helpers preserve the compact validation API while the local builders
below implement the paged, four-core hardware path locally.
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
    TensorParallelLayout,
    attention_prefix,
    as_phase,
    as_tp_layout,
    configured_run_mode,
    env_flag,
    paged_swa_source,
    project_attention_local_output,
    project_attention_output,
    project_attention_query,
    kernel_cache_enabled,
    kernel_cache_path,
    token_slice as slice_tokens,
)
from .attn_utils.swa_helper import swa_attention_paged
from .attn_utils.decode import decode_static_attention


SWA_TP_LAYOUT = TensorParallelLayout.TP2_ONE_CHIP
SWA_TP1_LAYOUT = TensorParallelLayout.TP1


def run_mode() -> ax.RunMode:
    return configured_run_mode()


def write_swa_cache(prefix, swa_kv_cache, swa_out_loc, core_id):
    """Persist current KV rows after every core has consumed the old ring."""
    axd.sync_all()
    if core_id == 0:
        head_dim = prefix.kv.shape[-1]
        ax.dsl.select_scatter(
            prefix.kv.reshape(-1, head_dim),
            ax.dsl.load(swa_out_loc),
            dim=0,
            r=swa_kv_cache.reshape(-1, head_dim),
        )
    axd.sync_all()


def window_indices(window_size: int, batch_size: int, seq_len: int, start_pos: int = 0) -> torch.Tensor:
    """Return causal logical window indices with -1 for unavailable slots."""
    if window_size <= 0 or batch_size <= 0 or seq_len <= 0 or start_pos < 0:
        raise ValueError("window_size, batch_size, and seq_len must be positive; start_pos must be non-negative")
    if start_pos > 0:
        slot = start_pos % window_size
        if start_pos >= window_size - 1:
            row = torch.cat((torch.arange(slot + 1, window_size), torch.arange(0, slot + 1)), dim=0)
        else:
            row = torch.nn.functional.pad(torch.arange(slot + 1), (0, window_size - slot - 1), value=-1)
        return row.to(torch.int32).reshape(1, 1, window_size).expand(batch_size, seq_len, window_size).clone()
    base = torch.arange(seq_len, dtype=torch.int32).reshape(seq_len, 1)
    columns = torch.arange(min(seq_len, window_size), dtype=torch.int32)
    result = (base - window_size + 1).clamp_min(0) + columns
    result = torch.where(result > base, torch.full_like(result, -1), result)
    if result.shape[-1] < window_size:
        result = torch.nn.functional.pad(result, (0, window_size - result.shape[-1]), value=-1)
    return result.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def swa_attention_value(q, kv_candidates, additive_mask, attn_sink, *, softmax_scale: float, output_dtype):
    """Record sink-normalized attention over gathered candidates."""
    batch, seq_len, heads, head_dim = q.shape
    candidate_count = kv_candidates.shape[2]
    q_value = q.reshape(batch * seq_len, heads, head_dim)
    kv_value = kv_candidates.reshape(batch * seq_len, candidate_count, head_dim)
    scores = ax.dsl.matmul(q_value, kv_value, transposed=True, output_dtype=ax.fp32)
    # This is a scalar graph attribute, not a runtime tensor.  ``from_numpy``
    # would emit a DDR->OCM load for every launch and is reported by CRUN as a
    # useless job when the scalar is folded into the graph.
    scores = ax.dsl.mul(
        scores,
        ax.dsl.full((1,), ax.fp32, fill_value=float(softmax_scale)),
        output_dtype=ax.fp32,
    )
    scores = ax.dsl.add(scores, additive_mask.reshape(batch * seq_len, 1, candidate_count), output_dtype=ax.fp32)
    sink_scores = ax.dsl.expand(attn_sink.reshape(1, heads, 1), shape=(batch * seq_len, heads, 1))
    scores = ax.dsl.concat(scores, sink_scores, dim=-1)
    score_count = candidate_count + 1
    aligned_count = (score_count + 7) // 8 * 8
    if aligned_count != score_count:
        scores = ax.dsl.concat(
            scores,
            ax.dsl.full((batch * seq_len, heads, aligned_count - score_count), ax.fp32, fill_value=-float("inf")),
            dim=-1,
        )
        score_count = aligned_count
    scores = scores.reshape(batch * seq_len * heads, score_count)
    max_score = ax.dsl.max(scores, dim=-1, keepdim=True)
    shifted = ax.dsl.sub(scores, max_score, output_dtype=ax.fp32, r=scores)
    denominator_weights = ax.dsl.exp(shifted, output_dtype=ax.fp32)
    numerator_weights = ax.dsl.cast(denominator_weights, dtype=output_dtype)
    denominator = ax.dsl.sum(denominator_weights, dim=-1, output_dtype=ax.fp32)
    numerator = ax.dsl.sum(
        ax.dsl.mul(
            numerator_weights[:, :candidate_count].reshape(batch * seq_len, heads, candidate_count, 1),
            kv_value.reshape(batch * seq_len, 1, candidate_count, head_dim),
            output_dtype=ax.fp32,
        ),
        dim=2,
        output_dtype=ax.fp32,
    )
    return ax.dsl.div(numerator, denominator.reshape(batch * seq_len, heads, 1), output_dtype=output_dtype).reshape(
        batch, seq_len, heads, head_dim
    )


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
)
def swa_attention_kernel(q, kv_candidates, additive_mask, attn_sink, output, softmax_scale):
    """Record sink-normalized attention over pre-gathered window candidates."""
    batch, seq_len, heads, head_dim = q.shape
    candidate_count = kv_candidates.shape[2]
    row_count = batch * seq_len
    row_slice = linear_program_slice(row_count)
    local_rows = row_slice.stop - row_slice.start
    q_rows = q.reshape(row_count, heads, head_dim)
    candidate_rows = kv_candidates.reshape(row_count, candidate_count, head_dim)
    mask_rows = additive_mask.reshape(row_count, candidate_count)
    output_rows = output.reshape(row_count, heads, head_dim)
    result = swa_attention_value(
        ax.dsl.load(q_rows[row_slice, :, :]).reshape(1, local_rows, heads, head_dim),
        ax.dsl.load(candidate_rows[row_slice, :, :]).reshape(1, local_rows, candidate_count, head_dim),
        ax.dsl.load(mask_rows[row_slice, :]).reshape(1, local_rows, candidate_count),
        ax.dsl.load(attn_sink),
        softmax_scale=softmax_scale,
        output_dtype=output.dtype,
    )
    ax.dsl.store(result.reshape(local_rows, heads, head_dim), r=output_rows[row_slice, :, :])
    return output


def swa_attention(
    q: torch.Tensor,
    kv_candidates: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    core_num: int = 1,
) -> torch.Tensor:
    """Run SWA attention on [B,S,W,D] candidates."""
    if q.ndim != 4 or kv_candidates.ndim != 4 or indices.ndim != 3:
        raise ValueError("q must be [B,S,H,D], kv_candidates [B,S,W,D], indices [B,S,W]")
    batch, seq_len, heads, head_dim = q.shape
    if tuple(kv_candidates.shape[:2]) != (batch, seq_len) or tuple(indices.shape) != tuple(kv_candidates.shape[:3]):
        raise ValueError("SWA query, candidate, and index shapes are incompatible")
    if attn_sink.shape != (heads,):
        raise ValueError(f"attn_sink must have shape ({heads},), got {tuple(attn_sink.shape)}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be INT32 or INT64")
    output = output if output is not None else torch.empty_like(q)
    if tuple(output.shape) != tuple(q.shape):
        raise ValueError("output must match q shape")
    valid = indices >= 0
    additive_mask = torch.where(
        valid,
        torch.zeros((), dtype=torch.float32, device=indices.device),
        torch.full((), -float("inf"), dtype=torch.float32, device=indices.device),
    )
    scale = float(softmax_scale) if softmax_scale is not None else head_dim ** -0.5
    validate_core_num(core_num, batch * seq_len, "SWA query rows")
    ax.init(chip_num=1, core_num=core_num)
    swa_attention_kernel[core_num](q, kv_candidates, additive_mask, attn_sink, output, scale)
    return output


def gather_window_candidates(kv_cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather candidates and zero invalid -1 slots."""
    if kv_cache.ndim != 3 or indices.ndim != 3 or kv_cache.shape[0] != indices.shape[0]:
        raise ValueError("kv_cache must be [B,L,D] and indices must be [B,S,W]")
    safe = indices.clamp_min(0).long()
    gathered = torch.gather(
        kv_cache[:, None, :, :].expand(-1, indices.shape[1], -1, -1),
        2,
        safe[..., None].expand(*safe.shape, kv_cache.shape[-1]),
    )
    return torch.where((indices >= 0)[..., None], gathered, torch.zeros_like(gathered))


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_swa_prefill(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
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
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    combine_residual=False,
):
    """Record one chip's SWA contribution from SGLang paged metadata."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)

    # prefix.state.post/collapsed: [B,S,4] FP32 / [B,S,D] BF16;
    # prefix.q_residual/kv: [B,S,Rq] / [B,S,Dh] BF16.
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
    # query/context: [B,S,Hq_core,Dh], BF16 private OCM. Each core owns its
    # heads for all query tokens, avoiding a full-chip query in SHARED_OCM.
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
    # context: [B,S,Hq_core,Dh], BF16 in the query allocation.
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
    # Outputs: branch_partial [B,S,4,D] BF16, residual_path [B,S,4,D] FP32,
    # and kv_out [B,S,Dh] BF16; production shapes [1,128,4,4096],
    # [1,128,4,4096], and [1,128,512].
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


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_swa_decode(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
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
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    combine_residual=False,
):
    """Record one SWA decode contribution for a runtime paged-window length."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    meta = KernelMeta(core_id, core_num, batch, seq_len, hidden)

    # prefix.q_residual/kv: [1,1,Rq]/[1,1,Dh] BF16 SHOCM.
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
    head_dim = prefix.kv.shape[-1]
    # Core 0 owns the S=1 KV row. Publish it before every head owner builds its
    # private right-aligned runtime window and early-context padding mask.
    # Synchronization only; no tensor output.
    axd.sync_all()
    source, source_mask = paged_swa_source(
        prefix,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
        name_prefix="swa_decode",
        mask_query_slice=slice(0, seq_len),
        source_alignment=window,
        cooperative=False,
    )

    heads_per_core = query.shape[2]
    sink_slice = sinks_chip[core_id * heads_per_core : (core_id + 1) * heads_per_core]
    # context: [1,1,8,Dh] BF16 private OCM.
    context = decode_static_attention(
        query,
        source.reshape(window, head_dim),
        sink_slice,
        source_mask.reshape(window),
        name_prefix="swa_decode",
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


def validate_swa_launch(
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
    *,
    phase,
    tp_layout,
    window,
    rotary_dim,
    core_num=None,
):
    """Validate the SWA launch contract for a one- or four-core grid."""
    core_num = attention_core_num(core_num)
    phase = as_phase(phase)
    tp_layout = as_tp_layout(tp_layout)
    if residual.ndim != 4:
        raise ValueError("SWA residual must have shape [B, S, 4, D]")
    batch, seq_len, streams, hidden = residual.shape
    if batch != 1 or streams != 4:
        raise ValueError("SWA kernels require residual shape [1, S, 4, D], " f"got {tuple(residual.shape)}")
    if phase is AttentionPhase.DECODE and seq_len != 1:
        raise ValueError(f"SWA decode requires S=1, got S={seq_len}")
    if phase is AttentionPhase.PREFILL and not 0 < seq_len <= window:
        raise ValueError(f"SWA prefill requires query lengths in [1, {window}], " f"got S={seq_len}")
    if window != 128:
        raise ValueError(f"SWA production cache contract requires window=128, got {window}")
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"SWA rotary_dim must be positive and even, got {rotary_dim}")
    expected_angles = (batch, seq_len, rotary_dim // 2)
    if tuple(cos.shape) != expected_angles or tuple(sin.shape) != expected_angles:
        raise ValueError(
            f"SWA cos/sin must both have shape {expected_angles}, " f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if phase is AttentionPhase.PREFILL and (positions is None or positions.numel() != batch * seq_len):
        raise ValueError("SWA prefill positions must contain one absolute position per token")
    if tuple(branch_partial.shape) != tuple(residual.shape):
        raise ValueError("SWA branch_partial must match residual shape")
    if tuple(residual_path.shape) != tuple(residual.shape):
        raise ValueError("SWA residual_path must match residual shape")
    if branch_partial.dtype != residual.dtype:
        raise ValueError("SWA branch_partial must match residual dtype")
    if residual_path.dtype != torch.float32:
        raise ValueError("SWA residual_path must be FP32 for model.py mHC Post semantics")
    if swa_kv_cache.ndim < 2:
        raise ValueError("SWA cache must expose a token axis and head dimension")
    head_dim = swa_kv_cache.shape[-1]
    if tuple(kv_out.shape) != (batch, seq_len, head_dim):
        raise ValueError(f"SWA kv_out must have shape {(batch, seq_len, head_dim)}, got {tuple(kv_out.shape)}")
    if kv_out.dtype != swa_kv_cache.dtype or kv_out.dtype != residual.dtype:
        raise ValueError("SWA residual, cache, and kv_out must share one dtype")
    if swa_page_indices.ndim != 2 or swa_page_indices.shape[0] != batch or swa_page_indices.shape[1] < window - 1:
        raise ValueError(f"SWA page indices must have shape [1, >= {window - 1}], got {tuple(swa_page_indices.shape)}")
    if swa_page_indices.dtype != torch.int32:
        raise ValueError("SWA page indices must be INT32")
    if swa_topk_lengths.ndim != 1 or swa_topk_lengths.shape[0] != batch:
        raise ValueError("SWA top-k lengths must have shape [1]")
    if swa_topk_lengths.dtype != torch.int32:
        raise ValueError("SWA top-k lengths must be INT32")
    if swa_out_loc.ndim != 1 or swa_out_loc.numel() != batch * seq_len:
        raise ValueError("SWA output locations must contain one slot per token")
    if swa_out_loc.dtype != torch.int32:
        raise ValueError("SWA output locations must be INT32")
    cache_tokens = swa_kv_cache.numel() // head_dim
    if swa_out_loc.numel() and (int(swa_out_loc.min()) < 0 or int(swa_out_loc.max()) >= cache_tokens):
        raise ValueError("SWA output locations must index valid cache token slots")

    if q_a_weight.ndim != 2 or q_a_weight.shape[1] != hidden:
        raise ValueError("SWA q_a weight must have shape [Rq, D]")
    if hidden % core_num:
        raise ValueError(f"SWA hidden size must split evenly over {core_num} cores")
    query_rank = q_a_weight.shape[0]
    if query_rank % core_num or query_rank // core_num < 16:
        raise ValueError(f"SWA query rank must split into at least 16 channels per core ({core_num=})")
    if head_dim % core_num or head_dim // core_num < 16:
        raise ValueError(f"SWA head dimension must split into at least 16 channels per core ({core_num=})")
    if rotary_dim > head_dim:
        raise ValueError("SWA rotary dimension cannot exceed the head dimension")
    if q_b_weight_chip.ndim != 2 or q_b_weight_chip.shape[1] != query_rank or q_b_weight_chip.shape[0] % head_dim:
        raise ValueError("SWA chip-local q_b weight has an incompatible shape")
    chip_heads = q_b_weight_chip.shape[0] // head_dim
    if chip_heads % core_num != 0 or tuple(sinks_chip.shape) != (chip_heads,):
        raise ValueError(f"SWA chip-local query heads and sinks must split evenly over {core_num} cores")
    if tuple(kv_weight.shape) != (head_dim, hidden):
        raise ValueError("SWA KV weight must have shape [Dh, D]")
    if o_a_weight_chip.ndim != 3:
        raise ValueError("SWA o_a weight must have shape [G, Ro, Hgroup * Dh]")
    groups = o_a_weight_chip.shape[0]
    expected_groups = 8 if tp_layout is SWA_TP1_LAYOUT else 4
    if groups != expected_groups:
        raise ValueError(f"SWA {tp_layout} requires {expected_groups} output groups, got {groups}")
    if groups < core_num or groups % core_num:
        raise ValueError(f"SWA output groups must split evenly over {core_num} cores")
    output_rank = o_a_weight_chip.shape[1]
    if o_a_weight_chip.shape[2] % head_dim:
        raise ValueError("SWA o_a input width must contain complete attention heads")
    heads_per_group = o_a_weight_chip.shape[2] // head_dim
    if chip_heads != groups * heads_per_group:
        raise ValueError("SWA query heads must match all o_a attention groups")
    if tuple(o_b_weight_chip.shape) != (hidden, groups * output_rank):
        raise ValueError("SWA o_b input width must match all o_a output groups")


def launch_swa_attention(
    residual,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    swa_out_loc,
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
    *,
    phase,
    tp_layout,
    positions=None,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    core_num=None,
):
    """Launch a DSV4 SWA builder on the configured one/four-core grid."""
    phase = as_phase(phase)
    tp_layout = as_tp_layout(tp_layout)
    core_num = attention_core_num(core_num)
    if source_chunk_size <= 0:
        raise ValueError("SWA source_chunk_size must be positive")
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
    ax.init(chip_num=1, core_num=core_num)
    if phase is AttentionPhase.PREFILL:
        attn_swa_prefill[core_num](
            residual,
            cos,
            sin,
            swa_kv_cache,
            swa_page_indices,
            swa_topk_lengths,
            swa_out_loc,
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
            window,
            source_chunk_size,
            rotary_dim,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            tp_layout is SWA_TP1_LAYOUT,
        )
    else:
        attn_swa_decode[core_num](
            residual,
            cos,
            sin,
            swa_kv_cache,
            swa_page_indices,
            swa_topk_lengths,
            swa_out_loc,
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
            window,
            rotary_dim,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            tp_layout is SWA_TP1_LAYOUT,
        )
    return branch_partial, residual_path, kv_out


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_swa_projection_prefill(
    residual,
    cos,
    sin,
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
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Produce the public tensors consumed by split SWA attention and Post-mHC."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    assert streams == 4 and batch == 1 and seq_len >= core_num
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
    return q_residual_out, post_out, residual_path_out, kv_out


@ax.builder(
    run_mode=configured_run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def attn_swa_core_prefill(
    q_residual,
    kv,
    cos,
    sin,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    local_output,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    sinks_chip,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    window=128,
    source_chunk_size=32,
    rotary_dim=64,
    rms_eps=1e-6,
):
    """Run SWA and store the pre-Post TP contribution `[B,S,D]`."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, hidden = local_output.shape
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
