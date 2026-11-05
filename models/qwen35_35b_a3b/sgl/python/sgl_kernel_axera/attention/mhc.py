"""DeepSeek-V4 multi-stream Hyper-Connection (mHC) kernels.

This module owns the reusable Hyper-Connection stages and the distributed
post-reduction entry point.  HCA-specific C128 compression and paged attention
remain in :mod:`sgl_kernel_axera.attention.hca`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import axnn.dsl as ax

from sgl_kernel_axera.attention.parallel_utils import (
    linear_program_slice,
    validate_core_num,
)
from .attn_utils.common import (
    configured_run_mode,
    env_flag,
    kernel_cache_enabled,
    kernel_cache_path,
    token_slice as slice_tokens,
)


def run_mode() -> ax.RunMode:
    return configured_run_mode()


def kernel_weight(value: torch.Tensor) -> torch.Tensor:
    """Use BF16 weights for AX8860 CRUN MatMul when required."""
    if run_mode() is ax.RunMode.CRUN and value.dtype == torch.float32:
        return value.to(torch.bfloat16)
    return value


def scalar(value: float):
    return ax.dsl.from_numpy(np.asarray([value], dtype=np.float32))


def rms_denominator(x, eps: float):
    """Record the FP32 RMS denominator used by mHC gates."""
    inverse_width = ax.dsl.full((1,), ax.fp32, fill_value=1.0 / x.shape[-1])
    square = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    mean = ax.dsl.mul(
        ax.dsl.sum(square, dim=-1, keepdim=True, output_dtype=ax.fp32),
        inverse_width,
        output_dtype=ax.fp32,
    )
    return ax.dsl.sqrt(ax.dsl.add(mean, scalar(eps), output_dtype=ax.fp32), output_dtype=ax.fp32)


def sinkhorn(comb_logits, hc_eps: float, sinkhorn_iters: int):
    """Record the alternating row/column normalization of the comb matrix."""
    comb = ax.dsl.softmax(comb_logits, dim=-1, output_dtype=ax.fp32)
    comb = ax.dsl.add(comb, scalar(hc_eps), output_dtype=ax.fp32)
    comb = ax.dsl.div(
        comb,
        ax.dsl.add(
            ax.dsl.sum(comb, dim=-2, keepdim=True, output_dtype=ax.fp32),
            scalar(hc_eps),
        ),
        output_dtype=ax.fp32,
    )
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = ax.dsl.div(
            comb,
            ax.dsl.add(
                ax.dsl.sum(comb, dim=-1, keepdim=True, output_dtype=ax.fp32),
                scalar(hc_eps),
            ),
            output_dtype=ax.fp32,
        )
        comb = ax.dsl.div(
            comb,
            ax.dsl.add(
                ax.dsl.sum(comb, dim=-2, keepdim=True, output_dtype=ax.fp32),
                scalar(hc_eps),
            ),
            output_dtype=ax.fp32,
        )
    return comb


def pre_hc_value(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    eps: float,
    sinkhorn_iters: int,
    output_dtype,
    normalize_dtype=None,
):
    """Record mHC Pre: gate generation, Sinkhorn comb, and stream collapse."""
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be at least one")
    batch, seq_len, hc_mult, hidden = x.shape
    x_flat = x.reshape(batch * seq_len, hc_mult * hidden)
    # Score-based routing can retain FP32 through the mHC normalized streams;
    # legacy callers keep the original activation dtype by default.
    normalized = ax.dsl.div(
        x_flat,
        rms_denominator(x_flat, eps),
        output_dtype=x.dtype if normalize_dtype is None else normalize_dtype,
    )
    streams = normalized.reshape(batch * seq_len, hc_mult, hidden)
    parts = []
    for stream_id in range(hc_mult):
        parts.append(
            ax.dsl.matmul(
                streams[:, stream_id : stream_id + 1, :].reshape(batch * seq_len, hidden),
                hc_fn[:, stream_id * hidden : (stream_id + 1) * hidden],
                transposed=True,
                output_dtype=ax.fp32,
            )
        )
    mixes = parts[0]
    for part in parts[1:]:
        mixes = ax.dsl.add(mixes, part, output_dtype=ax.fp32)
    pre = ax.dsl.add(
        ax.dsl.sigmoid(
            ax.dsl.add(
                ax.dsl.mul(mixes[:, :hc_mult], hc_scale[:1], output_dtype=ax.fp32),
                hc_base[:hc_mult],
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        ),
        scalar(eps),
        output_dtype=ax.fp32,
    )
    post = ax.dsl.mul(
        ax.dsl.sigmoid(
            ax.dsl.add(
                ax.dsl.mul(mixes[:, hc_mult : 2 * hc_mult], hc_scale[1:2], output_dtype=ax.fp32),
                hc_base[hc_mult : 2 * hc_mult],
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        ),
        scalar(2.0),
        output_dtype=ax.fp32,
    )
    comb_logits = mixes[:, 2 * hc_mult :].reshape(batch * seq_len, hc_mult, hc_mult)
    comb_logits = ax.dsl.add(
        ax.dsl.mul(comb_logits, hc_scale[2:3], output_dtype=ax.fp32),
        hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult),
        output_dtype=ax.fp32,
    )
    comb = sinkhorn(comb_logits, eps, sinkhorn_iters)
    reduced = ax.dsl.sum(
        ax.dsl.mul(pre.reshape(batch, seq_len, hc_mult, 1), x, output_dtype=ax.fp32),
        dim=2,
        output_dtype=output_dtype,
    )
    return (
        reduced,
        post.reshape(batch, seq_len, hc_mult),
        comb.reshape(batch, seq_len, hc_mult, hc_mult),
    )


def post_hc_value(x, residual, post, comb, *, output_dtype):
    """Record full mHC Post from an output, residual streams, and comb matrix."""
    batch, seq_len, hidden = x.shape
    hc_mult = residual.shape[2]
    value = x.reshape(batch, seq_len, 1, hidden)
    residual_value = residual.reshape(batch, seq_len, hc_mult, 1, hidden)
    post_value = post.reshape(batch, seq_len, hc_mult, 1)
    # model.py keeps the Sinkhorn matrix in FP32 through the residual reduction.
    comb_value = comb.reshape(batch, seq_len, hc_mult, hc_mult, 1)
    residual_mix = ax.dsl.sum(
        ax.dsl.mul(comb_value, residual_value, output_dtype=ax.fp32),
        dim=2,
        output_dtype=ax.fp32,
    )
    return ax.dsl.add(
        ax.dsl.mul(post_value, value, output_dtype=ax.fp32),
        residual_mix,
        output_dtype=output_dtype,
    )


def head_hc_value(x, hc_fn, hc_scale, hc_base, *, eps: float, output_dtype):
    """Record the final mHC head reduction from multiple streams."""
    batch, seq_len, hc_mult, hidden = x.shape
    flat = x.reshape(batch * seq_len, hc_mult * hidden)
    normalized = ax.dsl.div(flat, rms_denominator(flat, eps), output_dtype=x.dtype)
    mixes = ax.dsl.matmul(normalized, hc_fn, transposed=True, output_dtype=ax.fp32)
    pre = ax.dsl.add(
        ax.dsl.sigmoid(
            ax.dsl.add(
                ax.dsl.mul(mixes, hc_scale.reshape(1, 1), output_dtype=ax.fp32),
                hc_base.reshape(1, hc_mult),
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        ),
        scalar(eps),
        output_dtype=ax.fp32,
    )
    return ax.dsl.sum(
        ax.dsl.mul(pre.reshape(batch, seq_len, hc_mult, 1), x, output_dtype=ax.fp32),
        dim=2,
        output_dtype=output_dtype,
    )


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
)
def pre_hc_kernel(x, hc_fn, hc_scale, hc_base, reduced, post, comb, eps=1e-6, sinkhorn_iters=20):
    batch, seq_len, hc_mult, hidden = x.shape
    row_count = batch * seq_len
    row_slice = linear_program_slice(row_count)
    local_rows = row_slice.stop - row_slice.start
    reduced_value, post_value, comb_value = pre_hc_value(
        ax.dsl.load(x.reshape(row_count, hc_mult, hidden)[row_slice, :]).reshape(1, local_rows, hc_mult, hidden),
        ax.dsl.load(hc_fn),
        ax.dsl.load(hc_scale),
        ax.dsl.load(hc_base),
        eps=eps,
        sinkhorn_iters=sinkhorn_iters,
        output_dtype=reduced.dtype,
    )
    ax.dsl.store(
        reduced_value.reshape(local_rows, hidden),
        r=reduced.reshape(row_count, hidden)[row_slice, :],
    )
    ax.dsl.store(
        post_value.reshape(local_rows, hc_mult),
        r=post.reshape(row_count, hc_mult)[row_slice, :],
    )
    ax.dsl.store(
        comb_value.reshape(local_rows, hc_mult, hc_mult),
        r=comb.reshape(row_count, hc_mult, hc_mult)[row_slice, :, :],
    )
    return reduced, post, comb


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
)
def post_hc_kernel(x, residual, post, comb, output):
    batch, seq_len, hidden = x.shape
    hc_mult = residual.shape[2]
    row_count = batch * seq_len
    row_slice = linear_program_slice(row_count)
    local_rows = row_slice.stop - row_slice.start
    expanded = post_hc_value(
        ax.dsl.load(x.reshape(row_count, hidden)[row_slice, :]).reshape(1, local_rows, hidden),
        ax.dsl.load(residual.reshape(row_count, hc_mult, hidden)[row_slice, :, :]).reshape(
            1, local_rows, hc_mult, hidden
        ),
        ax.dsl.load(post.reshape(row_count, hc_mult)[row_slice, :]).reshape(1, local_rows, hc_mult),
        ax.dsl.load(comb.reshape(row_count, hc_mult, hc_mult)[row_slice, :, :]).reshape(
            1, local_rows, hc_mult, hc_mult
        ),
        output_dtype=output.dtype,
    )
    ax.dsl.store(
        expanded.reshape(local_rows, hc_mult, hidden),
        r=output.reshape(row_count, hc_mult, hidden)[row_slice, :, :],
    )
    return output


@ax.builder(
    run_mode=run_mode(),
    enable_cache=kernel_cache_enabled(),
    cache_path=kernel_cache_path(),
)
def head_hc_kernel(x, hc_fn, hc_scale, hc_base, output, eps=1e-6):
    batch, seq_len, hc_mult, hidden = x.shape
    row_count = batch * seq_len
    row_slice = linear_program_slice(row_count)
    local_rows = row_slice.stop - row_slice.start
    reduced = head_hc_value(
        ax.dsl.load(x.reshape(row_count, hc_mult, hidden)[row_slice, :, :]).reshape(1, local_rows, hc_mult, hidden),
        ax.dsl.load(hc_fn),
        ax.dsl.load(hc_scale),
        ax.dsl.load(hc_base),
        eps=eps,
        output_dtype=output.dtype,
    )
    ax.dsl.store(
        reduced.reshape(local_rows, hidden),
        r=output.reshape(row_count, hidden)[row_slice, :],
    )
    return output


def _check_pre_inputs(x, hc_fn, hc_scale, hc_base):
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, S, Hc, D], got {tuple(x.shape)}")
    batch, _, hc_mult, hidden = x.shape
    expected_mix = (2 + hc_mult) * hc_mult
    if (
        tuple(hc_fn.shape) != (expected_mix, hc_mult * hidden)
        or tuple(hc_scale.shape) != (3,)
        or tuple(hc_base.shape) != (expected_mix,)
    ):
        raise ValueError("mHC pre parameters have incompatible shapes")
    return batch, x.shape[1], hc_mult, hidden


# Keep the former helper name available for callers that imported it from
# ``attention.hca`` before the mHC code was moved into this module.
check_pre_inputs = _check_pre_inputs


def pre_hc(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    reduced: Optional[torch.Tensor] = None,
    post: Optional[torch.Tensor] = None,
    comb: Optional[torch.Tensor] = None,
    core_num: int = 1,
):
    """Run reusable mHC Pre on `[B,S,Hc,D]` input streams."""
    batch, seq_len, hc_mult, hidden = _check_pre_inputs(x, hc_fn, hc_scale, hc_base)
    reduced = reduced if reduced is not None else torch.empty((batch, seq_len, hidden), dtype=x.dtype, device=x.device)
    post = post if post is not None else torch.empty((batch, seq_len, hc_mult), dtype=torch.float32, device=x.device)
    comb = (
        comb
        if comb is not None
        else torch.empty((batch, seq_len, hc_mult, hc_mult), dtype=torch.float32, device=x.device)
    )
    validate_core_num(core_num, batch * seq_len, "mHC token rows")
    ax.init(chip_num=1, core_num=core_num)
    pre_hc_kernel[core_num](
        x,
        kernel_weight(hc_fn),
        hc_scale,
        hc_base,
        reduced,
        post,
        comb,
        eps,
        sinkhorn_iters,
    )
    return reduced, post, comb


def post_hc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    output=None,
    core_num: int = 1,
):
    """Run reusable mHC Post with the full residual and comb inputs."""
    if (
        x.ndim != 3
        or residual.ndim != 4
        or residual.shape[:2] != x.shape[:2]
        or post.shape != residual.shape[:3]
        or comb.shape != residual.shape[:3] + residual.shape[2:3]
    ):
        raise ValueError("mHC post inputs have incompatible shapes")
    output = output if output is not None else torch.empty_like(residual)
    validate_core_num(core_num, x.shape[0] * x.shape[1], "mHC token rows")
    ax.init(chip_num=1, core_num=core_num)
    post_hc_kernel[core_num](x, residual, post, comb, output)
    return output


def head_hc(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    eps: float = 1e-6,
    output: Optional[torch.Tensor] = None,
    core_num: int = 1,
):
    """Run final mHC head reduction from `[B,S,Hc,D]` to `[B,S,D]`."""
    if x.ndim != 4:
        raise ValueError("x must have shape [B, S, Hc, D]")
    batch, seq_len, hc_mult, hidden = x.shape
    if (
        tuple(hc_fn.shape) != (hc_mult, hc_mult * hidden)
        or tuple(hc_scale.shape) != (1,)
        or tuple(hc_base.shape) != (hc_mult,)
    ):
        raise ValueError("mHC head parameters have incompatible shapes")
    output = output if output is not None else torch.empty((batch, seq_len, hidden), dtype=x.dtype, device=x.device)
    validate_core_num(core_num, batch * seq_len, "mHC token rows")
    ax.init(chip_num=1, core_num=core_num)
    head_hc_kernel[core_num](x, kernel_weight(hc_fn), hc_scale, hc_base, output, eps)
    return output


@ax.builder(
    run_mode=run_mode(),
    enable_cache=False,
    dump_dsl_perf=env_flag("DUMP_DSL_PERF", False),
)
def post_mhc(global_output, post, residual_path, residual_out):
    """Apply mHC Post after all-reducing `[B,S,D]` chip outputs.

    ``residual_path`` is the precomputed ``comb @ residual`` term.  Keeping
    this distributed form separate avoids broadcasting the full comb matrix
    after the cross-chip reduction.
    """
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, hidden = global_output.shape
    streams = post.shape[-1]
    assert core_num == 4 and batch == 1 and seq_len >= core_num
    assert post.shape == (batch, seq_len, streams)
    assert residual_path.shape == residual_out.shape == (batch, seq_len, streams, hidden)
    token_range = slice_tokens(core_id, core_num, seq_len)
    token_count = token_range.stop - token_range.start
    global_rows = ax.dsl.load(global_output[:, token_range, :])
    post_rows = ax.dsl.load(post[:, token_range, :])
    residual_rows = ax.dsl.load(residual_path[:, token_range, :, :])
    branch = ax.dsl.mul(
        post_rows.reshape(batch, token_count, streams, 1),
        global_rows.reshape(batch, token_count, 1, hidden),
        output_dtype=ax.fp32,
    )
    result = ax.dsl.add(branch, residual_rows, output_dtype=residual_out.dtype)
    ax.dsl.store(result, r=residual_out[:, token_range, :, :])
    return residual_out


# Compatibility names used by the HCA module and existing callers.  The
# canonical public names are pre_hc/post_hc/head_hc because these operations
# are reusable mHC primitives, not HCA's C128 attention algorithm.
hca_pre_value = pre_hc_value
hca_post_value = post_hc_value
hca_head_value = head_hc_value
hca_pre_kernel = pre_hc_kernel
hca_post_kernel = post_hc_kernel
hca_head_kernel = head_hc_kernel
hca_pre = pre_hc
hca_post = post_hc
hca_head = head_hc
hc_pre = pre_hc
hc_post = post_hc
hc_head = head_hc


__all__ = [
    "pre_hc",
    "post_hc",
    "head_hc",
    "pre_hc_value",
    "post_hc_value",
    "head_hc_value",
    "pre_hc_kernel",
    "post_hc_kernel",
    "head_hc_kernel",
    "post_mhc",
    "check_pre_inputs",
    "run_mode",
    "kernel_weight",
    "scalar",
    "rms_denominator",
    "sinkhorn",
    "hca_pre_kernel",
    "hca_post_kernel",
    "hca_head_kernel",
    "hca_pre",
    "hca_post",
    "hca_head",
    "hca_pre_value",
    "hca_post_value",
    "hca_head_value",
    "hc_pre",
    "hc_post",
    "hc_head",
]
