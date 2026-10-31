"""HF-faithful rounding hooks around the reused SGL attention kernels."""

from __future__ import annotations

import os

import axnn.dsl as ax
from axnn.dsl import parallel as axd

from sgl_kernel_axera.attention.attn_utils import rmsnorm_sqrt
from sgl_kernel_axera.models.bank_access import gather_rows, load_tile


def rmsnorm(x, weight=None, eps=1e-6, output_dtype=None, r=None):
    """Qwen3 RMSNorm with HF's cast-to-input-dtype before weight multiply."""
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden_size = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=x.shape[-1])
    square_mean = ax.dsl.div(square_sum, hidden_size)
    epsilon = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=eps)
    denominator = ax.dsl.sqrt(ax.dsl.add(square_mean, epsilon))
    target_dtype = x.dtype if output_dtype is None else output_dtype
    inverse_rms = ax.dsl.reciprocal(denominator, output_dtype=ax.fp32)
    normalized = ax.dsl.mul(x, inverse_rms, output_dtype=x.dtype)
    if weight is None:
        return normalized if r is None else ax.dsl.move(normalized, r=r)
    return ax.dsl.mul(normalized, weight, output_dtype=target_dtype, r=r)


def rope_impl(
    x,
    cos,
    sin,
    position_ids=None,
    r=None,
    head_less=False,
    head_first=False,
    rotary_dim=None,
    banks=None,
):
    """RoPE with HF's BF16 cos/sin and per-multiply/add rounding."""
    assert not (head_less and head_first)
    if position_ids is not None:
        cos = gather_rows(cos, position_ids, dim=0, banks=banks)
        sin = gather_rows(sin, position_ids, dim=0, banks=banks)
    cos = ax.dsl.cast(cos, dtype=ax.bf16)
    sin = ax.dsl.cast(sin, dtype=ax.bf16)
    if not head_less:
        if head_first:
            cos = cos.reshape(1, cos.shape[0], cos.shape[1])
            sin = sin.reshape(1, sin.shape[0], sin.shape[1])
        else:
            cos = cos.reshape(cos.shape[0], 1, cos.shape[1])
            sin = sin.reshape(sin.shape[0], 1, sin.shape[1])
    head_dim = x.shape[-1]
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert cos.shape[-1] == sin.shape[-1] == rotary_dim // 2
    first_half = x[..., slice(0, rotary_dim // 2)]
    second_half = x[..., slice(rotary_dim // 2, rotary_dim)]
    first_cos = ax.dsl.mul(first_half, cos, output_dtype=x.dtype)
    second_sin = ax.dsl.mul(second_half, sin, output_dtype=x.dtype)
    second_cos = ax.dsl.mul(second_half, cos, output_dtype=x.dtype)
    first_sin = ax.dsl.mul(first_half, sin, output_dtype=x.dtype)
    if r is None:
        rotated_first = ax.dsl.sub(first_cos, second_sin, output_dtype=x.dtype)
        rotated_second = ax.dsl.add(second_cos, first_sin, output_dtype=x.dtype)
        rotated = ax.dsl.concat(rotated_first, rotated_second, dim=-1)
        if rotary_dim < head_dim:
            return ax.dsl.concat(rotated, x[..., slice(rotary_dim, head_dim)], dim=-1)
        return rotated
    ax.dsl.sub(first_cos, second_sin, r=r[..., slice(0, rotary_dim // 2)])
    ax.dsl.add(second_cos, first_sin, r=r[..., slice(rotary_dim // 2, rotary_dim)])
    if rotary_dim < head_dim:
        ax.dsl.move(x[..., slice(rotary_dim, head_dim)], r=r[..., slice(rotary_dim, head_dim)])
    return r


def prepare_shared_projection_input(
    x,
    input_norm_weight,
    query_row_slice,
    residual,
    post_residual_addition,
    core_count,
    query_indices=None,
    output=None,
    banks=None,
):
    """Prepare BF16-normalized projection input with HF rounding boundaries.

    The SGL default folds the RMS denominator past the projection.  That is
    algebraically equivalent, but not bit-equivalent to Qwen3's
    ``cast(normalize(x), BF16) * weight``.  This drop-in helper changes only
    the rounding boundary and returns a unit denominator to the caller.
    """
    query_len = x.shape[1] if query_indices is None else query_indices.shape_size
    tile_shape = (1, query_len, x.shape[-1])
    memory_type = ax.MemoryType.OCM if core_count == 1 else ax.MemoryType.SHARED_OCM
    shared_output = (
        output
        if output is not None and output.memory_type != ax.MemoryType.DDR
        else ax.dsl.create_tensor(tile_shape, x.dtype, memory_type, name="paged_prefill_output")
    )
    if query_indices is None:
        if x.memory_type == ax.MemoryType.DDR:
            local_x_rows = load_tile(x[query_row_slice], r=shared_output[query_row_slice], banks=banks)
        else:
            local_x_rows = shared_output[query_row_slice]
            ax.dsl.move(x[query_row_slice], r=local_x_rows)
    else:
        rows = query_indices[query_row_slice[1]]
        local_x_rows = gather_rows(
            x.reshape(-1, x.shape[-1]),
            rows,
            dim=0,
            r=shared_output[query_row_slice].reshape(rows.shape_size, x.shape[-1]),
            banks=banks,
        ).reshape(1, rows.shape_size, x.shape[-1])
    if residual is not None:
        ax.dsl.add(
            local_x_rows,
            (
                load_tile(residual[query_row_slice], banks=banks)
                if query_indices is None
                else gather_rows(residual.reshape(-1, x.shape[-1]), rows, dim=0, banks=banks).reshape(
                    *local_x_rows.shape
                )
            ),
            r=local_x_rows,
        )
    if post_residual_addition is not None:
        ax.dsl.add(
            local_x_rows,
            (
                load_tile(post_residual_addition[query_row_slice], banks=banks)
                if query_indices is None
                else gather_rows(
                    post_residual_addition.reshape(-1, x.shape[-1]),
                    rows,
                    dim=0,
                    banks=banks,
                ).reshape(*local_x_rows.shape)
            ),
            r=local_x_rows,
        )

    input_norm_weight = load_tile(input_norm_weight, banks=banks)
    scaled_x = ax.dsl.create_tensor(tile_shape, x.dtype, memory_type, name="paged_prefill_x_scaled")
    local_rms = ax.dsl.create_tensor(
        (1, query_len, 1),
        ax.fp32,
        memory_type if core_count == 1 else ax.MemoryType.OCM,
        name="paged_prefill_x_rms_local",
    )
    rmsnorm_sqrt(x=local_x_rows, r=local_rms[query_row_slice])
    inverse_rms = ax.dsl.reciprocal(local_rms[query_row_slice], output_dtype=ax.fp32)
    normalized = ax.dsl.mul(
        local_x_rows,
        inverse_rms,
        output_dtype=ax.bf16,
    )
    ax.dsl.mul(
        normalized,
        input_norm_weight,
        output_dtype=ax.bf16,
        r=scaled_x[query_row_slice],
    )
    unit_rms = ax.dsl.full((1, query_len, 1), ax.fp32, 1.0)
    if core_count > 1:
        for tensor in (shared_output, scaled_x):
            axd.pub(tensor)
            axd.sub(tensor)
    return shared_output, scaled_x, unit_rms


def install():
    """Install the rounding hook before the flash-attention builder is traced."""
    from sgl_kernel_axera.attention import flash_attn

    flash_attn.prepare_shared_projection_input = prepare_shared_projection_input
    if os.getenv("EMB_HF_QK", "0") == "1":
        flash_attn.rmsnorm = rmsnorm
    if os.getenv("EMB_HF_ROPE", "0") == "1":
        flash_attn.rope_impl = rope_impl
