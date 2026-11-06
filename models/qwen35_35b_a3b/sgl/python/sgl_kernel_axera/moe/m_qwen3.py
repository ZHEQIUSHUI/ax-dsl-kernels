"""Runtime-weight expert and token-parallel MoE kernels for Qwen3 and Qwen3.5."""


from collections.abc import Mapping
from math import gcd
import os
import pathlib

import numpy as np
import torch
from axnn.opset import datatype
import axnn.dsl as ax
from axnn.dsl import util as axdsl_util
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import linear
from sgl_kernel_axera.models.bank_access import (
    gather_rows,
    load_tile,
    scatter_rows,
    store_tile,
)
from sgl_kernel_axera.attention.attn_utils import rmsnorm

OCM_SIZE = 7 * 1024 * 1024


def _is_runtime_mxfp8_dtype_pair(weight_dtype, scale_dtype):
    """Recognize MXFP8 after its runtime weight marker has been normalized to FP8."""
    weight_typename = datatype.DType(weight_dtype).typename
    scale_typename = datatype.DType(scale_dtype).typename
    return weight_typename in ("MXFP8_E4M3", "FP8_E4M3") and scale_typename == "MXE8M0"


def _is_block_scale(scale, weight):
    """Block-quantized FP8 weight scale: [oc//block, ic//block] vs per-channel [oc, 1]."""
    scale_shape, weight_shape = scale.shape, weight.shape
    if len(scale_shape) != len(weight_shape):
        return False
    return (
        tuple(scale_shape) != tuple(weight_shape[:-1]) + (1,)
        and weight_shape[-1] == scale_shape[-1] * 128
        and weight_shape[-2] == scale_shape[-2] * 128
    )


def _has_reduction_block_scale(scale, weight):
    """Whether per-output-row scales are blocked along the reduction dim.

    Covers MXFP8 (32-element blocks) and NVFP4/MXFP4 (16-element blocks) as
    they are exported per expert: ``[experts, oc, ic // block]``.
    """
    return scale.shape[-2] == weight.shape[-2] and scale.shape[-1] > 1


def _scale_oc_slice(slice_oc, block_size=128):
    """Map an element-level oc slice to a block-scale oc slice."""
    assert (
        slice_oc.start % block_size == 0 and slice_oc.stop % block_size == 0
    ), f"oc slice {slice_oc} is not aligned to block size {block_size}"
    return slice(slice_oc.start // block_size, slice_oc.stop // block_size)


def _scale_dim_slice(element_slice, weight_size, scale_size):
    """Map an element-level weight slice to its quantization-scale dimension."""
    group_size = weight_size // scale_size
    return slice(
        element_slice.start // group_size,
        (element_slice.stop + group_size - 1) // group_size,
    )


def mlp_kernel_expert_parallel(
    x,  # [seq_len, hidden_dim]
    w_gate,  # [moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [moe_intermediate_size, 1] or [moe_intermediate_size//128, hidden_dim//128]
    w_scale2_gate,  # [1]
    w_up,  # [moe_intermediate_size, hidden_dim]
    w_scale_up,  # [moe_intermediate_size, 1] or [moe_intermediate_size//128, hidden_dim//128]
    w_scale2_up,  # [1]
    w_down,  # [hidden_dim, moe_intermediate_size]
    w_scale_down,  # [hidden_dim, 1] or [hidden_dim//128, moe_intermediate_size//128]
    w_scale2_down,  # [1]
    output_ax,  # [seq_len, hidden_dim] share ocm
    core_id,
    core_num,
):
    seq_len, hidden_dim = x.shape
    moe_intermediate_size = w_gate.shape[-2]
    w_dtype_num_bits = datatype.DType(w_gate.dtype).num_bits
    block_quant = _is_block_scale(w_scale_gate, w_gate)
    w_scale2_gate = ax.dsl.load(w_scale2_gate)
    w_scale2_up = ax.dsl.load(w_scale2_up)
    w_scale2_down = ax.dsl.load(w_scale2_down)
    moe_intermediate_size_per_core = (
        ((moe_intermediate_size + core_num - 1) // core_num * w_dtype_num_bits + 255) // 256 * 256
    ) // w_dtype_num_bits  # factor must 256 bits align
    slice_moe_intermediate = slice(
        core_id * moe_intermediate_size_per_core,
        min(
            core_id * moe_intermediate_size_per_core + moe_intermediate_size_per_core,
            moe_intermediate_size,
        ),
    )
    if block_quant:
        # Block-quantized FP8 (native): scale indexed by 128x128 blocks.
        slice_scale_gate = _scale_oc_slice(slice_moe_intermediate)
        w_scale_gate_cur = w_scale_gate[slice_scale_gate, :]
        w_scale_up_cur = w_scale_up[slice_scale_gate, :]
    else:
        w_scale_gate_cur = w_scale_gate[slice_moe_intermediate, :]
        w_scale_up_cur = w_scale_up[slice_moe_intermediate, :]

    # gate
    r_gate = ax.dsl.matmul(
        x,
        ax.dsl.load(w_gate[slice_moe_intermediate, :]),
        w_scale=ax.dsl.load(w_scale_gate_cur),
        w_scale2=w_scale2_gate,
        transposed=True,
        output_dtype=ax.bf16,
    )
    r_gate_silu = ax.dsl.silu(r_gate)
    # up
    r_up = ax.dsl.matmul(
        x,
        ax.dsl.load(w_up[slice_moe_intermediate, :]),
        w_scale=ax.dsl.load(w_scale_up_cur),
        w_scale2=w_scale2_up,
        transposed=True,
        output_dtype=ax.bf16,
    )
    current_hidden_states = ax.dsl.mul(r_gate_silu, r_up)  # [seq_len, moe_intermediate_size_per_core]
    # down
    x_running = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, ax.MemoryType.SHARED_OCM)
    if core_id == 0:
        ax.dsl.move(
            r=x_running,
            x=ax.dsl.full(shape=(seq_len, hidden_dim), dtype=x.dtype, fill_value=0.0),
        )
    axd.pub(x_running)
    axd.sub(x_running)
    hidden_dims_per_bank = (hidden_dim + core_num - 1) // core_num
    for j in range(core_num):
        # core interleave on hidden_dim for bank conflict
        hidden_dim_bank_offset = ((j + core_id) % core_num) * hidden_dims_per_bank
        slice_hidden_dim_bank = slice(
            hidden_dim_bank_offset,
            min(hidden_dim_bank_offset + hidden_dims_per_bank, hidden_dim),
        )
        if block_quant:
            # down scale: [hidden_dim//128, moe_intermediate//128]; slice both dims.
            slice_scale_down_oc = _scale_oc_slice(slice_hidden_dim_bank)
            slice_scale_down_ic = _scale_oc_slice(slice_moe_intermediate)
            w_scale_down_cur = w_scale_down[slice_scale_down_oc, slice_scale_down_ic]
        else:
            slice_scale_down_ic = _scale_dim_slice(slice_moe_intermediate, w_down.shape[-1], w_scale_down.shape[-1])
            w_scale_down_cur = w_scale_down[slice_hidden_dim_bank, slice_scale_down_ic]
        r_down = ax.dsl.matmul(
            current_hidden_states,
            ax.dsl.load(w_down[slice_hidden_dim_bank, slice_moe_intermediate]),
            w_scale=ax.dsl.load(w_scale_down_cur),
            w_scale2=w_scale2_down,
            transposed=True,
            output_dtype=ax.bf16,
        )
        # core reduce add
        r_sum_shocm = ax.dsl.create_tensor(
            x_running.shape,
            x_running.dtype,
            ax.MemoryType.SHARED_OCM,
            name=f"r_sum_shocm_{core_id}_{j}",
        )
        ax.dsl.add(
            x=r_down,
            y=x_running[:, slice_hidden_dim_bank],
            r=r_sum_shocm[:, slice_hidden_dim_bank],
        )
        axd.pub(r_sum_shocm)
        axd.sub(r_sum_shocm)
        x_running = r_sum_shocm
    ax.dsl.move(x=x_running, r=output_ax)


def _select_runtime_expert_tensor(tensor, expert_index, banks=None):
    """Select one expert from a stacked DDR tensor and remove its expert lane."""
    return gather_rows(tensor, expert_index, dim=0, banks=banks).reshape(*tensor.shape[1:])


def _select_runtime_expert_scale2(scale2, expert_index, banks=None):
    """Resolve a scalar or per-expert second-level scale at runtime."""
    if scale2.shape[0] == 1:
        return load_tile(scale2, banks=banks) if banks and scale2.name in banks else scale2
    return gather_rows(scale2, expert_index, dim=0, banks=banks)


def _hidden_parallel_rms_norm(x, w_norm, norm_eps, core_id, core_num, chunk_size):
    """Apply RMSNorm with the hidden dimension partitioned across cores.

    ``w_norm`` is the effective ``(1 + post_attention_layernorm.weight)``
    parameter prepared by the host wrapper.  Each core loads only its hidden
    shard, contributes that shard's FP32 square sum, and materializes that shard
    in a core-major shared workspace.
    """
    seq_len, hidden_dim = x.shape
    if hidden_dim % core_num:
        raise ValueError(
            "fused qwen3_moe RMSNorm requires hidden_dim divisible by core_num, "
            f"got hidden_dim={hidden_dim}, core_num={core_num}"
        )

    hidden_per_core = hidden_dim // core_num
    input_slice = slice(core_id * hidden_per_core, (core_id + 1) * hidden_per_core)
    w_norm_local = ax.dsl.load(w_norm[input_slice])
    x_norm_shards = ax.dsl.create_tensor(
        (core_num, seq_len, hidden_per_core),
        x.dtype,
        ax.MemoryType.SHARED_OCM,
        name="moe_rms_norm_shards",
    )
    hidden_count = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=hidden_dim)
    epsilon = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=norm_eps)
    chunk_size = min(int(chunk_size), seq_len) if chunk_size else seq_len

    for token_start in range(0, seq_len, chunk_size):
        token_stop = min(token_start + chunk_size, seq_len)
        token_slice = slice(token_start, token_stop)
        x_local = ax.dsl.load(x[token_slice, input_slice])

        token_count = token_stop - token_start
        square_sum_partials = ax.dsl.create_tensor(
            (core_num, token_count, 1),
            ax.fp32,
            ax.MemoryType.SHARED_OCM,
            name=f"moe_rms_partials_{token_start}",
        )
        ax.dsl.sum(
            ax.dsl.mul(x_local, x_local, output_dtype=ax.fp32),
            dim=-1,
            keepdim=True,
            output_dtype=ax.fp32,
            r=square_sum_partials[core_id : core_id + 1].reshape(token_count, 1),
        )
        axd.pub(square_sum_partials)
        axd.sub(square_sum_partials)
        square_sum_shared = ax.dsl.sum(square_sum_partials, dim=0, output_dtype=ax.fp32)

        rms = ax.dsl.sqrt(
            ax.dsl.add(
                ax.dsl.div(square_sum_shared, hidden_count, output_dtype=ax.fp32),
                epsilon,
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        )
        normalized_local = ax.dsl.div(x_local, rms, output_dtype=ax.fp32)
        ax.dsl.mul(
            normalized_local,
            w_norm_local,
            r=x_norm_shards[core_id : core_id + 1, token_slice, :].reshape(
                token_slice.stop - token_slice.start,
                hidden_per_core,
            ),
            output_dtype=x.dtype,
        )

    x_norm = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, ax.MemoryType.OCM)
    axd.pub(x_norm_shards)
    axd.sub(x_norm_shards)
    # Every MoE consumer needs complete rows. Rebuild the token-major
    # activation privately so no core reads a shared tensor while another
    # core is still reconstructing it.
    for owner in range(core_num):
        shard_slice = slice(owner * hidden_per_core, (owner + 1) * hidden_per_core)
        ax.dsl.move(
            x_norm_shards[owner : owner + 1].reshape(seq_len, hidden_per_core),
            r=x_norm[:, shard_slice],
        )
    return x_norm


def _expert_projection(x, weight, scale, scale2, zero_point=None, output=None):
    """Project with runtime weights and optional group-wise zero-point correction."""
    if datatype.DType(weight.dtype).typename in ("S4", "S8") and scale.shape[-1] > 1:
        # Native integer MatMul consumes one channel scale per output. Accumulate
        # runtime input groups in FP32, then emit the final BF16 result once.
        groups = scale.shape[-1]
        width = x.shape[-1] // groups
        total = None
        for group in range(groups):
            channels = slice(group * width, (group + 1) * width)
            # Combining the two runtime scales also produces the contiguous
            # channel vector required by native integer MatMul.
            group_scale = ax.dsl.mul(scale[:, group : group + 1], scale2, output_dtype=ax.fp32)
            part = ax.dsl.matmul(
                x[..., channels],
                weight[:, channels],
                w_scale=group_scale,
                transposed=True,
                output_dtype=ax.fp32,
            )
            if zero_point is not None:
                group_zero = zero_point[:, group : group + 1] if zero_point.shape[-1] > 1 else zero_point
                correction_scale = ax.dsl.mul(group_zero, group_scale)
                group_sum = ax.dsl.sum(x[..., channels], dim=-1, keepdim=True, output_dtype=ax.fp32)
                part = ax.dsl.sub(
                    part,
                    ax.dsl.mul(group_sum, correction_scale.reshape(1, -1)),
                    output_dtype=ax.fp32,
                )
            if group == 0:
                total = part
            elif group == groups - 1:
                return ax.dsl.add(total, part, output_dtype=ax.bf16, r=output)
            else:
                total = ax.dsl.add(total, part, output_dtype=ax.fp32)

    result = ax.dsl.matmul(
        x,
        weight,
        w_scale=scale,
        w_scale2=scale2,
        transposed=True,
        output_dtype=ax.bf16,
        r=output if zero_point is None else None,
    )
    if zero_point is None:
        return result
    # x[T,I] BF16 -> grouped sums[T,G] FP32. Zero points/scales[O,G]
    # remain runtime inputs; a scalar parameter broadcasts over output rows.
    correction_scale = ax.dsl.mul(ax.dsl.mul(zero_point, scale), scale2)
    groups = correction_scale.shape[-1]
    group_sum = ax.dsl.sum(
        x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups),
        dim=-1,
        output_dtype=ax.fp32,
    )
    # FP32 runtime weights are not a native MatMul operand. Broadcast the small
    # group dimension instead: [T,1,G] * [1,O,G] -> [T,O,G] -> correction[T,O].
    correction = ax.dsl.sum(
        ax.dsl.mul(
            group_sum.reshape(*x.shape[:-1], 1, groups),
            correction_scale.reshape(1, *correction_scale.shape),
        ),
        dim=-1,
        output_dtype=ax.fp32,
    )
    # BF16[T,O] - FP32[T,O] -> final BF16[T,O], without an extra cast or move.
    return ax.dsl.sub(result, correction, output_dtype=ax.bf16, r=output)


def mlp_kernel_expert_parallel_scf(
    x,  # [seq_len, hidden_dim]
    w_gate,  # [moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [moe_intermediate_size, 1] or [moe_intermediate_size//128, hidden_dim//128]
    w_scale2_gate,  # [1]
    w_up,  # [moe_intermediate_size, hidden_dim]
    w_scale_up,  # [moe_intermediate_size, 1] or [moe_intermediate_size//128, hidden_dim//128]
    w_scale2_up,  # [1]
    w_down,  # [hidden_dim, moe_intermediate_size]
    w_scale_down,  # [hidden_dim, 1] or [hidden_dim//128, moe_intermediate_size//128]
    w_scale2_down,  # [1]
    output_ax,  # [seq_len, hidden_dim] share ocm
    w_zeropoint_gate=None,
    w_zeropoint_up=None,
    w_zeropoint_down=None,
):
    # gate
    r_gate = _expert_projection(x, w_gate, w_scale_gate, w_scale2_gate, w_zeropoint_gate)
    r_gate_silu = ax.dsl.silu(r_gate)
    # up
    r_up = _expert_projection(x, w_up, w_scale_up, w_scale2_up, w_zeropoint_up)
    current_hidden_states = ax.dsl.mul(r_gate_silu, r_up)  # [seq_len, moe_intermediate_size_per_core]
    # down
    _expert_projection(
        current_hidden_states,
        w_down,
        w_scale_down,
        w_scale2_down,
        w_zeropoint_down,
        output_ax,
    )


def _runtime_shared_mlp(
    x,  # [seq_len, hidden_dim]
    w_gate,  # [moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [moe_intermediate_size, 1] or [moe_intermediate_size//128, hidden_dim//128]
    w_scale2_gate,  # [1]
    w_up,  # [moe_intermediate_size, hidden_dim]
    w_scale_up,  # [moe_intermediate_size, 1] or [moe_intermediate_size//128, hidden_dim//128]
    w_scale2_up,  # [1]
    w_down,  # [hidden_dim, moe_intermediate_size]
    w_scale_down,  # [hidden_dim, 1] or [hidden_dim//128, moe_intermediate_size//128]
    w_scale2_down,  # [1]
    output_ax,  # [seq_len, hidden_dim] share ocm
    core_id,
    core_num,
    banks=None,
):
    seq_len, hidden_dim = x.shape
    moe_intermediate_size = w_gate.shape[-2]
    w_dtype_num_bits = datatype.DType(w_gate.dtype).num_bits
    block_quant = _is_block_scale(w_scale_gate, w_gate)
    w_scale2_gate = load_tile(w_scale2_gate, banks=banks)
    w_scale2_up = load_tile(w_scale2_up, banks=banks)
    w_scale2_down = load_tile(w_scale2_down, banks=banks)
    moe_intermediate_size_per_core = (
        ((moe_intermediate_size + core_num - 1) // core_num * w_dtype_num_bits + 255) // 256 * 256
    ) // w_dtype_num_bits  # factor must 256 bits align
    slice_moe_intermediate = slice(
        core_id * moe_intermediate_size_per_core,
        min(
            core_id * moe_intermediate_size_per_core + moe_intermediate_size_per_core,
            moe_intermediate_size,
        ),
    )
    if block_quant:
        # Block-quantized FP8 (native): scale indexed by 128x128 blocks.
        slice_scale_gate = _scale_oc_slice(slice_moe_intermediate)
        w_scale_gate_cur = w_scale_gate[slice_scale_gate, :]
        w_scale_up_cur = w_scale_up[slice_scale_gate, :]
    else:
        w_scale_gate_cur = w_scale_gate[slice_moe_intermediate, :]
        w_scale_up_cur = w_scale_up[slice_moe_intermediate, :]

    # gate
    r_gate = ax.dsl.matmul(
        x,
        load_tile(w_gate[slice_moe_intermediate, :], banks=banks),
        w_scale=load_tile(w_scale_gate_cur, banks=banks),
        w_scale2=w_scale2_gate,
        transposed=True,
        output_dtype=ax.bf16,
    )
    r_gate_silu = ax.dsl.silu(r_gate)
    # up
    r_up = ax.dsl.matmul(
        x,
        load_tile(w_up[slice_moe_intermediate, :], banks=banks),
        w_scale=load_tile(w_scale_up_cur, banks=banks),
        w_scale2=w_scale2_up,
        transposed=True,
        output_dtype=ax.bf16,
    )
    if seq_len == 1:
        # Each core publishes its BF16[1,I/core_num] activation once. Down
        # then owns complete IC rows and disjoint output channels per core.
        activated = ax.dsl.create_tensor((1, moe_intermediate_size), x.dtype, ax.MemoryType.SHARED_OCM)
        ax.dsl.mul(r_gate_silu, r_up, r=activated[:, slice_moe_intermediate])
        axd.pub(activated)
        axd.sub(activated)
        hidden_per_core = (hidden_dim + core_num - 1) // core_num
        output_slice = slice(core_id * hidden_per_core, min((core_id + 1) * hidden_per_core, hidden_dim))
        scale_slice = _scale_oc_slice(output_slice) if block_quant else output_slice
        ax.dsl.matmul(
            activated,
            load_tile(w_down[output_slice, :], banks=banks),
            w_scale=load_tile(w_scale_down[scale_slice, :], banks=banks),
            w_scale2=w_scale2_down,
            transposed=True,
            output_dtype=x.dtype,
            r=output_ax[:, output_slice],
        )
        axd.pub(output_ax)
        axd.sub(output_ax)
        return

    current_hidden_states = ax.dsl.mul(r_gate_silu, r_up)  # [seq_len, moe_intermediate_size_per_core]
    # down
    # The first partial adds scalar BF16 zero. A full shared zero plane
    # would remain an immutable loop operand and occupy SHOCM across stages.
    x_running = ax.dsl.zeros((1,), x.dtype)
    hidden_dims_per_bank = (hidden_dim + core_num - 1) // core_num
    for j in range(core_num):
        # core interleave on hidden_dim for bank conflict
        hidden_dim_bank_offset = ((j + core_id) % core_num) * hidden_dims_per_bank
        slice_hidden_dim_bank = slice(
            hidden_dim_bank_offset,
            min(hidden_dim_bank_offset + hidden_dims_per_bank, hidden_dim),
        )
        if block_quant:
            # down scale: [hidden_dim//128, moe_intermediate//128]; slice both dims.
            slice_scale_down_oc = _scale_oc_slice(slice_hidden_dim_bank)
            slice_scale_down_ic = _scale_oc_slice(slice_moe_intermediate)
            w_scale_down_cur = w_scale_down[slice_scale_down_oc, slice_scale_down_ic]
        else:
            slice_scale_down_ic = _scale_dim_slice(slice_moe_intermediate, w_down.shape[-1], w_scale_down.shape[-1])
            w_scale_down_cur = w_scale_down[slice_hidden_dim_bank, slice_scale_down_ic]
        r_down = ax.dsl.matmul(
            current_hidden_states,
            load_tile(w_down[slice_hidden_dim_bank, slice_moe_intermediate], banks=banks),
            w_scale=load_tile(w_scale_down_cur, banks=banks),
            w_scale2=w_scale2_down,
            transposed=True,
            output_dtype=ax.bf16,
        )
        # core reduce add
        r_sum_shocm = (
            output_ax
            if j == core_num - 1
            else ax.dsl.create_tensor(
                (seq_len, hidden_dim),
                x.dtype,
                ax.MemoryType.SHARED_OCM,
                name=f"r_sum_shocm_{core_id}_{j}",
            )
        )
        ax.dsl.add(
            x=r_down,
            y=x_running if j == 0 else x_running[:, slice_hidden_dim_bank],
            r=r_sum_shocm[:, slice_hidden_dim_bank],
        )
        axd.pub(r_sum_shocm)
        axd.sub(r_sum_shocm)
        x_running = r_sum_shocm


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=bool(int(os.environ.get("DUMP_DSL_PERF", 0))),
    dump_path=pathlib.Path(os.environ.get("DSL_DUMP_PATH") or pathlib.Path.cwd() / "output"),
)
def moe_kernel_expert_parallel(
    x,  # [seq_len, hidden_dim]
    w_norm,  # [hidden_dim] effective post-attention norm weight; None disables fusion
    shared_expert_gate,  # [hidden_dim, 1]
    w_shared_expert_mlp_gate,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale_shared_expert_mlp_gate,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale2_shared_expert_mlp_gate,
    w_shared_expert_mlp_up,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale_shared_expert_mlp_up,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale2_shared_expert_mlp_up,
    w_shared_expert_mlp_down,  # [shared_expert_intermediate_size, hidden_dim]
    w_scale_shared_expert_mlp_down,  # [shared_expert_intermediate_size, hidden_dim]
    w_scale2_shared_expert_mlp_down,
    w_route,  # [num_experts, hidden_dim]
    w_scale_route,
    w_scale2_route,
    w_gate,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_gate,  # [1]
    w_zeropoint_gate,
    w_up,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_up,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_up,  # [1]
    w_zeropoint_up,
    w_down,  # [num_experts, hidden_dim, moe_intermediate_size]
    w_scale_down,  # [num_experts, hidden_dim, 1]
    w_scale2_down,  # [1]
    w_zeropoint_down,
    top_k,  # 8
    num_experts,  # 128
    norm_topk_prob,  # False
    output_ax,  # [seq_len, hidden_dim]
    chunk_size=64,
    norm_eps=1e-6,
):
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    x = x.reshape(-1, x.shape[-1])  # [B, seq_len, hidden_dim] -> [B*seq_len, hidden_dim]
    seq_len, hidden_dim = x.shape
    chunk_size = min(chunk_size, seq_len) if chunk_size is not None and chunk_size > 0 else seq_len
    seq_len_factor = (seq_len + core_num - 1) // core_num
    slice_seq_slice = slice(core_id * seq_len_factor, min((core_id + 1) * seq_len_factor, seq_len))
    slice_seq_cur_core = slice_seq_slice.stop - slice_seq_slice.start
    num_experts_per_core = (num_experts + core_num - 1) // core_num
    num_experts_offset_core = core_id * num_experts_per_core
    w_dtype = datatype.DType(w_gate.dtype)
    w_dtype_num_bits = w_dtype.num_bits
    # Runtime MXFP8 weights use an FP8_E4M3 payload dtype so MatMul can pair it
    # with the MXE8M0 block scale. The E8M0 scale distinguishes this path from
    # ordinary per-channel FP8 weights.
    mxfp8_runtime = _is_runtime_mxfp8_dtype_pair(w_gate.dtype, w_scale_gate.dtype)
    # Packed 4-bit, BF16, and MXFP8 weights use the sparse expert-chunk path.
    sparse_prefill = w_dtype_num_bits == 4 or w_dtype.typename in ("BF16", "MXFP8_E4M3") or mxfp8_runtime
    # BF16 weight tiles are large; selecting them inside each chunk branch is
    # unstable. Select once per expert for BF16 and lazily for packed dtypes.
    lazy_chunk_weight_select = sparse_prefill and w_dtype.typename != "BF16"
    slice_expert_core = slice(
        num_experts_offset_core,
        min(num_experts_offset_core + num_experts_per_core, num_experts),
    )
    # load tokens
    x_mem_scope = ax.MemoryType.SHARED_OCM
    if w_norm is None:
        x_shocm = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        if slice_seq_cur_core > 0:
            ax.dsl.load(r=x_shocm[slice_seq_slice, :], x=x[slice_seq_slice, :])
    else:
        x_shocm = _hidden_parallel_rms_norm(x, w_norm, norm_eps, core_id, core_num, chunk_size)
    axd.sync_all()

    if shared_expert_gate is not None:
        assert (
            w_shared_expert_mlp_gate is not None
            and w_shared_expert_mlp_up is not None
            and w_shared_expert_mlp_down is not None
        )
        # shared expert compute
        shared_expert_output = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        mlp_kernel_expert_parallel(
            x_shocm,
            w_shared_expert_mlp_gate,
            w_scale_shared_expert_mlp_gate,
            w_scale2_shared_expert_mlp_gate,
            w_shared_expert_mlp_up,
            w_scale_shared_expert_mlp_up,
            w_scale2_shared_expert_mlp_up,
            w_shared_expert_mlp_down,
            w_scale_shared_expert_mlp_down,
            w_scale2_shared_expert_mlp_down,
            shared_expert_output,
            core_id,
            core_num,
        )
        axd.sync_all()
        shared_expert_gate_output_sigmoid = ax.dsl.create_tensor((seq_len, 1), x.dtype, x_mem_scope)
        shared_expert_output_scaled = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        if core_id == 0:
            shared_expert_gate_output = ax.dsl.matmul(
                x_shocm,
                ax.dsl.load(shared_expert_gate),
                output_dtype=ax.bf16,
            )  # [seq_len, 1]
            ax.dsl.sigmoid(shared_expert_gate_output, r=shared_expert_gate_output_sigmoid)  # [seq_len, 1]
            ax.dsl.mul(
                shared_expert_output,
                shared_expert_gate_output_sigmoid,
                r=shared_expert_output_scaled,
            )  # [seq_len, hidden_dim]

        axd.sync_all()

    # route
    router_logits = ax.dsl.create_tensor((seq_len, num_experts), ax.fp32, x_mem_scope)
    ax.dsl.matmul(
        x_shocm,
        ax.dsl.load(w_route[slice_expert_core, :]),
        w_scale=ax.dsl.load(w_scale_route[slice_expert_core, :]),
        w_scale2=ax.dsl.load(w_scale2_route),
        transposed=True,
        output_dtype=ax.fp32,
        r=router_logits[:, slice_expert_core],
    )
    # Each core owns an expert-shard of logits, while every token-owner
    # reads all experts for softmax/top-k.
    axd.pub(router_logits)
    axd.sub(router_logits)

    top_k_weights = ax.dsl.create_tensor((seq_len, top_k), ax.fp32, x_mem_scope)
    top_k_index = ax.dsl.create_tensor((seq_len, top_k), ax.int32, x_mem_scope)
    if slice_seq_cur_core > 0:
        router_logits_ = ax.dsl.softmax(router_logits[slice_seq_slice, :], dim=-1)
        # [seq_len, top_k]@bf16 -> [seq_len, top_k]@bf16, [seq_len, top_k]@int32
        ax.dsl.topk(
            router_logits_,
            k=top_k,
            dim=-1,
            r_values=top_k_weights[slice_seq_slice, :],
            r_indices=top_k_index[slice_seq_slice, :],
        )
        if norm_topk_prob:
            top_k_weights_sum = ax.dsl.sum(top_k_weights[slice_seq_slice, :], dim=-1, keepdim=True)
            ax.dsl.div(
                top_k_weights[slice_seq_slice, :],
                top_k_weights_sum,
                r=top_k_weights[slice_seq_slice, :],
            )
    # Top-k results are token-sharded but consumed by every expert owner.
    axd.pub(top_k_weights)
    axd.pub(top_k_index)
    axd.sub(top_k_weights)
    axd.sub(top_k_index)

    # mlp
    expert_token_indices_t = ax.dsl.transpose(top_k_index, perm=(1, 0))
    token_expert_weights_sparse = ax.dsl.create_tensor(
        (num_experts, seq_len + 1), top_k_weights.dtype, ax.MemoryType.OCM
    )
    ax.dsl.move(
        r=token_expert_weights_sparse,
        x=ax.dsl.full(
            shape=token_expert_weights_sparse.shape,
            dtype=token_expert_weights_sparse.dtype,
            fill_value=0.0,
        ),
    )
    ax.dsl.scatter(
        r=token_expert_weights_sparse[:, :seq_len],
        indices=expert_token_indices_t,
        x=ax.dsl.transpose(top_k_weights, perm=(1, 0)),
        dim=0,
    )
    expert_token_map = ax.dsl.create_tensor((num_experts, seq_len), ax.uint8, ax.MemoryType.OCM)
    ax.dsl.move(
        r=expert_token_map,
        x=ax.dsl.full(shape=expert_token_map.shape, dtype=expert_token_map.dtype, fill_value=0),
    )
    ax.dsl.scatter(
        r=expert_token_map,
        indices=expert_token_indices_t,
        x=ax.dsl.full(shape=expert_token_indices_t.shape, dtype=ax.uint8, fill_value=1),
        dim=0,
    )
    if sparse_prefill:
        # Interleave expert ownership across cores. Routing can concentrate many
        # tokens on adjacent expert ids; contiguous ownership then leaves one
        # core with most work and the others waiting at the final reduction.
        real_expert_ids = np.arange(core_id, num_experts, core_num, dtype=np.int32)
        local_expert_count = real_expert_ids.size
        local_expert_ids = np.zeros(num_experts_per_core, dtype=np.int32)
        local_expert_ids[:local_expert_count] = real_expert_ids
        local_expert_id_tensor = ax.dsl.from_numpy(real_expert_ids) if local_expert_count else None
    else:
        local_expert_count = slice_expert_core.stop - slice_expert_core.start
        local_expert_ids = np.asarray(
            [
                expert_id if expert_id < num_experts else 0
                for expert_id in range(
                    num_experts_offset_core,
                    num_experts_offset_core + num_experts_per_core,
                )
            ],
            dtype=np.int32,
        )
        local_expert_id_tensor = None
    local_expert_token_map = ax.dsl.create_tensor((num_experts_per_core, seq_len), ax.uint8, ax.MemoryType.OCM)
    ax.dsl.move(
        r=local_expert_token_map,
        x=ax.dsl.full(shape=local_expert_token_map.shape, dtype=ax.uint8, fill_value=0),
    )
    if sparse_prefill and local_expert_count:
        ax.dsl.index_select(
            expert_token_map,
            local_expert_id_tensor,
            dim=0,
            r=local_expert_token_map[:local_expert_count],
        )
    else:
        ax.dsl.move(
            r=local_expert_token_map[:local_expert_count],
            x=expert_token_map[slice_expert_core],
        )
    local_token_weights = ax.dsl.create_tensor(
        (num_experts_per_core, seq_len + 1), top_k_weights.dtype, ax.MemoryType.OCM
    )
    ax.dsl.move(
        r=local_token_weights,
        x=ax.dsl.full(
            shape=local_token_weights.shape,
            dtype=local_token_weights.dtype,
            fill_value=0.0,
        ),
    )
    if sparse_prefill and local_expert_count:
        ax.dsl.index_select(
            token_expert_weights_sparse,
            local_expert_id_tensor,
            dim=0,
            r=local_token_weights[:local_expert_count],
        )
    else:
        ax.dsl.move(
            r=local_token_weights[:local_expert_count],
            x=token_expert_weights_sparse[slice_expert_core],
        )

    mlp_out = ax.dsl.create_tensor((seq_len + 1, hidden_dim), x.dtype, ax.MemoryType.OCM)
    ax.dsl.move(
        r=mlp_out,
        x=ax.dsl.full(shape=mlp_out.shape, dtype=mlp_out.dtype, fill_value=0.0),
    )
    x_collect = ax.dsl.create_tensor((seq_len + 1, hidden_dim), x.dtype, x_mem_scope)
    if core_id == 0:
        ax.dsl.move(
            r=x_collect,
            x=ax.dsl.full(shape=x_collect.shape, dtype=x_collect.dtype, fill_value=0.0),
        )
        ax.dsl.move(r=x_collect[:seq_len, :], x=x_shocm)
    axd.pub(x_collect)
    axd.sub(x_collect)

    compact_positions = ax.dsl.from_numpy(np.arange(seq_len, dtype=np.int32))
    # A Range body with OCM allocations must have the same trip count on every
    # core. Pad tail-core tables to the same capacity; padding routes are zero.
    expert_ids = ax.dsl.from_numpy(local_expert_ids)
    range_stop = ax.dsl.full(shape=(1,), dtype=ax.int32, fill_value=num_experts_per_core)
    has_expert_zeropoint = any(value is not None for value in (w_zeropoint_gate, w_zeropoint_up, w_zeropoint_down))

    def expert_body(expert_index, *loop_state):
        """Run one local expert and reuse the loop's token buffers."""
        (
            loop_expert_ids,
            loop_expert_token_map,
            loop_token_weights,
            loop_x_collect,
            loop_compact_positions,
            loop_w_gate,
            loop_w_scale_gate,
            loop_w_scale2_gate,
            loop_w_up,
            loop_w_scale_up,
            loop_w_scale2_up,
            loop_w_down,
            loop_w_scale_down,
            loop_w_scale2_down,
        ) = loop_state[:14]
        loop_zeropoints = loop_state[14:-1] if has_expert_zeropoint else ()
        loop_mlp_out = loop_state[-1]

        expert_idx = ax.dsl.index_select(loop_expert_ids, expert_index, dim=0)
        expert_token_indices, expert_token_num = ax.dsl.nonzero(
            ax.dsl.index_select(loop_expert_token_map, expert_index, dim=0).reshape(seq_len)
        )
        expert_weight_row = ax.dsl.index_select(
            loop_token_weights,
            expert_index,
            dim=0,
        ).reshape(seq_len + 1)

        if not sparse_prefill:
            expert_tokens = ax.dsl.create_tensor(
                (chunk_size, hidden_dim),
                x.dtype,
                ax.MemoryType.OCM,
            )
            mlp_out_tmp = ax.dsl.create_tensor(
                (chunk_size, hidden_dim),
                mlp_out.dtype,
                ax.MemoryType.OCM,
            )
            selected_w_gate = _select_runtime_expert_tensor(loop_w_gate, expert_idx)
            selected_w_up = _select_runtime_expert_tensor(loop_w_up, expert_idx)
            selected_w_down = _select_runtime_expert_tensor(loop_w_down, expert_idx)
            selected_w_scale_gate = _select_runtime_expert_tensor(loop_w_scale_gate, expert_idx)
            selected_w_scale_up = _select_runtime_expert_tensor(loop_w_scale_up, expert_idx)
            selected_w_scale_down = _select_runtime_expert_tensor(loop_w_scale_down, expert_idx)
            selected_w_scale2_gate = _select_runtime_expert_scale2(loop_w_scale2_gate, expert_idx)
            selected_w_scale2_up = _select_runtime_expert_scale2(loop_w_scale2_up, expert_idx)
            selected_w_scale2_down = _select_runtime_expert_scale2(loop_w_scale2_down, expert_idx)
            if has_expert_zeropoint:
                selected_zeropoints = tuple(
                    _select_runtime_expert_tensor(zeropoint, expert_idx) for zeropoint in loop_zeropoints
                )
            else:
                selected_zeropoints = (None, None, None)

            for i in range(0, seq_len, chunk_size):
                token_slice = slice(i, min(i + chunk_size, seq_len))
                chunk_len = token_slice.stop - token_slice.start
                batch_token_indices_raw = expert_token_indices[token_slice]
                batch_token_indices_raw = ax.dsl.minimum(
                    batch_token_indices_raw,
                    ax.dsl.full(
                        shape=batch_token_indices_raw.shape,
                        dtype=batch_token_indices_raw.dtype,
                        fill_value=seq_len,
                    ),
                )
                # Positions after expert_token_num are inactive lanes. Clamp
                # their mask to zero and send those lanes to the zero row.
                active_mask_i32 = ax.dsl.sub(
                    ax.dsl.cast(expert_token_num, dtype=ax.int32),
                    loop_compact_positions[token_slice],
                    output_dtype=ax.int32,
                )
                active_mask_i32 = ax.dsl.minimum(
                    active_mask_i32,
                    ax.dsl.full(
                        shape=active_mask_i32.shape,
                        dtype=active_mask_i32.dtype,
                        fill_value=1,
                    ),
                )
                active_mask_u32 = ax.dsl.maximum(
                    active_mask_i32,
                    ax.dsl.full(
                        shape=active_mask_i32.shape,
                        dtype=active_mask_i32.dtype,
                        fill_value=0,
                    ),
                    output_dtype=ax.uint32,
                )
                inactive_mask_u32 = ax.dsl.sub(
                    ax.dsl.full(
                        shape=active_mask_u32.shape,
                        dtype=active_mask_u32.dtype,
                        fill_value=1,
                    ),
                    active_mask_u32,
                    output_dtype=ax.uint32,
                )
                batch_token_indices_write = ax.dsl.add(
                    ax.dsl.mul(batch_token_indices_raw, active_mask_u32, output_dtype=ax.uint32),
                    ax.dsl.mul(
                        inactive_mask_u32,
                        ax.dsl.full(
                            shape=inactive_mask_u32.shape,
                            dtype=inactive_mask_u32.dtype,
                            fill_value=seq_len,
                        ),
                        output_dtype=ax.uint32,
                    ),
                    output_dtype=ax.uint32,
                )

                expert_token_view = expert_tokens[:chunk_len, :]
                mlp_out_view = mlp_out_tmp[:chunk_len, :]
                ax.dsl.index_select(
                    loop_x_collect,
                    batch_token_indices_write,
                    dim=0,
                    r=expert_token_view,
                )
                mlp_kernel_expert_parallel_scf(
                    expert_token_view,
                    selected_w_gate,
                    selected_w_scale_gate,
                    selected_w_scale2_gate,
                    selected_w_up,
                    selected_w_scale_up,
                    selected_w_scale2_up,
                    selected_w_down,
                    selected_w_scale_down,
                    selected_w_scale2_down,
                    mlp_out_view,
                    *selected_zeropoints,
                )
                expert_weights = ax.dsl.index_select(
                    expert_weight_row,
                    batch_token_indices_write,
                    dim=0,
                ).reshape(-1, 1)
                expert_weights = ax.dsl.cast(expert_weights, dtype=x.dtype)
                mlp_out_weighted = ax.dsl.mul(mlp_out_view, expert_weights)
                ax.dsl.index_add(
                    x=mlp_out_weighted,
                    indices=batch_token_indices_write,
                    r=loop_mlp_out,
                    dim=0,
                )

            return (*loop_state[:-1], loop_mlp_out)

        if sparse_prefill and not lazy_chunk_weight_select:
            # Select BF16 expert tiles once per expert. Branch-local 2 MiB
            # tiles produced wrong tokens and an intermittent cmodel gather
            # failure, so the chunk branch guards compute only.
            loop_w_gate = _select_runtime_expert_tensor(loop_w_gate, expert_idx)
            loop_w_scale_gate = _select_runtime_expert_tensor(loop_w_scale_gate, expert_idx)
            loop_w_scale2_gate = _select_runtime_expert_scale2(loop_w_scale2_gate, expert_idx)
            loop_w_up = _select_runtime_expert_tensor(loop_w_up, expert_idx)
            loop_w_scale_up = _select_runtime_expert_tensor(loop_w_scale_up, expert_idx)
            loop_w_scale2_up = _select_runtime_expert_scale2(loop_w_scale2_up, expert_idx)
            loop_w_down = _select_runtime_expert_tensor(loop_w_down, expert_idx)
            loop_w_scale_down = _select_runtime_expert_tensor(loop_w_scale_down, expert_idx)
            loop_w_scale2_down = _select_runtime_expert_scale2(loop_w_scale2_down, expert_idx)
            if loop_zeropoints:
                loop_zeropoints = tuple(
                    _select_runtime_expert_tensor(zeropoint, expert_idx) for zeropoint in loop_zeropoints
                )

        for i in range(0, seq_len, chunk_size):
            token_slice = slice(i, min(i + chunk_size, seq_len))
            chunk_len = token_slice.stop - token_slice.start
            # Experts own only their routed tokens, but the chunk loop is static
            # over the full sequence.  Skip a chunk when this expert has no
            # token at or after the chunk start, so padded MatMuls never run.
            chunk_active = ax.dsl.sub(
                ax.dsl.cast(expert_token_num, dtype=ax.int32),
                ax.dsl.full(shape=expert_token_num.shape, dtype=ax.int32, fill_value=i),
                output_dtype=ax.int32,
            )
            chunk_active = ax.dsl.minimum(
                chunk_active,
                ax.dsl.full(shape=chunk_active.shape, dtype=chunk_active.dtype, fill_value=1),
            )
            chunk_active = ax.dsl.maximum(
                chunk_active,
                ax.dsl.full(shape=chunk_active.shape, dtype=chunk_active.dtype, fill_value=0),
            )

            def true_func(*args):
                (
                    _cond,
                    chunk_expert_idx,
                    chunk_token_indices,
                    chunk_token_num,
                    positions,
                    x_collect,
                    weight_row,
                    w_gate_full,
                    w_scale_gate_full,
                    w_scale2_gate_full,
                    w_up_full,
                    w_scale_up_full,
                    w_scale2_up_full,
                    w_down_full,
                    w_scale_down_full,
                    w_scale2_down_full,
                    routed_out,
                ) = args[:17]
                zeropoints = args[17:]
                # Scratch is written inside this branch; it is not loop-carried
                # input state and needs no value before the first active chunk.
                token_buf = ax.dsl.create_tensor((chunk_len, hidden_dim), x.dtype, ax.MemoryType.OCM)
                out_buf = ax.dsl.create_tensor((chunk_len, hidden_dim), x.dtype, ax.MemoryType.OCM)
                if lazy_chunk_weight_select:
                    # Select the expert weights only when a chunk is active. Most
                    # experts own no token, so this avoids their DDR weight gathers.
                    w_gate_sel = _select_runtime_expert_tensor(w_gate_full, chunk_expert_idx)
                    w_up_sel = _select_runtime_expert_tensor(w_up_full, chunk_expert_idx)
                    w_down_sel = _select_runtime_expert_tensor(w_down_full, chunk_expert_idx)
                    w_scale_gate_sel = _select_runtime_expert_tensor(w_scale_gate_full, chunk_expert_idx)
                    w_scale_up_sel = _select_runtime_expert_tensor(w_scale_up_full, chunk_expert_idx)
                    w_scale_down_sel = _select_runtime_expert_tensor(w_scale_down_full, chunk_expert_idx)
                    w_scale2_gate_sel = _select_runtime_expert_scale2(w_scale2_gate_full, chunk_expert_idx)
                    w_scale2_up_sel = _select_runtime_expert_scale2(w_scale2_up_full, chunk_expert_idx)
                    w_scale2_down_sel = _select_runtime_expert_scale2(w_scale2_down_full, chunk_expert_idx)
                    if zeropoints:
                        selected_zeropoints = tuple(
                            _select_runtime_expert_tensor(zeropoint, chunk_expert_idx) for zeropoint in zeropoints
                        )
                    else:
                        selected_zeropoints = (None, None, None)
                else:
                    # BF16 weights were selected once per expert outside the
                    # branch; the branch only guards compute.
                    w_gate_sel = w_gate_full
                    w_up_sel = w_up_full
                    w_down_sel = w_down_full
                    w_scale_gate_sel = w_scale_gate_full
                    w_scale_up_sel = w_scale_up_full
                    w_scale_down_sel = w_scale_down_full
                    w_scale2_gate_sel = w_scale2_gate_full
                    w_scale2_up_sel = w_scale2_up_full
                    w_scale2_down_sel = w_scale2_down_full
                    selected_zeropoints = zeropoints if zeropoints else (None, None, None)
                batch_token_indices_raw = chunk_token_indices[token_slice]
                batch_token_indices_raw = ax.dsl.minimum(
                    batch_token_indices_raw,
                    ax.dsl.full(
                        shape=batch_token_indices_raw.shape,
                        dtype=batch_token_indices_raw.dtype,
                        fill_value=seq_len,
                    ),
                )
                # Positions after chunk_token_num are inactive lanes. Clamp
                # their mask to zero and send those lanes to the zero row.
                active_mask_i32 = ax.dsl.sub(
                    ax.dsl.cast(chunk_token_num, dtype=ax.int32),
                    positions[token_slice],
                    output_dtype=ax.int32,
                )
                active_mask_i32 = ax.dsl.minimum(
                    active_mask_i32,
                    ax.dsl.full(
                        shape=active_mask_i32.shape,
                        dtype=active_mask_i32.dtype,
                        fill_value=1,
                    ),
                )
                active_mask_u32 = ax.dsl.maximum(
                    active_mask_i32,
                    ax.dsl.full(
                        shape=active_mask_i32.shape,
                        dtype=active_mask_i32.dtype,
                        fill_value=0,
                    ),
                    output_dtype=ax.uint32,
                )
                inactive_mask_u32 = ax.dsl.sub(
                    ax.dsl.full(
                        shape=active_mask_u32.shape,
                        dtype=active_mask_u32.dtype,
                        fill_value=1,
                    ),
                    active_mask_u32,
                    output_dtype=ax.uint32,
                )
                batch_token_indices_write = ax.dsl.add(
                    ax.dsl.mul(batch_token_indices_raw, active_mask_u32, output_dtype=ax.uint32),
                    ax.dsl.mul(
                        inactive_mask_u32,
                        ax.dsl.full(
                            shape=inactive_mask_u32.shape,
                            dtype=inactive_mask_u32.dtype,
                            fill_value=seq_len,
                        ),
                        output_dtype=ax.uint32,
                    ),
                    output_dtype=ax.uint32,
                )

                expert_token_view = token_buf[:chunk_len, :]
                mlp_out_view = out_buf[:chunk_len, :]
                ax.dsl.index_select(
                    x_collect,
                    batch_token_indices_write,
                    dim=0,
                    r=expert_token_view,
                )
                mlp_kernel_expert_parallel_scf(
                    expert_token_view,
                    w_gate_sel,
                    w_scale_gate_sel,
                    w_scale2_gate_sel,
                    w_up_sel,
                    w_scale_up_sel,
                    w_scale2_up_sel,
                    w_down_sel,
                    w_scale_down_sel,
                    w_scale2_down_sel,
                    mlp_out_view,
                    *selected_zeropoints,
                )
                expert_weights = ax.dsl.index_select(
                    weight_row,
                    batch_token_indices_write,
                    dim=0,
                ).reshape(-1, 1)
                expert_weights = ax.dsl.cast(expert_weights, dtype=x.dtype)
                mlp_out_weighted = ax.dsl.mul(mlp_out_view, expert_weights)
                ax.dsl.index_add(
                    x=mlp_out_weighted,
                    indices=batch_token_indices_write,
                    r=routed_out,
                    dim=0,
                )
                return args

            def false_func(*args):
                return args

            ax.dsl.if_op(
                chunk_active,
                loop_mlp_out,
                true_func,
                false_func,
                chunk_active,
                expert_idx,
                expert_token_indices,
                expert_token_num,
                loop_compact_positions,
                loop_x_collect,
                expert_weight_row,
                loop_w_gate,
                loop_w_scale_gate,
                loop_w_scale2_gate,
                loop_w_up,
                loop_w_scale_up,
                loop_w_scale2_up,
                loop_w_down,
                loop_w_scale_down,
                loop_w_scale2_down,
                loop_mlp_out,
                *loop_zeropoints,
            )

        return (*loop_state[:-1], loop_mlp_out)

    loop_state = [
        expert_ids,
        local_expert_token_map,
        local_token_weights,
        x_collect,
        compact_positions,
        w_gate,
        w_scale_gate,
        w_scale2_gate,
        w_up,
        w_scale_up,
        w_scale2_up,
        w_down,
        w_scale_down,
        w_scale2_down,
    ]
    if has_expert_zeropoint:
        loop_state.extend((w_zeropoint_gate, w_zeropoint_up, w_zeropoint_down))
    loop_state.append(mlp_out)

    mlp_out = ax.dsl.range(
        ax.dsl.zeros(shape=(1,), dtype=ax.int32),
        ax.dsl.zeros(shape=(1,), dtype=ax.int32),
        range_stop,
        ax.dsl.full(shape=(1,), dtype=ax.int32, fill_value=1),
        expert_body,
        *loop_state,
        unroll_factor=1,
    )[-1]

    axd.sync_all()

    final_hidden_states = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
    if core_id == 0:
        ax.dsl.move(
            r=final_hidden_states,
            x=ax.dsl.full(
                shape=final_hidden_states.shape,
                dtype=final_hidden_states.dtype,
                fill_value=0.0,
            ),
        )
    axd.pub(final_hidden_states)
    axd.sub(final_hidden_states)

    hidden_dims_per_bank = (hidden_dim + core_num - 1) // core_num

    for bank_id in range(core_num):
        # core interleave on hidden_dim for bank conflict
        hidden_dim_bank_offset = ((bank_id + core_id) % core_num) * hidden_dims_per_bank
        slice_hidden_dim_bank = slice(
            hidden_dim_bank_offset,
            min(hidden_dim_bank_offset + hidden_dims_per_bank, hidden_dim),
        )
        r_sum_shocm = ax.dsl.create_tensor(
            final_hidden_states.shape,
            final_hidden_states.dtype,
            ax.MemoryType.SHARED_OCM,
            name=f"r_sum_shocm_{core_id}_{bank_id}",
        )
        # core reduce add
        ax.dsl.add(
            x=mlp_out[0:seq_len, slice_hidden_dim_bank],
            y=final_hidden_states[:, slice_hidden_dim_bank],
            r=r_sum_shocm[:, slice_hidden_dim_bank],
        )
        axd.pub(r_sum_shocm)
        axd.sub(r_sum_shocm)
        final_hidden_states = r_sum_shocm

    if core_id == 0:
        final_hidden_states_shared_expert = (
            ax.dsl.add(final_hidden_states, shared_expert_output_scaled)
            if shared_expert_gate is not None
            else final_hidden_states
        )
        if w_norm is None:
            ax.dsl.store(r=output_ax, x=final_hidden_states_shared_expert)
        else:
            for token_index in range(seq_len):
                fused_output = ax.dsl.add(
                    final_hidden_states_shared_expert[token_index : token_index + 1, :],
                    ax.dsl.load(x[token_index : token_index + 1, :]),
                )
                ax.dsl.store(
                    r=output_ax[token_index : token_index + 1, :],
                    x=fused_output,
                )

    axd.sync_all()

    return output_ax


def mlp_kernel_token_parallel(
    x,  # [1, hidden_dim]
    w_gate,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_gate,  # [1]
    w_up,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_up,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_up,  # [1]
    w_down,  # [num_experts, hidden_dim, moe_intermediate_size]
    w_scale_down,  # [num_experts, hidden_dim, 1]
    w_scale2_down,  # [1]
    core_id,
    core_num,
    top_k,
    top_k_experts,
    cur_top_k_weights,
    output_ax,  # [1, hidden_dim] share ocm
    i,
    merge_topk=False,
):
    seq_len, hidden_dim = x.shape
    num_experts, moe_intermediate_size, _ = w_gate.shape
    w_scale2_gate = ax.dsl.load(w_scale2_gate)
    w_scale2_up = ax.dsl.load(w_scale2_up)
    w_scale2_down = ax.dsl.load(w_scale2_down)
    w_dtype_num_bits = datatype.DType(w_gate.dtype).num_bits
    # MX/NV block-quantized weights carry scales along the reduction dim; they
    # cannot be applied as a post-matmul row scale.
    block_scale = _has_reduction_block_scale(w_scale_gate, w_gate)
    w_scale2_gate_topk = _gather_topk_scale2(w_scale2_gate, top_k_experts)
    w_scale2_up_topk = _gather_topk_scale2(w_scale2_up, top_k_experts)
    w_scale2_down_topk = _gather_topk_scale2(w_scale2_down, top_k_experts)
    moe_intermediate_size_per_core = (
        ((moe_intermediate_size + core_num - 1) // core_num * w_dtype_num_bits + 255) // 256 * 256
    ) // w_dtype_num_bits  # factor must 256 bits align
    slice_moe_intermediate = slice(
        core_id * moe_intermediate_size_per_core,
        min(
            core_id * moe_intermediate_size_per_core + moe_intermediate_size_per_core,
            moe_intermediate_size,
        ),
    )
    moe_intermediate_cur_size = slice_moe_intermediate.stop - slice_moe_intermediate.start
    # gate
    # Keep every gather payload at >= 256 B so packed 4-bit weights do not turn
    # each 128-element chunk into a 64 B DMA burst.
    hidden_dim_factor = max(128, 256 * 8 // w_dtype_num_bits)
    hidden_dim_parts = (hidden_dim + hidden_dim_factor - 1) // hidden_dim_factor
    w_gate_cur = ax.dsl.create_tensor(
        (top_k, moe_intermediate_cur_size, hidden_dim),
        dtype=datatype.DType(w_gate.dtype),
        name=f"w_gate_cur_{i}_{core_id}",
    )
    assert (np.prod(w_gate_cur.shape) * w_dtype_num_bits // 8) <= OCM_SIZE
    for hidden_dims_idx in range(hidden_dim_parts):
        hidden_dim_slice = slice(
            hidden_dims_idx * hidden_dim_factor,
            min((hidden_dims_idx + 1) * hidden_dim_factor, hidden_dim),
        )
        ax.dsl.index_select(
            w_gate[:, slice_moe_intermediate, hidden_dim_slice],
            top_k_experts,
            dim=0,
            r=w_gate_cur[:, :, hidden_dim_slice],
        )  # [top_k, moe_intermediate_size, hidden_dim]
    w_scale_gate_cur = _gather_topk_block_scale(w_scale_gate[:, slice_moe_intermediate, :], top_k_experts)
    # gate
    if block_scale:
        # One OC-stacked MatMul covers all top-k experts and shares a single
        # runtime B/K encode.  scale2 is a per-tensor factor, so applying it
        # to each output slot is equivalent to folding it into the weight.
        r_gate = ax.dsl.matmul(
            x,
            w_gate_cur.reshape(-1, hidden_dim),
            w_scale=w_scale_gate_cur.reshape(-1, w_scale_gate_cur.shape[-1]),
            transposed=True,
            output_dtype=ax.bf16,
        )
        r_gate = ax.dsl.mul(
            r_gate.reshape(seq_len, top_k, -1),
            ax.dsl.cast(_topk_scale2_broadcast(w_scale2_gate_topk), dtype=ax.bf16),
        ).reshape(seq_len, -1)
    else:
        w_scale_gate_cur_bf16 = ax.dsl.mul(
            w_scale_gate_cur, w_scale2_gate, output_dtype=ax.bf16
        )  # [top_k, moe_intermediate_size, 1]
        r_gate = ax.dsl.matmul(
            x,  # [seq_len, hidden_dim]
            w_gate_cur.reshape(-1, w_gate_cur.shape[-1]),  # [top_k * moe_intermediate_size_per_core, hidden_dim]
            transposed=True,
            output_dtype=ax.bf16,
        )  # [seq_len, top_k * moe_intermediate_size_per_core]
        r_gate = ax.dsl.mul(
            r_gate, w_scale_gate_cur_bf16.reshape(1, -1)
        )  # [seq_len, top_k * moe_intermediate_size_per_core]
    r_gate_silu = ax.dsl.silu(r_gate)  # [seq_len, top_k * moe_intermediate_size_per_core]

    # up
    w_up_cur = ax.dsl.create_tensor(
        (top_k, moe_intermediate_cur_size, hidden_dim),
        dtype=datatype.DType(w_gate.dtype),
        name=f"w_gate_cur_{i}_{core_id}",
    )
    assert (np.prod(w_gate_cur.shape) * w_dtype_num_bits // 8) <= OCM_SIZE
    for hidden_dims_idx in range(hidden_dim_parts):
        hidden_dim_slice = slice(
            hidden_dims_idx * hidden_dim_factor,
            min((hidden_dims_idx + 1) * hidden_dim_factor, hidden_dim),
        )
        ax.dsl.index_select(
            w_up[:, slice_moe_intermediate, hidden_dim_slice],
            top_k_experts,
            dim=0,
            r=w_up_cur[:, :, hidden_dim_slice],
        )  # [top_k, moe_intermediate_size, hidden_dim]
    w_scale_up_cur = _gather_topk_block_scale(w_scale_up[:, slice_moe_intermediate, :], top_k_experts)
    # up
    if block_scale:
        r_up = ax.dsl.matmul(
            x,
            w_up_cur.reshape(-1, hidden_dim),
            w_scale=w_scale_up_cur.reshape(-1, w_scale_up_cur.shape[-1]),
            transposed=True,
            output_dtype=ax.bf16,
        )
        r_up = ax.dsl.mul(
            r_up.reshape(seq_len, top_k, -1),
            ax.dsl.cast(_topk_scale2_broadcast(w_scale2_up_topk), dtype=ax.bf16),
        ).reshape(seq_len, -1)
    else:
        w_scale_up_cur_bf16 = ax.dsl.mul(
            w_scale_up_cur, w_scale2_up, output_dtype=ax.bf16
        )  # [top_k, moe_intermediate_size, 1]
        r_up = ax.dsl.matmul(
            x,  # [seq_len, hidden_dim]
            w_up_cur.reshape(-1, w_up_cur.shape[-1]),  # [top_k * moe_intermediate_size_per_core, hidden_dim]
            transposed=True,
            output_dtype=ax.bf16,
        )
        r_up = ax.dsl.mul(r_up, w_scale_up_cur_bf16.reshape(1, -1))  # [seq_len, top_k * moe_intermediate_size_per_core]
    current_hidden_states = ax.dsl.mul(r_gate_silu, r_up)  # [seq_len, top_k * moe_intermediate_size_per_core]
    current_hidden_states = current_hidden_states.reshape(seq_len, top_k, -1)
    current_hidden_states_mul = ax.dsl.mul(current_hidden_states, cur_top_k_weights.reshape(1, top_k, 1))
    if block_scale:
        # NVFP4's per-expert scale2 is shared by every Down output bank. Apply
        # it once to the routed activation instead of repeating the cast/mul
        # for each hidden-dimension chunk below.
        current_hidden_states_mul = ax.dsl.mul(
            current_hidden_states_mul,
            ax.dsl.cast(_topk_scale2_broadcast(w_scale2_down_topk), dtype=ax.bf16),
        )
    # [seq_len, top_k * moe_intermediate_size_per_core]
    current_hidden_states_mul_shape = (seq_len, -1) if merge_topk else (seq_len, top_k, -1)
    current_hidden_states_mul = current_hidden_states_mul.reshape(*current_hidden_states_mul_shape)

    # down
    bank_num = core_num
    # Each core owns a private row of the shared partial buffer. Keeping all
    # four Down output banks local until the end removes three cross-core
    # read-after-write barriers from the bank loop.
    partial_output = ax.dsl.create_tensor(
        (core_num, 1, hidden_dim),
        x.dtype,
        ax.MemoryType.SHARED_OCM,
        name=f"routed_partial_{i}_{core_id}",
    )
    hidden_dims_per_bank_org = (hidden_dim + bank_num - 1) // bank_num
    hidden_dims_per_bank = hidden_dims_per_bank_org
    # Packed block-quantized weights only keep `top_k` gathered experts live, so
    # a wider Down tile fits and halves the gather/MatMul count. BF16 keeps the
    # larger num_experts estimate because the wider tile regressed that path.
    down_bank_weight_factor = top_k if block_scale else num_experts + top_k
    while (
        hidden_dims_per_bank > 2
        and down_bank_weight_factor * hidden_dims_per_bank * moe_intermediate_size_per_core * w_dtype_num_bits // 8
        > OCM_SIZE
    ):
        hidden_dims_per_bank = hidden_dims_per_bank // 2
    assert (
        down_bank_weight_factor * hidden_dims_per_bank * moe_intermediate_size_per_core * w_dtype_num_bits // 8
        <= OCM_SIZE
    )
    real_bank_num = hidden_dim // hidden_dims_per_bank
    # core interleave on hidden_dim for bank conflict
    for j in range(core_num):
        hidden_dim_bank_offset_org = ((j + core_id) % core_num) * hidden_dims_per_bank_org
        slice_hidden_dim_bank_org = slice(
            hidden_dim_bank_offset_org,
            min(hidden_dim_bank_offset_org + hidden_dims_per_bank_org, hidden_dim),
        )
        r_down = ax.dsl.create_tensor(
            (1, slice_hidden_dim_bank_org.stop - slice_hidden_dim_bank_org.start),
            dtype=datatype.DType(x.dtype),
        )
        for k in range(real_bank_num // bank_num):
            hidden_dim_bank_offset = hidden_dim_bank_offset_org + k * hidden_dims_per_bank
            slice_hidden_dim_bank = slice(
                hidden_dim_bank_offset,
                min(hidden_dim_bank_offset + hidden_dims_per_bank, hidden_dim),
            )
            hidden_dims_per_core_cur_core = slice_hidden_dim_bank.stop - slice_hidden_dim_bank.start
            w_down_cur_bank = ax.dsl.index_select(
                w_down[:, slice_hidden_dim_bank, slice_moe_intermediate],
                top_k_experts,
                dim=0,
            )  # [top_k, hidden_dim, moe_intermediate_size]
            slice_scale_down_ic = _scale_dim_slice(slice_moe_intermediate, w_down.shape[-1], w_scale_down.shape[-1])
            w_scale_down_cur_bank = _gather_topk_block_scale(
                w_scale_down[:, slice_hidden_dim_bank, slice_scale_down_ic],
                top_k_experts,
            )  # [top_k, hidden_dim, 1]
            if merge_topk:
                if block_scale:
                    raise ValueError("token_parallel merge_topk does not support packed 4-bit weights")
                w_scale_down_cur_bank_bf16 = ax.dsl.mul(w_scale_down_cur_bank, w_scale2_down, output_dtype=ax.bf16)
                w_down_cur_bank_bf16 = ax.dsl.cast(w_down_cur_bank, dtype=ax.bf16)
                w_down_cur_bank_mul = ax.dsl.mul(
                    w_down_cur_bank_bf16, w_scale_down_cur_bank_bf16
                )  # [top_k, hidden_dim, moe_intermediate_size_per_core]
                w_down_cur_bank_t = ax.dsl.transpose(
                    w_down_cur_bank_mul, perm=(1, 0, 2)
                )  # [hidden_dim, top_k, moe_intermediate_size_per_core]
                ax.dsl.matmul(
                    current_hidden_states_mul,  # [seq_len, top_k * moe_intermediate_size_per_core]
                    w_down_cur_bank_t.reshape(
                        hidden_dims_per_core_cur_core, -1
                    ),  # [hidden_dim, top_k*moe_intermediate_size_per_core]
                    transposed=True,
                    output_dtype=ax.bf16,
                    r=r_down[:, k * hidden_dims_per_bank : (k + 1) * hidden_dims_per_bank],
                )  # [seq_len, hidden_dim]
            elif block_scale:
                # Concatenate the top-k experts along the reduction dim so the
                # down projection also needs a single runtime B/K encode.  The
                # per-expert scale2 was folded into the activation above,
                # keeping the block scale in its hardware (FP8 E4M3)
                # representation.
                w_down_cur_bank_t = ax.dsl.transpose(w_down_cur_bank, perm=(1, 0, 2))
                w_scale_down_cur_bank_t = ax.dsl.transpose(w_scale_down_cur_bank, perm=(1, 0, 2))
                ax.dsl.matmul(
                    current_hidden_states_mul.reshape(seq_len, -1),
                    w_down_cur_bank_t.reshape(hidden_dims_per_core_cur_core, -1),
                    w_scale=w_scale_down_cur_bank_t.reshape(hidden_dims_per_core_cur_core, -1),
                    transposed=True,
                    output_dtype=ax.bf16,
                    r=r_down[:, k * hidden_dims_per_bank : (k + 1) * hidden_dims_per_bank],
                )
            else:
                r_down_sum = ax.dsl.full(
                    (1, hidden_dims_per_core_cur_core),
                    dtype=datatype.DType(x.dtype),
                    fill_value=0.0,
                    prefix=f"r_down_sum_{i}_{core_id}_{j}_{k}",
                )
                w_scale_down_cur_bank_bf16 = ax.dsl.mul(w_scale_down_cur_bank, w_scale2_down, output_dtype=ax.bf16)
                for top_k_idx in range(top_k):
                    topk_down = ax.dsl.matmul(
                        current_hidden_states_mul[:, top_k_idx : top_k_idx + 1, :].reshape(
                            seq_len, -1
                        ),  # [seq_len, top_k * moe_intermediate_size_per_core]
                        w_down_cur_bank[top_k_idx : top_k_idx + 1, :].reshape(
                            hidden_dims_per_core_cur_core, -1
                        ),  # [hidden_dim, moe_intermediate_size_per_core]
                        transposed=True,
                        output_dtype=ax.bf16,
                    )  # [seq_len, hidden_dim]
                    topk_down_mul = ax.dsl.mul(
                        topk_down,
                        w_scale_down_cur_bank_bf16[top_k_idx : top_k_idx + 1, :].reshape(1, -1),
                    )
                    ax.dsl.add(topk_down_mul, r_down_sum, r=r_down_sum)
                ax.dsl.move(
                    r=r_down[:, k * hidden_dims_per_bank : (k + 1) * hidden_dims_per_bank],
                    x=r_down_sum,
                )
        ax.dsl.move(
            r=partial_output[core_id : core_id + 1, :, slice_hidden_dim_bank_org],
            x=r_down.reshape(1, 1, -1),
        )
    axd.pub(partial_output)
    axd.sub(partial_output)
    if core_id == 0:
        reduced = partial_output[0:1, :, :].reshape(1, -1)
        for reduce_core in range(1, core_num):
            reduced = ax.dsl.add(
                reduced,
                partial_output[reduce_core : reduce_core + 1, :, :].reshape(1, -1),
            )
        ax.dsl.move(x=reduced, r=output_ax)


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=bool(int(os.environ.get("DUMP_DSL_PERF", 0))),
    dump_path=pathlib.Path(os.environ.get("DSL_DUMP_PATH") or pathlib.Path.cwd() / "output"),
)
def moe_kernel_token_parallel(
    x,  # [seq_len, hidden_dim]
    w_norm,  # [hidden_dim] effective post-attention norm weight; None disables fusion
    shared_expert_gate,  # [hidden_dim, 1]
    w_shared_expert_mlp_gate,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale_shared_expert_mlp_gate,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale2_shared_expert_mlp_gate,
    w_shared_expert_mlp_up,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale_shared_expert_mlp_up,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale2_shared_expert_mlp_up,
    w_shared_expert_mlp_down,  # [shared_expert_intermediate_size, hidden_dim]
    w_scale_shared_expert_mlp_down,  # [shared_expert_intermediate_size, hidden_dim]
    w_scale2_shared_expert_mlp_down,
    w_route,  # [num_experts, hidden_dim]
    w_scale_route,
    w_scale2_route,
    w_gate,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_gate,  # [1]
    w_up,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_up,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_up,  # [1]
    w_down,  # [num_experts, hidden_dim, moe_intermediate_size]
    w_scale_down,  # [num_experts, hidden_dim, 1]
    w_scale2_down,  # [1]
    top_k,  # 8
    num_experts,  # 128
    norm_topk_prob,  # False
    output_ax,  # [seq_len, hidden_dim]
    chunk_size=None,
    norm_eps=1e-6,
):
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    x = x.reshape(-1, x.shape[-1])  # [B, seq_len, hidden_dim] -> [B*seq_len, hidden_dim]
    seq_len, hidden_dim = x.shape
    seq_len_factor = (seq_len + core_num - 1) // core_num
    slice_seq_slice = slice(core_id * seq_len_factor, min((core_id + 1) * seq_len_factor, seq_len))
    slice_seq_cur_core = slice_seq_slice.stop - slice_seq_slice.start
    num_experts_per_core = (num_experts + core_num - 1) // core_num
    num_experts_offset_core = core_id * num_experts_per_core
    slice_expert_core = slice(
        num_experts_offset_core,
        min(num_experts_offset_core + num_experts_per_core, num_experts),
    )
    # load tokens
    x_mem_scope = ax.MemoryType.SHARED_OCM
    if w_norm is None:
        x_shocm = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        if slice_seq_cur_core > 0:
            ax.dsl.load(r=x_shocm[slice_seq_slice, :], x=x[slice_seq_slice, :])
    else:
        x_shocm = _hidden_parallel_rms_norm(x, w_norm, norm_eps, core_id, core_num, chunk_size or 64)
    axd.sync_all()

    if shared_expert_gate is not None:
        assert (
            w_shared_expert_mlp_gate is not None
            and w_shared_expert_mlp_up is not None
            and w_shared_expert_mlp_down is not None
        )
        # shared expert compute
        shared_expert_output = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        mlp_kernel_expert_parallel(
            x_shocm,
            w_shared_expert_mlp_gate,
            w_scale_shared_expert_mlp_gate,
            w_scale2_shared_expert_mlp_gate,
            w_shared_expert_mlp_up,
            w_scale_shared_expert_mlp_up,
            w_scale2_shared_expert_mlp_up,
            w_shared_expert_mlp_down,
            w_scale_shared_expert_mlp_down,
            w_scale2_shared_expert_mlp_down,
            shared_expert_output,
            core_id,
            core_num,
        )
        axd.sync_all()
        shared_expert_gate_output_sigmoid = ax.dsl.create_tensor((seq_len, 1), x.dtype, x_mem_scope)
        shared_expert_output_scaled = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        if core_id == 0:
            shared_expert_gate_output = ax.dsl.matmul(
                x_shocm,
                ax.dsl.load(shared_expert_gate),
                output_dtype=ax.bf16,
            )  # [seq_len, 1]
            ax.dsl.sigmoid(shared_expert_gate_output, r=shared_expert_gate_output_sigmoid)  # [seq_len, 1]
            ax.dsl.mul(
                shared_expert_output,
                shared_expert_gate_output_sigmoid,
                r=shared_expert_output_scaled,
            )  # [seq_len, hidden_dim]

        axd.sync_all()

    # route
    router_logits = ax.dsl.create_tensor((seq_len, num_experts), x.dtype, x_mem_scope)
    ax.dsl.matmul(
        x_shocm,
        ax.dsl.load(w_route[slice_expert_core, :]),
        w_scale=ax.dsl.load(w_scale_route[slice_expert_core, :]),
        w_scale2=ax.dsl.load(w_scale2_route),
        transposed=True,
        output_dtype=ax.bf16,
        r=router_logits[:, slice_expert_core],
    )
    axd.sync_all()
    # topK
    top_k_weights = ax.dsl.create_tensor((seq_len, top_k), ax.bf16, x_mem_scope)
    top_k_index = ax.dsl.create_tensor((seq_len, top_k), ax.int32, x_mem_scope)
    if slice_seq_cur_core > 0:
        router_logits_ = ax.dsl.softmax(router_logits[slice_seq_slice, :], dim=-1)
        ax.dsl.topk(
            router_logits_,
            k=top_k,
            dim=-1,
            r_values=top_k_weights[slice_seq_slice, :],
            r_indices=top_k_index[slice_seq_slice, :],
        )
        if norm_topk_prob:
            top_k_weights_sum = ax.dsl.sum(top_k_weights[slice_seq_slice, :], dim=-1, keepdim=True)
            ax.dsl.div(
                top_k_weights[slice_seq_slice, :],
                top_k_weights_sum,
                r=top_k_weights[slice_seq_slice, :],
            )
    axd.sync_all()
    # moe process: 每个token收集top_k个专家的计算结果进行加权平均，最终得到每个token的输出
    # tokens: cluster or device parallel, mlp: core parallel
    for i in range(seq_len):
        final_hidden_states = ax.dsl.create_tensor((1, hidden_dim), x.dtype, x_mem_scope)
        mlp_kernel_token_parallel(
            x_shocm[i : i + 1, :],  # [1, hidden_dim]
            w_gate,  # [num_experts, moe_intermediate_size, hidden_dim]
            w_scale_gate,
            w_scale2_gate,
            w_up,  # [num_experts, moe_intermediate_size, hidden_dim]
            w_scale_up,
            w_scale2_up,
            w_down,  # [num_experts, hidden_dim, moe_intermediate_size]
            w_scale_down,
            w_scale2_down,
            core_id,
            core_num,
            top_k,
            top_k_index[i : i + 1, :].reshape(-1),
            top_k_weights[i : i + 1, :].reshape(-1),
            final_hidden_states,  # [1, hidden_dim]
            i,
        )
        axd.sync_all()
        if core_id == 0:
            final_hidden_states_shared_expert = (
                ax.dsl.add(final_hidden_states, shared_expert_output_scaled[i : i + 1, :])
                if shared_expert_gate is not None
                else final_hidden_states
            )
            if w_norm is not None:
                final_hidden_states_shared_expert = ax.dsl.add(
                    final_hidden_states_shared_expert, ax.dsl.load(x[i : i + 1, :])
                )
            ax.dsl.store(r=output_ax[i : i + 1, :], x=final_hidden_states_shared_expert)

    return output_ax


_MISSING = object()
_ONE_SCALE2 = np.array([1.0], dtype=np.float32)


def _gather_expert_rows(tensor, expert_ids, row_start, row_count, row_step=1, banks=None):
    """Gather DDR [E,R,C] tiles at runtime expert/row offsets into OCM [K,N,C]."""
    experts, rows, columns = tensor.shape
    expert_stride = ax.dsl.full((1,), ax.int32, fill_value=rows)
    offsets = ax.dsl.from_numpy(np.arange(row_count, dtype=np.int32) * row_step)
    indices = expert_ids.reshape(-1, 1) * expert_stride + row_start + offsets
    selected = gather_rows(tensor.reshape(experts * rows, columns), indices.reshape(-1), dim=0, banks=banks)
    return selected.reshape(expert_ids.shape_size, row_count, columns)


def _decode_prefetched_expert_groups(
    x,
    selected_ids,
    route_weights,
    w_gate,
    w_scale_gate,
    w_scale2_gate,
    w_down,
    w_scale_down,
    w_scale2_down,
    core_id,
    core_num,
    banks=None,
):
    """Use fixed alternating weight buffers, without carrying new payloads."""
    experts, hidden, intermediate = w_down.shape
    active = selected_ids.shape[-1]
    expert_tile = gcd(4, active)
    channels = intermediate // core_num
    width = (hidden + core_num - 1) // core_num
    output_slice = slice(core_id * width, min((core_id + 1) * width, hidden))
    width = output_slice.stop - output_slice.start
    # Select contiguous output-row blocks so short group-scale rows travel
    # in the same wide bursts as their weights, with far fewer row indices.
    down_row_group = gcd(64, hidden, width, output_slice.start)
    output_offset = ax.dsl.full((1,), ax.int32, output_slice.start // down_row_group)
    local_channels = slice(core_id * channels, (core_id + 1) * channels)
    activated = ax.dsl.create_tensor((active, intermediate), x.dtype, ax.MemoryType.SHARED_OCM)
    owned_activation = activated[:, local_channels]
    ax.dsl.expand(ax.dsl.zeros((1,), x.dtype), shape=owned_activation.shape, r=owned_activation)
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, 1)
    two = ax.dsl.full((1,), ax.int32, 2)
    unit_scale = ax.dsl.full((1,), ax.fp32, 1)
    offsets = ax.dsl.from_numpy(np.arange(expert_tile, dtype=np.int32))
    step = ax.dsl.full((1,), ax.int32, expert_tile)
    expert_stride = ax.dsl.full((1,), ax.int32, 2 * intermediate)
    down_stride = ax.dsl.full((1,), ax.int32, hidden // down_row_group)
    channel_offset = ax.dsl.full((1,), ax.int32, local_channels.start)
    rows = ax.dsl.from_numpy(
        np.array([0, intermediate], dtype=np.int32).reshape(1, 2, 1)
        + np.arange(channels, dtype=np.int32).reshape(1, 1, channels)
    )
    down_offsets = ax.dsl.from_numpy(np.arange(width // down_row_group, dtype=np.int32))
    all_scale2 = gather_rows(w_scale2_gate, selected_ids.reshape(-1), dim=0, banks=banks)
    all_down_scale2 = _gather_topk_scale2(w_scale2_down, selected_ids.reshape(-1), banks=banks)

    def buffers():
        return (
            ax.dsl.create_tensor((expert_tile * 2 * channels, hidden), w_gate.dtype),
            ax.dsl.create_tensor((expert_tile * 2 * channels, w_scale_gate.shape[-1]), w_scale_gate.dtype),
            ax.dsl.create_tensor((expert_tile, width, intermediate), w_down.dtype),
            ax.dsl.create_tensor((expert_tile, width, w_scale_down.shape[-1]), w_scale_down.dtype),
        )

    def prepare(group, gate, scale, down, down_scale):
        positions = group * step + offsets
        ids = gather_rows(selected_ids.reshape(-1), positions, dim=0, banks=banks)
        gate_rows = ids.reshape(expert_tile, 1, 1) * expert_stride + channel_offset + rows
        gather_rows(
            w_gate.reshape(experts * 2 * intermediate, hidden),
            gate_rows.reshape(-1),
            dim=0,
            r=gate,
            banks=banks,
        )
        gather_rows(
            w_scale_gate.reshape(experts * 2 * intermediate, w_scale_gate.shape[-1]),
            gate_rows.reshape(-1),
            dim=0,
            r=scale,
            banks=banks,
        )
        down_rows = ids.reshape(-1, 1) * down_stride + output_offset + down_offsets
        gather_rows(
            w_down.reshape(experts * hidden // down_row_group, down_row_group * intermediate),
            down_rows.reshape(-1),
            dim=0,
            r=down.reshape(-1, down_row_group * intermediate),
            banks=banks,
        )
        gather_rows(
            w_scale_down.reshape(experts * hidden // down_row_group, down_row_group * w_scale_down.shape[-1]),
            down_rows.reshape(-1),
            dim=0,
            r=down_scale.reshape(-1, down_row_group * w_scale_down.shape[-1]),
            banks=banks,
        )

    def compute(total, group, gate, scale, down, down_scale):
        positions = group * step + offsets
        scale2 = gather_rows(all_scale2, positions, dim=0, banks=banks)
        down_scale2 = gather_rows(all_down_scale2, positions, dim=0, banks=banks)
        projected = linear(
            x,
            gate,
            scale,
            "decode_gate_up",
            weight_scale2=unit_scale,
            ic_tile_size=min(512, hidden),
            oc_tile_size=256,
            output_dtype=ax.fp32,
        ).reshape(expert_tile, 2, channels)
        projected = ax.dsl.mul(projected, scale2.reshape(expert_tile, 2, 1), output_dtype=x.dtype)
        gate_value = projected[:, :1, :].reshape(expert_tile, channels)
        up_value = projected[:, 1:, :].reshape(expert_tile, channels)
        scatter_rows(
            ax.dsl.mul(ax.dsl.silu(gate_value), up_value),
            positions,
            dim=0,
            r=activated[:, local_channels],
            banks=banks,
        )
        axd.pub(activated)
        axd.sub(activated)
        values = gather_rows(activated, positions, dim=0, banks=banks)
        down_total = ax.dsl.zeros((expert_tile, 1, width), ax.fp32)
        ax.dsl.matmul(
            values.reshape(expert_tile, 1, intermediate),
            down,
            w_scale=down_scale,
            transposed=True,
            output_dtype=ax.fp32,
            r=down_total,
            output_add_assign=True,
        )
        result = ax.dsl.mul(down_total, down_scale2.reshape(-1, 1, 1), output_dtype=x.dtype)
        probabilities = gather_rows(route_weights.reshape(-1), positions, dim=0, banks=banks).reshape(expert_tile, 1, 1)
        weighted = ax.dsl.mul(result, probabilities, output_dtype=ax.fp32)
        # Keep the original two-expert FP32 reduction order.
        for begin in range(0, expert_tile, 2):
            contribution = ax.dsl.sum(weighted[begin : begin + 2], dim=0, output_dtype=ax.fp32).reshape(1, width)
            total = ax.dsl.add(total, contribution, output_dtype=ax.fp32)
        return total

    def body(pair, total, *owned):
        first, second = owned[:4], owned[4:]
        group = pair * two + one
        prepare(group + one, *first)
        total = compute(total, group, *second)
        prepare(group + two, *second)
        total = compute(total, group + one, *first)
        # Returning the same buffers preserves their OCM ownership. Only the
        # small FP32 sum changes allocation; no weight payload needs copying.
        return total, *owned

    first = buffers()
    prepare(zero, *first)
    total = ax.dsl.zeros((1, width), ax.fp32)
    groups = active // expert_tile
    if groups == 1:
        return compute(total, zero, *first)
    # Both loop operands need an initial value. Prime the two required groups
    # before entering Range; the first compute overlaps the second group read.
    second = buffers()
    prepare(one, *second)
    total = compute(total, zero, *first)
    pairs = (groups - 2) // 2
    if pairs:
        carried = ax.dsl.range(
            ax.dsl.zeros((1,), ax.int32),
            zero,
            ax.dsl.full((1,), ax.int32, pairs),
            one,
            body,
            total,
            *first,
            *second,
            unroll_factor=1,
        )
        total, first, second = carried[0], carried[1:5], carried[5:]
    if groups % 2:
        before_last = ax.dsl.full((1,), ax.int32, groups - 2)
        last = ax.dsl.full((1,), ax.int32, groups - 1)
        # The static tail needs its own tensor identity. Reusing the first
        # set here would let root-BODY scheduling overwrite an earlier read.
        final = buffers()
        prepare(last, *final)
        total = compute(total, before_last, *second)
        return compute(total, last, *final)
    return compute(total, ax.dsl.full((1,), ax.int32, groups - 1), *second)


def _decode_routed_experts(
    x,
    selected_ids,
    route_weights,
    w_gate,
    w_scale_gate,
    w_scale2_gate,
    w_up,
    w_scale_up,
    w_scale2_up,
    w_down,
    w_scale_down,
    w_scale2_down,
    shared_output,
    residual,
    output,
    fused_gate_up,
    core_id,
    core_num,
    banks=None,
):
    """Run expert groups and cooperative C waves through non-unrolled Ranges."""
    hidden_dim, intermediate = x.shape[-1], w_down.shape[-1]
    experts, active_experts = w_down.shape[0], selected_ids.shape[-1]
    expert_tile = gcd(2, active_experts)
    # C128/core covers I512 in one wave on four cores. A divisor tile keeps
    # every wave rectangular for other intermediate sizes as well.
    channel_tile = gcd(128, intermediate // core_num)
    wave_width = core_num * channel_tile
    assert intermediate % wave_width == 0
    wave_count = intermediate // wave_width
    hidden_per_core = (hidden_dim + core_num - 1) // core_num
    output_channels = slice(core_id * hidden_per_core, min((core_id + 1) * hidden_per_core, hidden_dim))
    output_width = output_channels.stop - output_channels.start
    scale_rows = _scale_oc_slice(output_channels) if _is_block_scale(w_scale_down, w_down) else output_channels

    # Root publications are BODY-local. Consume shared inputs into private OCM
    # before entering Range; only the current C wave is shared inside its BODY.
    local_x = ax.dsl.create_tensor(x.shape, x.dtype)
    local_ids = ax.dsl.create_tensor(selected_ids.shape, selected_ids.dtype)
    # Carry a private FP32 copy of routing probabilities into the expert Range.
    local_weights = ax.dsl.create_tensor(route_weights.shape, ax.fp32)
    ax.dsl.move(route_weights, r=local_weights)
    ax.dsl.move(x, r=local_x)
    ax.dsl.move(selected_ids, r=local_ids)
    if shared_output is not None:
        shared_local = ax.dsl.create_tensor((1, output_width), x.dtype)
        ax.dsl.move(shared_output[:, output_channels], r=shared_local)
    if residual is not None:
        residual_local = ax.dsl.create_tensor((1, output_width), x.dtype)
        ax.dsl.move(residual[:, output_channels], r=residual_local)

    buffered_experts = gcd(4, active_experts)
    groups = active_experts // buffered_experts
    buffer_count = min(2, groups) + int(groups > 1 and groups % 2 != 0)
    workspace = (
        buffer_count * buffered_experts * ((2 * channel_tile * hidden_dim + hidden_per_core * intermediate) * 9 // 16)
    )
    if (
        fused_gate_up
        and wave_count == 1
        and w_gate.dtype == ax.nvfp4
        and w_scale_gate.dtype == ax.fp8_e4m3
        and w_scale_down.shape[-1] == intermediate // 16
        and workspace <= 4 * 1024 * 1024
    ):
        total = _decode_prefetched_expert_groups(
            local_x,
            local_ids,
            local_weights,
            w_gate,
            w_scale_gate,
            w_scale2_gate,
            w_down,
            w_scale_down,
            w_scale2_down,
            core_id,
            core_num,
            banks=banks,
        )
    else:
        zero = ax.dsl.zeros((1,), dtype=ax.int32)
        one = ax.dsl.full((1,), ax.int32, fill_value=1)
        unit_scale2 = ax.dsl.full((1,), ax.fp32, fill_value=1)
        expert_offsets = ax.dsl.from_numpy(np.arange(expert_tile, dtype=np.int32))
        expert_step = ax.dsl.full((1,), ax.int32, fill_value=expert_tile)
        wave_stride = ax.dsl.full((1,), ax.int32, fill_value=wave_width)
        core_offset = ax.dsl.full((1,), ax.int32, fill_value=core_id * channel_tile)
        wave_stop = ax.dsl.full((1,), ax.int32, fill_value=wave_count)

        # Rows represent [Top-K,C-wave], with all cores' C shards contiguous.
        # Different loop iterations own distinct rows, so fast cores cannot overwrite
        # data still being consumed by a slower core. Scatter and gather move rows.
        activated = ax.dsl.create_tensor((active_experts * wave_count, wave_width), x.dtype, ax.MemoryType.SHARED_OCM)
        local_channels = slice(core_id * channel_tile, (core_id + 1) * channel_tile)
        owned_activation = activated[:, local_channels]
        ax.dsl.expand(
            ax.dsl.zeros((1,), dtype=x.dtype),
            shape=owned_activation.shape,
            r=owned_activation,
        )
        wave_count_tensor = ax.dsl.full((1,), ax.int32, fill_value=wave_count)

        def expert_group_body(group_index, total):
            # [K] INT32 runtime Top-K positions -> [K] INT32 logical expert IDs.
            positions = group_index * expert_step + expert_offsets
            ids = gather_rows(local_ids.reshape(-1), positions, dim=0, banks=banks)
            down_scale2 = _gather_topk_scale2(w_scale2_down, ids, banks=banks)
            if fused_gate_up:
                scales2 = gather_rows(w_scale2_gate, ids, dim=0, banks=banks)
            else:
                gate_scale2 = _gather_topk_scale2(w_scale2_gate, ids, banks=banks)
                up_scale2 = _gather_topk_scale2(w_scale2_up, ids, banks=banks)
            if w_scale_down.shape[-1] == 1:
                channel_scale = gather_rows(w_scale_down[:, scale_rows, :], ids, dim=0, banks=banks)

            def channel_body(wave_index, ids, positions, *state):
                # Parent-BODY results must be explicit operands of a nested Range.
                if fused_gate_up:
                    scales2 = state[0]
                else:
                    gate_scale2, up_scale2 = state[:2]
                if w_scale_down.shape[-1] == 1:
                    channel_scale = state[-2]
                down_total = state[-1]
                channel_start = wave_index * wave_stride + core_offset
                if fused_gate_up:
                    # Gather only this core's Gate/Up C rows for K runtime experts.
                    gate_up_rows = w_gate.reshape(experts, 2 * intermediate, hidden_dim)
                    gate_up_scales = w_scale_gate.reshape(experts, 2 * intermediate, w_scale_gate.shape[-1])
                    halves = ax.dsl.from_numpy(np.array([0, intermediate], dtype=np.int32))
                    row_offsets = halves.reshape(1, 2, 1) + ax.dsl.from_numpy(
                        np.arange(channel_tile, dtype=np.int32)
                    ).reshape(1, 1, channel_tile)
                    expert_stride = ax.dsl.full((1,), ax.int32, fill_value=2 * intermediate)
                    rows = ids.reshape(expert_tile, 1, 1) * expert_stride + channel_start + row_offsets
                    gate_up = gather_rows(
                        gate_up_rows.reshape(-1, hidden_dim),
                        rows.reshape(-1),
                        dim=0,
                        banks=banks,
                    )
                    scales = gather_rows(
                        gate_up_scales.reshape(-1, w_scale_gate.shape[-1]),
                        rows.reshape(-1),
                        dim=0,
                        banks=banks,
                    )
                    projected = linear(
                        local_x,
                        gate_up,
                        scales,
                        "decode_gate_up",
                        weight_scale2=unit_scale2,
                        ic_tile_size=min(512, hidden_dim),
                        oc_tile_size=256,
                        output_dtype=ax.fp32,
                    ).reshape(expert_tile, 2, channel_tile)
                    projected = ax.dsl.mul(projected, scales2.reshape(expert_tile, 2, 1), output_dtype=x.dtype)
                    gate = projected[:, :1, :].reshape(expert_tile, channel_tile)
                    up = projected[:, 1:, :].reshape(expert_tile, channel_tile)
                else:
                    gate_weight = _gather_expert_rows(w_gate, ids, channel_start, channel_tile, banks=banks)
                    up_weight = _gather_expert_rows(w_up, ids, channel_start, channel_tile, banks=banks)
                    gate_block = intermediate // w_scale_gate.shape[-2]
                    up_block = intermediate // w_scale_up.shape[-2]
                    gate_start = channel_start
                    if gate_block != 1:
                        gate_start = ax.dsl.div(
                            channel_start,
                            ax.dsl.full((1,), ax.int32, fill_value=gate_block),
                            output_dtype=ax.int32,
                        )
                    up_start = channel_start
                    if up_block != 1:
                        up_start = ax.dsl.div(
                            channel_start,
                            ax.dsl.full((1,), ax.int32, fill_value=up_block),
                            output_dtype=ax.int32,
                        )
                    gate_scale = _gather_expert_rows(
                        w_scale_gate,
                        ids,
                        gate_start,
                        channel_tile // gate_block,
                        banks=banks,
                    )
                    up_scale = _gather_expert_rows(w_scale_up, ids, up_start, channel_tile // up_block, banks=banks)
                    gate = ax.dsl.matmul(
                        local_x,
                        gate_weight.reshape(-1, hidden_dim),
                        w_scale=gate_scale.reshape(-1, gate_scale.shape[-1]),
                        transposed=True,
                        output_dtype=ax.fp32,
                    ).reshape(expert_tile, channel_tile)
                    up = ax.dsl.matmul(
                        local_x,
                        up_weight.reshape(-1, hidden_dim),
                        w_scale=up_scale.reshape(-1, up_scale.shape[-1]),
                        transposed=True,
                        output_dtype=ax.fp32,
                    ).reshape(expert_tile, channel_tile)
                    gate = ax.dsl.mul(gate, gate_scale2.reshape(-1, 1), output_dtype=x.dtype)
                    up = ax.dsl.mul(up, up_scale2.reshape(-1, 1), output_dtype=x.dtype)

                # [K,C/core] BF16 -> K disjoint [C] rows in preallocated SHOCM.
                activation_rows = positions * wave_count_tensor + wave_index
                scatter_rows(
                    ax.dsl.mul(ax.dsl.silu(gate), up),
                    activation_rows,
                    dim=0,
                    r=activated[:, local_channels],
                    banks=banks,
                )
                axd.pub(activated)
                axd.sub(activated)
                wave_values = gather_rows(activated, activation_rows, dim=0, banks=banks)
                # Flatten IC waves into rows so runtime C selection fetches each
                # Down payload once instead of gathering complete expert weights.
                down_rows = w_down.reshape(experts, hidden_dim * wave_count, wave_width)
                down_start = ax.dsl.full((1,), ax.int32, fill_value=output_channels.start * wave_count) + wave_index
                down = _gather_expert_rows(down_rows, ids, down_start, output_width, wave_count, banks=banks)
                if w_scale_down.shape[-1] == 1:
                    down_scale = channel_scale
                else:
                    scale_width = w_scale_down.shape[-1] // wave_count
                    scale_tensor = w_scale_down.reshape(experts, w_scale_down.shape[-2] * wave_count, scale_width)
                    scale_start = ax.dsl.full((1,), ax.int32, fill_value=scale_rows.start * wave_count) + wave_index
                    down_scale = _gather_expert_rows(
                        scale_tensor,
                        ids,
                        scale_start,
                        scale_rows.stop - scale_rows.start,
                        wave_count,
                        banks=banks,
                    )
                # [K,1,C] BF16 @ [K,H/core,C] weights -> [K,1,H/core] FP32 NMC.
                ax.dsl.matmul(
                    wave_values.reshape(expert_tile, 1, wave_width),
                    down,
                    w_scale=down_scale,
                    transposed=True,
                    output_dtype=ax.fp32,
                    r=down_total,
                    output_add_assign=True,
                )
                return (ids, positions, *state[:-1], down_total)

            down_total = ax.dsl.zeros((expert_tile, 1, output_width), dtype=ax.fp32)
            channel_inputs = [ids, positions, scales2] if fused_gate_up else [ids, positions, gate_scale2, up_scale2]
            if w_scale_down.shape[-1] == 1:
                channel_inputs.append(channel_scale)
            if wave_count == 1:
                # No loop boundary is needed for a single C wave; experts still
                # run through the non-unrolled Range below.
                down_total = channel_body(zero, *channel_inputs, down_total)[-1]
            else:
                down_total = ax.dsl.range(
                    ax.dsl.zeros((1,), dtype=ax.int32),
                    zero,
                    wave_stop,
                    one,
                    channel_body,
                    *channel_inputs,
                    down_total,
                    unroll_factor=1,
                )[-1]
            projected = ax.dsl.mul(down_total, down_scale2.reshape(-1, 1, 1), output_dtype=x.dtype)
            weights = gather_rows(local_weights.reshape(-1), positions, dim=0, banks=banks).reshape(expert_tile, 1, 1)
            weighted = ax.dsl.mul(projected, weights, output_dtype=ax.fp32)
            result = ax.dsl.sum(weighted, dim=0, output_dtype=ax.fp32).reshape(1, output_width)
            return (ax.dsl.add(total, result, output_dtype=ax.fp32),)

        total = ax.dsl.zeros((1, output_width), dtype=ax.fp32)
        total = ax.dsl.range(
            ax.dsl.zeros((1,), dtype=ax.int32),
            zero,
            ax.dsl.full((1,), ax.int32, fill_value=active_experts // expert_tile),
            one,
            expert_group_body,
            total,
            unroll_factor=1,
        )[0]
    if shared_output is not None:
        total = ax.dsl.add(total, shared_local)
    if residual is not None:
        # The final residual add writes directly to a fused on-chip handoff.
        destination = output[:, output_channels] if output.memory_type != ax.MemoryType.DDR else None
        total = ax.dsl.add(total, residual_local, output_dtype=output.dtype, r=destination)
    if output.memory_type == ax.MemoryType.DDR:
        if residual is None:
            total = ax.dsl.cast(total, dtype=output.dtype)
        store_tile(total, r=output[:, output_channels], banks=banks)
    elif residual is None:
        ax.dsl.cast(total, dtype=output.dtype, r=output[:, output_channels])


def _prefill_routed_experts(
    x,
    token_map,
    weight_map,
    accumulated,
    w_gate,
    w_scale_gate,
    w_scale2_gate,
    w_down,
    w_scale_down,
    w_scale2_down,
    output,
    core,
    cores,
    row_tile,
    keep_input_resident=False,
    banks=None,
):
    """Visit active experts in ID order, one expert at a time.

    All four cores cooperate on an expert: Gate/Up owns I/cores channels,
    Down owns H/cores output channels. Active experts load their weight
    shards once into private OCM and reuse them for every token tile.
    Discover nonempty experts once and prefetch the next expert's weights.
    For DDR-backed input, also prefetch its first token tile. The final
    expert consumes the cache without another lookahead load.
    """
    tokens = token_map.shape[1]
    experts, hidden, intermediate = w_down.shape
    hidden_width = hidden // cores
    channels = intermediate // cores
    prefetch_input = tokens > 2048 and not keep_input_resident
    hs = slice(core * hidden_width, (core + 1) * hidden_width)
    cs = slice(core * channels, (core + 1) * channels)
    local_input = x[:, hs] if prefetch_input else load_tile(x[:, hs], banks=banks)
    shared_input = ax.dsl.create_tensor((row_tile, hidden), ax.bf16, ax.MemoryType.SHARED_OCM)
    activated = ax.dsl.create_tensor((row_tile, intermediate), ax.bf16, ax.MemoryType.SHARED_OCM)
    epoch = ax.dsl.create_tensor(
        (cores, ((row_tile + 63) // 64) * ((hidden_width + 255) // 256)),
        ax.bf16,
        ax.MemoryType.SHARED_OCM,
    )
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    tile = ax.dsl.full((1,), ax.int32, fill_value=row_tile)
    sentinel = ax.dsl.full((1,), ax.int32, fill_value=tokens)
    expert_loop_index = ax.dsl.zeros((1,), ax.int32)
    token_loop_index = ax.dsl.zeros((1,), ax.int32)
    final_token_loop_index = ax.dsl.zeros((1,), ax.int32)
    zero_bf16 = ax.dsl.zeros((1,), ax.bf16)
    positions = ax.dsl.from_numpy(np.arange(row_tile, dtype=np.int32))
    gate_rows = ax.dsl.from_numpy(
        np.concatenate(
            (
                np.arange(cs.start, cs.stop),
                np.arange(intermediate + cs.start, intermediate + cs.stop),
            )
        ).astype(np.int32)
    )
    down_rows = ax.dsl.from_numpy(np.arange(hs.start, hs.stop, dtype=np.int32))
    down_stride = ax.dsl.full((1,), ax.int32, fill_value=hidden)
    gate_stride = ax.dsl.full((1,), ax.int32, fill_value=2 * intermediate)
    unit_scale = ax.dsl.full((1,), ax.fp32, fill_value=1)

    def prepare_tokens(indices, count, weight_row, slots, *, r=None):
        raw = gather_rows(indices.reshape(tokens), slots, dim=0, banks=banks)
        active = ax.dsl.minimum(ax.dsl.maximum(ax.dsl.sub(count, slots, output_dtype=ax.int32), zero), one)
        selected = raw * active
        # NonZero leaves padded indices unspecified. Gather a valid input row,
        # zero its probability, and scatter the inactive result to the sentinel.
        rows = selected + sentinel * (one - active)
        weights = gather_rows(weight_row.reshape(tokens), selected, dim=0, banks=banks)
        weights = ax.dsl.mul(weights, active, output_dtype=ax.bf16).reshape(-1, 1)
        loaded = gather_rows(local_input, selected, dim=0, r=r, banks=banks)
        return rows, weights, loaded

    def chunk_body(
        chunk,
        indices,
        count,
        weight_row,
        total,
        gate_weight,
        gate_scale,
        gate_scale2,
        down_weight,
        down_scale,
        down_scale2,
        prepared=None,
    ):
        if prepared is None:
            slots = chunk * tile + positions
            rows, weights, _ = prepare_tokens(indices, count, weight_row, slots, r=shared_input[:, hs])
        else:
            rows, weights, loaded_input = prepared
            ax.dsl.move(loaded_input, r=shared_input[:, hs])
        axd.pub(shared_input)
        axd.sub(shared_input)

        # One MatMul lets native IC jobs share an atomic NMC destination.
        projected = ax.dsl.expand(zero_bf16, shape=(row_tile, 2 * channels))
        ax.dsl.matmul(
            shared_input,
            gate_weight,
            w_scale=gate_scale,
            w_scale2=unit_scale,
            transposed=True,
            output_dtype=ax.bf16,
            output_add_assign=True,
            r=projected,
        )
        projected = ax.dsl.mul(projected.reshape(row_tile, 2, channels), gate_scale2, output_dtype=ax.bf16)
        gate = projected[:, :1].reshape(row_tile, channels)
        up = projected[:, 1:].reshape(row_tile, channels)
        swiglu = ax.dsl.mul(ax.dsl.silu(gate), up, output_dtype=ax.bf16)
        ax.dsl.mul(swiglu, weights, output_dtype=ax.bf16, r=activated[:, cs])
        axd.pub(activated)
        axd.sub(activated)

        result = ax.dsl.create_tensor((row_tile, hidden_width), ax.bf16)
        for m in range(0, row_tile, 64):
            m_stop = min(m + 64, row_tile)
            ax.dsl.matmul(
                activated[m:m_stop],
                down_weight,
                w_scale=down_scale,
                w_scale2=down_scale2,
                transposed=True,
                output_dtype=ax.bf16,
                r=result[m:m_stop],
            )
        ax.dsl.index_add(result, rows, dim=0, r=total)
        # Publish a completion value from every physical output tile.
        completed = [result[m : m + 1, o : o + 1] for m in range(0, row_tile, 64) for o in range(0, hidden_width, 256)]
        ax.dsl.concat(*completed, dim=1, r=epoch[core : core + 1])
        axd.pub(epoch)
        axd.sub(epoch)
        return (
            indices,
            count,
            weight_row,
            total,
            gate_weight,
            gate_scale,
            gate_scale2,
            down_weight,
            down_scale,
            down_scale2,
        )

    def load_weights(expert):
        wi = expert * gate_stride + gate_rows
        gate_weight = gather_rows(w_gate.reshape(experts * 2 * intermediate, hidden), wi, dim=0, banks=banks)
        gate_scale = gather_rows(w_scale_gate.reshape(experts * 2 * intermediate, -1), wi, dim=0, banks=banks)
        gate_scale2 = gather_rows(w_scale2_gate, expert, dim=0, banks=banks).reshape(1, 2, 1)
        di = expert * down_stride + down_rows
        down_weight = gather_rows(w_down.reshape(experts * hidden, intermediate), di, dim=0, banks=banks)
        down_scale = gather_rows(w_scale_down.reshape(experts * hidden, -1), di, dim=0, banks=banks)
        down_scale2 = gather_rows(w_scale2_down, expert, dim=0, banks=banks).reshape(1)
        return gate_weight, gate_scale, gate_scale2, down_weight, down_scale, down_scale2

    def load_routes(expert):
        indices, count = ax.dsl.nonzero(gather_rows(token_map, expert, dim=0, banks=banks))
        weight_row = gather_rows(weight_map, expert, dim=0, banks=banks)
        return indices, count.reshape(1), weight_row

    def expert_body(
        expert_index,
        total,
        indices,
        count,
        weight_row,
        gate_weight,
        gate_scale,
        gate_scale2,
        down_weight,
        down_scale,
        down_scale2,
        *prepared,
    ):
        first_tile = chunk_body(
            zero,
            indices,
            count,
            weight_row,
            total,
            gate_weight,
            gate_scale,
            gate_scale2,
            down_weight,
            down_scale,
            down_scale2,
            prepared=prepared if prefetch_input else None,
        )
        # Only immutable next-expert data is prepared ahead of computation.
        next_expert = gather_rows(active_ids, expert_index + one, dim=0, banks=banks).reshape(1)
        next_routes = load_routes(next_expert)
        next_weights = load_weights(next_expert)
        next_prepared = prepare_tokens(*next_routes, positions) if prefetch_input else ()
        stop = ax.dsl.ceil(ax.dsl.div(count, tile, output_dtype=ax.fp32))
        total = ax.dsl.range(token_loop_index, one, stop, one, chunk_body, *first_tile, unroll_factor=1)[3]
        return total, *next_routes, *next_weights, *next_prepared

    def active_experts(total, count):
        expert = gather_rows(active_ids, zero, dim=0, banks=banks).reshape(1)
        routes = load_routes(expert)
        cached_weights = load_weights(expert)
        prepared = prepare_tokens(*routes, positions) if prefetch_input else ()
        state = ax.dsl.range(
            expert_loop_index,
            zero,
            count - one,
            one,
            expert_body,
            total,
            *routes,
            *cached_weights,
            *prepared,
            unroll_factor=1,
        )
        # The final expert consumes the cache without an out-of-range prefetch.
        (
            total,
            indices,
            token_count,
            weight_row,
            gate_weight,
            gate_scale,
            gate_scale2,
            down_weight,
            down_scale,
            down_scale2,
            *prepared,
        ) = state
        first_tile = chunk_body(
            zero,
            indices,
            token_count,
            weight_row,
            total,
            gate_weight,
            gate_scale,
            gate_scale2,
            down_weight,
            down_scale,
            down_scale2,
            prepared=prepared if prefetch_input else None,
        )
        stop = ax.dsl.ceil(ax.dsl.div(token_count, tile, output_dtype=ax.fp32))
        total = ax.dsl.range(final_token_loop_index, one, stop, one, chunk_body, *first_tile, unroll_factor=1)[3]
        return (total,)

    def empty_experts(total, count):
        return (total,)

    counts = ax.dsl.sum(token_map, dim=1, output_dtype=ax.int32)
    active_ids, active_count = ax.dsl.nonzero(counts)
    active_count = ax.dsl.cast(active_count, dtype=ax.int32)
    accumulated = ax.dsl.if_op(
        active_count,
        None,
        active_experts,
        empty_experts,
        accumulated,
        active_count,
    )[0]
    store_tile(accumulated[:tokens], r=output[:, hs], banks=banks)
    return output


def _prefill_router_projection(x, weight):
    """Accumulate independent native IC partitions through one FP32 NMC call."""
    result = ax.dsl.create_tensor((*x.shape[:-1], weight.shape[0]), ax.fp32, name="prefill_router_accumulator")
    ax.dsl.expand(ax.dsl.zeros((1,), ax.fp32), shape=result.shape, r=result)
    # Keep all rows in each IC shard to amortize CONV startup. Native NMC
    # combines the independent shards without an explicit TENG reduction.
    return ax.dsl.matmul(
        x,
        weight,
        transposed=True,
        output_dtype=ax.fp32,
        output_add_assign=True,
        r=result,
    )


def _prefill_shared_gate_projection(x, weight, output_dtype=ax.fp32):
    """Keep the single-output gate on TENG with FP32 products and reduction."""
    product = ax.dsl.mul(x, weight.reshape(1, -1), output_dtype=ax.fp32)
    return ax.dsl.sum(product, dim=-1, keepdim=True, output_dtype=output_dtype)


def _large_prefill_moe(
    x,
    shared_expert_gate,
    w_shared_expert_mlp_gate,
    w_scale_shared_expert_mlp_gate,
    w_scale2_shared_expert_mlp_gate,
    w_shared_expert_mlp_up,
    w_scale_shared_expert_mlp_up,
    w_scale2_shared_expert_mlp_up,
    w_shared_expert_mlp_down,
    w_scale_shared_expert_mlp_down,
    w_scale2_shared_expert_mlp_down,
    w_route,
    w_scale_route,
    w_scale2_route,
    w_gate,
    w_scale_gate,
    w_scale2_gate,
    w_down,
    w_scale_down,
    w_scale2_down,
    top_k,
    num_experts,
    norm_topk_prob,
    output_ax,
    expert_tokens,
    input_norm_w=None,
    norm_eps=1e-06,
    add_residual=False,
    keep_input_resident=False,
    valid_tokens=None,
    banks=None,
):
    """Stream 256-token preparation tiles, then process routed experts in C shards.

    Four cores support complete 256-token blocks for T=512..4096. Expert tiles
    contain 64/128/128/128 rows for T=512/1024/2048/4096. The DDR output
    holds normalized input until all expert readers finish; its final write is
    therefore after the routed Range. Each core keeps [T+1,H/4] BF16 output,
    including shared-expert and residual contributions, in private OCM.
    """

    tokens, hidden = x.shape[-2:]
    core, cores = ax.program_id(0), ax.num_programs(0)
    experts = num_experts
    intermediate = w_down.shape[-1]
    # Full-model activations do not fit SHOCM. The output DDR buffer holds
    # normalized input until every expert has finished reading it.
    assert cores == 4 and tokens % 256 == 0 and 256 < tokens <= 4096
    assert expert_tokens > 0 and tokens % expert_tokens == 0
    assert w_gate.dtype == ax.nvfp4
    assert hidden % cores == 0 and intermediate % cores == 0 and experts % 4 == 0
    hs = slice(core * (hidden // cores), (core + 1) * (hidden // cores))
    cs = slice(core * (intermediate // cores), (core + 1) * (intermediate // cores))
    es = slice(core * (experts // cores), (core + 1) * (experts // cores))
    row_tile = 256
    ts = slice(core * (row_tile // cores), (core + 1) * (row_tile // cores))
    token_map = ax.dsl.create_tensor((experts, tokens), ax.uint8, ax.MemoryType.SHARED_OCM)
    weight_map = ax.dsl.create_tensor((experts, tokens), ax.bf16, ax.MemoryType.SHARED_OCM)
    zero_u8 = ax.dsl.zeros((1,), ax.uint8)
    zero_bf16 = ax.dsl.zeros((1,), ax.bf16)
    ax.dsl.expand(zero_u8, shape=(experts // cores, tokens), r=token_map[es])
    ax.dsl.expand(zero_bf16, shape=(experts // cores, tokens), r=weight_map[es])
    axd.pub(token_map)
    axd.pub(weight_map)
    axd.sub(token_map)
    axd.sub(weight_map)
    accumulated = ax.dsl.create_tensor((tokens + 1, hidden // cores), ax.bf16)
    # Preparation overwrites every real row before routing; only the inactive
    # scatter sentinel needs initialization.
    ax.dsl.expand(zero_bf16, shape=(1, hidden // cores), r=accumulated[tokens : tokens + 1])
    normed = ax.dsl.create_tensor((row_tile, hidden), ax.bf16, ax.MemoryType.SHARED_OCM)
    router = ax.dsl.create_tensor((row_tile, experts), ax.bf16, ax.MemoryType.SHARED_OCM)
    if shared_expert_gate is not None:
        activation = ax.dsl.create_tensor((row_tile, intermediate), ax.bf16, ax.MemoryType.SHARED_OCM)
        shared_gate = ax.dsl.create_tensor((row_tile, 1), ax.bf16, ax.MemoryType.SHARED_OCM)
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    stride = ax.dsl.full((1,), ax.int32, fill_value=row_tile)
    positions = ax.dsl.from_numpy(np.arange(row_tile, dtype=np.int32))
    token_stride = ax.dsl.full((1,), ax.int32, fill_value=tokens)
    unit = ax.dsl.full((1,), ax.fp32, fill_value=1)
    preparation_index = ax.dsl.zeros((1,), ax.int32)
    preparation_tiles = ax.dsl.full((1,), ax.int32, fill_value=tokens // row_tile)
    route_membership = ax.dsl.full((row_tile // cores * top_k,), ax.uint8, fill_value=1)
    # These operands are immutable across preparation tiles. Carry their OCM
    # views explicitly through the Range instead of rereading DDR each time.
    router_weight = load_tile(w_route[es], banks=banks)
    router_scale = ax.dsl.mul(
        load_tile(w_scale_route[es], banks=banks),
        load_tile(w_scale2_route, banks=banks),
    )
    # Shared-expert weights have the same lifetime as the cached router shard.
    shared_operands = ()
    if shared_expert_gate is not None:
        shared_operands = (
            ax.dsl.concat(
                load_tile(w_shared_expert_mlp_gate[cs], banks=banks),
                load_tile(w_shared_expert_mlp_up[cs], banks=banks),
                dim=0,
            ),
            ax.dsl.concat(
                load_tile(w_scale_shared_expert_mlp_gate[cs], banks=banks),
                load_tile(w_scale_shared_expert_mlp_up[cs], banks=banks),
                dim=0,
            ),
            ax.dsl.concat(
                load_tile(w_scale2_shared_expert_mlp_gate, banks=banks),
                load_tile(w_scale2_shared_expert_mlp_up, banks=banks),
                dim=0,
            ).reshape(1, 2, 1),
            load_tile(w_shared_expert_mlp_down[hs], banks=banks),
            load_tile(w_scale_shared_expert_mlp_down[hs], banks=banks),
            load_tile(w_scale2_shared_expert_mlp_down, banks=banks),
            load_tile(shared_expert_gate, banks=banks),
        )

    def prepare_tile(
        tile,
        loop_map,
        loop_weights,
        total,
        normalized_input,
        router_weight,
        router_scale,
        *shared_operands,
        first=False,
    ):
        rows = tile * stride + positions
        local = gather_rows(
            x,
            rows[ts],
            dim=0,
            banks=banks,
            r=normed[ts] if input_norm_w is None else None,
        )
        if input_norm_w is not None:
            rmsnorm(
                local,
                load_tile(input_norm_w, banks=banks),
                eps=norm_eps,
                r=normed[ts],
            )
        if shared_expert_gate is not None:
            wg, sg, scales, shared_down, shared_down_scale, shared_down_scale2, shared_gate_weight = shared_operands
            # This projection needs only this core's normalized row shard;
            # start it before waiting for the complete shared input tile.
            ax.dsl.sigmoid(
                _prefill_shared_gate_projection(normed[ts], shared_gate_weight, output_dtype=ax.bf16),
                r=shared_gate[ts],
            )
        axd.pub(normed)
        axd.sub(normed)
        if first:
            store_tile(normed[ts], r=normalized_input[ts], banks=banks)
        else:
            scatter_rows(normed[ts], rows[ts], dim=0, r=normalized_input, banks=banks)
        logits = _prefill_router_projection(normed, router_weight)
        ax.dsl.mul(logits, router_scale.reshape(1, -1), output_dtype=ax.bf16, r=router[:, es])
        axd.pub(router)
        axd.sub(router)
        if norm_topk_prob:
            # Normalizing selected softmax probabilities equals softmax on
            # selected logits; avoid exponentiating all experts for every row.
            logits, ids = ax.dsl.topk(router[ts], k=top_k, dim=-1)
            values = ax.dsl.softmax(logits, dim=-1, output_dtype=ax.bf16)
        else:
            probabilities = ax.dsl.softmax(router[ts], dim=-1, output_dtype=ax.fp32)
            values, ids = ax.dsl.topk(probabilities, k=top_k, dim=-1)
            values = ax.dsl.cast(values, dtype=ax.bf16)
        locations = (ids * token_stride + rows[ts].reshape(-1, 1)).reshape(-1)
        ax.dsl.scatter(values.reshape(-1), locations, dim=0, r=loop_weights.reshape(-1))
        ax.dsl.scatter(route_membership, locations, dim=0, r=loop_map.reshape(-1))
        if shared_expert_gate is not None:
            projected = linear(
                normed,
                wg,
                sg,
                "prefill_shared_gate_up",
                weight_scale2=unit,
                ic_tile_size=512,
                m_tile_size=64,
                output_dtype=ax.bf16,
            )
            projected = ax.dsl.mul(
                projected.reshape(row_tile, 2, intermediate // cores),
                scales,
                output_dtype=ax.bf16,
            )
            gate = projected[:, :1].reshape(row_tile, -1)
            up = projected[:, 1:].reshape(row_tile, -1)
            ax.dsl.mul(ax.dsl.silu(gate), up, r=activation[:, cs])
            axd.pub(activation)
            axd.pub(shared_gate)
            axd.sub(activation)
            axd.sub(shared_gate)
            result = linear(
                activation,
                shared_down,
                shared_down_scale,
                "prefill_shared_down",
                weight_scale2=shared_down_scale2,
                oc_tile_size=256,
                m_tile_size=64,
                output_dtype=ax.bf16,
            )
            result = ax.dsl.mul(result, shared_gate, output_dtype=ax.bf16)
        else:
            result = ax.dsl.expand(zero_bf16, shape=(row_tile, hidden // cores))
        if add_residual:
            result = ax.dsl.add(
                result,
                gather_rows(x[:, hs], rows, dim=0, banks=banks),
                output_dtype=ax.bf16,
            )
        scatter_rows(result, rows, dim=0, r=total, banks=banks)
        # Both input and shared-expert scratch are reused by the next tile.
        axd.pub(normed)
        axd.sub(normed)
        return loop_map, loop_weights, total, normalized_input, router_weight, router_scale, *shared_operands

    prepared = prepare_tile(
        zero, token_map, weight_map, accumulated, output_ax, router_weight, router_scale, *shared_operands, first=True
    )
    prepared = ax.dsl.range(
        preparation_index,
        one,
        preparation_tiles,
        one,
        prepare_tile,
        *prepared,
        unroll_factor=1,
    )
    token_map, weight_map, accumulated, normalized_input = prepared[:4]
    if valid_tokens is not None:
        # Rows past the live token count are only there to keep the fixed
        # 4096-row workspace shape.  Drop them from the routing map so they
        # neither activate experts nor add padded expert tiles.  Their output
        # rows are never read.
        valid = ax.dsl.load(valid_tokens)
        cols = ax.dsl.from_numpy(np.arange(tokens, dtype=np.int32))
        keep = ax.dsl.minimum(
            ax.dsl.maximum(ax.dsl.sub(valid, cols, output_dtype=ax.int32), zero), one, output_dtype=ax.uint8
        )
        ax.dsl.mul(token_map, keep.reshape(1, -1), output_dtype=ax.uint8, r=token_map)
    axd.pub(token_map)
    axd.pub(weight_map)
    axd.pub(output_ax)
    axd.sub(token_map)
    axd.sub(weight_map)
    axd.sub(output_ax)
    return _prefill_routed_experts(
        output_ax,
        token_map,
        weight_map,
        accumulated,
        w_gate,
        w_scale_gate,
        w_scale2_gate,
        w_down,
        w_scale_down,
        w_scale2_down,
        output_ax,
        core,
        cores,
        row_tile=expert_tokens,
        keep_input_resident=keep_input_resident,
        banks=banks,
    )


@ax.builder(run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")))
def moe_kernel_expert_parallel_v2_scf(
    x,  # [seq_len, hidden_dim]
    shared_expert_gate,  # [hidden_dim, 1]
    w_shared_expert_mlp_gate,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale_shared_expert_mlp_gate,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale2_shared_expert_mlp_gate,
    w_shared_expert_mlp_up,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale_shared_expert_mlp_up,  # [hidden_dim, shared_expert_intermediate_size]
    w_scale2_shared_expert_mlp_up,
    w_shared_expert_mlp_down,  # [shared_expert_intermediate_size, hidden_dim]
    w_scale_shared_expert_mlp_down,  # [shared_expert_intermediate_size, hidden_dim]
    w_scale2_shared_expert_mlp_down,
    w_route,  # [num_experts, hidden_dim]
    w_scale_route,
    w_scale2_route,
    w_gate,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_gate,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_gate,  # [1]
    w_up,  # [num_experts, moe_intermediate_size, hidden_dim]
    w_scale_up,  # [num_experts, moe_intermediate_size, 1]
    w_scale2_up,  # [1]
    w_down,  # [num_experts, hidden_dim, moe_intermediate_size]
    w_scale_down,  # [num_experts, hidden_dim, 1]
    w_scale2_down,  # [1]
    top_k,  # 8
    num_experts,  # 128
    norm_topk_prob,  # False
    output_ax,  # [seq_len, hidden_dim]
    chunk_size=None,
    prefill_expert_tokens=None,
    valid_tokens=None,
    input_norm_w=None,
    norm_eps=1e-6,
    add_residual=False,
    fused_gate_up=False,
    expert_unroll=4,
    prefill_input_resident=False,
    banks=None,
):
    """Run sparse experts from runtime weights, with optional input norm/skip.

    With fused_gate_up, gate/up share native [E,2I,H] weights and [E,2]
    global scales. Runtime expert IDs select the active experts' OCM tiles.
    With chunk_size=None, complete prefill batches of 256/512/1024/2048/4096
    tokens use 32/64/128/128/128 rows per expert tile. Other shapes use 64.
    prefill_expert_tokens overrides the token rows per expert tile for the
    large-prefill path (seq_len > 256).  valid_tokens (1,) int32 optionally
    hides workspace rows past the live token count from routing.
    Prefills above 256 tokens visit experts in ID order, with each expert's
    runtime token count controlling its inner Range. Empty experts skip all
    projection work; overloaded experts execute additional token tiles.
    expert_unroll controls expert-loop unrolling only for 256-token prefills.
    prefill_input_resident keeps the hidden input shard in private OCM. Its
    default preserves the standalone schedule; fused stages can opt in.
    """
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    x = x.reshape(-1, x.shape[-1])  # [B, T, H] -> [B*T, H], a metadata-only view.
    seq_len, hidden_dim = x.shape
    if chunk_size is None:
        chunk_size = {256: 32, 512: 64, 1024: 128, 2048: 128, 4096: 128}.get(seq_len, 64)
    if prefill_expert_tokens is not None:
        chunk_size = prefill_expert_tokens
    chunk_size = min(chunk_size, seq_len) if chunk_size > 0 else seq_len
    if seq_len > 256:
        assert fused_gate_up
        return _large_prefill_moe(
            x,
            shared_expert_gate,
            w_shared_expert_mlp_gate,
            w_scale_shared_expert_mlp_gate,
            w_scale2_shared_expert_mlp_gate,
            w_shared_expert_mlp_up,
            w_scale_shared_expert_mlp_up,
            w_scale2_shared_expert_mlp_up,
            w_shared_expert_mlp_down,
            w_scale_shared_expert_mlp_down,
            w_scale2_shared_expert_mlp_down,
            w_route,
            w_scale_route,
            w_scale2_route,
            w_gate,
            w_scale_gate,
            w_scale2_gate,
            w_down,
            w_scale_down,
            w_scale2_down,
            top_k,
            num_experts,
            norm_topk_prob,
            output_ax,
            chunk_size,
            input_norm_w,
            norm_eps,
            add_residual,
            prefill_input_resident,
            valid_tokens=valid_tokens,
            banks=banks,
        )
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    seq_len_factor = (seq_len + core_num - 1) // core_num
    slice_seq_slice = slice(core_id * seq_len_factor, min((core_id + 1) * seq_len_factor, seq_len))
    slice_seq_cur_core = slice_seq_slice.stop - slice_seq_slice.start
    num_experts_per_core = (num_experts + core_num - 1) // core_num
    num_experts_offset_core = core_id * num_experts_per_core
    w_dtype = datatype.DType(w_gate.dtype)
    w_dtype_num_bits = w_dtype.num_bits
    # Runtime MXFP8 weights use an FP8_E4M3 payload dtype so MatMul can pair it
    # with the MXE8M0 block scale. The E8M0 scale distinguishes this path from
    # ordinary per-channel FP8 weights.
    mxfp8_runtime = _is_runtime_mxfp8_dtype_pair(w_gate.dtype, w_scale_gate.dtype)
    # Packed 4-bit, BF16, and MXFP8 weights use the sparse expert-chunk path.
    sparse_prefill = w_dtype_num_bits == 4 or w_dtype.typename in ("BF16", "MXFP8_E4M3") or mxfp8_runtime
    slice_expert_core = slice(
        num_experts_offset_core,
        min(num_experts_offset_core + num_experts_per_core, num_experts),
    )
    # load tokens
    x_mem_scope = ax.MemoryType.SHARED_OCM
    x_shocm = x
    if x.memory_type == ax.MemoryType.DDR:
        x_shocm = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        if slice_seq_cur_core > 0:
            load_tile(r=x_shocm[slice_seq_slice, :], x=x[slice_seq_slice, :], banks=banks)
    if seq_len > 1:
        axd.pub(x_shocm)
        axd.sub(x_shocm)

    residual_input = x_shocm
    if input_norm_w is not None:
        x_norm = ax.dsl.create_tensor(x_shocm.shape, x.dtype, x_mem_scope)
        if slice_seq_cur_core > 0:
            rmsnorm(
                x_shocm[slice_seq_slice],
                load_tile(input_norm_w, banks=banks),
                eps=norm_eps,
                r=x_norm[slice_seq_slice],
            )
        if seq_len > 1:
            axd.pub(x_norm)
            axd.sub(x_norm)
        x_shocm = x_norm

    if seq_len == 1:
        axd.pub(x_shocm)
        axd.sub(x_shocm)

    # route
    router_logits = ax.dsl.create_tensor((seq_len, num_experts), x.dtype, x_mem_scope)
    if seq_len == 1 and w_route.dtype.typename == "BF16":
        # Keep each core's Router C shard resident, then expose the four IC
        # partials as independent NMC projections instead of one serial MatMul.
        route_weight = load_tile(w_route[slice_expert_core, :], banks=banks)
        projected = linear(
            x_shocm,
            route_weight,
            None,
            "decode_router",
            ic_tile_size=min(512, hidden_dim),
            output_dtype=ax.fp32,
        )
        scale = ax.dsl.mul(
            load_tile(w_scale_route[slice_expert_core, :], banks=banks),
            load_tile(w_scale2_route, banks=banks),
        )
        ax.dsl.mul(projected, scale.reshape(1, -1), r=router_logits[:, slice_expert_core])
    else:
        ax.dsl.matmul(
            x_shocm,
            load_tile(w_route[slice_expert_core, :], banks=banks),
            w_scale=load_tile(w_scale_route[slice_expert_core, :], banks=banks),
            w_scale2=load_tile(w_scale2_route, banks=banks),
            transposed=True,
            output_dtype=x.dtype,
            r=router_logits[:, slice_expert_core],
        )
    axd.pub(router_logits)
    axd.sub(router_logits)

    top_k_weights = ax.dsl.create_tensor((seq_len, top_k), ax.fp32, x_mem_scope)
    top_k_index = ax.dsl.create_tensor((seq_len, top_k), ax.int32, x_mem_scope)
    if slice_seq_cur_core > 0:
        router_logits_ = ax.dsl.softmax(router_logits[slice_seq_slice, :], dim=-1, output_dtype=ax.fp32)
        # Keep routing probabilities in FP32 through Top-K and normalization.
        ax.dsl.topk(
            router_logits_,
            k=top_k,
            dim=-1,
            r_values=top_k_weights[slice_seq_slice, :],
            r_indices=top_k_index[slice_seq_slice, :],
        )
        if norm_topk_prob:
            top_k_weights_sum = ax.dsl.sum(top_k_weights[slice_seq_slice, :], dim=-1, keepdim=True)
            ax.dsl.div(
                top_k_weights[slice_seq_slice, :],
                top_k_weights_sum,
                r=top_k_weights[slice_seq_slice, :],
            )
    if seq_len > 1:
        axd.pub(top_k_index)
        axd.pub(top_k_weights)
        axd.sub(top_k_index)
        axd.sub(top_k_weights)

    if shared_expert_gate is not None:
        assert (
            w_shared_expert_mlp_gate is not None
            and w_shared_expert_mlp_up is not None
            and w_shared_expert_mlp_down is not None
        )
        # shared expert compute
        shared_expert_output = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        _runtime_shared_mlp(
            x_shocm,
            w_shared_expert_mlp_gate,
            w_scale_shared_expert_mlp_gate,
            w_scale2_shared_expert_mlp_gate,
            w_shared_expert_mlp_up,
            w_scale_shared_expert_mlp_up,
            w_scale2_shared_expert_mlp_up,
            w_shared_expert_mlp_down,
            w_scale_shared_expert_mlp_down,
            w_scale2_shared_expert_mlp_down,
            shared_expert_output,
            core_id,
            core_num,
            banks=banks,
        )
        if seq_len > 1:
            axd.pub(shared_expert_output)
            axd.sub(shared_expert_output)
        shared_expert_gate_output_sigmoid = ax.dsl.create_tensor((seq_len, 1), x.dtype, x_mem_scope)
        shared_expert_output_scaled = ax.dsl.create_tensor((seq_len, hidden_dim), x.dtype, x_mem_scope)
        if core_id == 0:
            shared_expert_gate_output = ax.dsl.matmul(
                x_shocm,
                load_tile(shared_expert_gate, banks=banks),
                output_dtype=ax.bf16,
            )  # [seq_len, 1]
            ax.dsl.sigmoid(shared_expert_gate_output, r=shared_expert_gate_output_sigmoid)  # [seq_len, 1]
            ax.dsl.mul(
                shared_expert_output,
                shared_expert_gate_output_sigmoid,
                r=shared_expert_output_scaled,
            )  # [seq_len, hidden_dim]

        if seq_len > 1:
            axd.pub(shared_expert_output_scaled)
            axd.sub(shared_expert_output_scaled)

    if seq_len == 1:
        axd.pub(top_k_index)
        axd.pub(top_k_weights)
        axd.sub(top_k_index)
        axd.sub(top_k_weights)
        if shared_expert_gate is not None:
            axd.pub(shared_expert_output_scaled)
            axd.sub(shared_expert_output_scaled)
        _decode_routed_experts(
            x_shocm,
            top_k_index,
            top_k_weights,
            w_gate,
            w_scale_gate,
            w_scale2_gate,
            w_up,
            w_scale_up,
            w_scale2_up,
            w_down,
            w_scale_down,
            w_scale2_down,
            shared_expert_output_scaled if shared_expert_gate is not None else None,
            residual_input if add_residual else None,
            output_ax,
            fused_gate_up,
            core_id,
            core_num,
            banks=banks,
        )
        return output_ax

    # mlp
    # Invert [T,K] Top-K assignments directly into [E,T] routing maps.
    # FP32 probabilities round to the activation dtype once, before expert loops.
    token_rows = ax.dsl.from_numpy(np.repeat(np.arange(seq_len, dtype=np.int32), top_k))
    # Preserve the OCM row alignment while scattering through contiguous storage.
    weight_alignment = 256 // x.dtype.num_bits
    weight_stride = (seq_len + weight_alignment) // weight_alignment * weight_alignment
    weight_storage = ax.dsl.zeros((num_experts * weight_stride,), x.dtype)
    token_expert_weights_sparse = weight_storage.reshape(num_experts, weight_stride)[:, : seq_len + 1]
    # Write into packed storage: default [T,K] layouts can pad odd-K rows.
    weight_locations = ax.dsl.create_tensor((seq_len * top_k,), ax.int32)
    ax.dsl.mul(top_k_index, ax.dsl.full((1,), ax.int32, weight_stride), r=weight_locations.reshape(seq_len, top_k))
    ax.dsl.add(weight_locations, token_rows, r=weight_locations)
    routing_weights = ax.dsl.create_tensor((seq_len * top_k,), x.dtype)
    ax.dsl.cast(top_k_weights, dtype=x.dtype, r=routing_weights.reshape(seq_len, top_k))
    ax.dsl.scatter(
        routing_weights,
        weight_locations,
        dim=0,
        r=weight_storage,
    )
    map_stride = (seq_len + 31) // 32 * 32
    map_storage = ax.dsl.zeros((num_experts * map_stride,), ax.uint8)
    expert_token_map = map_storage.reshape(num_experts, map_stride)[:, :seq_len]
    map_locations = ax.dsl.create_tensor((seq_len * top_k,), ax.int32)
    ax.dsl.mul(top_k_index, ax.dsl.full((1,), ax.int32, map_stride), r=map_locations.reshape(seq_len, top_k))
    ax.dsl.add(map_locations, token_rows, r=map_locations)
    ax.dsl.scatter(
        ax.dsl.full((seq_len * top_k,), ax.uint8, fill_value=1),
        map_locations,
        dim=0,
        r=map_storage,
    )
    if sparse_prefill:
        # Interleave expert ownership across cores. Routing can concentrate many
        # tokens on adjacent expert ids; contiguous ownership then leaves one
        # core with most work and the others waiting at the final reduction.
        real_expert_ids = np.arange(core_id, num_experts, core_num, dtype=np.int32)
        local_expert_count = real_expert_ids.size
        local_expert_ids = np.zeros(num_experts_per_core, dtype=np.int32)
        local_expert_ids[:local_expert_count] = real_expert_ids
        local_expert_id_tensor = ax.dsl.from_numpy(real_expert_ids) if local_expert_count else None
    else:
        local_expert_count = slice_expert_core.stop - slice_expert_core.start
        local_expert_ids = np.asarray(
            [
                expert_id if expert_id < num_experts else 0
                for expert_id in range(
                    num_experts_offset_core,
                    num_experts_offset_core + num_experts_per_core,
                )
            ],
            dtype=np.int32,
        )
        local_expert_id_tensor = None
    local_expert_token_map = ax.dsl.zeros((num_experts_per_core, seq_len), ax.uint8)
    if sparse_prefill and local_expert_count:
        gather_rows(
            expert_token_map,
            local_expert_id_tensor,
            dim=0,
            r=local_expert_token_map[:local_expert_count],
            banks=banks,
        )
    else:
        ax.dsl.move(
            r=local_expert_token_map[:local_expert_count],
            x=expert_token_map[slice_expert_core],
        )
    local_token_weights = ax.dsl.zeros((num_experts_per_core, seq_len + 1), x.dtype)
    if sparse_prefill and local_expert_count:
        gather_rows(
            token_expert_weights_sparse,
            local_expert_id_tensor,
            dim=0,
            r=local_token_weights[:local_expert_count],
            banks=banks,
        )
    else:
        ax.dsl.move(
            r=local_token_weights[:local_expert_count],
            x=token_expert_weights_sparse[slice_expert_core],
        )

    mlp_out = ax.dsl.zeros((seq_len + 1, hidden_dim), x.dtype)
    # Expert branches can take different paths on each core. Keep the token
    # collection private so loop iterations do not consume cross-core signals.
    x_collect = ax.dsl.zeros((seq_len + 1, hidden_dim), x.dtype)
    ax.dsl.move(r=x_collect[:seq_len, :], x=x_shocm)

    compact_positions = ax.dsl.from_numpy(np.arange(chunk_size, dtype=np.int32))
    expert_ids = ax.dsl.from_numpy(local_expert_ids)
    # Compact each local expert's token IDs once: [local_E,T] INT32 IDs and
    # [local_E,1] UINT32 counts. All buffers have shape-only capacities.
    token_indices, token_counts = ax.dsl.nonzero(local_expert_token_map)
    offsets = np.arange(0, seq_len, chunk_size, dtype=np.int32)
    remaining = ax.dsl.sub(
        token_counts.reshape(num_experts_per_core, 1),
        ax.dsl.from_numpy(offsets.reshape(1, -1)),
        output_dtype=ax.int32,
    )
    active = ax.dsl.minimum(
        ax.dsl.maximum(remaining, ax.dsl.zeros((1,), ax.int32)),
        ax.dsl.full((1,), ax.int32, fill_value=1),
    )
    # Each active entry is one (local expert, token offset) task. For count 36
    # and tile 32 the list contains offsets 0 and 32; empty experts add no task.
    task_ids, task_count = ax.dsl.nonzero(active.reshape(-1))
    task_coords = ax.dsl.from_numpy(
        np.stack(
            (
                np.repeat(np.arange(num_experts_per_core, dtype=np.int32), offsets.size),
                np.tile(offsets, num_experts_per_core),
            ),
            axis=1,
        )
    )

    def expert_body(task_index, total):
        """Read immutable routing/weights and update only the expert accumulator."""
        task_id = gather_rows(task_ids, task_index, dim=0, banks=banks)
        task = gather_rows(task_coords, task_id, dim=0, banks=banks)
        expert_index, offset = task[:, :1].reshape(1), task[:, 1:].reshape(1)
        expert_idx = gather_rows(expert_ids, expert_index, dim=0, banks=banks)
        expert_token_indices = gather_rows(token_indices, expert_index, dim=0, banks=banks).reshape(seq_len)
        expert_token_num = gather_rows(token_counts, expert_index, dim=0, banks=banks).reshape(1)
        expert_weight_row = gather_rows(local_token_weights, expert_index, dim=0, banks=banks).reshape(seq_len + 1)

        token_buf = ax.dsl.create_tensor((chunk_size, hidden_dim), x.dtype)
        out_buf = ax.dsl.create_tensor((chunk_size, hidden_dim), x.dtype)
        if fused_gate_up:
            # Native FusedMoE stores [E,2I,H] gate/up together. Gather
            # that expert once; both projections then share its OCM tile.
            gate_up = _select_runtime_expert_tensor(w_gate, expert_idx, banks=banks)
            gate_up_scale = _select_runtime_expert_tensor(w_scale_gate, expert_idx, banks=banks)
            gate_up_scale2 = gather_rows(w_scale2_gate, expert_idx, dim=0, banks=banks).reshape(2)
            intermediate = gate_up.shape[0] // 2
            w_gate_sel, w_up_sel = (
                gate_up[:intermediate],
                gate_up[intermediate:],
            )
            w_scale_gate_sel, w_scale_up_sel = (
                gate_up_scale[:intermediate],
                gate_up_scale[intermediate:],
            )
            w_scale2_gate_sel, w_scale2_up_sel = (
                gate_up_scale2[:1],
                gate_up_scale2[1:],
            )
            w_down_sel = _select_runtime_expert_tensor(w_down, expert_idx, banks=banks)
            w_scale_down_sel = _select_runtime_expert_tensor(w_scale_down, expert_idx, banks=banks)
            w_scale2_down_sel = _select_runtime_expert_scale2(w_scale2_down, expert_idx, banks=banks)
        else:
            # Only active expert tiles reach this loop body.
            w_gate_sel = _select_runtime_expert_tensor(w_gate, expert_idx, banks=banks)
            w_up_sel = _select_runtime_expert_tensor(w_up, expert_idx, banks=banks)
            w_down_sel = _select_runtime_expert_tensor(w_down, expert_idx, banks=banks)
            w_scale_gate_sel = _select_runtime_expert_tensor(w_scale_gate, expert_idx, banks=banks)
            w_scale_up_sel = _select_runtime_expert_tensor(w_scale_up, expert_idx, banks=banks)
            w_scale_down_sel = _select_runtime_expert_tensor(w_scale_down, expert_idx, banks=banks)
            w_scale2_gate_sel = _select_runtime_expert_scale2(w_scale2_gate, expert_idx, banks=banks)
            w_scale2_up_sel = _select_runtime_expert_scale2(w_scale2_up, expert_idx, banks=banks)
            w_scale2_down_sel = _select_runtime_expert_scale2(w_scale2_down, expert_idx, banks=banks)
        # Runtime offset [1] INT32 + local row positions [chunk_size] INT32.
        # Clamp gather slots when the final tile extends past seq_len.
        slots = ax.dsl.add(offset, compact_positions, output_dtype=ax.int32)
        gather_slots = ax.dsl.minimum(slots, ax.dsl.full((1,), ax.int32, fill_value=seq_len - 1))
        batch_token_indices_raw = gather_rows(expert_token_indices, gather_slots, dim=0, banks=banks)
        batch_token_indices_raw = ax.dsl.minimum(
            batch_token_indices_raw,
            ax.dsl.full(
                shape=batch_token_indices_raw.shape,
                dtype=batch_token_indices_raw.dtype,
                fill_value=seq_len,
            ),
        )
        # Positions after expert_token_num are inactive lanes. Clamp
        # their mask to zero and send those lanes to the zero row.
        active_mask_i32 = ax.dsl.sub(
            expert_token_num,
            slots,
            output_dtype=ax.int32,
        )
        active_mask_i32 = ax.dsl.minimum(
            active_mask_i32,
            ax.dsl.full(
                shape=active_mask_i32.shape,
                dtype=active_mask_i32.dtype,
                fill_value=1,
            ),
        )
        active_mask_u32 = ax.dsl.maximum(
            active_mask_i32,
            ax.dsl.zeros((1,), ax.int32),
            output_dtype=ax.uint32,
        )
        inactive_mask_u32 = ax.dsl.sub(
            ax.dsl.full(
                shape=active_mask_u32.shape,
                dtype=active_mask_u32.dtype,
                fill_value=1,
            ),
            active_mask_u32,
            output_dtype=ax.uint32,
        )
        batch_token_indices_write = ax.dsl.add(
            ax.dsl.mul(batch_token_indices_raw, active_mask_u32, output_dtype=ax.uint32),
            ax.dsl.mul(
                inactive_mask_u32,
                ax.dsl.full(
                    shape=inactive_mask_u32.shape,
                    dtype=inactive_mask_u32.dtype,
                    fill_value=seq_len,
                ),
                output_dtype=ax.uint32,
            ),
            output_dtype=ax.uint32,
        )

        gather_rows(x_collect, batch_token_indices_write, dim=0, r=token_buf, banks=banks)
        mlp_kernel_expert_parallel_scf(
            token_buf,
            w_gate_sel,
            w_scale_gate_sel,
            w_scale2_gate_sel,
            w_up_sel,
            w_scale_up_sel,
            w_scale2_up_sel,
            w_down_sel,
            w_scale_down_sel,
            w_scale2_down_sel,
            out_buf,
        )
        expert_weights = gather_rows(expert_weight_row, batch_token_indices_write, dim=0, banks=banks).reshape(-1, 1)
        mlp_out_weighted = ax.dsl.mul(out_buf, expert_weights)
        ax.dsl.index_add(
            x=mlp_out_weighted,
            indices=batch_token_indices_write,
            r=total,
            dim=0,
        )
        return (total,)

    mlp_out = ax.dsl.range(
        ax.dsl.zeros(shape=(1,), dtype=ax.int32),
        ax.dsl.zeros(shape=(1,), dtype=ax.int32),
        ax.dsl.cast(task_count, dtype=ax.int32),
        ax.dsl.full(shape=(1,), dtype=ax.int32, fill_value=1),
        expert_body,
        mlp_out,
        # Bounded unrolling pipelines adjacent expert tasks without specializing
        # on their runtime count. Tiny prefills keep the single-body path.
        unroll_factor=expert_unroll if seq_len == 256 and fused_gate_up and w_dtype_num_bits == 4 else 1,
    )[-1]

    # Keep the original BF16 addition order without a full SHOCM zero tensor.
    final_hidden_states = ax.dsl.zeros((1,), x.dtype)
    hidden_dims_per_bank = (hidden_dim + core_num - 1) // core_num

    for bank_id in range(core_num):
        # core interleave on hidden_dim for bank conflict
        hidden_dim_bank_offset = ((bank_id + core_id) % core_num) * hidden_dims_per_bank
        slice_hidden_dim_bank = slice(
            hidden_dim_bank_offset,
            min(hidden_dim_bank_offset + hidden_dims_per_bank, hidden_dim),
        )
        r_sum_shocm = ax.dsl.create_tensor(
            (seq_len, hidden_dim),
            x.dtype,
            ax.MemoryType.SHARED_OCM,
            name=f"r_sum_shocm_{core_id}_{bank_id}",
        )
        # core reduce add
        ax.dsl.add(
            x=mlp_out[0:seq_len, slice_hidden_dim_bank],
            y=final_hidden_states if bank_id == 0 else final_hidden_states[:, slice_hidden_dim_bank],
            r=r_sum_shocm[:, slice_hidden_dim_bank],
        )
        axd.pub(r_sum_shocm)
        axd.sub(r_sum_shocm)
        final_hidden_states = r_sum_shocm

    # Retain the reduction buffer through both BF16 additions, then publish
    # its final version before every core writes its owned hidden shard.
    if core_id == 0:
        if shared_expert_gate is not None:
            ax.dsl.add(final_hidden_states, shared_expert_output_scaled, r=final_hidden_states)
        if add_residual:
            ax.dsl.add(final_hidden_states, residual_input, r=final_hidden_states)
    axd.pub(final_hidden_states)
    axd.sub(final_hidden_states)
    owned_hidden = slice(
        core_id * hidden_dims_per_bank,
        min((core_id + 1) * hidden_dims_per_bank, hidden_dim),
    )
    if output_ax.memory_type == ax.MemoryType.DDR:
        store_tile(final_hidden_states[:, owned_hidden], r=output_ax[:, owned_hidden], banks=banks)
    else:
        ax.dsl.move(final_hidden_states[:, owned_hidden], r=output_ax[:, owned_hidden])

    return output_ax


_MOE_KERNELS = {
    "expert_parallel": moe_kernel_expert_parallel,
    "token_parallel": moe_kernel_token_parallel,
}


def _get_value(obj, name, default=_MISSING):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _require_value(obj, name):
    value = _get_value(obj, name)
    if value is _MISSING:
        raise ValueError(f"qwen3_moe inputs must provide {name}")
    return value


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = axdsl_util.torch2np(value)
    array = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(array)


def _runtime_arg(value):
    """Convert host metadata or ordinary NumPy arrays to Torch launch inputs."""
    if torch.is_tensor(value):
        return value
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return axdsl_util.np2torch(np.ascontiguousarray(value))


def _runtime_weight_arg(value):
    """Convert exported weight storage to a dtype accepted by runtime DSL IO."""
    if torch.is_tensor(value):
        return value
    value = np.ascontiguousarray(value)
    typename = datatype.DType(value.dtype).typename
    if typename in ("NVFP4", "MXFP4"):
        if value.shape[-1] % 2:
            raise ValueError(f"{typename} runtime weight width must be even, got {value.shape}")
        storage = value.view(np.uint8)
        packed = storage[..., 0::2] | (storage[..., 1::2] << 4)
        return torch.from_numpy(packed.copy()).view(torch.float4_e2m1fn_x2)
    if typename == "MXFP8_E4M3":
        # Preserve the payload bytes while retaining the FP8 marker required by
        # MatMul to combine these weights with their E8M0 block scales.
        return torch.from_numpy(value.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    if typename == "MXE8M0":
        return torch.from_numpy(value.view(np.uint8).copy()).view(torch.float8_e8m0fnu)
    if typename == "S4":
        # Torch has no signed-int4 dtype; preserve the exported sign-extended
        # values in S8 storage for the runtime MatMul path.
        return torch.from_numpy(value.astype(np.int8, copy=False)).clone()
    return _runtime_arg(value)


def _normalize_weight_scale(weight, scale, name):
    scale = _as_numpy(scale)
    weight_typename = datatype.DType(weight.dtype).typename
    if weight_typename in ("S4", "S8") and scale.ndim == weight.ndim:
        # Checkpoint group parameters may be [..., groups, outputs]. Runtime
        # MatMul and zero-point correction both consume [..., outputs, groups].
        if scale.shape[:-1] == weight.shape[:-1] and weight.shape[-1] % scale.shape[-1] == 0:
            return np.ascontiguousarray(scale, dtype=np.float32)
        if (
            scale.shape[:-2] == weight.shape[:-2]
            and scale.shape[-1] == weight.shape[-2]
            and weight.shape[-1] % scale.shape[-2] == 0
        ):
            return np.ascontiguousarray(scale.swapaxes(-1, -2), dtype=np.float32)
    if weight_typename in ("MXFP4", "MXFP8_E4M3"):
        expected_shape = tuple(weight.shape[:-1]) + (weight.shape[-1] // 32,)
        if scale.shape != expected_shape:
            raise ValueError(f"{name} shape {scale.shape} must match {weight_typename} group-32 shape {expected_shape}")
        return np.ascontiguousarray(scale, dtype=datatype.mxe8m0)
    expected_shape = tuple(weight.shape[:-1]) + (1,)
    if scale.ndim == 0 or scale.size == 1:
        return np.full(expected_shape, float(scale.reshape(-1)[0]), dtype=np.float32)
    if scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
        broadcast_shape = (scale.shape[0],) + (1,) * (weight.ndim - 1)
        return np.ascontiguousarray(
            np.broadcast_to(scale.reshape(broadcast_shape), expected_shape),
            dtype=np.float32,
        )
    if tuple(scale.shape) == expected_shape:
        return np.ascontiguousarray(scale, dtype=np.float32)
    if tuple(scale.shape) == tuple(weight.shape[:-1]):
        return np.ascontiguousarray(scale[..., None], dtype=np.float32)
    # Block-quantized FP8 scale: [..., oc//block, ic//block] with ic//block > 1.
    if (
        scale.ndim == weight.ndim
        and scale.ndim >= 2
        and tuple(scale.shape[:-2]) == tuple(weight.shape[:-2])
        and scale.shape[-1] > 1
        and weight.shape[-2] % scale.shape[-2] == 0
        and weight.shape[-1] % scale.shape[-1] == 0
    ):
        # NVFP4 block scales use FP8 E4M3; the backend needs this dtype to
        # select the correct post-K format when building the MatMul command.
        return np.ascontiguousarray(scale)
    raise ValueError(f"{name} shape {tuple(scale.shape)} is incompatible with weight shape {tuple(weight.shape)}")


def _normalize_scale2(scale2):
    scale2 = _as_numpy(scale2)
    if scale2.ndim == 0:
        scale2 = scale2.reshape(1)
    return np.ascontiguousarray(scale2)


def _normalize_expert_scale2(scale2, num_experts, name):
    if scale2.size == 1:
        return scale2
    if scale2.ndim == 2 and scale2.shape == (num_experts, 1):
        scale2 = scale2.reshape(num_experts)
    if scale2.ndim != 1 or scale2.shape[0] != num_experts:
        raise ValueError(f"{name} must be scalar or have shape ({num_experts},), got {scale2.shape}")
    return np.ascontiguousarray(scale2)


def _gather_topk_scale2(scale2, top_k_experts, banks=None):
    """Resolve the per-expert second-level scale for the selected top-k experts."""
    if scale2.shape[0] == 1:
        return load_tile(scale2, banks=banks) if banks and scale2.name in banks else scale2
    return gather_rows(scale2, top_k_experts, dim=0, banks=banks)


def _gather_topk_block_scale(scale, top_k_experts):
    """Gather only the routed experts' block scales directly from DDR."""
    return ax.dsl.index_select(scale, top_k_experts, dim=0)


def _topk_scale2_broadcast(scale2):
    """Broadcast per-expert scale2 over [top_k, oc, ic/block] block scales."""
    return scale2.reshape(1, -1, 1)


def _topk_scale2(scale2, top_k_idx):
    """Pick one top-k slot, keeping a shared scalar scale2 broadcastable."""
    if scale2.shape[0] == 1:
        return scale2
    return scale2[top_k_idx : top_k_idx + 1]


def _const_inputs(attrs):
    if isinstance(attrs, Mapping):
        return attrs["const_inputs"]
    return attrs.const_inputs


def _extract_fc(inputs, attrs_name, weight_name, scale_name, scale2_name, required=True):
    attrs = _get_value(inputs, attrs_name, None)
    if attrs is not None:
        const_inputs = _const_inputs(attrs)
        weight = _as_numpy(const_inputs["w"])
        scale = const_inputs.get("w_scale", 1.0)
        scale2 = const_inputs.get("w_scale2", _ONE_SCALE2)
    else:
        weight = _get_value(inputs, weight_name)
        if weight is _MISSING:
            if required:
                raise ValueError(f"qwen3_moe inputs must provide {attrs_name} or {weight_name}")
            return None, None, None
        weight = _as_numpy(weight)
        scale = _get_value(inputs, scale_name, 1.0)
        scale2 = _get_value(inputs, scale2_name, _ONE_SCALE2)

    return (
        weight,
        _normalize_weight_scale(weight, scale, scale_name),
        _normalize_scale2(scale2),
    )


def _extract_zeropoint(inputs, attrs_name, zeropoint_name, weight):
    attrs = _get_value(inputs, attrs_name, None)
    if attrs is not None:
        zeropoint = _const_inputs(attrs).get("w_zeropoint", None)
    else:
        zeropoint = _get_value(inputs, zeropoint_name, None)
    if zeropoint is None:
        return None
    return _normalize_weight_scale(weight, zeropoint, zeropoint_name)


def _prepare_hidden_states_and_output(hidden_states, output):
    if hidden_states.ndim < 2:
        raise ValueError("qwen3_moe hidden states must have rank >= 2")

    if output is None:
        output = torch.empty_like(hidden_states)
    elif tuple(output.shape) != tuple(hidden_states.shape):
        raise ValueError(f"output shape {tuple(output.shape)} must match hidden_states {tuple(hidden_states.shape)}")
    elif output.dtype != hidden_states.dtype:
        raise ValueError(f"output dtype {output.dtype} must match hidden_states dtype {hidden_states.dtype}")

    hidden_size = hidden_states.shape[-1]
    return (
        hidden_states.reshape(-1, hidden_size),
        output,
        output.reshape(-1, hidden_size),
    )


def _select_moe_mode(num_tokens):
    """Single-token decode uses the per-core expert split; batches use expert_parallel."""
    return "token_parallel" if num_tokens == 1 else "expert_parallel"


def _resident_runtime_args(inputs, x, output, mode, w_norm=None):
    """Bind prepared resident BF16/E4M3 tensors without host weight export."""
    projections = [
        (
            _require_value(inputs, "w_" + name),
            _require_value(inputs, "w_scale_" + name),
            name,
        )
        for name in ("gate", "up", "down")
    ]
    down = projections[2][0]
    experts, hidden, intermediate = down.shape
    for weight, scale, name in projections:
        shape = (experts, hidden, intermediate) if name == "down" else (experts, intermediate, hidden)
        if tuple(weight.shape) != shape:
            raise ValueError("resident MoE requires gate/up [E,I,H] and down [E,H,I]")
        if weight.dtype != down.dtype or weight.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise ValueError("resident MoE supports BF16 or E4M3 weights")
        if not weight.is_contiguous() or not scale.is_contiguous():
            raise ValueError("resident weights and scales must be contiguous")
        if scale.dtype != torch.float32 or tuple(scale.shape[:-1]) != tuple(weight.shape[:-1]):
            raise ValueError("resident MoE requires F32 per-output-row scales")
        blocks = weight.shape[-1] // 128 if weight.dtype == torch.float8_e4m3fn else 1
        if scale.shape[-1] != blocks:
            raise ValueError("resident FP8 scales must retain the block128 reduction axis")
    shared_gate = _get_value(inputs, "shared_expert_gate", None)
    args = [x, w_norm, shared_gate]
    for projection in ("gate", "up", "down"):
        for prefix in ("w_", "w_scale_", "w_scale2_"):
            name = prefix + "shared_expert_mlp_" + projection
            args.append(_require_value(inputs, name) if shared_gate is not None else None)
    args.extend(_require_value(inputs, name) for name in ("w_route", "w_scale_route", "w_scale2_route"))
    for weight, scale, name in projections:
        args.extend((weight, scale, _require_value(inputs, "w_scale2_" + name)))
        if mode == "expert_parallel":
            args.append(None)
    args.extend(
        (
            int(_require_value(inputs, "top_k")),
            experts,
            bool(_get_value(inputs, "norm_topk_prob", False)),
            output,
        )
    )
    for value in args:
        if torch.is_tensor(value) and (value.device != x.device or not value.is_contiguous()):
            raise ValueError("resident runtime operands must be contiguous on the input device")
    return args


def qwen3_moe(
    inputs,
    output=None,
    *,
    mode="auto",
    core_num=1,
    chunk_size=None,
    resident_weights=False,
):
    """Run the Qwen3 MoE block.

    Providing ``inputs.w_norm`` fuses the decoder contract
    ``x + moe(post_attention_layernorm(x))``.  The norm parameter uses the
    Qwen3.5 ``(1 + weight)`` convention.

    ``resident_weights=True`` accepts resident ``w_gate/w_up [E,I,H]``,
    ``w_down [E,H,I]`` and matching F32 per-output-row scales.
    FP8 scales retain their reduction-block128 axis; BF16 scales have one
    column. Router/shared tensors and scale2 values use the separate API's
    runtime names. All operands must already be contiguous on the input
    device. The default separate/export API remains unchanged.
    """

    hidden_states = _get_value(inputs, "x", None)
    if hidden_states is None:
        hidden_states = _require_value(inputs, "hidden_states")
    x_flat, output, output_flat = _prepare_hidden_states_and_output(hidden_states, output)

    w_norm = _get_value(inputs, "w_norm", None)
    norm_eps = _get_value(inputs, "norm_eps", None)
    norm_eps = 1e-6 if norm_eps is None else float(norm_eps)
    if w_norm is not None:
        w_norm = _runtime_arg(w_norm).to(device=x_flat.device, dtype=torch.float32).reshape(-1)
        if w_norm.shape != (x_flat.shape[-1],):
            raise ValueError(f"qwen3_moe w_norm shape {w_norm.shape} must match hidden size {x_flat.shape[-1]}")
        # Qwen3.5 stores the zero-centered delta from one; materialize the
        # effective multiplier once before recording the kernel.
        w_norm = w_norm + 1.0

    if mode == "auto":
        mode = _select_moe_mode(x_flat.shape[0])
    kernel = _MOE_KERNELS.get(mode)
    if kernel is None:
        choices = ", ".join(sorted(_MOE_KERNELS))
        raise ValueError(f"unsupported qwen3_moe mode {mode!r}; expected one of: {choices}, auto")

    if resident_weights:
        resident_norm = _runtime_arg(w_norm).to(x_flat.device) if w_norm is not None else None
        args = _resident_runtime_args(inputs, x_flat, output_flat, mode, resident_norm)
        ax.init(chip_num=1)
        kernel[core_num, core_num](
            *args,
            chunk_size=64 if chunk_size is None else chunk_size,
            norm_eps=norm_eps,
        )
        return output

    w_route, w_scale_route, w_scale2_route = _extract_fc(
        inputs, "attrs_route", "w_route", "w_scale_route", "w_scale2_route"
    )
    w_gate, w_scale_gate, w_scale2_gate = _extract_fc(inputs, "attrs_gate", "w_gate", "w_scale_gate", "w_scale2_gate")
    if chunk_size is None:
        # MXFP8 expert tiles are twice as wide as NVFP4 tiles. A 128-token
        # chunk halves repeated weight gathers and measured faster than 64.
        chunk_size = 128 if _is_runtime_mxfp8_dtype_pair(w_gate.dtype, w_scale_gate.dtype) else 64
    w_up, w_scale_up, w_scale2_up = _extract_fc(inputs, "attrs_up", "w_up", "w_scale_up", "w_scale2_up")
    w_down, w_scale_down, w_scale2_down = _extract_fc(inputs, "attrs_down", "w_down", "w_scale_down", "w_scale2_down")
    w_zeropoint_gate = _extract_zeropoint(inputs, "attrs_gate", "w_zeropoint_gate", w_gate)
    w_zeropoint_up = _extract_zeropoint(inputs, "attrs_up", "w_zeropoint_up", w_up)
    w_zeropoint_down = _extract_zeropoint(inputs, "attrs_down", "w_zeropoint_down", w_down)
    has_expert_zeropoint = any(value is not None for value in (w_zeropoint_gate, w_zeropoint_up, w_zeropoint_down))
    if has_expert_zeropoint and (mode != "expert_parallel" or core_num != 1):
        raise ValueError("Qwen3 AWQ INT4 MoE currently requires mode='expert_parallel' and core_num=1")

    shared_expert_gate = _get_value(inputs, "shared_expert_gate", None)
    if shared_expert_gate is None:
        w_shared_expert_mlp_gate = w_scale_shared_expert_mlp_gate = w_scale2_shared_expert_mlp_gate = None
        w_shared_expert_mlp_up = w_scale_shared_expert_mlp_up = w_scale2_shared_expert_mlp_up = None
        w_shared_expert_mlp_down = w_scale_shared_expert_mlp_down = w_scale2_shared_expert_mlp_down = None
    else:
        shared_expert_gate = _as_numpy(shared_expert_gate, dtype=np.float32)
        (w_shared_expert_mlp_gate, w_scale_shared_expert_mlp_gate, w_scale2_shared_expert_mlp_gate,) = _extract_fc(
            inputs,
            "attrs_shared_expert_mlp_gate",
            "w_shared_expert_mlp_gate",
            "w_scale_shared_expert_mlp_gate",
            "w_scale2_shared_expert_mlp_gate",
        )
        (w_shared_expert_mlp_up, w_scale_shared_expert_mlp_up, w_scale2_shared_expert_mlp_up,) = _extract_fc(
            inputs,
            "attrs_shared_expert_mlp_up",
            "w_shared_expert_mlp_up",
            "w_scale_shared_expert_mlp_up",
            "w_scale2_shared_expert_mlp_up",
        )
        (w_shared_expert_mlp_down, w_scale_shared_expert_mlp_down, w_scale2_shared_expert_mlp_down,) = _extract_fc(
            inputs,
            "attrs_shared_expert_mlp_down",
            "w_shared_expert_mlp_down",
            "w_scale_shared_expert_mlp_down",
            "w_scale2_shared_expert_mlp_down",
        )

    top_k = int(_require_value(inputs, "top_k"))
    num_experts = int(_get_value(inputs, "num_experts", w_gate.shape[0]))
    norm_topk_prob = bool(_get_value(inputs, "norm_topk_prob", False))
    w_scale2_gate = _normalize_expert_scale2(w_scale2_gate, num_experts, "w_scale2_gate")
    w_scale2_up = _normalize_expert_scale2(w_scale2_up, num_experts, "w_scale2_up")
    w_scale2_down = _normalize_expert_scale2(w_scale2_down, num_experts, "w_scale2_down")

    # Builder inputs are ordinary Torch tensors.  Keep all shape/layout
    # normalization above in NumPy, then convert once at the launch boundary;
    # this also repacks the two FP4 nibbles required by Torch runtime storage.
    shared_expert_gate = (
        _runtime_arg(np.asarray(shared_expert_gate, dtype=datatype.bf16)) if shared_expert_gate is not None else None
    )
    w_norm = _runtime_arg(w_norm) if w_norm is not None else None
    w_route, w_scale_route, w_scale2_route = (
        _runtime_weight_arg(w_route),
        _runtime_arg(w_scale_route),
        _runtime_arg(w_scale2_route),
    )
    w_gate, w_scale_gate, w_scale2_gate = (
        _runtime_weight_arg(w_gate),
        _runtime_arg(w_scale_gate),
        _runtime_arg(w_scale2_gate),
    )
    w_up, w_scale_up, w_scale2_up = (
        _runtime_weight_arg(w_up),
        _runtime_arg(w_scale_up),
        _runtime_arg(w_scale2_up),
    )
    w_down, w_scale_down, w_scale2_down = (
        _runtime_weight_arg(w_down),
        _runtime_arg(w_scale_down),
        _runtime_arg(w_scale2_down),
    )
    w_zeropoint_gate = _runtime_arg(w_zeropoint_gate) if w_zeropoint_gate is not None else None
    w_zeropoint_up = _runtime_arg(w_zeropoint_up) if w_zeropoint_up is not None else None
    w_zeropoint_down = _runtime_arg(w_zeropoint_down) if w_zeropoint_down is not None else None
    if w_shared_expert_mlp_gate is not None:
        (w_shared_expert_mlp_gate, w_scale_shared_expert_mlp_gate, w_scale2_shared_expert_mlp_gate,) = (
            _runtime_weight_arg(w_shared_expert_mlp_gate),
            _runtime_arg(w_scale_shared_expert_mlp_gate),
            _runtime_arg(w_scale2_shared_expert_mlp_gate),
        )
        (w_shared_expert_mlp_up, w_scale_shared_expert_mlp_up, w_scale2_shared_expert_mlp_up,) = (
            _runtime_weight_arg(w_shared_expert_mlp_up),
            _runtime_arg(w_scale_shared_expert_mlp_up),
            _runtime_arg(w_scale2_shared_expert_mlp_up),
        )
        (w_shared_expert_mlp_down, w_scale_shared_expert_mlp_down, w_scale2_shared_expert_mlp_down,) = (
            _runtime_weight_arg(w_shared_expert_mlp_down),
            _runtime_arg(w_scale_shared_expert_mlp_down),
            _runtime_arg(w_scale2_shared_expert_mlp_down),
        )

    ax.init(chip_num=1)
    kernel_args = [
        x_flat,
        w_norm,
        shared_expert_gate,
        w_shared_expert_mlp_gate,
        w_scale_shared_expert_mlp_gate,
        w_scale2_shared_expert_mlp_gate,
        w_shared_expert_mlp_up,
        w_scale_shared_expert_mlp_up,
        w_scale2_shared_expert_mlp_up,
        w_shared_expert_mlp_down,
        w_scale_shared_expert_mlp_down,
        w_scale2_shared_expert_mlp_down,
        w_route,
        w_scale_route,
        w_scale2_route,
        w_gate,
        w_scale_gate,
        w_scale2_gate,
    ]
    if mode == "expert_parallel":
        kernel_args.append(w_zeropoint_gate)
    kernel_args.extend([w_up, w_scale_up, w_scale2_up])
    if mode == "expert_parallel":
        kernel_args.append(w_zeropoint_up)
    kernel_args.extend([w_down, w_scale_down, w_scale2_down])
    if mode == "expert_parallel":
        kernel_args.append(w_zeropoint_down)
    kernel_args.extend([top_k, num_experts, norm_topk_prob, output_flat])
    kernel[core_num, core_num](*kernel_args, chunk_size=chunk_size, norm_eps=norm_eps)
    return output
