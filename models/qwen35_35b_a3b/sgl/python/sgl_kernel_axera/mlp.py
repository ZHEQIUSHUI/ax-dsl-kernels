"""Fused MLP compute kernel and allocation-free runtime-tensor entry point."""
import math
import os
import torch
import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.opset import datatype

_PACKED_FP4_WEIGHT_TYPENAMES = frozenset(("MXFP4_X2", "NVFP4_X2"))


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def mlp_kernel(
    x,
    w_norm,
    w1,
    w_scale1,
    w_scale2_1,
    w2,
    w_scale2,
    w_scale2_2,
    w3,
    w_scale3,
    w_scale2_3,
    w_zp_scale1,
    w_zp_scale2,
    w_zp_scale3,
    input_permutation,
    residual,
    output,
    tile_size,
):
    # MLP parallelism is tensor parallel: every core owns one intermediate
    # (gate/up) shard and contributes a partial down projection.  The launch
    # grid is therefore one-dimensional (one program per core), rather than a
    # batch/sequence grid that leaves only one core active for decode.
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    reduction_memory = ax.MemoryType.SHARED_OCM

    w_dtype_name = datatype.DType(w3.dtype).typename
    prefill_fp4 = x.shape[1] > 1 and w_dtype_name in ("NVFP4", "MXFP4", "NVFP4_X2", "MXFP4_X2")
    # OC256 x IC768 fills the 96KiB native FP4 weight buffer.
    bc = 256 if prefill_fp4 else 512
    packed_fp4 = w_dtype_name in _PACKED_FP4_WEIGHT_TYPENAMES
    ic_slice_factor = 2 if packed_fp4 else 1
    B, N, De = x.shape
    token_count = B * N
    x_flat = x.reshape(token_count, De)
    output_flat = output.reshape(token_count, De)
    # Merge before RMSNorm and retain the same value for the output skip.
    residual_flat = None if residual is None else residual.reshape(token_count, De)
    Dh, _ = w1.shape
    w3_logical_ic = w3.shape[-1] * ic_slice_factor
    if w3_logical_ic != Dh:
        raise ValueError(f"w3 logical input dim {w3_logical_ic} does not match w1 dim {Dh}")
    if Dh % core_num != 0 or De % core_num != 0:
        raise ValueError("MLP TP requires both hidden dimensions to be divisible by the core count")

    hidden_per_core = Dh // core_num
    input_per_core = De // core_num
    output_per_core = De // core_num
    hidden_start = core_id * hidden_per_core
    input_start = core_id * input_per_core
    input_stop = input_start + input_per_core
    input_slice = slice(input_start, input_stop)
    token_tile = int(tile_size[0])
    input_permutation_indices = ax.dsl.load(input_permutation) if input_permutation.shape_size > 1 else None
    w_scale2_1_ocm = ax.dsl.load(w_scale2_1)
    w_scale2_2_ocm = ax.dsl.load(w_scale2_2)
    w_scale2_3_ocm = ax.dsl.load(w_scale2_3)
    w_norm_local = ax.dsl.load(w_norm[input_slice])

    # Flatten batch and sequence into one token axis.  All cores walk the same
    # token tiles, so a batched decode and a prefill use the same launch shape.
    for token_start in range(0, token_count, token_tile):
        token_stop = min(token_count, token_start + token_tile)
        x_local = ax.dsl.load(x_flat[slice(token_start, token_stop), input_slice])
        if residual_flat is not None:
            residual_local = ax.dsl.load(residual_flat[slice(token_start, token_stop), input_slice])
            ax.dsl.add(x_local, residual_local, r=x_local)
        token_tile_count, _ = x_local.shape

        # RMSNorm is reduced over hidden shards, then reconstructed in shared
        # OCM so every core sees the same normalized full vector.
        square_sum_local = ax.dsl.sum(
            ax.dsl.mul(x_local, x_local, output_dtype=ax.fp32),
            dim=-1,
            keepdim=True,
            output_dtype=ax.fp32,
        )
        square_sum_shared = ax.dsl.create_tensor(
            square_sum_local.shape,
            ax.fp32,
            reduction_memory,
            name=f"mlp_rms_sum_{token_start}",
        )
        for owner in range(core_num):
            if core_id == owner:
                if owner == 0:
                    ax.dsl.move(square_sum_local, r=square_sum_shared)
                else:
                    ax.dsl.move(square_sum_local, r=square_sum_shared, op="add")
            axd.sync_all()

        rms = ax.dsl.sqrt(
            ax.dsl.add(
                ax.dsl.div(
                    square_sum_shared,
                    ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=De),
                    output_dtype=ax.fp32,
                ),
                ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1e-6),
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        )
        x_norm_shared = ax.dsl.create_tensor(
            (token_tile_count, De),
            ax.bf16,
            reduction_memory,
            name=f"mlp_norm_{token_start}",
        )
        normalized_local = ax.dsl.div(x_local, rms, output_dtype=ax.fp32)
        ax.dsl.mul(
            normalized_local,
            w_norm_local,
            r=x_norm_shared[:, input_slice],
            output_dtype=ax.bf16,
        )
        axd.sync_all()

        norm_tile_quant = x_norm_shared
        if input_permutation_indices is not None:
            norm_tile_quant = ax.dsl.index_select(x_norm_shared, input_permutation_indices, dim=-1)
        mul_local = ax.dsl.create_tensor((token_tile_count, hidden_per_core), ax.bf16, ax.MemoryType.OCM)
        local_chunk_count = math.ceil(hidden_per_core / bc)

        fc_ic_split = min(De, 768) if prefill_fp4 else ((De // 2) + 511) // 512 * 512

        def load_fc_tile(weight, scale, zp_scale, chunk_id, ic_start, ic_stop):
            local_start = chunk_id * bc
            local_stop = min(local_start + bc, hidden_per_core)
            global_slice = slice(hidden_start + local_start, hidden_start + local_stop)
            weight_slice = slice(ic_start // ic_slice_factor, ic_stop // ic_slice_factor)
            weight_tile = ax.dsl.load(weight[global_slice, weight_slice])
            if scale.shape_size > 1:
                scale_block = De // scale.shape[1]
                scale_slice = slice(ic_start // scale_block, (ic_stop + scale_block - 1) // scale_block)
                scale_tile = ax.dsl.load(scale[global_slice, scale_slice])
            else:
                scale_tile = ax.dsl.load(scale)
            if zp_scale.shape_size > 1:
                zp_block = De // zp_scale.shape[1]
                zp_slice = slice(ic_start // zp_block, (ic_stop + zp_block - 1) // zp_block)
                zp_tile = ax.dsl.load(zp_scale[global_slice, zp_slice])
            else:
                zp_tile = None
            return weight_tile, scale_tile, zp_tile

        direct_fc = prefill_fp4 and w_zp_scale1.shape_size == 1 and w_zp_scale2.shape_size == 1
        # Gate and up projections share the same input and IC tiling.
        for chunk_id in range(local_chunk_count):
            local_start = chunk_id * bc
            local_stop = min(local_start + bc, hidden_per_core)
            local_slice = slice(local_start, local_stop)
            hidden_tile = local_stop - local_start
            r1_tile = ax.dsl.create_tensor((token_tile_count, hidden_tile), ax.bf16, ax.MemoryType.OCM)
            r2_tile = ax.dsl.create_tensor((token_tile_count, hidden_tile), ax.bf16, ax.MemoryType.OCM)
            # Small hidden sizes have one IC tile; never emit an empty FC.
            for ic_start in range(0, De, fc_ic_split):
                ic_stop = min(ic_start + fc_ic_split, De)
                w1_tile, w_scale1_tile, w_zp_scale1_tile = load_fc_tile(
                    w1, w_scale1, w_zp_scale1, chunk_id, ic_start, ic_stop
                )
                w2_tile, w_scale2_tile, w_zp_scale2_tile = load_fc_tile(
                    w2, w_scale2, w_zp_scale2, chunk_id, ic_start, ic_stop
                )
                x_ic = norm_tile_quant[:, slice(ic_start, ic_stop)]
                gate_part = ax.dsl.matmul(
                    x_ic,
                    w1_tile,
                    w_scale=w_scale1_tile,
                    w_scale2=w_scale2_1_ocm,
                    transposed=True,
                    output_dtype=ax.bf16,
                    output_add_assign=direct_fc and ic_start != 0,
                    r=r1_tile if direct_fc or (ic_start == 0 and w_zp_scale1_tile is None) else None,
                )
                up_part = ax.dsl.matmul(
                    x_ic,
                    w2_tile,
                    w_scale=w_scale2_tile,
                    w_scale2=w_scale2_2_ocm,
                    transposed=True,
                    output_dtype=ax.bf16,
                    output_add_assign=direct_fc and ic_start != 0,
                    r=r2_tile if direct_fc or (ic_start == 0 and w_zp_scale2_tile is None) else None,
                )
                if w_zp_scale1_tile is not None:
                    # Preserve the projection's BF16 rounding, then subtract
                    # FP32 compensation and round directly into the BF16 tile.
                    gate_part = ax.dsl.sub(
                        gate_part,
                        _apply_zero_point_compensation(x_ic, w_zp_scale1_tile),
                        output_dtype=ax.bf16,
                        r=r1_tile if ic_start == 0 else None,
                    )
                if w_zp_scale2_tile is not None:
                    up_part = ax.dsl.sub(
                        up_part,
                        _apply_zero_point_compensation(x_ic, w_zp_scale2_tile),
                        output_dtype=ax.bf16,
                        r=r2_tile if ic_start == 0 else None,
                    )
                if not direct_fc and ic_start != 0:
                    ax.dsl.move(gate_part, r=r1_tile, op="add")
                    ax.dsl.move(up_part, r=r2_tile, op="add")
            ax.dsl.mul(
                ax.dsl.silu(r1_tile),
                r2_tile,
                r=mul_local[:, local_slice],
                output_dtype=ax.bf16,
            )

        # Down projection is reduced across cores without a core-sized
        # temporary: rotate output shards so each synchronization round has
        # exactly one writer per shared-OCM slice.
        down_shared = ax.dsl.create_tensor(
            (token_tile_count, De),
            ax.bf16,
            reduction_memory,
            name=f"mlp_down_sum_{token_start}",
        )

        down_ic_tile = 768 if prefill_fp4 else 512

        def load_down_tile(chunk_id, step):
            local_start = chunk_id * down_ic_tile
            local_stop = min(local_start + down_ic_tile, hidden_per_core)
            logical_ic_start = hidden_start + local_start
            logical_ic_stop = hidden_start + local_stop
            output_owner = (core_id + step) % core_num
            output_start = output_owner * output_per_core
            output_stop = output_start + output_per_core
            output_slice = slice(output_start, output_stop)
            w3_ic_slice = slice(logical_ic_start // ic_slice_factor, logical_ic_stop // ic_slice_factor)
            w3_tile = ax.dsl.load(w3[output_slice, w3_ic_slice])
            if w_scale3.shape_size > 1:
                scale_block_k = w3_logical_ic // w_scale3.shape[1]
                scale_start = logical_ic_start // scale_block_k
                scale_end = (logical_ic_stop + scale_block_k - 1) // scale_block_k
                w_scale3_tile = ax.dsl.load(w_scale3[output_slice, slice(scale_start, scale_end)])
            else:
                w_scale3_tile = ax.dsl.load(w_scale3)
            if w_zp_scale3.shape_size > 1:
                zp_block_k = w3_logical_ic // w_zp_scale3.shape[1]
                zp_start = logical_ic_start // zp_block_k
                zp_end = (logical_ic_stop + zp_block_k - 1) // zp_block_k
                w_zp_scale3_tile = ax.dsl.load(w_zp_scale3[output_slice, slice(zp_start, zp_end)])
            else:
                w_zp_scale3_tile = None
            return output_slice, w3_tile, w_scale3_tile, w_zp_scale3_tile

        for chunk_id in range(math.ceil(hidden_per_core / down_ic_tile)):
            local_start = chunk_id * down_ic_tile
            local_stop = min(local_start + down_ic_tile, hidden_per_core)
            local_slice = slice(local_start, local_stop)
            mul_tile = mul_local[:, local_slice]
            output_slice, w3_tile, w_scale3_tile, w_zp_scale3_tile = load_down_tile(chunk_id, 0)
            for step in range(core_num):
                first_partial = chunk_id == 0 and step == 0
                down_tile = ax.dsl.matmul(
                    mul_tile,
                    w3_tile,
                    w_scale=w_scale3_tile,
                    w_scale2=w_scale2_3_ocm,
                    transposed=True,
                    output_dtype=ax.bf16,
                    r=down_shared[:, output_slice] if first_partial and w_zp_scale3_tile is None else None,
                )
                if w_zp_scale3_tile is not None:
                    down_tile = ax.dsl.sub(
                        down_tile,
                        _apply_zero_point_compensation(mul_tile, w_zp_scale3_tile),
                        output_dtype=ax.bf16,
                        r=down_shared[:, output_slice] if first_partial else None,
                    )
                if not first_partial:
                    ax.dsl.move(down_tile, r=down_shared[:, output_slice], op="add")
                axd.sync_all()

                if step + 1 < core_num:
                    (
                        output_slice,
                        w3_tile,
                        w_scale3_tile,
                        w_zp_scale3_tile,
                    ) = load_down_tile(chunk_id, step + 1)

        # Each core owns one output (De) shard.  The reduction above has
        # synchronized all contributors, so the shards can be written to DDR
        # independently instead of having core 0 store the full tensor.
        ax.dsl.add(down_shared[:, input_slice], x_local, r=x_local)
        ax.dsl.store(r=output_flat[slice(token_start, token_stop), input_slice], x=x_local)

    return output


def _apply_zero_point_compensation(x_tile, w_zp_scale_tile):
    num_groups = w_zp_scale_tile.shape[1]
    if x_tile.shape[1] % num_groups != 0:
        raise ValueError(f"input dim {x_tile.shape[1]} is not divisible by zeropoint groups {num_groups}")

    group_size = x_tile.shape[1] // num_groups
    x_group_sum = ax.dsl.sum(
        x_tile.reshape(x_tile.shape[0], num_groups, group_size),
        dim=-1,
        output_dtype=x_tile.dtype,
    )
    return ax.dsl.matmul(
        x_group_sum,
        w_zp_scale_tile,
        transposed=True,
        output_dtype=ax.fp32,
    )


MLP_WEIGHT_NAMES = (
    "w_norm",
    "w1",
    "w_scale1",
    "w_scale2_1",
    "w2",
    "w_scale2",
    "w_scale2_2",
    "w3",
    "w_scale3",
    "w_scale2_3",
    "w_zp_scale1",
    "w_zp_scale2",
    "w_zp_scale3",
    "input_permutation",
)


def mlp(inputs, output, core_num=int(os.getenv("SGLANG_AXERA_KERNEL_CORE_NUM", "4"))):
    """Launch complete token blocks using caller-owned runtime tensors.

    Inputs carry x, residual (or None), and the named MLP_WEIGHT_NAMES tensors.
    The kernel computes z = x + residual (or x), then MLP(RMSNorm(z)) + z.
    An already merged stream is passed as x with residual=None.
    Projection scales have shape [output_channels, groups], including unit scales.
    Weight normalization, device transfer and block preparation belong to SGLang.
    """
    if not isinstance(core_num, int) or core_num < 1:
        raise ValueError(f"core_num must be positive, got {core_num}")
    x = inputs.x
    if x.shape[1] != 1 and x.shape[1] % 128:
        raise ValueError("MLP requires decode rows or complete 128-token tiles")
    if output is None or output.shape != x.shape:
        raise ValueError("MLP output must match the complete input block")
    weights = [getattr(inputs, name) for name in MLP_WEIGHT_NAMES]
    for name, value in zip(MLP_WEIGHT_NAMES, weights):
        if not isinstance(value, torch.Tensor) or value.device != x.device:
            raise TypeError(f"{name} must be a runtime tensor on {x.device}")
    token_tile = 256 if x.shape[1] > 1 and inputs.w1.dtype == torch.float4_e2m1fn_x2 else 128
    args = (x, *weights, inputs.residual, output, [min(x.shape[0] * x.shape[1], token_tile)])
    mlp_kernel[core_num, core_num](*args)
    return output
