"""Dense Qwen3.8 MLP with on-chip activation and residual ownership."""


import os

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.opset import datatype

from sgl_kernel_axera.models.bank_access import gather_rows, load_tile, store_tile
from sgl_kernel_axera.mlp import _apply_zero_point_compensation

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
    residual_buffer=None,
    banks=None,
):
    # All four cores process the same token tile. Each owns one gate/up
    # shard and contributes to every down-projection output shard.
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    assert core_num == 4, "Qwen3.8 MLP requires four cores"

    w_dtype_name = datatype.DType(w3.dtype).typename
    prefill_fp4 = x.shape[1] > 1 and w_dtype_name in (
        "NVFP4",
        "MXFP4",
        "NVFP4_X2",
        "MXFP4_X2",
    )
    # OC256 x IC768 fills the 96KiB native FP4 weight buffer.
    fc_oc_tile = 256 if prefill_fp4 else 512
    packed_fp4 = w_dtype_name in _PACKED_FP4_WEIGHT_TYPENAMES
    ic_slice_factor = 2 if packed_fp4 else 1
    batch_size, sequence_length, hidden_size = x.shape
    token_count = batch_size * sequence_length
    x_flat = x.reshape(token_count, hidden_size)
    output_flat = output.reshape(token_count, hidden_size)
    # Merge before RMSNorm and retain the same value for the output skip.
    residual_flat = None if residual is None else residual.reshape(token_count, hidden_size)
    intermediate_size = w1.shape[0]
    w3_logical_ic = w3.shape[-1] * ic_slice_factor
    if w3_logical_ic != intermediate_size:
        raise ValueError(f"w3 logical input dim {w3_logical_ic} does not match w1 dim {intermediate_size}")
    if intermediate_size % core_num != 0 or hidden_size % core_num != 0:
        raise ValueError("MLP TP requires both hidden dimensions to be divisible by the core count")

    intermediate_per_core = intermediate_size // core_num
    hidden_per_core = hidden_size // core_num
    intermediate_start = core_id * intermediate_per_core
    input_start = core_id * hidden_per_core
    input_stop = input_start + hidden_per_core
    input_slice = slice(input_start, input_stop)
    token_tile = int(tile_size[0])
    input_permutation_indices = load_tile(input_permutation, banks=banks) if input_permutation.shape_size > 1 else None
    w_scale2_1_ocm = load_tile(w_scale2_1, banks=banks)
    w_scale2_2_ocm = load_tile(w_scale2_2, banks=banks)
    w_scale2_3_ocm = load_tile(w_scale2_3, banks=banks)
    w_norm_local = load_tile(w_norm[input_slice], banks=banks)

    # Flatten batch and sequence into one token axis.  All cores walk the same
    # token tiles, so a batched decode and a prefill use the same launch shape.
    for token_start in range(0, token_count, token_tile):
        token_stop = min(token_count, token_start + token_tile)
        input_tile = x_flat[slice(token_start, token_stop), input_slice]
        # Retain only the owned residual shard; the full shared activation can
        # expire after RMSNorm, before down-projection partials need SHOCM.
        if x.memory_type == ax.MemoryType.DDR:
            x_local = load_tile(input_tile, banks=banks)
        else:
            # The fused loop has consumed its private carry at BODY entry.
            # Reuse that storage for the owned residual instead of keeping an
            # idle carry plus a second equally sized residual allocation.
            x_local = (
                ax.dsl.create_tensor(input_tile.shape, input_tile.dtype)
                if residual_buffer is None
                else residual_buffer[token_start:token_stop]
            )
            ax.dsl.move(input_tile, r=x_local)
        if residual_flat is not None:
            residual_local = load_tile(residual_flat[slice(token_start, token_stop), input_slice], banks=banks)
            ax.dsl.add(x_local, residual_local, r=x_local)
        token_tile_count, _ = x_local.shape

        # RMSNorm is reduced over hidden shards, then reconstructed in shared
        # OCM so every core sees the same normalized full vector.
        square_sum_shared = ax.dsl.create_tensor(
            (core_num, token_tile_count, 1),
            ax.fp32,
            ax.MemoryType.SHARED_OCM,
            name=f"mlp_rms_squares_{token_start}",
        )
        ax.dsl.sum(
            ax.dsl.mul(x_local, x_local, output_dtype=ax.fp32),
            dim=-1,
            keepdim=True,
            output_dtype=ax.fp32,
            r=square_sum_shared[core_id : core_id + 1].reshape(token_tile_count, 1),
        )
        axd.pub(square_sum_shared)
        axd.sub(square_sum_shared)
        # Each core reduces the same ordered, immutable producer slots.
        square_sum = square_sum_shared[0:1].reshape(token_tile_count, 1)
        for owner in range(1, core_num):
            square_sum = ax.dsl.add(
                square_sum,
                square_sum_shared[owner : owner + 1].reshape(token_tile_count, 1),
            )

        rms = ax.dsl.sqrt(
            ax.dsl.add(
                ax.dsl.div(
                    square_sum,
                    ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=hidden_size),
                    output_dtype=ax.fp32,
                ),
                ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1e-6),
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        )
        x_norm_shared = ax.dsl.create_tensor(
            (token_tile_count, hidden_size),
            ax.bf16,
            ax.MemoryType.SHARED_OCM,
            name=f"mlp_norm_{token_start}",
        )
        normalized_local = ax.dsl.div(x_local, rms, output_dtype=ax.fp32)
        ax.dsl.mul(
            normalized_local,
            w_norm_local,
            r=x_norm_shared[:, input_slice],
            output_dtype=ax.bf16,
        )
        axd.pub(x_norm_shared)
        axd.sub(x_norm_shared)

        norm_tile_quant = x_norm_shared
        if input_permutation_indices is not None:
            norm_tile_quant = gather_rows(x_norm_shared, input_permutation_indices, dim=-1, banks=banks)
        mul_local = ax.dsl.create_tensor((token_tile_count, intermediate_per_core), ax.bf16, ax.MemoryType.OCM)

        fc_ic_tile = 768 if prefill_fp4 else ((hidden_size // 2) + 511) // 512 * 512

        def load_fc_tile(weight, scale, zp_scale, global_slice, ic_start, ic_stop, full_scale):
            weight_slice = slice(ic_start // ic_slice_factor, ic_stop // ic_slice_factor)
            weight_tile = load_tile(weight[global_slice, weight_slice], banks=banks)
            if scale.shape_size > 1:
                scale_block = hidden_size // scale.shape[1]
                scale_slice = slice(ic_start // scale_block, (ic_stop + scale_block - 1) // scale_block)
                scale_tile = (
                    full_scale[:, scale_slice]
                    if full_scale is not None
                    else load_tile(scale[global_slice, scale_slice], banks=banks)
                )
            else:
                scale_tile = load_tile(scale, banks=banks)
            if zp_scale.shape_size > 1:
                zp_block = hidden_size // zp_scale.shape[1]
                zp_slice = slice(ic_start // zp_block, (ic_stop + zp_block - 1) // zp_block)
                zp_tile = load_tile(zp_scale[global_slice, zp_slice], banks=banks)
            else:
                zp_tile = None
            return weight_tile, scale_tile, zp_tile

        direct_fc = prefill_fp4 and w_zp_scale1.shape_size == 1 and w_zp_scale2.shape_size == 1
        # Gate and up projections share the same input and IC tiling.
        for local_start in range(0, intermediate_per_core, fc_oc_tile):
            local_stop = min(local_start + fc_oc_tile, intermediate_per_core)
            local_slice = slice(local_start, local_stop)
            global_slice = slice(intermediate_start + local_start, intermediate_start + local_stop)
            hidden_tile = local_stop - local_start
            r1_tile = ax.dsl.create_tensor((token_tile_count, hidden_tile), ax.bf16, ax.MemoryType.OCM)
            r2_tile = ax.dsl.create_tensor((token_tile_count, hidden_tile), ax.bf16, ax.MemoryType.OCM)
            # Decode reuses complete scale rows across the original IC waves.
            full_scales = [
                load_tile(scale[global_slice, :], banks=banks)
                if token_count == 1 and scale.shape_size > 1 and hidden_tile * scale.shape[-1] <= 256 * 1024
                else None
                for scale in (w_scale1, w_scale2)
            ]
            for ic_start in range(0, hidden_size, fc_ic_tile):
                ic_stop = min(ic_start + fc_ic_tile, hidden_size)
                w1_tile, w_scale1_tile, w_zp_scale1_tile = load_fc_tile(
                    w1, w_scale1, w_zp_scale1, global_slice, ic_start, ic_stop, full_scales[0]
                )
                w2_tile, w_scale2_tile, w_zp_scale2_tile = load_fc_tile(
                    w2, w_scale2, w_zp_scale2, global_slice, ic_start, ic_stop, full_scales[1]
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

        # Rotate output owners in the same cyclic BF16 accumulation order.
        # Shared output stays in OCM through each published reduction round.
        on_chip_output = output.memory_type != ax.MemoryType.DDR
        if on_chip_output:
            down_shared = output_flat[slice(token_start, token_stop)]
        else:
            down_sum = ax.dsl.create_tensor((token_tile_count, hidden_per_core), ax.bf16)

        down_ic_tile = 768 if prefill_fp4 else 512

        # Prefill keeps only one OC tile of partial results live. Decode and
        # DDR outputs retain the full owned shard and the same BF16 sum order.
        down_oc_tile = fc_oc_tile if on_chip_output and token_count > 1 else hidden_per_core
        full_down_scale = None
        if w_scale3.shape_size > 1:
            scale_block_k = w3_logical_ic // w_scale3.shape[1]
        if token_count == 1 and w_scale3.shape_size > 1:
            owned_scale_start = intermediate_start // scale_block_k
            owned_scale_stop = (intermediate_start + intermediate_per_core + scale_block_k - 1) // scale_block_k
            owned_scale_width = owned_scale_stop - owned_scale_start
            if (
                w_scale3.shape[0] == hidden_size
                and owned_scale_width > 128
                and hidden_size * owned_scale_width * w_scale3.dtype.num_bits <= 2 * 1024 * 1024 * 8
            ):
                # All rounds consume this core's entire IC shard exactly once.
                # Reuse the slab for the original OC/IC accumulation sequence.
                full_down_scale = load_tile(w_scale3[:, owned_scale_start:owned_scale_stop], banks=banks)
        for chunk_id, local_start in enumerate(range(0, intermediate_per_core, down_ic_tile)):
            local_stop = min(local_start + down_ic_tile, intermediate_per_core)
            mul_tile = mul_local[:, slice(local_start, local_stop)]
            logical_ic_start = intermediate_start + local_start
            logical_ic_stop = intermediate_start + local_stop
            w3_ic_slice = slice(logical_ic_start // ic_slice_factor, logical_ic_stop // ic_slice_factor)
            if w_scale3.shape_size > 1:
                scale_slice = slice(
                    logical_ic_start // scale_block_k, (logical_ic_stop + scale_block_k - 1) // scale_block_k
                )
            if w_zp_scale3.shape_size > 1:
                zp_block_k = w3_logical_ic // w_zp_scale3.shape[1]
                zp_slice = slice(logical_ic_start // zp_block_k, (logical_ic_stop + zp_block_k - 1) // zp_block_k)
            for step in range(core_num):
                first_partial = chunk_id == 0 and step == 0
                if not on_chip_output:
                    down_shared = ax.dsl.create_tensor(
                        (token_tile_count, hidden_size),
                        ax.bf16,
                        ax.MemoryType.SHARED_OCM,
                        name=f"mlp_down_partial_{token_start}_{chunk_id}_{step}",
                    )
                output_owner = (core_id + step) % core_num
                owner_start = output_owner * hidden_per_core
                for output_start in range(0, hidden_per_core, down_oc_tile):
                    output_stop = min(output_start + down_oc_tile, hidden_per_core)
                    output_slice = slice(owner_start + output_start, owner_start + output_stop)
                    w3_tile = load_tile(w3[output_slice, w3_ic_slice], banks=banks)
                    if full_down_scale is not None:
                        local_scale_slice = slice(
                            scale_slice.start - owned_scale_start, scale_slice.stop - owned_scale_start
                        )
                        w_scale3_tile = full_down_scale[output_slice, local_scale_slice]
                    else:
                        scale_tile = w_scale3[output_slice, scale_slice] if w_scale3.shape_size > 1 else w_scale3
                        w_scale3_tile = load_tile(scale_tile, banks=banks)
                    w_zp_scale3_tile = (
                        load_tile(w_zp_scale3[output_slice, zp_slice], banks=banks)
                        if w_zp_scale3.shape_size > 1
                        else None
                    )
                    destination = down_shared[:, output_slice]
                    initialize = not on_chip_output or first_partial
                    down_tile = ax.dsl.matmul(
                        mul_tile,
                        w3_tile,
                        w_scale=w_scale3_tile,
                        w_scale2=w_scale2_3_ocm,
                        transposed=True,
                        output_dtype=ax.bf16,
                        r=destination if initialize and w_zp_scale3_tile is None else None,
                    )
                    if w_zp_scale3_tile is not None:
                        down_tile = ax.dsl.sub(
                            down_tile,
                            _apply_zero_point_compensation(mul_tile, w_zp_scale3_tile),
                            output_dtype=ax.bf16,
                            r=destination if initialize else None,
                        )
                    if on_chip_output and not first_partial:
                        # Preserve BF16 partial rounding and let the next
                        # MatMul overlap this independent NMC update.
                        ax.dsl.move(down_tile, r=destination, op="add")
                axd.pub(down_shared)
                axd.sub(down_shared)
                if not on_chip_output:
                    contribution = down_shared[:, input_slice]
                    if first_partial:
                        # Snapshot the shared contribution before the private sum.
                        ax.dsl.move(contribution, r=down_sum)
                    else:
                        ax.dsl.add(down_sum, contribution, r=down_sum)

        # Each core adds its residual and writes only its owned hidden shard.
        destination = output_flat[slice(token_start, token_stop), input_slice]
        if output.memory_type == ax.MemoryType.DDR:
            ax.dsl.add(down_sum, x_local, r=x_local)
            store_tile(r=destination, x=x_local, banks=banks)
        else:
            ax.dsl.add(down_shared[:, input_slice], x_local, r=destination)

    return output
