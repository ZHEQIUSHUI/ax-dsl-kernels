"""Runtime-weight GDN prefill and decode kernels with FP32 recurrent state."""

from __future__ import annotations


import os
import pathlib
from typing import Any, Mapping

import ml_dtypes
import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import util as ax_util
from axnn.dsl import parallel as axd
from sgl_kernel_axera.models.bank_access import (
    gather_rows,
    load_tile,
    load_shared_scale_rows,
    scatter_rows,
    store_tile,
)
from sgl_kernel_axera.attention.attn_utils import (
    rmsnorm,
    rmsnorm_sqrt,
)

GDN_MAX_SEQ_LEN = 256


_REQUIRED_WEIGHTS = frozenset(
    {
        "input_norm_w",
        "w_q",
        "w_k",
        "w_v",
        "w_z",
        "w_a",
        "w_b",
        "w_q_conv",
        "w_k_conv",
        "w_v_conv",
        "a_log",
        "dt_bias",
        "o_norm_w",
        "w_o",
    }
)
_OPTIONAL_SCALES = (
    "w_q_scale",
    "w_q_scale2",
    "w_k_scale",
    "w_k_scale2",
    "w_v_scale",
    "w_v_scale2",
    "w_z_scale",
    "w_z_scale2",
    "w_a_scale",
    "w_a_scale2",
    "w_b_scale",
    "w_b_scale2",
    "w_o_scale",
    "w_o_scale2",
)
_FC_SCALE_WEIGHTS = {
    "w_q_scale": "w_q",
    "w_k_scale": "w_k",
    "w_v_scale": "w_v",
    "w_z_scale": "w_z",
    "w_a_scale": "w_a",
    "w_b_scale": "w_b",
    "w_o_scale": "w_o",
}


def is_nvfp4_weight(weight) -> bool:
    """Return whether a host or DSL FC tensor uses packed NVFP4 storage."""
    dtype = weight.dtype
    if isinstance(dtype, torch.dtype):
        return dtype == GDN_NVFP4_DTYPE
    return dtype == ax.nvfp4


def is_mxfp8_scale(scale) -> bool:
    """Return whether a runtime FC scale uses the E8M0 MX block format."""
    if scale is None:
        return False
    dtype = scale.dtype
    if isinstance(dtype, torch.dtype):
        return dtype == torch.float8_e8m0fnu
    return dtype == ax.mxe8m0


def is_mxfp8_weight(weight, scale) -> bool:
    """Return whether an FP8 weight is paired with an E8M0 MXFP8 block scale."""
    if not is_mxfp8_scale(scale):
        return False
    dtype = weight.dtype
    if isinstance(dtype, torch.dtype):
        return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    return dtype in (ax.fp8_e4m3, ax.fp8_e5m2, ax.mxfp8_e4m3)


def is_mxfp4_weight(weight, scale) -> bool:
    """Return whether a packed FP4 weight uses the E8M0 MX block format."""
    if not is_mxfp8_scale(scale):
        return False
    dtype = weight.dtype
    if isinstance(dtype, torch.dtype):
        return dtype == GDN_NVFP4_DTYPE
    return dtype in (ax.nvfp4, ax.mxfp4)


def prepare_projection_folded_rmsnorm(pre_attn, input_norm_weight, eps, banks=None):
    """Prepare the scaled input and reciprocal RMS denominator shared by projections.

    The reciprocal remains separate from the BF16 scaled input so callers can apply it after
    each FC without materializing a full normalized hidden tensor.
    """
    hidden = pre_attn.shape[-1]
    input_scale = load_tile(input_norm_weight, banks=banks).reshape(1, 1, hidden)
    x_scaled = ax.dsl.mul(pre_attn, input_scale, output_dtype=pre_attn.dtype)
    x_sqrt = rmsnorm_sqrt(pre_attn, eps=eps)
    return x_scaled, ax.dsl.reciprocal(x_sqrt, output_dtype=ax.fp32)


def slice_fc_scale(scale, slice_oc, slice_ic=None, full_ic=None):
    """Slice per-output or grouped FC scales consistently with an ``[OC, IC]`` weight shard."""
    if scale is None:
        return None
    if scale.shape_size <= 1:
        return scale
    # Batched MatMul expects a leading batch dimension on a runtime scale.
    # Keep the shared scale explicit as [1, OC, IC/group] so each batch slice
    # retains its per-output/grouped scale after DSL slicing.
    batched_runtime_scale = scale.ndim == 3 and scale.shape[0] == 1
    oc_axis = 1 if batched_runtime_scale else 0
    group_axis = 2 if batched_runtime_scale else 1

    def sliced_scale(output_slice, group_slice):
        if batched_runtime_scale:
            return scale[:, output_slice, group_slice]
        return scale[output_slice, group_slice]

    if slice_ic is None or full_ic is None:
        return sliced_scale(slice_oc, slice(None))

    scale_group_count = scale.shape[group_axis]
    if scale_group_count == 1:
        return sliced_scale(slice_oc, slice(None))
    if scale_group_count == full_ic // 16:
        group_size = 16
    elif scale_group_count == full_ic // 32:
        group_size = 32
    elif scale_group_count == full_ic:
        group_size = 1
    else:
        raise AssertionError(f"unsupported FC scale shape {scale.shape} for input width {full_ic}")

    ic_start = 0 if slice_ic.start is None else slice_ic.start
    ic_stop = full_ic if slice_ic.stop is None else slice_ic.stop
    assert ic_start % group_size == 0, "FC input slice start must align to its scale group"
    assert ic_stop % group_size == 0, "FC input slice stop must align to its scale group"
    scale_ic = slice(ic_start // group_size, ic_stop // group_size)
    return sliced_scale(slice_oc, scale_ic)


def slice_wave_scale(scale, output_slice, group_slice):
    """Slice one OC/IC wave and return the 2-D MatMul scale layout."""
    if scale.ndim == 3 and scale.shape[0] == 1:
        return scale[0, output_slice, group_slice]
    return scale[output_slice, group_slice]


def scale_for_matmul(scale, x_ndim):
    """Match a runtime scale rank to a 2-D or batched MatMul input rank."""
    if scale is None:
        return None
    if x_ndim == 3 and scale.ndim == 2:
        return scale.reshape(1, scale.shape[0], scale.shape[1])
    if x_ndim == 2 and scale.ndim == 3 and scale.shape[0] == 1:
        return scale.reshape(scale.shape[1], scale.shape[2])
    return scale


def stateful_causal_depthwise_conv1d_silu(
    projected, weight, state_in, state_out, conv_size, name_prefix, banks=None, history_rows=None
):
    """Run causal Conv1D from a runtime checkpoint-layout weight and update state."""
    batch, seq_len, channels = projected.shape
    state_len = conv_size - 1
    conv_input = ax.dsl.create_tensor(
        (batch, state_len + seq_len, channels),
        projected.dtype,
        name=f"{name_prefix}_conv_input",
    )
    load_tile(x=state_in, r=conv_input[:, :state_len, :], banks=banks)
    # Conv consumes one contiguous history-plus-projection window.
    ax.dsl.move(x=projected, r=conv_input[:, state_len:, :])

    # Both PyTorch Conv1d checkpoints and DSL Conv use oldest-to-newest taps.
    # Keep their order when exposing the runtime rank-three filter.
    if seq_len == 1:
        # transpose: [B,K,C] BF16 OCM -> [B,C,K] BF16, with contiguous taps.
        window = ax.dsl.transpose(conv_input, perm=(0, 2, 1))
        # load/reshape: runtime [C,1,K] BF16 -> [1,C,K] BF16 OCM.
        taps = load_tile(weight, banks=banks).reshape(1, channels, conv_size)
        # mul: aligned [B,C,K] operands -> [B,C,K] FP32 tap products.
        products = ax.dsl.mul(window, taps, output_dtype=ax.fp32)
        # sum: reduce K taps in FP32, emit [B,C] BF16 before SiLU.
        conv_value = ax.dsl.sum(
            products,
            dim=-1,
            keepdim=False,
            output_dtype=projected.dtype,
        ).reshape(batch, 1, channels)

    else:
        # History is already prepended: valid Conv emits exactly the S new rows.
        # conv: [B,K-1+S,C] BF16 * [C,1,K] BF16 -> [B,S,C] BF16 OCM.
        conv_value = ax.dsl.conv(
            conv_input,
            load_tile(weight, banks=banks),
            stride=[1],
            padding=[0, 0],
            dilation=[1],
            group=channels,
            layout="NLC",
            output_dtype=projected.dtype,
        )

    # Explicit row IDs preserve the core-owned channel slice and update only
    # this layer's history in the retained state pool.
    rows = ax.dsl.from_numpy(np.arange(state_len, dtype=np.int32))
    if history_rows is None:
        next_history = conv_input[:, seq_len : seq_len + state_len, :].reshape(state_len, channels)
    else:
        # The window starts at the real token count, including old history
        # when fewer than conv_size - 1 new tokens are valid.
        next_history = ax.dsl.index_select(conv_input.reshape(state_len + seq_len, channels), history_rows, dim=0)
    scatter_rows(
        next_history,
        rows,
        dim=0,
        r=state_out.reshape(state_len, channels),
        banks=banks,
    )

    # silu: [B,S,C] BF16 -> [B,S,C] BF16, after Conv output rounding.
    return ax.dsl.silu(conv_value, output_dtype=projected.dtype)


def balanced_aligned_tile_slices(length, max_tile, alignment):
    """Split channels into balanced aligned tiles when one tile is not enough."""
    assert length > 0 and max_tile > 0
    if length <= max_tile:
        return [slice(0, length)]

    assert max_tile >= alignment > 0
    assert length % alignment == 0 and max_tile % alignment == 0
    tile_count = (length + max_tile - 1) // max_tile
    total_units = length // alignment
    base_units, extra_units = divmod(total_units, tile_count)

    slices = []
    start = 0
    for tile_index in range(tile_count):
        tile_units = base_units + (1 if tile_index < extra_units else 0)
        stop = start + tile_units * alignment
        slices.append(slice(start, stop))
        start = stop
    return slices


def load_pre_attention_residual(x, residual_in, core_id, core_num, name, token_indices=None, banks=None):
    """Assemble shared [1,S,H] BF16 input from disjoint hidden shards."""
    if x.memory_type != ax.MemoryType.DDR and residual_in is None and token_indices is None:
        # The fused producer already published the complete shared activation.
        return x
    batch, full_seq_len, hidden = x.shape
    seq_len = full_seq_len if token_indices is None else token_indices.shape_size
    hidden_per_core = hidden // core_num
    hidden_slice = slice(core_id * hidden_per_core, (core_id + 1) * hidden_per_core)
    memory_type = ax.MemoryType.SHARED_OCM
    pre_attn = ax.dsl.create_tensor((batch, seq_len, hidden), x.dtype, memory_type, name=name)
    destination = pre_attn[:, :, hidden_slice]

    if token_indices is None:
        if residual_in is None:
            # load: DDR hidden shard -> its final shared BF16 slice.
            load_tile(x[:, :, hidden_slice], r=destination, banks=banks)
        else:
            ax.dsl.add(
                load_tile(x[:, :, hidden_slice], banks=banks),
                load_tile(residual_in[:, :, hidden_slice], banks=banks),
                output_dtype=x.dtype,
                r=destination,
            )
    elif residual_in is None:
        # index_select: DDR rows -> their final [S,H/core] BF16 shared view.
        gather_rows(
            x.reshape(full_seq_len, hidden)[:, hidden_slice],
            token_indices,
            dim=0,
            r=destination.reshape(seq_len, hidden_per_core),
            banks=banks,
        )
    else:
        x_hidden = gather_rows(
            x.reshape(full_seq_len, hidden)[:, hidden_slice],
            token_indices,
            dim=0,
            banks=banks,
        )
        residual_hidden = gather_rows(
            residual_in.reshape(full_seq_len, hidden)[:, hidden_slice],
            token_indices,
            dim=0,
            banks=banks,
        )
        ax.dsl.add(
            x_hidden,
            residual_hidden,
            output_dtype=x.dtype,
            r=destination.reshape(seq_len, hidden_per_core),
        )

    # All projections read the complete input after disjoint shard writes.
    axd.pub(pre_attn)
    axd.sub(pre_attn)
    return pre_attn


def store_row_parallel_o_proj_residual_nmc(
    input_fc,
    w_o,
    w_o_scale,
    w_o_scale2,
    residual_base,
    output,
    residual_out,
    slice_ic,
    core_id,
    core_num,
    name_prefix,
    ic_tile=None,
    oc_tile=None,
    token_indices=None,
    banks=None,
):
    """NMC-accumulate a row-parallel output projection and store one hidden shard.

    Once all cores have finished consuming the shared ``residual_base``, they
    cooperatively initialize one shared accumulator by hidden shard. Each round
    then computes one rotated output-hidden shard and uses OCM NMC add to update
    that accumulator in place. The rotation keeps all cores on disjoint
    destination slices within a round. Optional aligned IC/OC tiles expose the
    physical FC split in the DSL graph so each partial can accumulate directly
    into the final shared tensor.
    """
    batch, tile_seq_len, hidden = residual_base.shape
    hidden_per_core = hidden // core_num
    local_input_width = input_fc.shape[-1]
    assert hidden % core_num == 0
    assert input_fc.shape == (batch * tile_seq_len, local_input_width)
    if token_indices is None:
        assert residual_base.shape == residual_out.shape
    else:
        assert token_indices.shape_size == tile_seq_len
    assert slice_ic.start is not None and slice_ic.stop is not None
    assert slice_ic.stop - slice_ic.start == local_input_width

    ic_slices = (
        [slice(0, local_input_width)]
        if ic_tile is None
        else balanced_aligned_tile_slices(local_input_width, ic_tile, 128)
    )
    oc_slices = (
        [slice(0, hidden_per_core)] if oc_tile is None else balanced_aligned_tile_slices(hidden_per_core, oc_tile, 128)
    )

    memory_type = ax.MemoryType.SHARED_OCM
    accumulator = (
        residual_out
        if residual_out.memory_type != ax.MemoryType.DDR
        else ax.dsl.create_tensor(
            residual_base.shape,
            residual_base.dtype,
            memory_type,
            name=f"{name_prefix}_residual_accumulator",
        )
    )
    owned_hidden_start = core_id * hidden_per_core
    owned_hidden_stop = owned_hidden_start + hidden_per_core
    owned_hidden = slice(owned_hidden_start, owned_hidden_stop)
    ax.dsl.move(
        x=residual_base[:, :, owned_hidden],
        r=accumulator[:, :, owned_hidden],
    )
    # Publish the initialized residual before another core adds its projection.
    axd.pub(accumulator)
    axd.sub(accumulator)

    scale2 = None if w_o_scale2 is None else load_tile(w_o_scale2, banks=banks)
    staged_scale = None
    block_scale = (
        len(ic_slices) > 1
        and is_nvfp4_weight(w_o)
        and w_o_scale is not None
        and w_o_scale.shape[-1] == w_o.shape[1] // 16
        and local_input_width // 16 >= 64
    )
    if block_scale:
        scale_bytes = w_o_scale.shape_size * w_o_scale.dtype.num_bits // 8
        if batch * tile_seq_len > 1 and 1024 * 1024 < scale_bytes <= 2 * 1024 * 1024:
            # Full physical rows are shared across all IC owners without
            # loading unused columns or changing the BF16 arithmetic waves.
            shared_scale = load_shared_scale_rows(w_o_scale, banks=banks)
            staged_scale = slice_fc_scale(shared_scale, slice(None), slice_ic, w_o.shape[1])
        elif batch * tile_seq_len == 1 and hidden * (local_input_width // 16) <= 256 * 1024:
            staged_scale = load_tile(slice_fc_scale(w_o_scale, slice(None), slice_ic, w_o.shape[1]), banks=banks)
    for reduction_round in range(core_num):
        hidden_start = ((reduction_round + core_id) * hidden_per_core) % hidden
        for oc_slice in oc_slices:
            slice_hidden = slice(hidden_start + oc_slice.start, hidden_start + oc_slice.stop)
            output_width = oc_slice.stop - oc_slice.start
            for local_ic_slice in ic_slices:
                global_ic_slice = slice(
                    slice_ic.start + local_ic_slice.start,
                    slice_ic.start + local_ic_slice.stop,
                )
                if staged_scale is not None:
                    w_o_scale_value = scale_for_matmul(
                        slice_fc_scale(staged_scale, slice_hidden, local_ic_slice, local_input_width), 2
                    )
                else:
                    w_o_scale_value = (
                        None
                        if w_o_scale is None
                        else scale_for_matmul(
                            load_tile(
                                slice_fc_scale(
                                    w_o_scale,
                                    slice_hidden,
                                    global_ic_slice,
                                    w_o.shape[1],
                                ),
                                banks=banks,
                            ),
                            2,
                        )
                    )
                projected = ax.dsl.matmul(
                    input_fc[:, local_ic_slice],
                    load_tile(w_o[slice_hidden, global_ic_slice], banks=banks),
                    w_scale=w_o_scale_value,
                    w_scale2=scale2,
                    transposed=True,
                    output_dtype=residual_out.dtype,
                ).reshape(batch, tile_seq_len, output_width)
                ax.dsl.move(
                    x=projected,
                    r=accumulator[:, :, slice_hidden],
                    op="add",
                )
        # The next core must consume this round's completed partial sum.
        axd.pub(accumulator)
        axd.sub(accumulator)

    residual_value = accumulator[:, :, owned_hidden]
    if token_indices is None:
        if residual_out.memory_type == ax.MemoryType.DDR:
            store_tile(x=residual_value, r=residual_out[:, :, owned_hidden], banks=banks)
        if output is not None:
            store_tile(x=residual_value, r=output[:, :, owned_hidden], banks=banks)
    else:
        rows = residual_value.reshape(tile_seq_len, hidden_per_core)
        scatter_rows(
            rows,
            token_indices,
            dim=0,
            r=residual_out.reshape(-1, hidden)[:, owned_hidden],
            banks=banks,
        )
        if output is not None:
            scatter_rows(
                rows,
                token_indices,
                dim=0,
                r=output.reshape(-1, hidden)[:, owned_hidden],
                banks=banks,
            )


def project_normed_input(
    x_scaled,
    x_rstd,
    weight,
    weight_scale,
    weight_scale2,
    nvfp4_oc_tile_width=None,
    banks=None,
):
    """Record a GDN runtime projection scaled by the shared input RMSNorm reciprocal."""
    original_shape = x_scaled.shape
    if is_nvfp4_weight(weight):
        # Packed FP4 weights cover NVFP4 (E4M3 group-16 plus scale2) and MXFP4
        # (E8M0 group-32, one scale); MatMul selects MXFP4 from the scale dtype.
        # The AX8860 FP4 path needs a 2-D token matrix, so stage narrow
        # output-channel tiles and compute the previous tile while the next
        # tile is loading; a full production weight forms one long SDMA barrier.
        x_rows = x_scaled.reshape(-1, original_shape[-1])
        raw = ax.dsl.create_tensor(
            (x_rows.shape[0], weight.shape[0]),
            x_scaled.dtype,
            name=f"{weight.name}_nvfp4_raw",
        )
        scale2_value = None if weight_scale2 is None else load_tile(weight_scale2, banks=banks)
        pending = None
        oc_tile_width = nvfp4_oc_tile_width or GDN_NVFP4_PROJECTION_TILE_WIDTH
        for output_start in range(0, weight.shape[0], oc_tile_width):
            output_stop = min(output_start + oc_tile_width, weight.shape[0])
            output_slice = slice(output_start, output_stop)
            staged_weight = load_tile(weight[output_slice, :], banks=banks)
            staged_scale = None if weight_scale is None else load_tile(weight_scale[output_slice, :], banks=banks)
            if pending is not None:
                previous_slice, previous_weight, previous_scale = pending
                ax.dsl.matmul(
                    x_rows,
                    previous_weight,
                    w_scale=previous_scale,
                    w_scale2=scale2_value,
                    transposed=True,
                    output_dtype=x_scaled.dtype,
                    r=raw[:, previous_slice],
                )
            pending = output_slice, staged_weight, staged_scale

        if pending is not None:
            output_slice, staged_weight, staged_scale = pending
            ax.dsl.matmul(
                x_rows,
                staged_weight,
                w_scale=staged_scale,
                w_scale2=scale2_value,
                transposed=True,
                output_dtype=x_scaled.dtype,
                r=raw[:, output_slice],
            )
        projected = ax.dsl.mul(
            raw,
            x_rstd.reshape(-1, 1),
            output_dtype=x_scaled.dtype,
        )
        return projected.reshape(*original_shape[:-1], weight.shape[0])
    else:
        x_rows = x_scaled.reshape(-1, original_shape[-1])
        weight_scale = None if weight_scale is None else load_tile(scale_for_matmul(weight_scale, 2), banks=banks)
        weight_scale2 = None if weight_scale2 is None else load_tile(weight_scale2, banks=banks)
        projected = ax.dsl.matmul(
            x_rows,
            load_tile(weight, banks=banks),
            w_scale=weight_scale,
            w_scale2=weight_scale2,
            transposed=True,
            output_dtype=x_scaled.dtype,
        )
        projected = ax.dsl.mul(
            projected,
            x_rstd.reshape(-1, 1),
            output_dtype=x_scaled.dtype,
        )
        return projected.reshape(*original_shape[:-1], weight.shape[0])


def project_normed_inputs_scaled_waves(
    x_scaled,
    x_rstd,
    projections,
    ic_tile_width,
    scale_group_size,
    oc_tile_width=None,
    raw_buffers=None,
    banks=None,
):
    """Record scaled projections with balanced OC owners and source-level IC waves.

    The baseline backend already splits MatMul IC internally, but its weight staging is a full-IC
    256-output-channel tensor. Explicit smaller IC waves reduce staging allocation pressure. Owner
    tiles from independent projections are interleaved into common IC waves rather than
    sequential projection lifetimes. ``scale_group_size`` is 16 for NVFP4 and 32 for MXFP8.
    """
    assert scale_group_size in (16, 32)
    assert ic_tile_width > 0 and ic_tile_width % scale_group_size == 0
    assert projections
    if oc_tile_width is None:
        oc_tile_width = GDN_NVFP4_PROJECTION_TILE_WIDTH

    original_shape = x_scaled.shape
    x_rows = x_scaled.reshape(-1, original_shape[-1])

    outputs = []
    owner_groups = []
    input_width = projections[0][1].shape[1]
    for projection_index, (raw_name, weight, scale, scale2) in enumerate(projections):
        assert scale is not None, f"{raw_name} must provide a block scale"
        assert weight.shape[1] % scale_group_size == 0
        assert weight.shape[1] == input_width, "waved projections must share input width"
        # The fused loop may lend its consumed carry to the two Q/K
        # projections. Keep their BF16 values there through the convolution.
        raw = (
            ax.dsl.create_tensor((x_rows.shape[0], weight.shape[0]), x_scaled.dtype, name=raw_name)
            if raw_buffers is None
            else raw_buffers[projection_index]
        )
        outputs.append(raw)
        scale2 = None if scale2 is None else load_tile(scale2, banks=banks)

        owners = []
        for output_start in range(0, weight.shape[0], oc_tile_width):
            output_stop = min(output_start + oc_tile_width, weight.shape[0])
            owners.append((raw, slice(output_start, output_stop), weight, scale, scale2))
        owner_groups.append(owners)

    # For Q/K this produces Q0, K0, Q1, K1; a single Z call produces Z0, Z1, Z2, Z3.
    owners = []
    for tile_index in range(max(len(group) for group in owner_groups)):
        for group in owner_groups:
            if tile_index < len(group):
                owners.append(group[tile_index])

    ic_tiles = [slice(start, min(start + ic_tile_width, input_width)) for start in range(0, input_width, ic_tile_width)]

    owner_row_bytes = max(
        (output_slice.stop - output_slice.start)
        * (weight.shape[1] * weight.dtype.num_bits + scale.shape[-1] * scale.dtype.num_bits)
        // 8
        for _, output_slice, weight, scale, _ in owners
    )
    if x_rows.shape[0] > 1 and scale_group_size == 16 and owner_row_bytes <= 512 * 1024:
        # Consume one complete OC tile's scale rows before moving to the next.
        # Full weight rows allow the same bounded double-buffered IO schedule.
        _, first_slice, first_weight, first_scale, _ = owners[0]
        weight = load_tile(first_weight[first_slice, :], banks=banks)
        scale = load_tile(slice_wave_scale(first_scale, first_slice, slice(None)), banks=banks)
        for owner_index, (raw, output_slice, _, _, scale2) in enumerate(owners):
            next_weight = next_scale = None
            if owner_index + 1 < len(owners):
                _, next_slice, source_weight, source_scale, _ = owners[owner_index + 1]
                next_weight = load_tile(source_weight[next_slice, :], banks=banks)
                next_scale = load_tile(slice_wave_scale(source_scale, next_slice, slice(None)), banks=banks)
            for wave_index, input_slice in enumerate(ic_tiles):
                group_slice = slice(input_slice.start // scale_group_size, input_slice.stop // scale_group_size)
                ax.dsl.matmul(
                    x_rows[:, input_slice],
                    weight[:, input_slice],
                    w_scale=slice_wave_scale(scale, slice(None), group_slice),
                    w_scale2=scale2,
                    transposed=True,
                    output_dtype=x_scaled.dtype,
                    r=raw[:, output_slice],
                    output_add_assign=wave_index > 0,
                )
            weight, scale = next_weight, next_scale
    else:
        # Two pending IC waves already retain the complete payload. Load bounded
        # full rows once to avoid fragmented DDR gathers, keeping the same compute
        # waves and BF16 rounding. Reserve the remaining OCM for state and staging.
        full_row_bytes = sum(
            weight.shape_size * weight.dtype.num_bits // 8 + scale.shape_size * scale.dtype.num_bits // 8
            for _, weight, scale, _ in projections
        )
        stage_rows = (
            x_rows.shape[0] == 1 and scale_group_size == 16 and len(ic_tiles) <= 2 and full_row_bytes <= 3 * 1024 * 1024
        )
        cached_weights, cached_scales = [], []
        if stage_rows:
            for _, output_slice, weight, scale, _ in owners:
                cached_weights.append(load_tile(weight[output_slice, :], banks=banks))
                cached_scales.append(load_tile(slice_wave_scale(scale, output_slice, slice(None)), banks=banks))
        else:
            scale_bytes = sum(scale.shape_size * scale.dtype.num_bits // 8 for _, _, scale, _ in projections)
            if x_rows.shape[0] > 1 and len(ic_tiles) >= 4 and scale_bytes <= 1024 * 1024:
                cached_scales = [
                    load_tile(slice_wave_scale(scale, output_slice, slice(None)), banks=banks)
                    for _, output_slice, _, scale, _ in owners
                ]

        def load_wave(wave_index):
            loaded = []
            input_slice = ic_tiles[wave_index]
            group_slice = slice(
                input_slice.start // scale_group_size,
                input_slice.stop // scale_group_size,
            )
            for owner_index, (raw, output_slice, weight, scale, scale2) in enumerate(owners):
                staged_weight = (
                    cached_weights[owner_index][:, input_slice]
                    if cached_weights
                    else load_tile(weight[output_slice, input_slice], banks=banks)
                )
                staged_scale = (
                    slice_wave_scale(cached_scales[owner_index], slice(None), group_slice)
                    if cached_scales
                    else load_tile(slice_wave_scale(scale, output_slice, group_slice), banks=banks)
                )
                loaded.append((staged_weight, staged_scale))
            return loaded

        pending_wave = load_wave(0)
        for wave_index in range(len(ic_tiles)):
            # Stage the next IC wave before consuming the current wave so SDMA can overlap CONV work.
            next_wave = load_wave(wave_index + 1) if wave_index + 1 < len(ic_tiles) else None
            for (raw, output_slice, _, _, scale2), (staged_weight, staged_scale) in zip(owners, pending_wave):
                ax.dsl.matmul(
                    x_rows[:, ic_tiles[wave_index]],
                    staged_weight,
                    w_scale=staged_scale,
                    w_scale2=scale2,
                    transposed=True,
                    output_dtype=x_scaled.dtype,
                    r=raw[:, output_slice],
                    output_add_assign=wave_index > 0,
                )
            pending_wave = next_wave

    return tuple(
        ax.dsl.mul(
            raw,
            x_rstd.reshape(-1, 1),
            output_dtype=x_scaled.dtype,
            r=None if raw_buffers is None else raw,
        ).reshape(*original_shape[:-1], raw.shape[-1])
        for raw in outputs
    )


def project_normed_pair(
    x_scaled,
    x_rstd,
    first_weight,
    first_scale,
    first_scale2,
    second_weight,
    second_scale,
    second_scale2,
    *,
    ic_tile_width,
    mxfp4_oc_tile_width=None,
    mxfp8_ic_tile_width=None,
    nvfp4_ic_tile_width=None,
    nvfp4_oc_tile_width=None,
    raw_buffers=None,
    banks=None,
):
    """Interleave Q/K or V/Z projections using the pair's runtime weight format."""
    if is_mxfp4_weight(first_weight, first_scale) and is_mxfp4_weight(second_weight, second_scale):
        group_size = 32
        oc_tile_width = mxfp4_oc_tile_width
        first_scale2 = second_scale2 = None
    elif is_nvfp4_weight(first_weight) and is_nvfp4_weight(second_weight):
        group_size = 16
        ic_tile_width = nvfp4_ic_tile_width or GDN_NVFP4_WIDE_PROJECTION_IC_TILE_WIDTH
        oc_tile_width = nvfp4_oc_tile_width
    elif is_mxfp8_weight(first_weight, first_scale) and is_mxfp8_weight(second_weight, second_scale):
        group_size = 32
        ic_tile_width = mxfp8_ic_tile_width or ic_tile_width
        oc_tile_width = GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH
        first_scale2 = second_scale2 = None
    else:
        return (
            project_normed_input(
                x_scaled,
                x_rstd,
                first_weight,
                first_scale,
                first_scale2,
                nvfp4_oc_tile_width,
                banks=banks,
            ),
            project_normed_input(
                x_scaled,
                x_rstd,
                second_weight,
                second_scale,
                second_scale2,
                nvfp4_oc_tile_width,
                banks=banks,
            ),
        )

    return project_normed_inputs_scaled_waves(
        x_scaled,
        x_rstd,
        (
            (f"{first_weight.name}_pair_raw", first_weight, first_scale, first_scale2),
            (
                f"{second_weight.name}_pair_raw",
                second_weight,
                second_scale,
                second_scale2,
            ),
        ),
        ic_tile_width,
        group_size,
        oc_tile_width,
        raw_buffers,
        banks=banks,
    )


def project_normed_inputs_unscaled(x_scaled, x_rstd, weight, output_widths, ic_tile_width=None, banks=None):
    """Record unscaled pre-attention projections and split contiguous outputs."""
    if ic_tile_width is None:
        ic_tile_width = GDN_PRE_PROJECTION_IC_TILE_WIDTH
    if max(output_widths) <= GDN_PRE_PROJECTION_TILE_WIDTH:
        projected = ax.dsl.matmul(
            x_scaled,
            load_tile(weight, banks=banks),
            transposed=True,
            output_dtype=x_scaled.dtype,
        )
        projected = ax.dsl.mul(projected, x_rstd, output_dtype=x_scaled.dtype)

        outputs = []
        channel_start = 0
        for output_width in output_widths:
            channel_stop = channel_start + output_width
            outputs.append(projected[..., channel_start:channel_stop])
            channel_start = channel_stop
        return tuple(outputs)

    # Qwen3.5-9B fuses 3088 output channels per core. A whole-core BF16 weight
    # slice needs over 25 MB of OCM, so split every projection into 256-output-
    # channel tiles. A production tile is 2 MiB and takes about 33 us to load;
    # split it again over input channels so a 512 KiB wave can double-buffer
    # below that full-tile OCM footprint. Stage the next wave before computing
    # the current wave, then flatten all projection tiles into that queue.
    input_width = weight.shape[1]
    input_slices = [
        slice(start, min(start + ic_tile_width, input_width)) for start in range(0, input_width, ic_tile_width)
    ]

    tiles = []
    channel_start = 0
    for projection_index, output_width in enumerate(output_widths):
        channel_stop = channel_start + output_width
        for channel_offset in range(channel_start, channel_stop, GDN_PRE_PROJECTION_TILE_WIDTH):
            tile_stop = min(channel_offset + GDN_PRE_PROJECTION_TILE_WIDTH, channel_stop)
            tiles.append(
                {
                    "projection": projection_index,
                    "channel_slice": slice(channel_offset, tile_stop),
                    "raw": ax.dsl.create_tensor(
                        x_scaled.shape[:-1] + (tile_stop - channel_offset,),
                        x_scaled.dtype,
                        name=f"gdn_pre_projection_{projection_index}_{channel_offset}",
                    ),
                }
            )
        channel_start = channel_stop

    waves = []
    for tile in tiles:
        for wave_index, input_slice in enumerate(input_slices):
            waves.append(
                {
                    "tile": tile,
                    "input_slice": input_slice,
                    "first": wave_index == 0,
                    "last": wave_index + 1 == len(input_slices),
                }
            )

    pending_weight = load_tile(weight[tiles[0]["channel_slice"], input_slices[0]], banks=banks)
    for wave_index, wave in enumerate(waves):
        next_weight = None
        if wave_index + 1 < len(waves):
            next_wave = waves[wave_index + 1]
            next_weight = load_tile(
                weight[next_wave["tile"]["channel_slice"], next_wave["input_slice"]],
                banks=banks,
            )
        ax.dsl.matmul(
            x_scaled[..., wave["input_slice"]],
            pending_weight,
            transposed=True,
            output_dtype=x_scaled.dtype,
            r=wave["tile"]["raw"],
            output_add_assign=not wave["first"],
        )
        if wave["last"]:
            wave["tile"]["projected"] = ax.dsl.mul(wave["tile"]["raw"], x_rstd, output_dtype=x_scaled.dtype)
        pending_weight = next_weight

    outputs = []
    for projection_index in range(len(output_widths)):
        parts = [tile["projected"] for tile in tiles if tile["projection"] == projection_index]
        outputs.append(parts[0] if len(parts) == 1 else ax.dsl.concat(*parts, dim=-1))
    return tuple(outputs)


def load_shared_gate_parameters(a_log, dt_bias, core_id, banks=None):
    """Load GDN gate parameters once and synchronize all participating cores."""
    memory_type = ax.MemoryType.SHARED_OCM
    a_log_shared = ax.dsl.create_tensor(a_log.shape, a_log.dtype, memory_type, name="gdn_a_log")
    dt_bias_shared = ax.dsl.create_tensor(dt_bias.shape, dt_bias.dtype, memory_type, name="gdn_dt_bias")
    if core_id == 0:
        load_tile(x=a_log, r=a_log_shared, banks=banks)
        load_tile(x=dt_bias, r=dt_bias_shared, banks=banks)
    axd.pub(a_log_shared)
    axd.pub(dt_bias_shared)
    axd.sub(a_log_shared)
    axd.sub(dt_bias_shared)
    return a_log_shared, dt_bias_shared


def l2norm_lastdim(x, eps=1e-6):
    """Record the GDN direct-divide L2Norm formulation."""
    x_square = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(x_square, dim=-1, keepdim=True, output_dtype=ax.fp32)
    denominator = ax.dsl.sqrt(
        ax.dsl.add(
            square_sum,
            ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=eps),
            output_dtype=ax.fp32,
        ),
        output_dtype=ax.fp32,
    )
    return ax.dsl.div(x, denominator, output_dtype=x.dtype)


def gdn_gate(a_tile, b_tile, a_log_head, dt_bias_head):
    """Record GDN decay and write gates for one value-head tile."""
    shifted_a = ax.dsl.add(a_tile, dt_bias_head, output_dtype=ax.fp32)
    decay_rate = ax.dsl.exp(a_log_head, output_dtype=ax.fp32)
    decay_rate = ax.dsl.mul(
        decay_rate,
        ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=-1.0),
        output_dtype=ax.fp32,
    )
    g = ax.dsl.mul(
        decay_rate,
        ax.dsl.softplus(shifted_a, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    beta = ax.dsl.sigmoid(b_tile, output_dtype=ax.bf16)
    return g, beta


def gdn_recurrent_step(q_scaled, k, v, g, beta, state, output_dtype):
    """Update independent heads and read their updated FP32 states.

    Q/K are BF16 [H,1,K], V is BF16 [H,1,V], gates are [H,1,1], and
    state is FP32 [H,K,V]. Keep the two BF16 state views required by the
    projection math; the persistent state and outer-product update stay FP32.
    """
    state = ax.dsl.mul(ax.dsl.exp(g, output_dtype=ax.fp32), state, output_dtype=ax.fp32)
    prediction = ax.dsl.matmul(
        k,
        ax.dsl.cast(state, dtype=ax.bf16),
        output_dtype=v.dtype,
    )
    # Keep the small prediction residual in FP32 until the FP32 state update.
    error = ax.dsl.mul(
        ax.dsl.sub(v, prediction, output_dtype=ax.fp32),
        beta,
        output_dtype=ax.fp32,
    )
    # IC=1 is an outer product; broadcast multiplication avoids CONV IC padding.
    # Moving the singleton token axis only needs a view: [H,1,K] -> [H,K,1].
    state_delta = ax.dsl.mul(
        k.reshape(k.shape[0], k.shape[-1], 1),
        error,
        output_dtype=ax.fp32,
    )
    state = ax.dsl.add(state, state_delta, output_dtype=ax.fp32)
    output = ax.dsl.sum(
        ax.dsl.mul(
            q_scaled.reshape(q_scaled.shape[0], q_scaled.shape[-1], 1),
            ax.dsl.cast(state, dtype=ax.bf16),
            output_dtype=ax.fp32,
        ),
        dim=1,
        keepdim=True,
        output_dtype=output_dtype,
    )
    return output, state


def chunk_prefix_sum_fp32(x, chunk_size, name):
    """Compute independent FP32 prefix sums for every sequence chunk and head."""
    batch, seq_len, heads = x.shape
    prefix = ax.dsl.create_tensor(x.shape, ax.fp32, name=name)
    full_chunk_count = seq_len // chunk_size
    full_seq_len = full_chunk_count * chunk_size

    if full_chunk_count:
        x_full = x[:, :full_seq_len, :].reshape(batch, full_chunk_count, chunk_size, heads)
        prefix_full = prefix[:, :full_seq_len, :].reshape(batch, full_chunk_count, chunk_size, heads)
        ax.dsl.cumsum(x_full, dim=2, output_dtype=ax.fp32, r=prefix_full)

    if full_seq_len < seq_len:
        x_tail = x[:, full_seq_len:, :]
        prefix_tail = prefix[:, full_seq_len:, :]
        ax.dsl.cumsum(x_tail, dim=1, output_dtype=ax.fp32, r=prefix_tail)

    return prefix


def small_matmul_fp32(x, y):
    """Multiply FP32 matrices with bounded scratch and balanced accumulation."""
    assert x.shape[:-2] == y.shape[:-2]
    rows, reduction = x.shape[-2:]
    assert y.shape[-2] == reduction
    columns = y.shape[-1]
    if reduction < 16:
        products = ax.dsl.mul(
            x.reshape(*x.shape[:-2], rows, reduction, 1),
            y.reshape(*y.shape[:-2], 1, reduction, columns),
            output_dtype=ax.fp32,
        )
        return ax.dsl.sum(products, dim=-2, keepdim=False, output_dtype=ax.fp32)
    # [*B,M,Kt,1] FP32 * [*B,1,Kt,N] FP32 -> [*B,M,Kt,N] FP32.
    # Bound the reduction scratch; independent partials can use different TENGs.
    partials = []
    for start in range(0, reduction, 4):
        stop = min(start + 4, reduction)
        width = stop - start
        products = ax.dsl.mul(
            x[..., start:stop].reshape(*x.shape[:-2], rows, width, 1),
            y[..., start:stop, :].reshape(*y.shape[:-2], 1, width, columns),
            output_dtype=ax.fp32,
        )
        # Reduce Kt: [*B,M,Kt,N] FP32 -> [*B,M,N] FP32.
        partial = ax.dsl.sum(products, dim=-2, keepdim=False, output_dtype=ax.fp32)
        partials.append(partial)
    # Add independent [*B,M,N] FP32 partials as a balanced tree.
    while len(partials) > 1:
        partials = [
            ax.dsl.add(partials[i], partials[i + 1], output_dtype=ax.fp32) if i + 1 < len(partials) else partials[i]
            for i in range(0, len(partials), 2)
        ]
    return partials[0]


def inverse_unit_lower_blocks(negative_lower, name_prefix):
    """Recursively invert independent FP32 unit-lower triangular blocks.

    ``negative_lower`` is a batch of strict lower-triangular matrices ``N``;
    the result is ``(I - N)^-1``.  The input size is a power of two and the
    two diagonal halves are merged along the batch axis at each recursive
    level.  This preserves independent matrices while allowing the same DSL
    graph to solve all value-head blocks together.
    """
    batches, size, _ = negative_lower.shape
    if size <= 2:
        identity = ax.dsl.from_numpy(
            np.broadcast_to(np.eye(size, dtype=np.float32), negative_lower.shape).copy(),
            prefix=f"{name_prefix}_identity",
        )
        return ax.dsl.add(negative_lower, identity, output_dtype=ax.fp32)

    half = size // 2
    diagonal = ax.dsl.concat(
        negative_lower[:, :half, :half],
        negative_lower[:, half:, half:],
        dim=0,
    )
    inverse = inverse_unit_lower_blocks(diagonal, f"{name_prefix}_{half}")
    top = inverse[:batches]
    bottom = inverse[batches:]

    product = small_matmul_fp32(negative_lower[:, half:, :half], top)
    lower = small_matmul_fp32(bottom, product)
    zero = ax.dsl.from_numpy(
        np.zeros((batches, half, half), dtype=np.float32),
        prefix=f"{name_prefix}_zero",
    )
    result = ax.dsl.create_tensor((batches, size, size), ax.fp32, name=f"{name_prefix}_inverse")
    ax.dsl.concat(top, zero, dim=-1, r=result[:, :half, :])
    ax.dsl.concat(lower, bottom, dim=-1, r=result[:, half:, :])
    return result


def inverse_unit_lower_recursive(negative_lower, name_prefix):
    """Invert FP32 strict-lower matrices of any valid chunk length.

    Leading dimensions are flattened into independent matrices.  A non-power-
    of-two chunk is zero-padded to the next power of two; because the added
    rows and columns are zero, the padded system contains independent identity
    entries and cropping the inverse is mathematically exact.
    """
    size = negative_lower.shape[-1]
    assert negative_lower.shape[-2] == size and size >= 1
    assert negative_lower.dtype == ax.fp32
    batch_shape = negative_lower.shape[:-2]
    batches = int(np.prod(batch_shape))
    padded = 1 << (size - 1).bit_length()
    matrix = negative_lower.reshape(batches, size, size)

    if padded != size:
        zero_columns = ax.dsl.from_numpy(
            np.zeros((batches, size, padded - size), dtype=np.float32),
            prefix=f"{name_prefix}_pad_columns",
        )
        zero_rows = ax.dsl.from_numpy(
            np.zeros((batches, padded - size, padded), dtype=np.float32),
            prefix=f"{name_prefix}_pad_rows",
        )
        matrix = ax.dsl.concat(
            ax.dsl.concat(matrix, zero_columns, dim=-1),
            zero_rows,
            dim=-2,
        )

    inverse = inverse_unit_lower_blocks(matrix, name_prefix)
    return inverse[:, :size, :size].reshape(*batch_shape, size, size)


def gdn_stable_chunk(
    q_scaled,
    k_tile,
    v_tile,
    beta,
    state,
    prefix_g,
    exp_prefix_g,
    name_prefix,
    lower_mask=None,
    negative_strict_lower_mask=None,
    prepared_solver=None,
):
    """Advance independent head states through one causal chunk.

    Q/K are [Ht,C,K] BF16, V is [Ht,C,V] BF16, state is [Ht,K,V]
    FP32, and gate prefixes are [Ht,C,1] FP32. The caller provides either
    shared triangular masks or a precomputed (BF16 inverse, FP32 decay)
    pair [Ht,C,C] for this chunk.
    """
    head_count, chunk_len, value_dim = v_tile.shape
    key_dim = k_tile.shape[-1]
    assert q_scaled.shape == k_tile.shape == (head_count, chunk_len, key_dim)
    assert state.shape == (head_count, key_dim, value_dim)
    assert beta.shape == prefix_g.shape == exp_prefix_g.shape == (head_count, chunk_len, 1)
    if prepared_solver is None:
        gate_difference = ax.dsl.sub(
            prefix_g,
            prefix_g.reshape(*prefix_g.shape[:-2], 1, chunk_len),
            output_dtype=ax.fp32,
        )
        gate_difference = ax.dsl.mul(gate_difference, lower_mask, output_dtype=ax.fp32)
        decay = ax.dsl.exp(gate_difference, output_dtype=ax.fp32)
        decay = ax.dsl.mul(decay, lower_mask, output_dtype=ax.fp32)
    else:
        inverse, decay = prepared_solver

    k_beta = ax.dsl.mul(k_tile, beta, output_dtype=ax.fp32)
    if prepared_solver is None:
        lower = ax.dsl.matmul(k_tile, k_tile, transposed=True, output_dtype=ax.fp32)
        lower = ax.dsl.mul(lower, beta, output_dtype=ax.fp32)
        lower = ax.dsl.mul(lower, decay, output_dtype=ax.fp32)
        negative_lower = ax.dsl.mul(lower, negative_strict_lower_mask, output_dtype=ax.fp32)
        # The solve remains FP32; BF16 is required only at the NNCore boundary.
        inverse = ax.dsl.cast(
            inverse_unit_lower_recursive(negative_lower, f"{name_prefix}_solve"),
            dtype=v_tile.dtype,
        )

    solver_values = ax.dsl.concat(
        ax.dsl.mul(v_tile, beta, output_dtype=v_tile.dtype),
        ax.dsl.mul(k_beta, exp_prefix_g, output_dtype=k_tile.dtype),
        dim=-1,
    )
    transformed = ax.dsl.matmul(inverse, solver_values, output_dtype=v_tile.dtype)
    transformed_v = transformed[..., :value_dim]
    transformed_k = transformed[..., value_dim:]

    state_compute = ax.dsl.cast(state, dtype=ax.bf16)
    # One BF16 NNCore read computes both independent FP32 state products.
    queries = ax.dsl.concat(transformed_k, q_scaled, dim=-2)
    state_products = ax.dsl.matmul(queries, state_compute, output_dtype=ax.fp32)
    v_from_state = state_products[..., :chunk_len, :]
    base_output = state_products[..., chunk_len:, :]
    v_new = ax.dsl.sub(transformed_v, v_from_state, output_dtype=ax.fp32)

    read_weight = ax.dsl.matmul(q_scaled, k_tile, transposed=True, output_dtype=ax.fp32)
    # Match the reference solver's BF16 rounding boundary before the output delta.
    read_weight = ax.dsl.mul(read_weight, decay, output_dtype=v_tile.dtype)
    base_output = ax.dsl.mul(base_output, exp_prefix_g, output_dtype=ax.fp32)
    output_delta = ax.dsl.matmul(read_weight, ax.dsl.cast(v_new, dtype=v_tile.dtype), output_dtype=ax.fp32)
    output_tile = ax.dsl.add(base_output, output_delta, output_dtype=v_tile.dtype)

    final_exp_prefix_g = exp_prefix_g[..., chunk_len - 1 : chunk_len, :]
    final_decay = decay[..., chunk_len - 1 : chunk_len, :].reshape(*decay.shape[:-2], chunk_len, 1)
    # Keep the old state intact until both independent state reads finish.
    state_decay = ax.dsl.mul(
        final_exp_prefix_g,
        state,
        output_dtype=ax.fp32,
    )
    update_value = ax.dsl.mul(v_new, final_decay, output_dtype=v_tile.dtype)
    # MatMul can transpose only its right input, so the key transpose is required.
    ax.dsl.matmul(
        ax.dsl.transpose(k_tile, perm=(0, 2, 1)),
        update_value,
        output_dtype=ax.fp32,
        r=state_decay,
        output_add_assign=True,
    )
    return output_tile, state_decay


GDN_PREFILL_TILE_SIZE = 256
GDN_BASELINE_CHUNK_SIZE = 32
# Keep production-size unscaled pre-projection tiles within the OCM budget.
GDN_PRE_PROJECTION_TILE_WIDTH = 256
# BF16 pre-projection weights stage 2 MiB per 256-OC full-IC tile. 1024-IC
# waves keep the double-buffered staging footprint below that single tile.
GDN_PRE_PROJECTION_IC_TILE_WIDTH = 1024
# Baseline and optimized packed projection tiles are selected by sequence length
# at the GDN wrapper boundary so decode-only tuning does not affect prefill.
GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH = 256
GDN_NVFP4_PROJECTION_TILE_WIDTH = 1024
# Full-IC Q/K staging creates large OCM tiles and delayed address reuse. Source-level
# IC waves make staging finer while the backend preserves output accumulation semantics.
GDN_NVFP4_WIDE_PROJECTION_IC_TILE_WIDTH = 1024
GDN_NVFP4_BASELINE_WIDE_PROJECTION_IC_TILE_WIDTH = 512
# MXFP8 stages one byte per weight. A 256-IC wave is 64 KiB per 256-OC owner;
# 128-IC regresses because MatMul and wave setup cost overtakes the staging gain.
GDN_MXFP8_IC_TILE_WIDTH = 256
# Decode has a single token, so larger waves trade compute overlap for fewer
# MatMul/wave boundaries while staying far below the full-weight OCM footprint.
GDN_MXFP8_DECODE_IC_TILE_WIDTH = 1024
# MXFP4 stages four bits per weight. The selected prefill path uses a 512-IC wave
# with one 1024-OC owner per projection; decode retains its separately tuned width.
GDN_MXFP4_IC_TILE_WIDTH = 1024
GDN_MXFP4_OC_TILE_WIDTH = 512
GDN_MXFP4_DECODE_IC_TILE_WIDTH = 1024
GDN_NVFP4_DTYPE = torch.float4_e2m1fn_x2


def validate_gdn_prefill_contract(
    x,
    residual_in,
    input_norm_w,
    w_q,
    w_k,
    w_v,
    w_z,
    w_a,
    w_b,
    w_pre,
    w_q_conv,
    w_k_conv,
    w_v_conv,
    a_log,
    dt_bias,
    o_norm_w,
    w_o,
    state_in,
    q_conv_state_in,
    k_conv_state_in,
    v_conv_state_in,
    output,
    state_out,
    q_conv_state_out,
    k_conv_state_out,
    v_conv_state_out,
    residual_out,
    core_num,
    conv_size,
    chunk_size,
):
    batch, seq_len, hidden = x.shape
    value_heads = state_in.shape[1]
    key_dim = state_in.shape[2]
    value_dim = state_in.shape[3]
    key_heads = w_q.shape[0] // key_dim
    conv_state_len = conv_size - 1
    nvfp4_weight = is_nvfp4_weight(w_q)

    assert batch == 1, "GDN prefill currently supports one request per launch"
    assert seq_len >= 1
    assert conv_size > 1, "GDN causal convolution requires conv_size > 1"
    assert 1 <= chunk_size <= GDN_PREFILL_TILE_SIZE
    assert hidden % core_num == 0
    assert key_heads > 0 and value_heads % key_heads == 0
    assert key_heads % core_num == 0 and value_heads % core_num == 0

    if residual_in is not None:
        assert residual_in.shape == x.shape
    assert input_norm_w.shape == (hidden,)
    assert w_q.shape == (key_heads * key_dim, hidden)
    assert w_k.shape == w_q.shape
    assert w_v.shape == (value_heads * value_dim, hidden)
    assert w_z.shape == w_v.shape
    assert w_a.shape == (value_heads, hidden)
    assert w_b.shape == w_a.shape
    # Scaled projections consume their individual runtime matrices. Only the
    # unscaled fused path has a concatenated per-core projection matrix.
    if w_pre is not None:
        assert w_pre.shape == (
            2 * w_q.shape[0] + 2 * w_v.shape[0] + 2 * value_heads,
            hidden,
        )
    assert w_q_conv.shape == (key_heads * key_dim, 1, conv_size)
    assert w_k_conv.shape == w_q_conv.shape
    assert w_v_conv.shape == (value_heads * value_dim, 1, conv_size)
    assert a_log.shape == (value_heads,)
    assert dt_bias.shape == (value_heads,)
    assert o_norm_w.shape == (value_dim,)
    assert w_o.shape == (hidden, value_heads * value_dim)
    assert state_out.shape == state_in.shape
    assert q_conv_state_in.shape == (batch, conv_state_len, key_heads * key_dim)
    assert k_conv_state_in.shape == q_conv_state_in.shape
    assert v_conv_state_in.shape == (batch, conv_state_len, value_heads * value_dim)
    assert q_conv_state_out.shape == q_conv_state_in.shape
    assert k_conv_state_out.shape == k_conv_state_in.shape
    assert v_conv_state_out.shape == v_conv_state_in.shape
    assert residual_out.shape == x.shape
    if output is not None:
        assert output.shape == x.shape

    assert x.dtype == ax.bf16
    if residual_in is not None:
        assert residual_in.dtype == x.dtype
    assert state_in.dtype == ax.fp32 and state_out.dtype == state_in.dtype
    assert q_conv_state_in.dtype == x.dtype and k_conv_state_in.dtype == x.dtype
    assert v_conv_state_in.dtype == x.dtype
    assert q_conv_state_out.dtype == x.dtype and k_conv_state_out.dtype == x.dtype
    assert v_conv_state_out.dtype == x.dtype
    assert residual_out.dtype == x.dtype
    if output is not None:
        assert output.dtype == x.dtype
    assert a_log.dtype == ax.fp32
    assert dt_bias.dtype in (ax.bf16, ax.fp32)
    if nvfp4_weight:
        nvfp4_weights = (w_q, w_k, w_v, w_z, w_a, w_b, w_o)
        assert all(is_nvfp4_weight(weight) for weight in nvfp4_weights)
        assert hidden % 16 == 0 and value_heads * value_dim % 16 == 0
    return batch, seq_len, hidden, key_heads, value_heads, key_dim, value_dim


def gdn_prefill_impl(
    x,
    residual_in,
    input_norm_w,
    w_q,
    w_k,
    w_v,
    w_z,
    w_a,
    w_b,
    w_pre,
    w_q_conv,
    w_k_conv,
    w_v_conv,
    a_log,
    dt_bias,
    o_norm_w,
    w_o,
    state_in,
    q_conv_state_in,
    k_conv_state_in,
    v_conv_state_in,
    output,
    state_out,
    q_conv_state_out,
    k_conv_state_out,
    v_conv_state_out,
    residual_out,
    w_q_scale=None,
    w_q_scale2=None,
    w_k_scale=None,
    w_k_scale2=None,
    w_v_scale=None,
    w_v_scale2=None,
    w_z_scale=None,
    w_z_scale2=None,
    w_a_scale=None,
    w_a_scale2=None,
    w_b_scale=None,
    w_b_scale2=None,
    w_o_scale=None,
    w_o_scale2=None,
    conv_size=4,
    chunk_size=GDN_BASELINE_CHUNK_SIZE,
    input_norm_eps=1e-6,
    o_norm_eps=1e-6,
    token_indices=None,
    core_id=None,
    core_num=None,
    projection_buffer=None,
    banks=None,
    valid_tokens=None,
):
    """GDN prefill attention sublayer; post-attention LayerNorm belongs to MLP."""
    core_id = ax.program_id(0) if core_id is None else core_id
    core_num = ax.num_programs(0) if core_num is None else core_num
    (batch, seq_len, hidden, key_heads, value_heads, key_dim, value_dim,) = validate_gdn_prefill_contract(
        x,
        residual_in,
        input_norm_w,
        w_q,
        w_k,
        w_v,
        w_z,
        w_a,
        w_b,
        w_pre,
        w_q_conv,
        w_k_conv,
        w_v_conv,
        a_log,
        dt_bias,
        o_norm_w,
        w_o,
        state_in,
        q_conv_state_in,
        k_conv_state_in,
        v_conv_state_in,
        output,
        state_out,
        q_conv_state_out,
        k_conv_state_out,
        v_conv_state_out,
        residual_out,
        core_num,
        conv_size,
        chunk_size,
    )
    seq_len = seq_len if token_indices is None else token_indices.shape_size
    assert seq_len <= GDN_PREFILL_TILE_SIZE
    assert input_norm_eps > 0 and o_norm_eps > 0
    history_rows = None
    if valid_tokens is not None:
        assert seq_len == GDN_PREFILL_TILE_SIZE and token_indices is None
        assert valid_tokens.shape == (1,) and valid_tokens.dtype == ax.int32
        count = ax.dsl.load(valid_tokens)
        # Both consumers share one row vector, even for a long Conv history.
        rows = ax.dsl.from_numpy(np.arange(max(seq_len, conv_size - 1), dtype=np.int32))
        history_rows = count + rows[: conv_size - 1]
        keep = ax.dsl.minimum(
            ax.dsl.maximum(count - rows[:seq_len], ax.dsl.zeros((1,), ax.int32)),
            ax.dsl.full((1,), ax.int32, 1),
            output_dtype=ax.uint8,
        )

    key_heads_per_core = key_heads // core_num
    value_heads_per_core = value_heads // core_num
    value_heads_per_key = value_heads // key_heads
    key_head_start = core_id * key_heads_per_core
    key_head_stop = key_head_start + key_heads_per_core
    value_head_start = core_id * value_heads_per_core
    value_head_stop = value_head_start + value_heads_per_core
    local_key_width = key_heads_per_core * key_dim
    local_value_width = value_heads_per_core * value_dim
    slice_key = slice(key_head_start * key_dim, key_head_stop * key_dim)
    slice_value = slice(value_head_start * value_dim, value_head_stop * value_dim)
    slice_value_head = slice(value_head_start, value_head_stop)
    pre_attn = load_pre_attention_residual(
        x,
        residual_in,
        core_id,
        core_num,
        "gdn_prefill_pre_attn",
        token_indices=token_indices,
        banks=banks,
    )
    # Each token's input scaling and RMS denominator are computed once and
    # published to all head owners. Private OCM no longer holds four copies.
    shared_memory = ax.MemoryType.SHARED_OCM
    x_scaled = ax.dsl.create_tensor(pre_attn.shape, x.dtype, shared_memory, name="gdn_prefill_scaled_input")
    x_rstd = ax.dsl.create_tensor((batch, seq_len, 1), ax.fp32, shared_memory, name="gdn_prefill_input_rstd")
    rows_per_core = (seq_len + core_num - 1) // core_num
    row_start = min(core_id * rows_per_core, seq_len)
    row_stop = min(row_start + rows_per_core, seq_len)
    if row_start < row_stop:
        owned_rows = slice(row_start, row_stop)
        # load: [H] BF16 DDR -> [1,1,H] BF16 OCM.
        input_scale = load_tile(input_norm_w, banks=banks).reshape(1, 1, hidden)
        # mul: [B,Tc,H] BF16 inputs -> owned [B,Tc,H] BF16 shared output.
        ax.dsl.mul(
            pre_attn[:, owned_rows, :],
            input_scale,
            output_dtype=x.dtype,
            r=x_scaled[:, owned_rows, :],
        )
        # RMS denominator: [B,Tc,H] BF16 -> [B,Tc,1] FP32 OCM.
        x_sqrt = rmsnorm_sqrt(pre_attn[:, owned_rows, :], eps=input_norm_eps)
        # reciprocal: [B,Tc,1] FP32 -> owned FP32 shared output.
        ax.dsl.reciprocal(x_sqrt, output_dtype=ax.fp32, r=x_rstd[:, owned_rows, :])
    axd.pub(x_scaled)
    axd.pub(x_rstd)
    axd.sub(x_scaled)
    axd.sub(x_rstd)

    pre_scales = (
        w_q_scale,
        w_q_scale2,
        w_k_scale,
        w_k_scale2,
        w_v_scale,
        w_v_scale2,
        w_z_scale,
        w_z_scale2,
        w_a_scale,
        w_a_scale2,
        w_b_scale,
        w_b_scale2,
    )
    local_pre_width = 2 * local_key_width + 2 * local_value_width + 2 * value_heads_per_core
    pre_slice = slice(
        core_id * local_pre_width,
        (core_id + 1) * local_pre_width,
    )
    if all(scale is None for scale in pre_scales):
        (q_projected, k_projected, v_projected, z, a, b,) = project_normed_inputs_unscaled(
            x_scaled,
            x_rstd,
            w_pre[pre_slice, :],
            (
                local_key_width,
                local_key_width,
                local_value_width,
                local_value_width,
                value_heads_per_core,
                value_heads_per_core,
            ),
            banks=banks,
        )
        z = z.reshape(batch, seq_len, value_heads_per_core, value_dim)
    else:
        local_q_weight = w_q[slice_key, :]
        local_k_weight = w_k[slice_key, :]
        local_v_weight = w_v[slice_value, :]
        local_z_weight = w_z[slice_value, :]
        q_projected, k_projected = project_normed_pair(
            x_scaled,
            x_rstd,
            local_q_weight,
            slice_fc_scale(w_q_scale, slice_key),
            w_q_scale2,
            local_k_weight,
            slice_fc_scale(w_k_scale, slice_key),
            w_k_scale2,
            ic_tile_width=GDN_MXFP4_IC_TILE_WIDTH,
            mxfp4_oc_tile_width=GDN_MXFP4_OC_TILE_WIDTH,
            mxfp8_ic_tile_width=GDN_MXFP8_IC_TILE_WIDTH,
            nvfp4_ic_tile_width=GDN_NVFP4_BASELINE_WIDE_PROJECTION_IC_TILE_WIDTH,
            nvfp4_oc_tile_width=GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH,
            raw_buffers=None
            if projection_buffer is None
            else (
                projection_buffer[:, :local_key_width],
                projection_buffer[:, local_key_width : 2 * local_key_width],
            ),
            banks=banks,
        )
        v_projected, z = project_normed_pair(
            x_scaled,
            x_rstd,
            local_v_weight,
            slice_fc_scale(w_v_scale, slice_value),
            w_v_scale2,
            local_z_weight,
            slice_fc_scale(w_z_scale, slice_value),
            w_z_scale2,
            ic_tile_width=GDN_MXFP8_IC_TILE_WIDTH,
            mxfp4_oc_tile_width=GDN_MXFP4_OC_TILE_WIDTH,
            mxfp8_ic_tile_width=GDN_MXFP8_IC_TILE_WIDTH,
            nvfp4_ic_tile_width=GDN_NVFP4_BASELINE_WIDE_PROJECTION_IC_TILE_WIDTH,
            nvfp4_oc_tile_width=GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH,
            banks=banks,
        )
        z = z.reshape(batch, seq_len, value_heads_per_core, value_dim)
        a = project_normed_input(
            x_scaled,
            x_rstd,
            w_a[slice_value_head, :],
            slice_fc_scale(w_a_scale, slice_value_head),
            w_a_scale2,
            GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH,
            banks=banks,
        )
        b = project_normed_input(
            x_scaled,
            x_rstd,
            w_b[slice_value_head, :],
            slice_fc_scale(w_b_scale, slice_value_head),
            w_b_scale2,
            GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH,
            banks=banks,
        )

    q = stateful_causal_depthwise_conv1d_silu(
        q_projected,
        w_q_conv[slice_key, :],
        q_conv_state_in[:, :, slice_key],
        q_conv_state_out[:, :, slice_key],
        conv_size,
        "gdn_prefill_q",
        banks=banks,
        history_rows=history_rows,
    ).reshape(batch, seq_len, key_heads_per_core, key_dim)
    k = stateful_causal_depthwise_conv1d_silu(
        k_projected,
        w_k_conv[slice_key, :],
        k_conv_state_in[:, :, slice_key],
        k_conv_state_out[:, :, slice_key],
        conv_size,
        "gdn_prefill_k",
        banks=banks,
        history_rows=history_rows,
    ).reshape(batch, seq_len, key_heads_per_core, key_dim)
    v = stateful_causal_depthwise_conv1d_silu(
        v_projected,
        w_v_conv[slice_value, :],
        v_conv_state_in[:, :, slice_value],
        v_conv_state_out[:, :, slice_value],
        conv_size,
        "gdn_prefill_v",
        banks=banks,
        history_rows=history_rows,
    ).reshape(batch, seq_len, value_heads_per_core, value_dim)
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=key_dim**-0.5)
    # Normalize each local Q/K row once over the full launch sequence. Chunk
    # views below reuse these tensors for every value head mapped to a key head.
    q = ax.dsl.mul(l2norm_lastdim(q), q_scale, output_dtype=x.dtype)
    k = l2norm_lastdim(k)

    a_log_shared, dt_bias_shared = load_shared_gate_parameters(a_log, dt_bias, core_id, banks=banks)
    # Gate equations are token-local and independent of the recurrent state. Build each
    # local value head once for the full sequence; C32 chunks consume views below.
    g_all, beta_all = gdn_gate(
        a,
        b,
        a_log_shared[slice_value_head],
        dt_bias_shared[slice_value_head],
    )
    if valid_tokens is not None:
        # A padded step must be the identity recurrence: log decay zero and
        # no delta update. Mask after Conv/SiLU and gate biases, before cumsum.
        zero = ax.dsl.zeros((1,), x.dtype)
        mask = keep.reshape(1, seq_len, 1, 1)
        q = ax.dsl.where(mask, q, zero, output_dtype=x.dtype)
        k = ax.dsl.where(mask, k, zero, output_dtype=x.dtype)
        v = ax.dsl.where(mask, v, zero, output_dtype=x.dtype)
        mask = keep.reshape(1, seq_len, 1)
        beta_all = ax.dsl.where(mask, beta_all, zero, output_dtype=x.dtype)
        g_all = ax.dsl.where(mask, g_all, ax.dsl.zeros((1,), ax.fp32), output_dtype=ax.fp32)
    prefix_g_all = chunk_prefix_sum_fp32(g_all, chunk_size, "gdn_prefill_prefix_g")
    # The batched solve and recurrent chunks share one head-major layout.
    q_heads = ax.dsl.transpose(q, perm=(0, 2, 1, 3)).reshape(key_heads_per_core, 1, seq_len, key_dim)
    k_heads = ax.dsl.transpose(k, perm=(0, 2, 1, 3)).reshape(key_heads_per_core, 1, seq_len, key_dim)
    v_heads = ax.dsl.transpose(v, perm=(0, 2, 1, 3)).reshape(value_heads_per_core, seq_len, value_dim)
    # Compute exponentials after the layout change so both tensors share one transpose.
    prefix_heads = ax.dsl.transpose(prefix_g_all, perm=(0, 2, 1)).reshape(value_heads_per_core, seq_len)
    exp_prefix_heads = ax.dsl.exp(prefix_heads, output_dtype=ax.fp32).reshape(value_heads_per_core, seq_len, 1)
    prefix_heads = prefix_heads.reshape(value_heads_per_core, seq_len, 1)
    beta_heads = ax.dsl.transpose(beta_all, perm=(0, 2, 1)).reshape(value_heads_per_core, seq_len, 1)
    # Batch state-independent solves across uniform chunks. The recurrent
    # state loop below selects [Hv,C,C] views in causal order.
    prepared_inverse = None
    prepared_decay = None
    if seq_len > chunk_size and seq_len % chunk_size == 0:
        chunk_count = seq_len // chunk_size
        keys = k_heads.reshape(key_heads_per_core * chunk_count, chunk_size, key_dim)
        # One correlation matrix per key head/chunk serves all mapped value heads.
        correlations = ax.dsl.matmul(keys, keys, transposed=True, output_dtype=ax.fp32)
        lower_mask = ax.dsl.from_numpy(
            np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32)).astype(ml_dtypes.bfloat16),
            prefix="gdn_prefill_batched_lower",
        )
        negative_mask = ax.dsl.from_numpy(
            (-np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32), k=-1)).astype(ml_dtypes.bfloat16),
            prefix="gdn_prefill_batched_negative_lower",
        )
        # [Hv,N,C,1] - [Hv,N,1,C] -> causal FP32 decay.
        difference = ax.dsl.sub(
            prefix_heads.reshape(value_heads_per_core, chunk_count, chunk_size, 1),
            prefix_heads.reshape(value_heads_per_core, chunk_count, 1, chunk_size),
            output_dtype=ax.fp32,
        )
        difference = ax.dsl.mul(difference, lower_mask, output_dtype=ax.fp32)
        prepared_decay = ax.dsl.mul(
            ax.dsl.exp(difference, output_dtype=ax.fp32),
            lower_mask,
            output_dtype=ax.fp32,
        )
        # mul: one KK correlation per key head broadcasts over its value heads.
        lower = ax.dsl.mul(
            correlations.reshape(key_heads_per_core, 1, chunk_count, chunk_size, chunk_size),
            beta_heads.reshape(key_heads_per_core, value_heads_per_key, chunk_count, chunk_size, 1),
            output_dtype=ax.fp32,
        ).reshape(value_heads_per_core, chunk_count, chunk_size, chunk_size)
        lower = ax.dsl.mul(lower, prepared_decay, output_dtype=ax.fp32)
        negative_lower = ax.dsl.mul(lower, negative_mask, output_dtype=ax.fp32)
        # The solve stays FP32 until the NNCore input boundary.
        prepared_inverse = ax.dsl.cast(
            inverse_unit_lower_recursive(negative_lower, "gdn_prefill_batched_solve"),
            dtype=x.dtype,
        )

    gdn_output = ax.dsl.create_tensor(
        (batch, seq_len, value_heads_per_core, value_dim),
        x.dtype,
        name="gdn_prefill_core_output",
    )

    # Keep projection GEMMs large, then interleave independent four-value-head
    # recurrent chains so one chain's TENG work can overlap another's CONV work.
    q_heads = ax.dsl.expand(q_heads, shape=(key_heads_per_core, value_heads_per_key, seq_len, key_dim)).reshape(
        value_heads_per_core, seq_len, key_dim
    )
    k_heads = ax.dsl.expand(k_heads, shape=(key_heads_per_core, value_heads_per_key, seq_len, key_dim)).reshape(
        value_heads_per_core, seq_len, key_dim
    )
    solver_masks = {}
    for head_start in range(0, value_heads_per_core, 4):
        head_stop = min(head_start + 4, value_heads_per_core)
        head_count = head_stop - head_start
        head_slice = slice(head_start, head_stop)
        global_heads = slice(value_head_start + head_start, value_head_start + head_stop)
        # load: owned [1,Ht,K,V] FP32 DDR -> [Ht,K,V] FP32 OCM.
        states = load_tile(state_in[:, global_heads, :, :], banks=banks).reshape(head_count, key_dim, value_dim)
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_stop - chunk_start
            chunk_slice = slice(chunk_start, chunk_stop)
            q_tile = q_heads[head_slice, chunk_slice, :]
            k_tile = k_heads[head_slice, chunk_slice, :]
            v_tile = v_heads[head_slice, chunk_slice, :]
            prefix_g = prefix_heads[head_slice, chunk_slice, :]
            exp_prefix_g = exp_prefix_heads[head_slice, chunk_slice, :]
            beta = beta_heads[head_slice, chunk_slice, :]
            lower_mask = None
            negative_strict_lower_mask = None
            if prepared_inverse is None:
                if chunk_len not in solver_masks:
                    solver_masks[chunk_len] = (
                        ax.dsl.from_numpy(
                            np.tril(np.ones((chunk_len, chunk_len), dtype=np.float32)).astype(ml_dtypes.bfloat16),
                            prefix=f"gdn_prefill_c{chunk_len}_lower_mask_shared",
                        ),
                        ax.dsl.from_numpy(
                            (
                                -np.tril(
                                    np.ones((chunk_len, chunk_len), dtype=np.float32),
                                    k=-1,
                                )
                            ).astype(ml_dtypes.bfloat16),
                            prefix=f"gdn_prefill_c{chunk_len}_negative_strict_lower_mask_shared",
                        ),
                    )
                lower_mask, negative_strict_lower_mask = solver_masks[chunk_len]
            prepared_solver = None
            if prepared_inverse is not None:
                chunk_index = chunk_start // chunk_size
                prepared_solver = (
                    prepared_inverse[head_slice, chunk_index : chunk_index + 1, :, :].reshape(
                        head_count, chunk_len, chunk_len
                    ),
                    prepared_decay[head_slice, chunk_index : chunk_index + 1, :, :].reshape(
                        head_count, chunk_len, chunk_len
                    ),
                )
            output_tile, states = gdn_stable_chunk(
                q_tile,
                k_tile,
                v_tile,
                beta,
                states,
                prefix_g,
                exp_prefix_g,
                f"gdn_prefill_h{head_start}_c{chunk_start}",
                lower_mask=lower_mask,
                negative_strict_lower_mask=negative_strict_lower_mask,
                prepared_solver=prepared_solver,
            )
            # transpose: [Ht,C,V] BF16 -> token-major owned output view.
            ax.dsl.transpose(
                output_tile.reshape(1, head_count, chunk_len, value_dim),
                perm=(0, 2, 1, 3),
                r=gdn_output[:, chunk_slice, head_slice, :],
            )
        # store: final [Ht,K,V] FP32 -> the same owned DDR state slice.
        store_tile(
            states.reshape(1, head_count, key_dim, value_dim),
            r=state_out[:, global_heads, :, :],
            banks=banks,
        )

    o_norm_scale = load_tile(o_norm_w, banks=banks).reshape(1, 1, 1, value_dim)
    normed = rmsnorm(gdn_output, o_norm_scale, eps=o_norm_eps)
    normed = ax.dsl.mul(normed, ax.dsl.silu(z, output_dtype=x.dtype), output_dtype=x.dtype)
    store_row_parallel_o_proj_residual_nmc(
        normed.reshape(batch * seq_len, local_value_width),
        w_o,
        w_o_scale,
        w_o_scale2,
        pre_attn,
        output,
        residual_out,
        slice_value,
        core_id,
        core_num,
        name_prefix="gdn_prefill",
        ic_tile=512,
        # Dense OC256 tiles keep each runtime NVFP4 FC result at offset zero.
        oc_tile=256 if is_nvfp4_weight(w_o) else 512,
        token_indices=token_indices,
        banks=banks,
    )
    return (
        output,
        state_out,
        q_conv_state_out,
        k_conv_state_out,
        v_conv_state_out,
        residual_out,
    )


def gdn_decode_impl(
    x,
    residual_in,
    input_norm_w,
    w_q,
    w_k,
    w_v,
    w_z,
    w_a,
    w_b,
    w_pre,
    w_q_conv,
    w_k_conv,
    w_v_conv,
    a_log,
    dt_bias,
    o_norm_w,
    w_o,
    state_in,
    q_conv_state_in,
    k_conv_state_in,
    v_conv_state_in,
    output,
    state_out,
    q_conv_state_out,
    k_conv_state_out,
    v_conv_state_out,
    residual_out,
    w_q_scale=None,
    w_q_scale2=None,
    w_k_scale=None,
    w_k_scale2=None,
    w_v_scale=None,
    w_v_scale2=None,
    w_z_scale=None,
    w_z_scale2=None,
    w_a_scale=None,
    w_a_scale2=None,
    w_b_scale=None,
    w_b_scale2=None,
    w_o_scale=None,
    w_o_scale2=None,
    conv_size=4,
    input_norm_eps=1e-6,
    o_norm_eps=1e-6,
    banks=None,
):
    """GDN single-token attention decode using the exact recurrent state update."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    (batch, seq_len, hidden, key_heads, value_heads, key_dim, value_dim,) = validate_gdn_prefill_contract(
        x,
        residual_in,
        input_norm_w,
        w_q,
        w_k,
        w_v,
        w_z,
        w_a,
        w_b,
        w_pre,
        w_q_conv,
        w_k_conv,
        w_v_conv,
        a_log,
        dt_bias,
        o_norm_w,
        w_o,
        state_in,
        q_conv_state_in,
        k_conv_state_in,
        v_conv_state_in,
        output,
        state_out,
        q_conv_state_out,
        k_conv_state_out,
        v_conv_state_out,
        residual_out,
        core_num,
        conv_size,
        chunk_size=1,
    )
    assert seq_len == 1, "GDN decode requires exactly one token per launch (S == 1)"
    assert input_norm_eps > 0 and o_norm_eps > 0

    key_heads_per_core = key_heads // core_num
    value_heads_per_core = value_heads // core_num
    value_heads_per_key = value_heads // key_heads
    key_head_start = core_id * key_heads_per_core
    value_head_start = core_id * value_heads_per_core
    local_key_width = key_heads_per_core * key_dim
    local_value_width = value_heads_per_core * value_dim
    slice_key = slice(key_head_start * key_dim, (key_head_start + key_heads_per_core) * key_dim)
    slice_value = slice(
        value_head_start * value_dim,
        (value_head_start + value_heads_per_core) * value_dim,
    )
    slice_value_head = slice(value_head_start, value_head_start + value_heads_per_core)
    pre_attn = load_pre_attention_residual(x, residual_in, core_id, core_num, "gdn_decode_pre_attn", banks=banks)
    x_scaled, x_rstd = prepare_projection_folded_rmsnorm(pre_attn, input_norm_w, input_norm_eps, banks=banks)

    pre_scales = (
        w_q_scale,
        w_q_scale2,
        w_k_scale,
        w_k_scale2,
        w_v_scale,
        w_v_scale2,
        w_z_scale,
        w_z_scale2,
        w_a_scale,
        w_a_scale2,
        w_b_scale,
        w_b_scale2,
    )
    local_pre_width = 2 * local_key_width + 2 * local_value_width + 2 * value_heads_per_core
    pre_slice = slice(core_id * local_pre_width, (core_id + 1) * local_pre_width)
    if all(scale is None for scale in pre_scales):
        (q_projected, k_projected, v_projected, z, a, b,) = project_normed_inputs_unscaled(
            x_scaled,
            x_rstd,
            w_pre[pre_slice, :],
            (
                local_key_width,
                local_key_width,
                local_value_width,
                local_value_width,
                value_heads_per_core,
                value_heads_per_core,
            ),
            banks=banks,
        )
        z = z.reshape(batch, 1, value_heads_per_core, value_dim)
    else:
        local_q_weight = w_q[slice_key, :]
        local_k_weight = w_k[slice_key, :]
        local_v_weight = w_v[slice_value, :]
        local_z_weight = w_z[slice_value, :]
        q_projected, k_projected = project_normed_pair(
            x_scaled,
            x_rstd,
            local_q_weight,
            slice_fc_scale(w_q_scale, slice_key),
            w_q_scale2,
            local_k_weight,
            slice_fc_scale(w_k_scale, slice_key),
            w_k_scale2,
            ic_tile_width=GDN_MXFP4_DECODE_IC_TILE_WIDTH,
            mxfp4_oc_tile_width=GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH,
            mxfp8_ic_tile_width=GDN_MXFP8_DECODE_IC_TILE_WIDTH,
            nvfp4_ic_tile_width=1536,
            nvfp4_oc_tile_width=GDN_NVFP4_PROJECTION_TILE_WIDTH,
            banks=banks,
        )
        v_projected, z = project_normed_pair(
            x_scaled,
            x_rstd,
            local_v_weight,
            slice_fc_scale(w_v_scale, slice_value),
            w_v_scale2,
            local_z_weight,
            slice_fc_scale(w_z_scale, slice_value),
            w_z_scale2,
            ic_tile_width=GDN_MXFP4_DECODE_IC_TILE_WIDTH,
            mxfp4_oc_tile_width=GDN_NVFP4_BASELINE_PROJECTION_TILE_WIDTH,
            mxfp8_ic_tile_width=GDN_MXFP8_DECODE_IC_TILE_WIDTH,
            nvfp4_ic_tile_width=1536,
            nvfp4_oc_tile_width=GDN_NVFP4_PROJECTION_TILE_WIDTH,
            banks=banks,
        )
        z = z.reshape(batch, 1, value_heads_per_core, value_dim)
        a = project_normed_input(
            x_scaled,
            x_rstd,
            w_a[slice_value_head, :],
            slice_fc_scale(w_a_scale, slice_value_head),
            w_a_scale2,
            GDN_NVFP4_PROJECTION_TILE_WIDTH,
            banks=banks,
        )
        b = project_normed_input(
            x_scaled,
            x_rstd,
            w_b[slice_value_head, :],
            slice_fc_scale(w_b_scale, slice_value_head),
            w_b_scale2,
            GDN_NVFP4_PROJECTION_TILE_WIDTH,
            banks=banks,
        )

    q = stateful_causal_depthwise_conv1d_silu(
        q_projected,
        w_q_conv[slice_key, :],
        q_conv_state_in[:, :, slice_key],
        q_conv_state_out[:, :, slice_key],
        conv_size,
        "gdn_decode_q",
        banks=banks,
    ).reshape(batch, 1, key_heads_per_core, key_dim)
    k = stateful_causal_depthwise_conv1d_silu(
        k_projected,
        w_k_conv[slice_key, :],
        k_conv_state_in[:, :, slice_key],
        k_conv_state_out[:, :, slice_key],
        conv_size,
        "gdn_decode_k",
        banks=banks,
    ).reshape(batch, 1, key_heads_per_core, key_dim)
    v = stateful_causal_depthwise_conv1d_silu(
        v_projected,
        w_v_conv[slice_value, :],
        v_conv_state_in[:, :, slice_value],
        v_conv_state_out[:, :, slice_value],
        conv_size,
        "gdn_decode_v",
        banks=banks,
    ).reshape(batch, 1, value_heads_per_core, value_dim)

    a_log_shared, dt_bias_shared = load_shared_gate_parameters(a_log, dt_bias, core_id, banks=banks)
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=key_dim**-0.5)
    # Batch all independently owned heads: Q/K [HKc, R, D] expands each
    # key head to its R value heads; recurrent state is [HVc, K, V] FP32.
    q_token = ax.dsl.mul(l2norm_lastdim(q), q_scale, output_dtype=x.dtype)
    k_token = l2norm_lastdim(k)
    q_token = ax.dsl.expand(
        q_token.reshape(key_heads_per_core, 1, key_dim),
        shape=(key_heads_per_core, value_heads_per_key, key_dim),
    ).reshape(value_heads_per_core, 1, key_dim)
    k_token = ax.dsl.expand(
        k_token.reshape(key_heads_per_core, 1, key_dim),
        shape=(key_heads_per_core, value_heads_per_key, key_dim),
    ).reshape(value_heads_per_core, 1, key_dim)
    g, beta = gdn_gate(
        a.reshape(value_heads_per_core, 1, 1),
        b.reshape(value_heads_per_core, 1, 1),
        a_log_shared[slice_value_head].reshape(value_heads_per_core, 1, 1),
        dt_bias_shared[slice_value_head].reshape(value_heads_per_core, 1, 1),
    )
    state = load_tile(state_in[:, slice_value_head, :, :], banks=banks).reshape(
        value_heads_per_core, key_dim, value_dim
    )
    core_output, state = gdn_recurrent_step(
        q_token,
        k_token,
        v.reshape(value_heads_per_core, 1, value_dim),
        g,
        beta,
        state,
        x.dtype,
    )
    core_output = core_output.reshape(batch, 1, value_heads_per_core, value_dim)
    store_tile(
        state.reshape(batch, value_heads_per_core, key_dim, value_dim),
        r=state_out[:, slice_value_head, :, :],
        banks=banks,
    )

    o_norm_scale = load_tile(o_norm_w, banks=banks).reshape(1, 1, 1, value_dim)
    # Gated RMSNorm keeps FP32 vectors until the completed gate product.
    normed = rmsnorm(ax.dsl.cast(core_output, dtype=ax.fp32), o_norm_scale, eps=o_norm_eps)
    normed = ax.dsl.mul(normed, ax.dsl.silu(z, output_dtype=ax.fp32), output_dtype=x.dtype)
    store_row_parallel_o_proj_residual_nmc(
        normed.reshape(batch, local_value_width),
        w_o,
        w_o_scale,
        w_o_scale2,
        pre_attn,
        output,
        residual_out,
        slice_value,
        core_id,
        core_num,
        name_prefix="gdn_decode",
        ic_tile=512 if is_nvfp4_weight(w_o) else None,
        banks=banks,
    )
    return (
        output,
        state_out,
        q_conv_state_out,
        k_conv_state_out,
        v_conv_state_out,
        residual_out,
    )


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=bool(int(os.environ.get("DUMP_DSL_PERF", 0))),
    dump_path=pathlib.Path(os.environ.get("DSL_DUMP_PATH") or pathlib.Path.cwd() / "output"),
)
def GDNKernel(
    x,
    residual_in,
    input_norm_w,
    w_q,
    w_k,
    w_v,
    w_z,
    w_a,
    w_b,
    w_pre,
    w_q_conv,
    w_k_conv,
    w_v_conv,
    a_log,
    dt_bias,
    o_norm_w,
    w_o,
    state_in,
    q_conv_state_in,
    k_conv_state_in,
    v_conv_state_in,
    output,
    state_out,
    q_conv_state_out,
    k_conv_state_out,
    v_conv_state_out,
    residual_out,
    w_q_scale=None,
    w_q_scale2=None,
    w_k_scale=None,
    w_k_scale2=None,
    w_v_scale=None,
    w_v_scale2=None,
    w_z_scale=None,
    w_z_scale2=None,
    w_a_scale=None,
    w_a_scale2=None,
    w_b_scale=None,
    w_b_scale2=None,
    w_o_scale=None,
    w_o_scale2=None,
    conv_size=4,
    chunk_size=32,
    input_norm_eps=1e-6,
    o_norm_eps=1e-6,
    projection_buffer=None,
    banks=None,
    valid_tokens=None,
):
    """Record GDN with one INT32 valid_tokens ABI for 256-row prefill.

    The caller supplies 1..256 valid rows. Only their outputs are defined;
    recurrent and convolution states stop at the real sequence end.
    """
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    if x.shape[1] == GDN_PREFILL_TILE_SIZE:
        assert valid_tokens is not None, "256-row prefill requires a runtime valid token count, including full blocks"
    if valid_tokens is not None:
        assert x.shape[1] == GDN_PREFILL_TILE_SIZE
    args = (
        x,
        residual_in,
        input_norm_w,
        w_q,
        w_k,
        w_v,
        w_z,
        w_a,
        w_b,
        w_pre,
        w_q_conv,
        w_k_conv,
        w_v_conv,
        a_log,
        dt_bias,
        o_norm_w,
        w_o,
        state_in,
        q_conv_state_in,
        k_conv_state_in,
        v_conv_state_in,
        output,
        state_out,
        q_conv_state_out,
        k_conv_state_out,
        v_conv_state_out,
        residual_out,
    )
    scales = {
        "w_q_scale": w_q_scale,
        "w_q_scale2": w_q_scale2,
        "w_k_scale": w_k_scale,
        "w_k_scale2": w_k_scale2,
        "w_v_scale": w_v_scale,
        "w_v_scale2": w_v_scale2,
        "w_z_scale": w_z_scale,
        "w_z_scale2": w_z_scale2,
        "w_a_scale": w_a_scale,
        "w_a_scale2": w_a_scale2,
        "w_b_scale": w_b_scale,
        "w_b_scale2": w_b_scale2,
        "w_o_scale": w_o_scale,
        "w_o_scale2": w_o_scale2,
    }
    if x.shape[1] == 1:
        gdn_decode_impl(
            *args,
            **scales,
            conv_size=conv_size,
            input_norm_eps=input_norm_eps,
            o_norm_eps=o_norm_eps,
            banks=banks,
        )
    elif x.shape[1] <= GDN_PREFILL_TILE_SIZE:
        gdn_prefill_impl(
            *args,
            **scales,
            conv_size=conv_size,
            chunk_size=chunk_size,
            input_norm_eps=input_norm_eps,
            o_norm_eps=o_norm_eps,
            projection_buffer=projection_buffer,
            banks=banks,
            valid_tokens=valid_tokens,
        )
    else:
        assert x.shape[1] % GDN_PREFILL_TILE_SIZE == 0
        core_id, core_num = ax.program_id(0), ax.num_programs(0)
        offsets = ax.dsl.from_numpy(np.arange(GDN_PREFILL_TILE_SIZE, dtype=np.int32))
        stride = ax.dsl.full((1,), ax.int32, fill_value=GDN_PREFILL_TILE_SIZE)
        one = ax.dsl.full((1,), ax.int32, fill_value=1)
        gdn_prefill_impl(
            *args,
            **scales,
            conv_size=conv_size,
            chunk_size=chunk_size,
            input_norm_eps=input_norm_eps,
            o_norm_eps=o_norm_eps,
            token_indices=offsets,
            core_id=core_id,
            core_num=core_num,
            banks=banks,
        )

        def tile_body(tile, loop_state, loop_q, loop_k, loop_v, loop_output):
            gdn_prefill_impl(
                x,
                residual_in,
                input_norm_w,
                w_q,
                w_k,
                w_v,
                w_z,
                w_a,
                w_b,
                w_pre,
                w_q_conv,
                w_k_conv,
                w_v_conv,
                a_log,
                dt_bias,
                o_norm_w,
                w_o,
                loop_state,
                loop_q,
                loop_k,
                loop_v,
                output,
                loop_state,
                loop_q,
                loop_k,
                loop_v,
                loop_output,
                **scales,
                conv_size=conv_size,
                chunk_size=chunk_size,
                input_norm_eps=input_norm_eps,
                o_norm_eps=o_norm_eps,
                token_indices=tile * stride + offsets,
                core_id=core_id,
                core_num=core_num,
                banks=banks,
            )
            return loop_state, loop_q, loop_k, loop_v, loop_output

        ax.dsl.range(
            ax.dsl.zeros((1,), ax.int32),
            one,
            ax.dsl.full((1,), ax.int32, fill_value=x.shape[1] // GDN_PREFILL_TILE_SIZE),
            one,
            tile_body,
            state_out,
            q_conv_state_out,
            k_conv_state_out,
            v_conv_state_out,
            residual_out,
            unroll_factor=1,
        )
    return (
        output,
        state_out,
        q_conv_state_out,
        k_conv_state_out,
        v_conv_state_out,
        residual_out,
    )


def as_runtime_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not torch.is_tensor(value):
        if not isinstance(value, np.ndarray):
            tensor = torch.as_tensor(value, dtype=dtype)
            return tensor.detach().contiguous()
        value = value.copy()
        try:
            value = ax_util.np2torch(value)
        except (KeyError, TypeError):
            dtype_name = str(value.dtype)
            if "int4" in dtype_name:
                # Torch has no signed-int4 dtype. Runtime MatMul accepts the
                # sign-extended learned values in signed-int8 storage.
                return torch.from_numpy(value.astype(np.int8)).clone()
            if "bfloat16" not in dtype_name and "float8_e4m3" not in dtype_name:
                raise
            target_dtype = torch.bfloat16 if "bfloat16" in dtype_name else torch.float8_e4m3fn
            value = torch.from_numpy(value.astype(np.float32)).to(target_dtype)
    tensor = value
    return tensor.detach().to(dtype=dtype).contiguous() if dtype is not None else tensor.detach().contiguous()


@ax.builder(run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")))
def reset_gdn_states(state, q_conv, k_conv, v_conv, zero_value):
    """Clear a single active request's complete recurrent history on its stream."""
    core_id, core_num = ax.program_id(0), ax.num_programs(0)
    # load: [1] FP32 DDR -> OCM. AXCL requires an input to expose a shape group.
    value = ax.dsl.load(zero_value)
    for tensor in (state, q_conv, k_conv, v_conv):
        flat = tensor.reshape(-1)
        shard = (flat.shape[0] + core_num - 1) // core_num
        start, stop = core_id * shard, min((core_id + 1) * shard, flat.shape[0])
        for offset in range(start, stop, 65536):
            # full -> [tile] FP32 for recurrent state, BF16 for convolution history.
            zero = ax.dsl.full((min(65536, stop - offset),), tensor.dtype, fill_value=0)
            # add: [tile] + [1] -> [tile], directly in the destination dtype.
            zero = ax.dsl.add(zero, value, output_dtype=tensor.dtype)
            # store: [tile] OCM -> this core's disjoint [tile] DDR state slice.
            ax.dsl.store(zero, r=flat[offset : offset + zero.shape[0]])
    return state, q_conv, k_conv, v_conv


def runtime_conv_weight(value: Any) -> torch.Tensor:
    """Convert a checkpoint Conv weight to runtime DSL ``[C, 1, K]`` layout."""
    weight = as_runtime_tensor(value)
    if weight.ndim != 2:
        raise ValueError(f"checkpoint Conv weight must be rank 2, got {tuple(weight.shape)}")
    return weight.unsqueeze(1).contiguous()


def runtime_pre_projection_weight(weights: Mapping[str, torch.Tensor | None], core_num: int) -> torch.Tensor | None:
    """Build a per-core-contiguous unscaled pre-attention projection weight."""
    # Per-channel FP8 and packed FP4 projections use their separate matrices.
    # In particular, FP8 Q/K/V/Z and BF16 A/B cannot be concatenated together.
    if any(
        weights[f"{name}_scale"] is not None or weights[f"{name}_scale2"] is not None
        for name in ("w_q", "w_k", "w_v", "w_z", "w_a", "w_b")
    ):
        return None
    q = weights["w_q"]
    k = weights["w_k"]
    v = weights["w_v"]
    z = weights["w_z"]
    a = weights["w_a"]
    b = weights["w_b"]
    local_key_width = q.shape[0] // core_num
    local_value_width = v.shape[0] // core_num
    local_value_heads = a.shape[0] // core_num
    parts = []
    for core_id in range(core_num):
        key_start = core_id * local_key_width
        value_start = core_id * local_value_width
        head_start = core_id * local_value_heads
        parts.extend(
            (
                q[key_start : key_start + local_key_width],
                k[key_start : key_start + local_key_width],
                v[value_start : value_start + local_value_width],
                z[value_start : value_start + local_value_width],
                a[head_start : head_start + local_value_heads],
                b[head_start : head_start + local_value_heads],
            )
        )
    return torch.cat(parts, dim=0).contiguous()


def normalized_fc_scale(value: Any, weight: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize an FC scale to the runtime per-output or grouped layout."""
    scale = as_runtime_tensor(value)
    # NVFP4 group scales remain E4M3 runtime operands. Legacy scalar and
    # per-channel callers continue to use FP32.
    if scale.dtype not in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        scale = scale.float()
    logical_input_channels = weight.shape[-1] * (2 if weight.dtype == GDN_NVFP4_DTYPE else 1)
    output_channels, input_channels = weight.shape[0], logical_input_channels
    if scale.numel() == 1:
        return scale.reshape(1, 1, 1).expand(1, output_channels, 1).contiguous()
    if scale.ndim == 1 and scale.shape[0] == output_channels:
        return scale.reshape(1, output_channels, 1).contiguous()
    if scale.ndim == 3 and scale.shape[0] == 1 and scale.shape[1] == output_channels:
        scale = scale.reshape(output_channels, scale.shape[2])

    valid_group_counts = {1, input_channels, input_channels // 16, input_channels // 32}
    # AxMatMul identifies NVFP4/MXFP8 by a 2-D [OC, IC/group] scale. The extra
    # batch dimension used by per-channel runtime scales would be interpreted
    # as OC and yield an invalid IC group size.
    if scale.ndim == 2 and scale.shape[1] in (
        input_channels // 16,
        input_channels // 32,
    ):
        return scale.contiguous()
    if scale.ndim != 2 or scale.shape[0] != output_channels or scale.shape[1] not in valid_group_counts:
        expected = f"(), ({output_channels},), (1, {output_channels}, 1), or grouped [OC, IC/group] shapes"
        raise ValueError(f"{name} shape {tuple(scale.shape)} is incompatible with {expected}")
    return scale.unsqueeze(0).contiguous()


def normalized_scale2(value: Any, name: str) -> torch.Tensor:
    """Normalize an optional secondary FC scale to one runtime FP32 value."""
    scale2 = as_runtime_tensor(value, dtype=torch.float32)
    if scale2.numel() != 1:
        raise ValueError(f"{name} must contain exactly one value, got {tuple(scale2.shape)}")
    return scale2.reshape(1).contiguous()


def runtime_weights(weights: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
    """Normalize host weights and optional scales into runtime launch tensors."""
    missing = _REQUIRED_WEIGHTS.difference(weights)
    if missing:
        raise ValueError(f"GDN weights are missing: {sorted(missing)}")
    values = {}
    for name in _REQUIRED_WEIGHTS:
        if name in {"w_q_conv", "w_k_conv", "w_v_conv"}:
            values[name] = runtime_conv_weight(weights[name])
        else:
            dtype = torch.float32 if name == "a_log" else None
            values[name] = as_runtime_tensor(weights[name], dtype=dtype)
    for name in _OPTIONAL_SCALES:
        value = weights.get(name)
        if value is None:
            values[name] = None
        elif name in _FC_SCALE_WEIGHTS:
            values[name] = normalized_fc_scale(value, values[_FC_SCALE_WEIGHTS[name]], name)
        else:
            values[name] = normalized_scale2(value, name)
    return values


def split_conv_state(conv_state: torch.Tensor, key_width: int, value_width: int) -> tuple[torch.Tensor, ...]:
    expected_width = 2 * key_width + value_width
    if conv_state.ndim != 3 or conv_state.shape[-1] != expected_width:
        raise ValueError(
            "GDN combined Conv state must have shape "
            f"[B, conv_size - 1, {expected_width}], got {tuple(conv_state.shape)}"
        )
    return torch.split(conv_state, (key_width, key_width, value_width), dim=-1)


def contiguous_conv_state_shards(
    conv_state: torch.Tensor, key_width: int, value_width: int
) -> tuple[torch.Tensor, ...]:
    """Split the packed Conv pool into launchable contiguous q/k/v shards.

    ``split_conv_state`` returns column views of the packed ``[B, K-1, Q+K+V]``
    tensor. The compiled kernel addresses launch tensors as contiguous memory,
    so passing those views through unchanged makes every shard read and write
    the q region of the pool: decode then continues from the wrong Conv history
    and the second generated token diverges.
    """
    return tuple(part.contiguous() for part in split_conv_state(conv_state, key_width, value_width))


def gdn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    weights: Mapping[str, Any],
    temporal_state: torch.Tensor,
    conv_state: torch.Tensor,
    *,
    core_num: int | None = None,
    chunk_size: int = 32,
    input_norm_eps: float = 1e-6,
    o_norm_eps: float = 1e-6,
    valid_tokens: torch.Tensor | None = None,
):
    """Run GDN with runtime weights and return residual and recurrent states.

    With valid_tokens (INT32 [1]), hidden_states retains 256 physical rows;
    only the specified 1..256 leading rows affect the returned states.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("GDN AXDSL supports one dense request per launch")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("GDN AXDSL requires BF16 hidden states")
    if core_num is None:
        core_num = int(os.getenv("SGLANG_AXERA_KERNEL_CORE_NUM", "4"))
    if core_num != 4:
        raise ValueError("GDN AXDSL requires four cores")
    if temporal_state.ndim != 4 or temporal_state.shape[0] != 1 or temporal_state.dtype != torch.float32:
        raise ValueError("GDN temporal state must be FP32 [1, value_heads, key_dim, value_dim]")
    if any(dim <= 0 or dim & (dim - 1) for dim in temporal_state.shape[-2:]):
        raise ValueError("GDN AXDSL requires power-of-two key and value head dimensions")
    if conv_state.dtype != torch.bfloat16:
        raise ValueError("GDN Conv state must be BF16")
    if residual is not None and (residual.shape != hidden_states.shape or residual.dtype != torch.bfloat16):
        raise ValueError("GDN residual must be BF16 and match hidden_states")
    if input_norm_eps <= 0 or o_norm_eps <= 0:
        raise ValueError("GDN RMSNorm epsilons must be positive")

    weight_args = runtime_weights(weights)
    if hidden_states.shape[1] == GDN_PREFILL_TILE_SIZE and valid_tokens is None:
        valid_tokens = torch.full((1,), GDN_PREFILL_TILE_SIZE, dtype=torch.int32, device=hidden_states.device)
    w_pre = runtime_pre_projection_weight(weight_args, core_num)
    temporal_state = as_runtime_tensor(temporal_state, dtype=torch.float32)
    q_width = int(weight_args["w_q"].shape[0])
    value_width = int(weight_args["w_v"].shape[0])
    q_conv_in, k_conv_in, v_conv_in = contiguous_conv_state_shards(conv_state, q_width, value_width)
    output = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(hidden_states)
    state_out = torch.empty_like(temporal_state)
    q_conv_out = torch.empty_like(q_conv_in)
    k_conv_out = torch.empty_like(k_conv_in)
    v_conv_out = torch.empty_like(v_conv_in)

    ax.init(chip_num=1)
    kernel_args = [
        hidden_states,
        residual,
        weight_args["input_norm_w"],
        weight_args["w_q"],
        weight_args["w_k"],
        weight_args["w_v"],
        weight_args["w_z"],
        weight_args["w_a"],
        weight_args["w_b"],
        w_pre,
        weight_args["w_q_conv"],
        weight_args["w_k_conv"],
        weight_args["w_v_conv"],
        weight_args["a_log"],
        weight_args["dt_bias"],
        weight_args["o_norm_w"],
        weight_args["w_o"],
        temporal_state,
        q_conv_in,
        k_conv_in,
        v_conv_in,
        output,
        state_out,
        q_conv_out,
        k_conv_out,
        v_conv_out,
        residual_out,
    ]
    kernel_args.extend(weight_args[name] for name in _OPTIONAL_SCALES)
    GDNKernel[core_num, core_num](
        *kernel_args,
        conv_size=conv_state.shape[1] + 1,
        chunk_size=chunk_size,
        input_norm_eps=input_norm_eps,
        o_norm_eps=o_norm_eps,
        valid_tokens=valid_tokens,
    )
    # The model layer consumes residual_out; output remains the separate
    # attention contribution for compatibility with the existing SGL contract.
    return (
        torch.zeros_like(output),
        residual_out,
        state_out,
        torch.cat((q_conv_out, k_conv_out, v_conv_out), dim=-1),
    )


__all__ = [
    "GDNKernel",
    "as_runtime_tensor",
    "gdn",
    "reset_gdn_states",
    "runtime_conv_weight",
    "runtime_weights",
    "split_conv_state",
]
