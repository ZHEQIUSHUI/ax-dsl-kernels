"""Paged attention compute kernels with runtime-tensor inputs."""

from __future__ import annotations


import os

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.opset import datatype
from sgl_kernel_axera.models.bank_access import (
    gather_rows,
    load_tile,
    load_shared_scale_rows,
    scatter_rows,
    store_tile,
)
from sgl_kernel_axera.attention.attn_utils import (
    gather_mrope_cos_sin,
    rmsnorm,
    rmsnorm_sqrt,
    rope_impl,
)

FLASH_ATTN_BR = 128
BASELINE_PAGE_SIZE = 256
# Compact SGLang decode often has one history page. SDMA gather on a size-1 or
# size-2 page axis / page-id table lowers to index ``2**31``, which is
# ``tensor_base + 2**31 * row_bytes`` and misses the mapped DDR window. The
# axnn decode tests keep ``MAX_NUM_PAGES >= 32``; match that floor so both the
# 5D page axis and the 1-D page-id table stay out of that lowering.
_MIN_PAGED_GATHER_PAGES = 32
# Qwen3-0.6B is GQA=2, so each KV head attends two query heads. Concat-padding
# that size-2 view is still a size-2 SDMA gather source (8B is GQA=4 and already
# clears it). Select dummy rows from the full local Q instead, then drop them.
_MIN_DECODE_QUERY_HEADS = 4
_KV_PAGE_SIZE = BASELINE_PAGE_SIZE
_SUPPORTED_QUERY_LENGTHS = (128, 256)
_QKV_WEIGHT_OC_TILE = 128
_QKV_WEIGHT_IC_LOAD_TILE = 1024
_KV_PROJECTION_TOKEN_TILE = 128
_PAGE_UNROLL_FACTOR = int(os.environ.get("PAGED_ATTN_PAGE_UNROLL", "1"))
_PACKED_FP4_WEIGHT_TYPENAMES = frozenset(("MXFP4_X2", "NVFP4_X2"))
_ATTENTION_OC_TILE = 64
_PROJECTION_WEIGHT_OC_LOAD_TILE = 512
_OUTPUT_PROJECTION_WEIGHT_IC_LOAD_TILE = 512
_OUTPUT_PROJECTION_WEIGHT_OC_LOAD_TILE = 512


def store_paged_kv(values, cache, location, plane, head_start, banks=None):
    """Scatter this core's token/head rows directly into the retained KV pool."""
    # T=tokens, Hc=local_heads, D=head_dim; cache[P,2,L,H,D], location[2] int32.
    # values[T,Hc,D] and cache use A=values.dtype (BF16 for Qwen3); S=P*2*L*H.
    tokens, local_heads, head_dim = values.shape
    heads = cache.shape[3]
    page_size = cache.shape[2]
    # load: location[2] int32 DDR -> target[2] int32 OCM (page ID, token offset).
    target = load_tile(location, banks=banks) if location.memory_type == ax.MemoryType.DDR else location
    # from_numpy: np[1] int32 -> page_rows[1] int32.
    page_rows = ax.dsl.from_numpy(np.array([2 * page_size], dtype=np.int32))
    # from_numpy: np[1] int32 -> plane_rows[1] int32 (K or V plane offset).
    plane_rows = ax.dsl.from_numpy(np.array([plane * page_size], dtype=np.int32))
    # from_numpy: np[1] int32 -> head_count[1] int32.
    head_count = ax.dsl.from_numpy(np.array([heads], dtype=np.int32))
    # mul/add: target slices[1], page_rows[1], plane_rows[1], head_count[1] int32 -> first_row[1] int32.
    first_row = (target[:1] * page_rows + plane_rows + target[1:2]) * head_count
    local_rows = (
        np.arange(tokens, dtype=np.int32)[:, None] * heads
        + np.arange(head_start, head_start + local_heads, dtype=np.int32)[None, :]
    )
    # from_numpy: local_rows.reshape(-1)[T*Hc] int32 -> [T*Hc] int32 OCM.
    # add: first_row[1], local rows[T*Hc] int32 -> row_indices[T*Hc] uint32.
    row_indices = ax.dsl.add(first_row, ax.dsl.from_numpy(local_rows.reshape(-1)), output_dtype=ax.uint32)
    # select_scatter: values[T*Hc,D] A, indices[T*Hc] uint32 -> cache[S,D] A DDR, in place.
    scatter_rows(
        values.reshape(tokens * local_heads, head_dim),
        row_indices,
        dim=0,
        r=cache.reshape(-1, head_dim),
        banks=banks,
    )


def _history_tail_mask(page_index, slot_positions, page_size_scalar, valid_kv_len, page_size, dtype):
    """Additive ``[1, page_size]`` mask that hides slots past the valid history.

    A partially filled last page carries uninitialized cache rows after
    ``valid_kv_len``; masking their scores with a large negative bias keeps
    them out of the online softmax. Returns ``None`` when no masking is needed.
    The DSL has no comparison op, so ``keep`` is built with clamped integer
    arithmetic: ``keep = 1`` for valid slots and ``0`` otherwise.
    """
    if valid_kv_len is None:
        return None
    global_slot_pos = ax.dsl.add(ax.dsl.mul(page_index, page_size_scalar), slot_positions)
    diff = ax.dsl.sub(valid_kv_len, global_slot_pos, output_dtype=dtype)
    one = ax.dsl.from_numpy(np.array([1], dtype=dtype.nptype))
    zero = ax.dsl.from_numpy(np.array([0], dtype=dtype.nptype))
    keep = ax.dsl.minimum(ax.dsl.maximum(diff, zero), one)
    neg_bias = ax.dsl.from_numpy(np.array([3e38], dtype=dtype.nptype))
    return ax.dsl.mul(ax.dsl.sub(keep, one), neg_bias).reshape(1, page_size)


def paged_attn_update(
    q,
    k_tiles,
    v_tiles,
    max_state,
    normalizer,
    accumulator,
    state_scores,
    active_mask=None,
    tail_mask=None,
):
    """Merge one history page or causal current block into online-softmax state.

    ``q`` is ``[query_rows, head_dim]``. K is tiled by rows and V by output
    channels. A missing ``active_mask`` denotes a complete history page and
    writes the updated maximum. A supplied additive ``[query_rows, kv_len]``
    mask denotes the final causal block, where the maximum is no longer
    consumed. Active positions are zero and inactive positions are negative.
    ``tail_mask`` is an additive ``[1, kv_len]`` mask that hides the invalid
    tail of a partially filled history page while keeping the history-page
    maximum-writing behaviour.
    """
    query_rows, head_dim = q.shape
    kv_len = sum(k_tile.shape[0] for k_tile in k_tiles)
    ax.dsl.move(max_state, r=state_scores[:, :1])
    if active_mask is None:
        state_scores = state_scores[:, : 1 + kv_len + 1]
        ax.dsl.move(ax.dsl.zeros((query_rows, 1), dtype=q.dtype), r=state_scores[:, -1:])
    else:
        state_scores = state_scores[:, : 1 + kv_len]

    # Keep Q@K and the causal/tail masks on this 2-D slice of ``state_scores``.
    # Reshaping the same storage to 3-D before MatMul/Add can drop the alias,
    # so online-softmax then reduces uninitialized scores to Inf.
    score_rows = state_scores[:, 1 : 1 + kv_len]
    score_start = 0
    for k_tile in k_tiles:
        score_stop = score_start + k_tile.shape[0]
        ax.dsl.matmul(q, k_tile, transposed=True, r=score_rows[:, score_start:score_stop])
        score_start = score_stop

    if active_mask is not None:
        ax.dsl.add(score_rows, active_mask, r=score_rows)
    if tail_mask is not None:
        ax.dsl.add(score_rows, tail_mask, r=score_rows)

    updated_max = ax.dsl.max(state_scores[:, : 1 + kv_len], dim=-1, keepdim=True)
    ax.dsl.sub(state_scores, updated_max, r=state_scores)
    shifted_scores = state_scores[:, 1 : 1 + kv_len].reshape(1, query_rows, kv_len)

    # Keep BF16 probabilities for P @ V and FP32 probabilities for the softmax sum.
    probabilities = ax.dsl.exp(shifted_scores)
    probabilities_fp32 = ax.dsl.exp(shifted_scores, output_dtype=ax.fp32)
    block_sum = ax.dsl.sum(probabilities_fp32, dim=-1, output_dtype=ax.fp32)
    previous_state_ratio = ax.dsl.exp(state_scores[:, :1], output_dtype=ax.fp32).reshape(1, query_rows)
    scaled_previous_sum = ax.dsl.mul(previous_state_ratio, normalizer, output_dtype=normalizer.dtype)
    ax.dsl.add(scaled_previous_sum, block_sum, r=normalizer, output_dtype=normalizer.dtype)

    weighted_values = ax.dsl.create_tensor((1, query_rows, head_dim), dtype=accumulator.dtype)
    value_start = 0
    for v_tile in v_tiles:
        value_stop = value_start + v_tile.shape[1]
        ax.dsl.matmul(
            probabilities.reshape(query_rows, kv_len),
            v_tile,
            r=weighted_values[:, :, value_start:value_stop].reshape(query_rows, v_tile.shape[1]),
        )
        value_start = value_stop

    scaled_previous_output = ax.dsl.mul(
        previous_state_ratio.reshape(1, query_rows, 1),
        accumulator,
        output_dtype=ax.fp32,
    )
    ax.dsl.add(
        scaled_previous_output,
        weighted_values,
        r=accumulator,
        output_dtype=accumulator.dtype,
    )

    if active_mask is None:
        minus_one = ax.dsl.from_numpy(np.full((1,), -1, dtype=max_state.dtype.nptype))
        ax.dsl.mul(state_scores[:, -1:], minus_one, r=max_state)


def gather_rope_cos_sin(
    cos,
    sin,
    position_ids,
    query_len,
    mrope_section,
    mrope_interleaved,
    has_mrope,
    cos_sin_cache=None,
    banks=None,
):
    """Gather rotary rows, optionally from the framework's merged cache."""
    valid_tokens = position_ids.shape[-1]
    if valid_tokens < query_len:
        # Padded queries follow all valid queries, so their positions cannot
        # affect valid causal outputs. Position zero keeps RoPE gathers in range.
        padding = ax.dsl.zeros((*position_ids.shape[:-1], query_len - valid_tokens), dtype=ax.int32)
        position_ids = ax.dsl.concat(position_ids, padding, dim=len(position_ids.shape) - 1)
    if has_mrope:
        return gather_mrope_cos_sin(
            cos,
            sin,
            position_ids,
            mrope_section,
            interleaved=mrope_interleaved,
            cos_sin_cache=cos_sin_cache,
            banks=banks,
        )
    if cos_sin_cache is not None:
        rows = gather_rows(cos_sin_cache, position_ids, dim=0, banks=banks)
        half = cos_sin_cache.shape[-1] // 2
        return rows[:, :half], rows[:, half:]
    return (
        gather_rows(cos, position_ids, dim=0, banks=banks),
        gather_rows(sin, position_ids, dim=0, banks=banks),
    )


def project_component(
    x,
    input_rms,
    weight,
    weight_scale,
    weight_scale2,
    global_oc_start,
    output_channel_count,
    token_tile_size,
    name,
    banks=None,
    r=None,
):
    """Project one core's complete local Q, Q/Gate, K, or V shard.

    Q consumes each output-channel tile immediately after loading it. K and V
    keep their two local output-channel tiles resident and consume them in
    token-major order. The latter order lets both CONV jobs for one token tile
    become ready together while reusing the resident weights. Once all MatMuls
    for this component are recorded, apply the folded input RMS denominator in
    one pass. The caller owns the model-specific head layout and postprocessing.
    """
    token_count = x.shape[0] * x.shape[1]
    input_dim = x.shape[-1]
    ic_slice_factor = 2 if datatype.DType(weight.dtype).typename in _PACKED_FP4_WEIGHT_TYPENAMES else 1
    assert input_dim % ic_slice_factor == 0
    weight_oc_tile = min(
        output_channel_count,
        256
        if datatype.DType(weight.dtype).typename in ("NVFP4", "MXFP4", "NVFP4_X2", "MXFP4_X2")
        else _QKV_WEIGHT_OC_TILE,
    )
    local_oc_count = output_channel_count
    assert local_oc_count % weight_oc_tile == 0
    assert token_count % token_tile_size == 0

    projected = ax.dsl.create_tensor(
        (x.shape[0], x.shape[1], local_oc_count),
        dtype=ax.bf16,
        name=f"{name}_projected",
    )
    projected_2d = projected.reshape(token_count, local_oc_count)
    x_2d = x.reshape(token_count, input_dim)

    weight_scale2_local = load_tile(weight_scale2, banks=banks)
    local_oc_starts = tuple(range(0, local_oc_count, weight_oc_tile))

    # Keep every local scale tile live before MatMulV2 starts. Loading later
    # tiles inside the weight loop lets the allocator alias them with earlier
    # MatMulV2 scratch, which serializes SDMA behind the preceding CONV wave.
    if weight_scale.shape_size > 1:
        weight_scale_tiles = []
        for local_oc_start in local_oc_starts:
            global_tile_start = global_oc_start + local_oc_start
            global_oc_slice = slice(global_tile_start, global_tile_start + weight_oc_tile)
            weight_scale_tiles.append(load_tile(weight_scale[global_oc_slice, :], banks=banks))
    else:
        weight_scale_local = load_tile(weight_scale, banks=banks)
        weight_scale_tiles = [weight_scale_local] * len(local_oc_starts)

    reduction_tile = min(input_dim, 1024) if weight.dtype.typename in ("NVFP4", "NVFP4_X2") else input_dim

    def load_weight_tiles(local_oc_start):
        global_tile_start = global_oc_start + local_oc_start
        global_oc_slice = slice(global_tile_start, global_tile_start + weight_oc_tile)
        tiles = []
        for ic_start in range(0, input_dim, reduction_tile):
            ic_stop = min(ic_start + reduction_tile, input_dim)
            # Runtime NVFP4 MatMul consumes a dense reduction tile. A column
            # view into the full-K weight has a larger row stride and corrupts
            # some channels during native FC accumulation. Load the exact DDR
            # tile directly, retaining it for all token tiles without a copy.
            weight_tile = ax.dsl.create_tensor(
                (weight_oc_tile, (ic_stop - ic_start) // ic_slice_factor),
                dtype=weight.dtype,
                name=f"{name}_weight_oc{local_oc_start}_ic{ic_start}",
            )
            for load_start in range(ic_start, ic_stop, _QKV_WEIGHT_IC_LOAD_TILE):
                load_stop = min(load_start + _QKV_WEIGHT_IC_LOAD_TILE, ic_stop)
                source = slice(load_start // ic_slice_factor, load_stop // ic_slice_factor)
                destination = slice(
                    (load_start - ic_start) // ic_slice_factor,
                    (load_stop - ic_start) // ic_slice_factor,
                )
                load_tile(
                    weight[global_oc_slice, source],
                    r=weight_tile[:, destination],
                    banks=banks,
                )
            tiles.append((slice(ic_start, ic_stop), weight_tile))
        return tiles

    def project_tile(activation, weight_tiles, scale_tile, result):
        # [M,Kt] BF16 @ [OC,Kt] NVFP4 -> [M,OC] BF16; OC256/K1024
        # occupies 128KiB of weights and 16KiB of group-16 E4M3 scales.
        scale_group = input_dim // scale_tile.shape[-1] if scale_tile.shape_size > 1 else input_dim
        for channels, weight_tile in weight_tiles:
            scale = (
                scale_tile[
                    :,
                    channels.start // scale_group : (channels.stop + scale_group - 1) // scale_group,
                ]
                if scale_tile.shape_size > 1
                else scale_tile
            )
            ax.dsl.matmul(
                activation[:, channels],
                weight_tile,
                w_scale=scale,
                w_scale2=weight_scale2_local,
                transposed=True,
                output_dtype=ax.bf16,
                output_add_assign=channels.start != 0,
                r=result,
            )

    if token_tile_size == token_count:
        # Q keeps the full token dimension M and splits only output channels N:
        # load and consume all dense IC tiles before proceeding to the next
        # OC tile. FP4 uses OC256; FP8/BF16 use OC128.
        for tile_index, local_oc_start in enumerate(local_oc_starts):
            local_oc_slice = slice(local_oc_start, local_oc_start + weight_oc_tile)
            weight_tile = load_weight_tiles(local_oc_start)
            project_tile(
                x_2d,
                weight_tile,
                weight_scale_tiles[tile_index],
                projected_2d[:, local_oc_slice],
            )

    else:
        # K/V split both M and N:
        #   [M=token_tile_size, K=input_dim] @ [N=OC128, K=input_dim]^T.
        # Keep the local KV weights resident across the M128 token tiles.
        # Record token-major so independent M/OC shards can use the CONV EUs.
        resident_tiles = [
            (
                load_weight_tiles(local_oc_start),
                weight_scale_tiles[tile_index],
                local_oc_start,
            )
            for tile_index, local_oc_start in enumerate(local_oc_starts)
        ]
        for token_start in range(0, token_count, token_tile_size):
            token_slice = slice(token_start, token_start + token_tile_size)
            for weight_tile, weight_scale_tile, local_oc_start in resident_tiles:
                local_oc_slice = slice(local_oc_start, local_oc_start + weight_oc_tile)
                project_tile(
                    x_2d[token_slice, :],
                    weight_tile,
                    weight_scale_tile,
                    projected_2d[token_slice, local_oc_slice],
                )

    destination = projected if r is None else r.reshape(*projected.shape)
    return ax.dsl.div(projected, input_rms, output_dtype=ax.bf16, r=destination)


def prepare_shared_projection_input(
    x,
    input_norm_weight,
    query_row_slice,
    residual,
    post_residual_addition,
    query_indices=None,
    output=None,
    banks=None,
):
    """Load residual rows and prepare the folded-RMSNorm projection input."""
    query_len = x.shape[1] if query_indices is None else query_indices.shape_size
    tile_shape = (1, query_len, x.shape[-1])
    memory_type = ax.MemoryType.SHARED_OCM
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
    scaled_x = ax.dsl.create_tensor(
        tile_shape,
        x.dtype,
        memory_type,
        name="paged_prefill_x_scaled",
    )
    input_rms = ax.dsl.create_tensor(
        (1, query_len, 1),
        ax.fp32,
        memory_type,
        name="paged_prefill_x_rms",
    )
    ax.dsl.mul(local_x_rows, input_norm_weight, r=scaled_x[query_row_slice])
    rmsnorm_sqrt(x=local_x_rows, r=input_rms[query_row_slice])
    for tensor in (shared_output, scaled_x, input_rms):
        axd.pub(tensor)
        axd.sub(tensor)
    return shared_output, scaled_x, input_rms


def paged_flash_attention(
    q,
    k,
    v,
    paged_kv_cache,
    indices,
    page_start,
    history_page_count,
    kv_head_start,
    banks=None,
):
    """Update all local Q-head states over history and current KV.

    Complete history pages need no causal mask: every history token precedes
    every query token. The current query block has a causal bias. Independent
    per-head state tensors expose the parallel head chains to the scheduler.
    """
    query_len, q_head_count, head_dim = q.shape
    kv_head_count = k.shape[1]
    assert v.shape == k.shape
    assert q_head_count % kv_head_count == 0
    kv_repeat = q_head_count // kv_head_count

    active_mask_np = np.zeros((query_len, query_len), dtype=q.dtype.nptype)
    active_mask_np[np.triu_indices(query_len, k=1)] = -3e38
    active_mask = ax.dsl.from_numpy(active_mask_np)

    # Per head: max [T,1] BF16, normalizer [1,T] FP32, output [1,T,D] BF16.
    states = []
    for local_q_head in range(q_head_count):
        states.extend(
            (
                ax.dsl.full((query_len, 1), dtype=q.dtype, fill_value=-3e38),
                ax.dsl.zeros((1, query_len), dtype=ax.fp32),
                ax.dsl.zeros((1, query_len, head_dim), dtype=q.dtype),
            )
        )

    # Scratch is captured by the range body instead of returned as loop state:
    # every update overwrites its active view, while a loop-carried in-place
    # scratch tensor would require a full range writeback.
    packed_attention_workspaces = []
    for local_q_head in range(q_head_count):
        packed_attention_workspaces.append(
            ax.dsl.create_tensor(
                (query_len, 1 + _KV_PAGE_SIZE + 1),
                dtype=q.dtype,
                name=f"paged_attn_state_scores_h{local_q_head}",
            )
        )

    page_index = ax.dsl.zeros((1,), dtype=ax.int32)
    range_start = ax.dsl.zeros((1,), dtype=ax.int32)
    range_step = ax.dsl.from_numpy(np.ones((1,), dtype=np.int32))

    def page_loop_body(
        page_index,
        paged_kv_cache,
        paged_kv_indices,
        page_start,
        q,
        *head_states,
    ):
        physical_page = gather_rows(paged_kv_indices, page_start + page_index, dim=0, banks=banks)

        # index_select records AxGather. Independent single-head gathers can
        # occupy SDMA0 and SDMA1 together, while each source view still carries
        # the full token-major cache stride.
        kv_page_heads = []
        for local_kv_head in range(kv_head_count):
            global_kv_head = kv_head_start + local_kv_head
            kv_page_heads.append(
                gather_rows(
                    paged_kv_cache[:, :, :, global_kv_head : global_kv_head + 1, :],
                    physical_page,
                    dim=0,
                    banks=banks,
                ).reshape(2, _KV_PAGE_SIZE, 1, head_dim)
            )

        for local_q_head in range(q_head_count):
            local_kv_head = local_q_head // kv_repeat
            q_head = q[:, local_q_head : local_q_head + 1, :].reshape(query_len, head_dim)
            kv_page = kv_page_heads[local_kv_head]
            k_page = kv_page[0:1, :, :, :].reshape(_KV_PAGE_SIZE, head_dim)
            v_page = kv_page[1:2, :, :, :].reshape(_KV_PAGE_SIZE, head_dim)
            max_state, normalizer_state, accumulator_state = head_states[local_q_head * 3 : local_q_head * 3 + 3]
            paged_attn_update(
                q_head,
                (k_page,),
                (v_page,),
                max_state,
                normalizer_state,
                accumulator_state,
                packed_attention_workspaces[local_q_head],
            )

        return (
            paged_kv_cache,
            paged_kv_indices,
            page_start,
            q,
            *head_states,
        )

    # Zero history keeps the initialized softmax state through a zero-trip Range.
    loop_results = ax.dsl.range(
        page_index,
        range_start,
        history_page_count,
        range_step,
        page_loop_body,
        paged_kv_cache,
        indices,
        page_start,
        q,
        *states,
        unroll_factor=_PAGE_UNROLL_FACTOR,
    )
    final_states = loop_results[-3 * q_head_count :]
    # Final divisions write directly to [T,Hlocal*D] BF16 channel views;
    # the output projection needs no head concat or layout-copy pass.
    attention = ax.dsl.create_tensor((query_len, q_head_count * head_dim), dtype=q.dtype, name="prefill_attention")
    for local_q_head in range(q_head_count):
        local_kv_head = local_q_head // kv_repeat
        max_state, normalizer, accumulator = final_states[local_q_head * 3 : local_q_head * 3 + 3]
        paged_attn_update(
            q[:, local_q_head : local_q_head + 1, :].reshape(query_len, head_dim),
            (k[:, local_kv_head : local_kv_head + 1, :].reshape(query_len, head_dim),),
            (v[:, local_kv_head : local_kv_head + 1, :].reshape(query_len, head_dim),),
            max_state,
            normalizer,
            accumulator,
            packed_attention_workspaces[local_q_head][:, : 1 + query_len],
            active_mask,
        )
        target = attention[:, local_q_head * head_dim : (local_q_head + 1) * head_dim].reshape(1, query_len, head_dim)
        ax.dsl.div(accumulator, normalizer.reshape(1, query_len, 1), r=target)

    return attention


def project_attention_output(
    attention,
    weight,
    weight_scale,
    weight_scale2,
    shared_output,
    output_dtype,
    core_id,
    core_count,
    global_ic_start,
    hidden_dim_per_core,
    banks=None,
):
    """Project a contiguous local head shard and add it to output banks."""
    query_len, local_input_dim = attention.shape
    ic_slice_factor = 2 if datatype.DType(weight.dtype).typename in _PACKED_FP4_WEIGHT_TYPENAMES else 1
    weight_input_dim = local_input_dim // ic_slice_factor
    # Bound each resident weight tile to about 1MiB on every supported grid.
    capacity_oc = (8 * 1024 * 1024) // (weight_input_dim * weight.dtype.num_bits)
    oc_tile = min(hidden_dim_per_core, max(128, min(1024, capacity_oc // 128 * 128)))
    weight_ic = slice(
        global_ic_start // ic_slice_factor,
        (global_ic_start + local_input_dim) // ic_slice_factor,
    )
    weight_scale2_local = load_tile(weight_scale2, banks=banks)
    for reduction_round in range(core_count):
        target_shard = (core_id + reduction_round) % core_count
        shard_start = target_shard * hidden_dim_per_core
        for local_oc in range(0, hidden_dim_per_core, oc_tile):
            oc_count = min(oc_tile, hidden_dim_per_core - local_oc)
            target_slice = slice(shard_start + local_oc, shard_start + local_oc + oc_count)
            local_weight = load_tile(weight[target_slice, weight_ic], banks=banks)
            if weight_scale.shape_size == 1:
                local_scale = load_tile(weight_scale, banks=banks)
            elif weight_scale.shape[-1] == 1:
                local_scale = load_tile(weight_scale[target_slice, :], banks=banks)
            else:
                scale_group = weight.shape[-1] * ic_slice_factor // weight_scale.shape[-1]
                group_slice = slice(
                    global_ic_start // scale_group,
                    (global_ic_start + local_input_dim) // scale_group,
                )
                local_scale = load_tile(weight_scale[target_slice, group_slice], banks=banks)
            projected = ax.dsl.matmul(
                attention,
                local_weight,
                w_scale=local_scale,
                w_scale2=weight_scale2_local,
                transposed=True,
                output_dtype=output_dtype,
            )
            contribution = projected.reshape(1, query_len, oc_count)
            residual = shared_output[:, :, target_slice]
            ax.dsl.move(contribution, r=residual, op="add")
        axd.pub(shared_output)
        axd.sub(shared_output)


def prefill_attention_tile(
    x,  # [1, query_len, hidden_dim]
    x_norm_w,  # [hidden_dim]
    w_q,  # ungated [Hq * D, M]; gated packed [Hq * 2D, M]
    w_scale_q,
    w_scale2_q,
    w_zeropoint_q,
    w_k,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_k,
    w_scale2_k,
    w_zeropoint_k,
    w_v,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_v,
    w_scale2_v,
    w_zeropoint_v,
    q_norm_w,  # [head_dim]
    k_norm_w,  # [head_dim]
    cos,  # [max_position, rotary_dim // 2]
    sin,  # [max_position, rotary_dim // 2]
    position_ids,  # [valid_query_tokens] or [3, valid_query_tokens] int32
    w_o,  # [hidden_dim, num_heads_q * head_dim]
    w_scale_o,
    w_scale2_o,
    w_zeropoint_o,
    kv_page_out,  # BF16: [2, 256, num_kv_heads, head_dim]
    output,  # [1, query_len, hidden_dim]
    paged_kv_cache,  # BF16: [max_pages, 2, 256, num_kv_heads, head_dim]
    indptr,  # [2] int32
    indices,  # [history_pages] int32
    kv_write_page_cursor,  # [1] uint32; current tile start within the output page
    attn_output_gate=False,
    partial_rotary_factor=1.0,
    mrope_section=(16, 24, 24),
    mrope_interleaved=False,
    has_mrope=False,
    residual=None,
    post_residual_addition=None,
    cache_location=None,
    *,
    cos_sin_cache=None,
    core_id,
    core_count,
    query_indices=None,
    banks=None,
):
    """Record one bounded page of head-sharded attention.

    The graph has four stages: row-parallel input preparation, head-sharded Q
    projection, paged attention, and IC-sharded output projection with residual
    accumulation. K/V projections are head-sharded when there are at least as
    many KV heads as cores; otherwise adjacent cores share one projection.
    This fragment processes query lengths 128 and 256; single-token decode remains owned
    by the decode kernel. Optional gated Q rows use per-head [Q, Gate] packing,
    and partial RoPE rotates only the prefix selected by
    ``partial_rotary_factor``.
    """
    batch_size, total_query_len, hidden_dim = x.shape
    query_len = total_query_len if query_indices is None else query_indices.shape_size
    head_dim = q_norm_w.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)
    q_gate_row_factor = 2 if attn_output_gate else 1
    assert w_q.shape[0] % (q_gate_row_factor * head_dim) == 0
    q_head_count = w_q.shape[0] // (q_gate_row_factor * head_dim)
    kv_head_count = w_k.shape[0] // head_dim

    assert _PAGE_UNROLL_FACTOR > 0
    assert batch_size == 1
    assert indptr.shape == (2,)
    assert query_len in _SUPPORTED_QUERY_LENGTHS
    assert paged_kv_cache.shape[1] == 2
    assert paged_kv_cache.shape[2] == _KV_PAGE_SIZE
    assert paged_kv_cache.shape[3] == kv_head_count
    assert paged_kv_cache.dtype == ax.bf16
    assert w_k.shape == w_v.shape
    assert q_head_count % kv_head_count == 0
    assert q_head_count % core_count == 0
    assert hidden_dim % core_count == 0
    assert query_len % core_count == 0
    assert output.shape == x.shape
    q_heads_per_core = q_head_count // core_count
    shared_kv_projection = kv_head_count < core_count
    if shared_kv_projection:
        kv_heads_per_core = 1
        kv_head_start = core_id // 2
        pair_rank = core_id % 2
    else:
        assert kv_head_count % core_count == 0
        kv_heads_per_core = kv_head_count // core_count
        kv_head_start = core_id * kv_heads_per_core
        pair_rank = 0
    assert q_heads_per_core % kv_heads_per_core == 0
    assert paged_kv_cache.shape[4:] == (head_dim,)
    if cache_location is None:
        assert kv_page_out.shape == (2, _KV_PAGE_SIZE, kv_head_count, head_dim)
        assert kv_page_out.dtype == ax.bf16
    assert 0 < partial_rotary_factor <= 1.0
    assert rotary_dim % 2 == 0
    if cos_sin_cache is None:
        assert cos.shape[-1] == sin.shape[-1] == rotary_dim // 2
    else:
        assert cos_sin_cache.shape[-1] == rotary_dim

    query_rows_per_core = query_len // core_count
    query_row_start = core_id * query_rows_per_core
    query_row_slice = (
        slice(None),
        slice(query_row_start, query_row_start + query_rows_per_core),
        slice(None),
    )
    hidden_dim_per_core = hidden_dim // core_count

    # Stage 1: row-parallel input preparation. The shared tensors become
    # complete after the helper's barrier and serve both Q/K/V and residual.
    shared_output, shared_x_scaled, shared_x_rms = prepare_shared_projection_input(
        x,
        x_norm_w,
        query_row_slice,
        residual,
        post_residual_addition,
        query_indices,
        output,
        banks=banks,
    )

    q_head_start = core_id * q_heads_per_core
    q_head_stop = q_head_start + q_heads_per_core
    q_gate_dim_slice = slice(
        q_head_start * q_gate_row_factor * head_dim,
        q_head_stop * q_gate_row_factor * head_dim,
    )
    kv_head_stop = kv_head_start + kv_heads_per_core
    kv_dim_slice = slice(kv_head_start * head_dim, kv_head_stop * head_dim)

    q_norm_w = load_tile(q_norm_w, banks=banks)
    k_norm_w = load_tile(k_norm_w, banks=banks)
    indptr = load_tile(indptr, banks=banks) if indptr.memory_type == ax.MemoryType.DDR else indptr
    indices = load_tile(indices, banks=banks)
    if cache_location is None:
        kv_write_page_cursor_u32 = load_tile(kv_write_page_cursor, banks=banks)
    position_ids = (
        load_tile(position_ids, banks=banks)
        if query_indices is None
        else gather_rows(position_ids, query_indices, dim=1 if has_mrope else 0, banks=banks)
    )

    page_start_value = indptr[:1]
    history_page_count = indptr[1:2] - page_start_value

    # A slice view does not retain a stable base address across AX8860 Range
    # runtime re-entry. Materialize only the page-table start; computed scalar
    # results can be consumed directly and do not need loop-carried aliases.
    page_start = ax.dsl.create_tensor(page_start_value.shape, dtype=page_start_value.dtype)
    ax.dsl.move(x=page_start_value, r=page_start)

    rope_cos, rope_sin = gather_rope_cos_sin(
        cos,
        sin,
        position_ids,
        query_len,
        mrope_section,
        mrope_interleaved,
        has_mrope,
        cos_sin_cache,
        banks=banks,
    )

    # Stage 2: OC-parallel Q/K/V projection. Gated checkpoints keep each
    # head's Q and Gate rows adjacent, so one packed projection preserves the
    # checkpoint layout without host-side repacking.
    projected_q_gate = project_component(
        shared_x_scaled,
        shared_x_rms,
        w_q,
        w_scale_q,
        w_scale2_q,
        q_gate_dim_slice.start,
        q_gate_dim_slice.stop - q_gate_dim_slice.start,
        query_len,
        "q_gate" if attn_output_gate else "q",
        banks=banks,
    )
    if attn_output_gate:
        projected_q_gate = projected_q_gate.reshape(query_len, q_heads_per_core, 2 * head_dim)
        projected_q = projected_q_gate[..., :head_dim]
        gate = projected_q_gate[..., head_dim:]
    else:
        projected_q = projected_q_gate.reshape(query_len, q_heads_per_core, head_dim)
        gate = None

    if shared_kv_projection:
        # The even core in each pair computes one K/V projection and publishes it to its partner.
        shared_projected_k = [
            ax.dsl.create_tensor(
                (query_len, 1, head_dim),
                x.dtype,
                ax.MemoryType.SHARED_OCM,
                name=f"shared_projected_k_h{kv_head}",
            )
            for kv_head in range(kv_head_count)
        ]
        shared_projected_v = [
            ax.dsl.create_tensor(
                (query_len, 1, head_dim),
                x.dtype,
                ax.MemoryType.SHARED_OCM,
                name=f"shared_projected_v_h{kv_head}",
            )
            for kv_head in range(kv_head_count)
        ]
        if pair_rank == 0:
            project_component(
                shared_x_scaled,
                shared_x_rms,
                w_k,
                w_scale_k,
                w_scale2_k,
                kv_dim_slice.start,
                head_dim,
                min(_KV_PROJECTION_TOKEN_TILE, query_len),
                "k",
                banks=banks,
                r=shared_projected_k[kv_head_start],
            )
            project_component(
                shared_x_scaled,
                shared_x_rms,
                w_v,
                w_scale_v,
                w_scale2_v,
                kv_dim_slice.start,
                head_dim,
                min(_KV_PROJECTION_TOKEN_TILE, query_len),
                "v",
                banks=banks,
                r=shared_projected_v[kv_head_start],
            )
        for tensor in (*shared_projected_k, *shared_projected_v):
            axd.pub(tensor)
        for tensor in (*shared_projected_k, *shared_projected_v):
            axd.sub(tensor)

        projected_k = shared_projected_k[kv_head_start]
        v = shared_projected_v[kv_head_start]
    else:
        projected_k = project_component(
            shared_x_scaled,
            shared_x_rms,
            w_k,
            w_scale_k,
            w_scale2_k,
            kv_dim_slice.start,
            kv_heads_per_core * head_dim,
            min(_KV_PROJECTION_TOKEN_TILE, query_len),
            "k",
            banks=banks,
        ).reshape(query_len, kv_heads_per_core, head_dim)
        v = project_component(
            shared_x_scaled,
            shared_x_rms,
            w_v,
            w_scale_v,
            w_scale2_v,
            kv_dim_slice.start,
            kv_heads_per_core * head_dim,
            min(_KV_PROJECTION_TOKEN_TILE, query_len),
            "v",
            banks=banks,
        ).reshape(query_len, kv_heads_per_core, head_dim)

    projected_k = projected_k.reshape(query_len, kv_heads_per_core, head_dim)
    normalized_k = rmsnorm(projected_k, k_norm_w)
    k = ax.dsl.create_tensor(projected_k.shape, dtype=ax.bf16, name="k")
    rope_impl(normalized_k, rope_cos, rope_sin, r=k, rotary_dim=rotary_dim, banks=banks)

    # K is the final producer on the Q@K critical path. Record its postprocess
    # before Q so the scheduler prioritizes K once its projection completes;
    # Q is ready earlier and can use otherwise idle TENG slots in the meantime.
    normalized_q = rmsnorm(projected_q, q_norm_w)
    q = ax.dsl.create_tensor(projected_q.shape, dtype=ax.bf16, name="q")
    rope_impl(normalized_q, rope_cos, rope_sin, r=q, rotary_dim=rotary_dim, banks=banks)

    # Flatten token/head into cache rows and scatter the runtime-selected rows
    # directly to DDR. Each core owns disjoint row indices, so this uses the
    # existing AxScatter path without requiring strided DynamicSt support.
    if not shared_kv_projection or pair_rank == 0:
        if cache_location is not None:
            store_paged_kv(k, paged_kv_cache, cache_location, 0, kv_head_start, banks=banks)
            store_paged_kv(v, paged_kv_cache, cache_location, 1, kv_head_start, banks=banks)
        else:
            local_kv_row_indices = (
                np.arange(query_len, dtype=np.uint32)[:, None] * kv_head_count
                + np.arange(kv_head_start, kv_head_stop, dtype=np.uint32)[None, :]
            ).reshape(-1)
            kv_write_row_indices = ax.dsl.add(
                kv_write_page_cursor_u32 * ax.dsl.from_numpy(np.array([kv_head_count], dtype=np.uint32)),
                ax.dsl.from_numpy(local_kv_row_indices),
            )
            for kv_plane, projected_kv in enumerate((k, v)):
                cache_rows = kv_page_out[kv_plane : kv_plane + 1].reshape(_KV_PAGE_SIZE * kv_head_count, head_dim)
                scatter_rows(
                    projected_kv.reshape(query_len * kv_heads_per_core, head_dim),
                    kv_write_row_indices,
                    dim=0,
                    r=cache_rows,
                    banks=banks,
                )

    # Stage 3: statically expanded, head-parallel paged FlashAttention.
    local_attention_heads = paged_flash_attention(
        q,
        k,
        v,
        paged_kv_cache,
        indices,
        page_start,
        history_page_count,
        kv_head_start,
        banks=banks,
    )
    if gate is not None:
        # Gate rows are strided views of packed [Q, Gate] projections.
        # Sigmoid produces contiguous [T,H,D] BF16 before the view is flattened.
        gate = ax.dsl.sigmoid(gate, output_dtype=x.dtype).reshape(query_len, q_heads_per_core * head_dim)
        ax.dsl.mul(
            local_attention_heads,
            gate,
            r=local_attention_heads,
            output_dtype=x.dtype,
        )

    # Stage 4: consume all local head channels in each W_o projection and add
    # the complete per-core contribution to each residual bank through NMC.
    project_attention_output(
        local_attention_heads,
        w_o,
        w_scale_o,
        w_scale2_o,
        shared_output,
        output.dtype,
        core_id,
        core_count,
        q_head_start * head_dim,
        hidden_dim_per_core,
        banks=banks,
    )

    # NMC completion is not a normal tensor producer edge. Close all remote
    # additions before the participating cores store disjoint output shards.
    axd.pub(shared_output)
    axd.sub(shared_output)
    if query_indices is None:
        if output.memory_type == ax.MemoryType.DDR:
            store_tile(x=shared_output[query_row_slice], r=output[query_row_slice], banks=banks)
    else:
        # Close every NMC contributor before the next tile reuses shared storage.
        axd.pub(shared_output)
        axd.sub(shared_output)
        axd.pub(paged_kv_cache)
        axd.sub(paged_kv_cache)
        scatter_rows(
            shared_output[query_row_slice].reshape(query_rows_per_core, hidden_dim),
            query_indices[query_row_slice[1]],
            dim=0,
            r=output.reshape(total_query_len, hidden_dim),
            banks=banks,
        )
    return output, kv_page_out


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def flash_attn_with_kvcache(
    x,  # [1, query_len, hidden_dim]
    x_norm_w,  # [hidden_dim]
    w_q,  # ungated [Hq * D, M]; gated packed [Hq * 2D, M]
    w_scale_q,
    w_scale2_q,
    w_zeropoint_q,
    w_k,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_k,
    w_scale2_k,
    w_zeropoint_k,
    w_v,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_v,
    w_scale2_v,
    w_zeropoint_v,
    q_norm_w,  # [head_dim]
    k_norm_w,  # [head_dim]
    cos,  # [max_position, rotary_dim // 2]
    sin,  # [max_position, rotary_dim // 2]
    position_ids,  # [valid_query_tokens] or [3, valid_query_tokens] int32
    w_o,  # [hidden_dim, num_heads_q * head_dim]
    w_scale_o,
    w_scale2_o,
    w_zeropoint_o,
    kv_page_out,  # BF16: [2, 256, num_kv_heads, head_dim]
    output,  # [1, query_len, hidden_dim]
    paged_kv_cache,  # BF16: [max_pages, 2, 256, num_kv_heads, head_dim]
    indptr,  # [2] int32
    indices,  # [history_pages] int32
    kv_write_page_cursor,  # [1] uint32; current tile start within the output page
    attn_output_gate=False,
    partial_rotary_factor=1.0,
    mrope_section=(16, 24, 24),
    mrope_interleaved=False,
    has_mrope=False,
    residual=None,
    post_residual_addition=None,
    cache_location=None,
    cos_sin_cache=None,
    banks=None,
):
    """Run one request using bounded 256-token pages for long prefill.

    Long prefill requires complete pages, a zero write offset, full positions,
    and page-table entries for the history and every new page.
    """
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    core_id, core_count = ax.program_id(0), ax.num_programs(0)
    projection_args = (
        x,
        x_norm_w,
        w_q,
        w_scale_q,
        w_scale2_q,
        w_zeropoint_q,
        w_k,
        w_scale_k,
        w_scale2_k,
        w_zeropoint_k,
        w_v,
        w_scale_v,
        w_scale2_v,
        w_zeropoint_v,
        q_norm_w,
        k_norm_w,
        cos,
        sin,
        position_ids,
        w_o,
        w_scale_o,
        w_scale2_o,
        w_zeropoint_o,
    )
    options = (
        attn_output_gate,
        partial_rotary_factor,
        mrope_section,
        mrope_interleaved,
        has_mrope,
    )
    if x.shape[1] <= 256:
        return prefill_attention_tile(
            *projection_args,
            kv_page_out,
            output,
            paged_kv_cache,
            indptr,
            indices,
            kv_write_page_cursor,
            *options,
            residual,
            post_residual_addition,
            cache_location,
            cos_sin_cache=cos_sin_cache,
            core_id=core_id,
            core_count=core_count,
            banks=banks,
        )
    assert x.shape[1] % 256 == 0
    assert cache_location is not None
    assert indices.shape_size >= x.shape[1] // 256
    offsets = ax.dsl.from_numpy(np.arange(256, dtype=np.int32))
    stride = ax.dsl.full((1,), ax.int32, fill_value=256)
    start = load_tile(indptr[:1], banks=banks)
    prefix_end = load_tile(indptr[1:2], banks=banks)
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    # The first page can have empty history; subsequent pages always have it.
    prefill_attention_tile(
        *projection_args,
        kv_page_out,
        output,
        paged_kv_cache,
        indptr,
        indices,
        kv_write_page_cursor,
        *options,
        residual,
        post_residual_addition,
        cache_location,
        cos_sin_cache=cos_sin_cache,
        core_id=core_id,
        core_count=core_count,
        query_indices=offsets,
        banks=banks,
    )

    def tile_body(tile, loop_cache, loop_output):
        query_indices = tile * stride + offsets
        tile_indptr = ax.dsl.concat(start, prefix_end + tile, dim=0)
        page = gather_rows(load_tile(indices, banks=banks), prefix_end + tile, dim=0, banks=banks)
        tile_location = ax.dsl.concat(page, zero, dim=0)
        prefill_attention_tile(
            *projection_args,
            kv_page_out,
            loop_output,
            loop_cache,
            tile_indptr,
            indices,
            kv_write_page_cursor,
            *options,
            residual,
            post_residual_addition,
            tile_location,
            cos_sin_cache=cos_sin_cache,
            core_id=core_id,
            core_count=core_count,
            query_indices=query_indices,
            banks=banks,
        )
        return loop_cache, loop_output

    ax.dsl.range(
        ax.dsl.zeros((1,), ax.int32),
        one,
        ax.dsl.full((1,), ax.int32, fill_value=x.shape[1] // 256),
        one,
        tile_body,
        paged_kv_cache,
        output,
        unroll_factor=1,
    )
    return output, kv_page_out


def project_token(
    x,
    weight,
    weight_scale,
    weight_scale2,
    output_channel_start,
    output_channel_count,
    name,
    banks=None,
    r=None,
):
    """Project one core's Q, K, or V output-channel shard."""
    output_shape = (x.shape[0], x.shape[1], output_channel_count)
    destination = None if r is None else r.reshape(-1, output_channel_count)
    if weight.dtype.typename in ("NVFP4", "NVFP4_X2") and weight_scale.dtype.typename == "FP8_E4M3":
        # Dynamic blocks expose static views plus explicit bank coordinates.
        # Stage every weight/scale through those coordinates; ordinary linear
        # loads would keep selecting block zero. Retain FP32 NMC and IC tiles.
        input_dim = x.shape[-1]
        packed_factor = 2 if weight.dtype.typename == "NVFP4_X2" else 1
        input_tiles = [slice(start, min(start + 1024, input_dim)) for start in range(0, input_dim, 1024)]
        if len(input_tiles) > 1 and input_tiles[-1].stop - input_tiles[-1].start < 1024:
            input_tiles[-2:] = [slice(input_tiles[-2].start, input_dim)]
        output_tiles = [
            slice(start, min(start + 512, output_channel_count)) for start in range(0, output_channel_count, 512)
        ]
        if len(output_tiles) > 1 and output_tiles[-1].stop - output_tiles[-1].start < 512:
            output_tiles[-2:] = [slice(output_tiles[-2].start, output_channel_count)]
        projected = ax.dsl.create_tensor((x.shape[0], output_channel_count), ax.fp32, name=f"{name}_accumulator")
        split_ic = len(input_tiles) > 1
        if split_ic:
            zero = ax.dsl.zeros((1,), ax.fp32)
            for output_slice in output_tiles:
                tile = projected[:, output_slice]
                ax.dsl.expand(zero, shape=tile.shape, r=tile)
        scale2 = load_tile(weight_scale2, banks=banks)
        specs = [(output_slice, input_slice) for output_slice in output_tiles for input_slice in input_tiles]
        pending = None
        for spec in [*specs, None]:
            next_tile = None
            if spec is not None:
                output_slice, input_slice = spec
                channels = slice(
                    output_channel_start + output_slice.start,
                    output_channel_start + output_slice.stop,
                )
                packed = slice(
                    input_slice.start // packed_factor,
                    input_slice.stop // packed_factor,
                )
                groups = slice(input_slice.start // 16, input_slice.stop // 16)
                next_tile = (
                    output_slice,
                    input_slice,
                    load_tile(weight[channels, packed], banks=banks),
                    load_tile(weight_scale[channels, groups], banks=banks),
                )
            # Stage the next tile before consuming the pending one, preserving
            # the existing overlap between bank transfers and MatMul.
            if pending is not None:
                output_slice, input_slice, values, scales = pending
                ax.dsl.matmul(
                    x.reshape(-1, input_dim)[:, input_slice],
                    values,
                    w_scale=scales,
                    w_scale2=scale2,
                    transposed=True,
                    output_dtype=ax.fp32,
                    output_add_assign=split_ic,
                    r=projected[:, output_slice],
                )
            pending = next_tile
        # IC waves accumulate in FP32 and round once at the projection boundary.
        return ax.dsl.cast(projected, dtype=x.dtype, r=destination).reshape(*output_shape)
    input_dim = x.shape[-1]
    ic_slice_factor = 2 if datatype.DType(weight.dtype).typename in _PACKED_FP4_WEIGHT_TYPENAMES else 1
    assert input_dim % ic_slice_factor == 0
    weight_input_dim = input_dim // ic_slice_factor
    matmul_oc_tile = min(output_channel_count, 256 if ic_slice_factor == 2 else 128)
    # Keep each runtime NVFP4 FC weight at offset zero, as in prefill's dense
    # reduction tiles. Other formats retain the larger OC prefetch group.
    weight_load_oc_tile = (
        matmul_oc_tile
        if weight.dtype.typename in ("NVFP4", "NVFP4_X2")
        else min(output_channel_count, _PROJECTION_WEIGHT_OC_LOAD_TILE)
    )
    assert output_channel_count % matmul_oc_tile == 0

    projected = (
        ax.dsl.create_tensor(
            (x.shape[0] * x.shape[1], output_channel_count),
            dtype=ax.bf16,
            name=f"{name}_projected",
        )
        if destination is None
        else destination
    )
    x_2d = x.reshape(-1, input_dim)

    scale2 = load_tile(weight_scale2, banks=banks)

    local_oc_starts = tuple(range(0, output_channel_count, matmul_oc_tile))
    if weight_scale.shape_size == 1:
        scalar_scale = load_tile(weight_scale, banks=banks)
        scale_tiles = [scalar_scale] * len(local_oc_starts)
    else:
        scale_tiles = []
        for local_oc_start in local_oc_starts:
            global_oc_start = output_channel_start + local_oc_start
            global_oc_slice = slice(global_oc_start, global_oc_start + matmul_oc_tile)
            scale_tiles.append(load_tile(weight_scale[global_oc_slice, :], banks=banks))

    # Consume one OC group before loading the next. Keeping a complete 27B Q
    # shard live would make the allocator swap its MXFP8 weights through DDR.
    for load_oc_start in range(0, output_channel_count, weight_load_oc_tile):
        load_oc_count = min(weight_load_oc_tile, output_channel_count - load_oc_start)
        assert load_oc_count % matmul_oc_tile == 0
        global_oc_start = output_channel_start + load_oc_start
        global_oc_slice = slice(global_oc_start, global_oc_start + load_oc_count)
        loaded_weight = ax.dsl.create_tensor(
            (load_oc_count, weight_input_dim),
            dtype=weight.dtype,
            name=f"{name}_weight_oc{load_oc_start}",
        )
        # Load the complete IC row once. Splitting a packed DDR row into
        # strided IC views makes DMA fetch the full source row for each view.
        load_tile(weight[global_oc_slice, :], r=loaded_weight, banks=banks)
        for tile_oc_start in range(0, load_oc_count, matmul_oc_tile):
            local_oc_start = load_oc_start + tile_oc_start
            local_oc_slice = slice(local_oc_start, local_oc_start + matmul_oc_tile)
            tile_index = local_oc_start // matmul_oc_tile
            ax.dsl.matmul(
                x_2d,
                loaded_weight[tile_oc_start : tile_oc_start + matmul_oc_tile, :],
                w_scale=scale_tiles[tile_index],
                w_scale2=scale2,
                transposed=True,
                output_dtype=ax.bf16,
                r=projected[:, local_oc_slice],
            )
    return projected.reshape(*output_shape)


def _decode_query_head_ids(head_start, head_count, min_heads=_MIN_DECODE_QUERY_HEADS):
    """Row ids for one KV head's queries, repeating the first id to the gather floor."""
    ids = np.arange(head_start, head_start + head_count, dtype=np.int32)
    if ids.size < min_heads:
        ids = np.concatenate([ids, np.full(min_heads - ids.size, int(ids[0]), dtype=np.int32)])
    return ids


def _select_decode_query_rows(q_heads, head_start, head_count, banks=None):
    """Take query rows from the full local Q, never a size-1/2 slice.

    GQA=2 would slice two heads and concat dummy rows. That size-2 source is the
    same SDMA gather lowering as a short page table. Index-select from the parent
    keeps the gather axis at ``local_query_head_count``. Dummy rows repeat the
    first real head and are discarded after softmax.
    """
    if head_count >= _MIN_DECODE_QUERY_HEADS:
        return q_heads[head_start : head_start + head_count]
    return gather_rows(
        q_heads,
        ax.dsl.from_numpy(_decode_query_head_ids(head_start, head_count)),
        dim=0,
        banks=banks,
    )


def _gather_history_page_tiles(
    paged_kv_cache,
    physical_page,
    page_size,
    kv_head_index,
    head_dim,
    value_tile=_ATTENTION_OC_TILE,
    banks=None,
):
    """Gather one KV head from flattened cache rows.

    Compact SGLang decode often has a single history page. A dim-0 gather on
    that size-1 page axis (or on a short page-id table) lowers to an SDMA
    gather whose index is not the runtime page id, which shows up as a DDR
    address about ``2**31`` strides past the tensor. Flatten the contiguous
    ``[page, 2, PAGE_SIZE, kv_head, head_dim]`` cache first so the gather axis
    is the large row dim. K and V are selected by row-index offsets, not by
    slicing the size-2 K/V plane: that view has a V-sized gap between pages,
    so DSL reshape cannot explode the page dim into flattened rows. Host
    padding still keeps the page axis and the page-id table at
    ``_MIN_PAGED_GATHER_PAGES``.
    """
    kv_heads = paged_kv_cache.shape[3]
    token_row_offsets = np.arange(page_size, dtype=np.int32) * kv_heads + kv_head_index
    k_row_offsets = ax.dsl.from_numpy(token_row_offsets)
    v_row_offsets = ax.dsl.from_numpy(page_size * kv_heads + token_row_offsets)
    page_row_stride = ax.dsl.from_numpy(np.array([2 * page_size * kv_heads], dtype=np.int32))
    page_row_base = physical_page * page_row_stride
    flat_cache = paged_kv_cache.reshape(-1, head_dim)
    k_partitions = [
        gather_rows(
            flat_cache,
            page_row_base + k_row_offsets[tile_start : tile_start + _ATTENTION_OC_TILE],
            dim=0,
            banks=banks,
        )
        for tile_start in range(0, page_size, _ATTENTION_OC_TILE)
    ]
    v_page = gather_rows(flat_cache, page_row_base + v_row_offsets, dim=0, banks=banks)
    v_partitions = [v_page[:, start : start + value_tile] for start in range(0, head_dim, value_tile)]
    return k_partitions, v_partitions


def paged_attention(
    q,
    paged_kv_cache,
    paged_kv_indices,
    page_start,
    page_count,
    current_k,
    current_v,
    output,
    page_size,
    kv_head_index,
    accumulator_dtype,
    valid_kv_len,
    banks=None,
):
    """Attend all local KV heads in one runtime history-page loop.

    H is the number of local KV heads, G their padded GQA query count, D the
    head dimension, and P the page size. Each head owns separate maximum,
    FP32 normalizer and accumulator tensors. Only page IDs and lengths are
    runtime values; neither controls the compiled tensor shapes.
    """
    local_query_head_count, head_dim = q.shape
    kv_head_count = current_k.shape[0]
    real_query_head_count = local_query_head_count // kv_head_count
    query_head_count = max(real_query_head_count, _MIN_DECODE_QUERY_HEADS)
    slot_positions = ax.dsl.from_numpy(np.arange(page_size, dtype=np.int32))
    page_size_scalar = ax.dsl.from_numpy(np.array([page_size], dtype=np.int32))
    assert page_size % _ATTENTION_OC_TILE == 0
    assert head_dim % _ATTENTION_OC_TILE == 0

    # Each triple is [G, 1] BF16, [1, G] FP32, [1, G, D] accumulator_dtype.
    states = []
    for head in range(kv_head_count):
        states.extend(
            (
                ax.dsl.full((query_head_count, 1), dtype=q.dtype, fill_value=-3e38),
                ax.dsl.zeros((1, query_head_count), dtype=ax.fp32),
                ax.dsl.zeros((1, query_head_count, head_dim), dtype=accumulator_dtype),
            )
        )
    # Compute current-token QK early; merge it after the history recurrence.
    current_scores = []
    for head in range(kv_head_count):
        local_q = _select_decode_query_rows(q, head * real_query_head_count, real_query_head_count, banks=banks)
        # [G, D] BF16 * [1, D] BF16 -> [G, D] FP32.
        score_products = ax.dsl.mul(
            local_q,
            current_k[head : head + 1].reshape(1, head_dim),
            output_dtype=ax.fp32,
        )
        score = ax.dsl.sum(score_products, dim=-1, keepdim=True, output_dtype=q.dtype)
        current_scores.append(score)
    page_index = ax.dsl.zeros((1,), dtype=ax.int32)
    range_start = ax.dsl.zeros((1,), dtype=ax.int32)
    range_step = ax.dsl.from_numpy(np.ones((1,), dtype=np.int32))

    def load_page(logical_page, indices, start, cache):
        # Runtime [1] INT32 table lookup -> [1] INT32 physical page ID.
        physical_page = gather_rows(indices, start + logical_page, dim=0, banks=banks)
        flat_cache = cache.reshape(-1, head_dim)
        kv_heads = cache.shape[3]
        # Physical page ID -> [1] INT32 flattened cache-row base.
        page_stride = ax.dsl.from_numpy(np.array([2 * page_size * kv_heads], dtype=np.int32))
        page_base = physical_page * page_stride
        buffers = []
        for head in range(kv_head_count):
            rows = np.arange(page_size, dtype=np.int32) * kv_heads + kv_head_index + head
            # [P] INT32 offsets select complete [P, D] BF16 K and V pages.
            k_rows = ax.dsl.from_numpy(rows)
            v_rows = ax.dsl.from_numpy(rows + page_size * kv_heads)
            k_page = gather_rows(flat_cache, page_base + k_rows, dim=0, banks=banks)
            v_page = gather_rows(flat_cache, page_base + v_rows, dim=0, banks=banks)
            buffers.extend((k_page, v_page))
        return buffers

    def consume_page(logical_page, local_q, valid_len, positions, buffers, states):
        # [1, P] BF16 additive mask shared by all heads; invalid slots are hidden.
        tail_mask = _history_tail_mask(
            logical_page,
            positions,
            page_size_scalar,
            valid_len,
            page_size,
            local_q.dtype,
        )
        for head in range(kv_head_count):
            k_page, v_page = buffers[head * 2 : head * 2 + 2]
            # Views only: K tiles [64, D], V tiles [P, 64], both BF16.
            k_tiles = [k_page[i : i + _ATTENTION_OC_TILE] for i in range(0, page_size, _ATTENTION_OC_TILE)]
            v_tiles = [v_page[:, i : i + _ATTENTION_OC_TILE] for i in range(0, head_dim, _ATTENTION_OC_TILE)]
            scores = ax.dsl.create_tensor((query_head_count, page_size + 2), dtype=local_q.dtype)
            head_q = _select_decode_query_rows(
                local_q,
                head * real_query_head_count,
                real_query_head_count,
                banks=banks,
            )
            # [G, D] BF16 Q + page tiles update only this head's state triple.
            paged_attn_update(
                head_q,
                k_tiles,
                v_tiles,
                *states[head * 3 : head * 3 + 3],
                scores,
                tail_mask=tail_mask,
            )

    # Keep gathered pages inside their consuming iteration. Carrying primed
    # page buffers through the exported Range produces Inf on the board for
    # single-page history, despite a finite simulator result.

    def history_body(page_index, indices, start, cache, local_q, valid_len, positions, *states):
        buffers = load_page(page_index, indices, start, cache)
        consume_page(page_index, local_q, valid_len, positions, buffers, states)
        return indices, start, cache, local_q, valid_len, positions, *states

    results = ax.dsl.range(
        page_index,
        range_start,
        page_count,
        range_step,
        history_body,
        paged_kv_indices,
        page_start,
        paged_kv_cache,
        q,
        valid_kv_len,
        slot_positions,
        *states,
    )
    states = results[-3 * kv_head_count :]
    current_values, output_heads = current_v, output
    for head in range(kv_head_count):
        current_score = current_scores[head]
        max_state, normalizer, accumulator = states[head * 3 : head * 3 + 3]
        current_v = current_values[head : head + 1].reshape(1, head_dim)
        output = output_heads[:, head * real_query_head_count : (head + 1) * real_query_head_count]
        # The current KV row has N=1. Compute all query heads as vector operations
        # instead of padding one QK MatMul per head to the backend's N=64 tile.
        updated_max = ax.dsl.maximum(max_state, current_score)
        shifted_previous_max = ax.dsl.sub(max_state, updated_max)
        shifted_current_score = ax.dsl.sub(current_score, updated_max)

        previous_ratio = ax.dsl.exp(shifted_previous_max, output_dtype=ax.fp32).reshape(1, query_head_count)
        current_probability = ax.dsl.exp(shifted_current_score)
        current_probability_fp32 = ax.dsl.exp(shifted_current_score, output_dtype=ax.fp32).reshape(1, query_head_count)
        scaled_previous_sum = ax.dsl.mul(previous_ratio, normalizer, output_dtype=normalizer.dtype)
        ax.dsl.add(
            scaled_previous_sum,
            current_probability_fp32,
            r=normalizer,
            output_dtype=normalizer.dtype,
        )

        weighted_current_value = ax.dsl.mul(
            current_probability.reshape(1, query_head_count, 1),
            current_v.reshape(1, 1, head_dim),
            output_dtype=accumulator.dtype,
        )
        scaled_previous_output = ax.dsl.mul(
            previous_ratio.reshape(1, query_head_count, 1),
            accumulator,
            output_dtype=ax.fp32,
        )
        ax.dsl.add(
            scaled_previous_output,
            weighted_current_value,
            r=accumulator,
            output_dtype=accumulator.dtype,
        )
        ax.dsl.div(
            accumulator[:, :real_query_head_count],
            normalizer[:, :real_query_head_count].reshape(1, real_query_head_count, 1),
            r=output,
        )


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def decode_paged_kv_kernel(
    x,  # [B, 1, hidden_dim]
    x_norm_w,  # [1, hidden_dim]
    w_q,  # ungated [Hq * D, M]; gated packed [Hq * 2D, M]
    w_scale_q,
    w_scale2_q,
    w_zeropoint_q,
    w_k,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_k,
    w_scale2_k,
    w_zeropoint_k,
    w_v,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_v,
    w_scale2_v,
    w_zeropoint_v,
    q_norm_w,  # [1, head_dim]
    k_norm_w,  # [1, head_dim]
    cos,  # [max_seq_len, rotary_dim // 2]
    sin,  # [max_seq_len, rotary_dim // 2]
    position_ids,  # [1] or [3, 1] int32
    w_o,  # [hidden_dim, num_heads_qo * head_dim]
    w_scale_o,
    w_scale2_o,
    w_zeropoint_o,
    k_cache_out,  # [batch_size, max_seq_len, num_heads_kv, head_dim]
    v_cache_out,  # [batch_size, max_seq_len, num_heads_kv, head_dim]
    output,
    paged_kv_cache,  # [MAX_NUM_PAGES, 2, PAGE_SIZE, kv_head_num, head_dims]
    indptr,  # [BATCH+1], e.g. [0, 2, 5, 6] int32
    indices,  # [PAGES USED BY BATCH], e.g. [5, 7, 2, 9, 11, 14] int32
    last_page_len,  # [BATCH], e.g. [3, 1, 4] int32
    PAGE_SIZE,
    kv_offset=0,
    attn_output_gate=False,
    partial_rotary_factor=1.0,
    mrope_section=(16, 24, 24),
    mrope_interleaved=False,
    has_mrope=False,
    residual=None,
    post_residual_addition=None,
    cache_location=None,
    cos_sin_cache=None,
    banks=None,
):
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)

    B, seq_len, hidden_dim = x.shape
    assert B == 1
    assert seq_len == 1
    q_gate_row_factor = 2 if attn_output_gate else 1
    assert w_q.shape[0] % (q_gate_row_factor * q_norm_w.shape[-1]) == 0
    query_projection_dim = w_q.shape[0] // q_gate_row_factor
    kv_projection_dim = w_k.shape[0]
    head_dim = q_norm_w.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)
    assert hidden_dim % core_num == 0
    assert _PAGE_UNROLL_FACTOR > 0
    assert 0 < partial_rotary_factor <= 1.0
    assert rotary_dim % 2 == 0
    if cos_sin_cache is None:
        assert cos.shape[-1] == sin.shape[-1] == rotary_dim // 2
    else:
        assert cos_sin_cache.shape[-1] == rotary_dim
    if cache_location is None:
        assert kv_offset + 1 <= k_cache_out.shape[1]
        k_cache_out_cur = k_cache_out[:, kv_offset : kv_offset + 1, :, :]
        v_cache_out_cur = v_cache_out[:, kv_offset : kv_offset + 1, :, :]

    query_head_count = query_projection_dim // head_dim
    kv_head_count = kv_projection_dim // head_dim
    query_heads_per_kv = query_head_count // kv_head_count
    assert query_head_count % core_num == 0
    assert kv_head_count >= core_num
    assert kv_head_count % core_num == 0
    local_query_head_count = query_head_count // core_num
    local_kv_head_count = kv_head_count // core_num
    kv_head_start = core_id * local_kv_head_count
    assert local_query_head_count == local_kv_head_count * query_heads_per_kv

    # Each projection core loads one contiguous hidden-dimension shard. The
    # first barrier publishes the complete token to core 0 for normalization.
    hidden_dims_per_core = hidden_dim // core_num
    hidden_slice = slice(core_id * hidden_dims_per_core, (core_id + 1) * hidden_dims_per_core)
    if x.memory_type == ax.MemoryType.DDR:
        x_shocm = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM)
        load_tile(r=x_shocm[:, :, hidden_slice], x=x[:, :, hidden_slice], banks=banks)
    else:
        # The fused producer published the complete shared activation.
        assert residual is None and post_residual_addition is None
        x_shocm = x
    if residual is not None:
        ax.dsl.add(
            x_shocm[:, :, hidden_slice],
            load_tile(residual[:, :, hidden_slice], banks=banks),
            r=x_shocm[:, :, hidden_slice],
        )
    if post_residual_addition is not None:
        ax.dsl.add(
            x_shocm[:, :, hidden_slice],
            load_tile(post_residual_addition[:, :, hidden_slice], banks=banks),
            r=x_shocm[:, :, hidden_slice],
        )

    # Only core 0 records the full RMSNorm. The second barrier publishes the
    # normalized token before every core starts its local Q/K/V projections.
    r_norm_tmp = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM)
    if core_id == 0:
        x_norm_ocm = load_tile(x_norm_w, banks=banks)
    if x.memory_type == ax.MemoryType.DDR:
        axd.pub(x_shocm)
        axd.sub(x_shocm)
    if core_id == 0:
        squared = ax.dsl.mul(x_shocm, x_shocm, output_dtype=ax.fp32)
        square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
        hidden_size = ax.dsl.full((1,), dtype=ax.fp32, fill_value=hidden_dim)
        square_mean = ax.dsl.div(square_sum, hidden_size)
        epsilon = ax.dsl.full((1,), dtype=ax.fp32, fill_value=1e-6)
        denominator = ax.dsl.sqrt(ax.dsl.add(square_mean, epsilon))
        normalized = ax.dsl.div(x_shocm, denominator, output_dtype=ax.fp32)
        ax.dsl.mul(normalized, x_norm_ocm, output_dtype=x.dtype, r=r_norm_tmp)
    axd.pub(r_norm_tmp)
    axd.sub(r_norm_tmp)

    # Query and KV heads are split evenly across cores. Each core projects,
    # stores, and attends its complete local KV-head group.
    query_head_start = core_id * local_query_head_count
    query_head_stop = query_head_start + local_query_head_count
    query_channel_slice = slice(query_head_start * head_dim, query_head_stop * head_dim)
    q_gate_channel_slice = slice(
        query_head_start * q_gate_row_factor * head_dim,
        query_head_stop * q_gate_row_factor * head_dim,
    )
    kv_channel_start = kv_head_start * head_dim
    kv_channel_count = local_kv_head_count * head_dim
    kv_head_slice = slice(kv_head_start, kv_head_start + local_kv_head_count)
    projected_q_gate = project_token(
        r_norm_tmp,
        w_q,
        w_scale_q,
        w_scale2_q,
        q_gate_channel_slice.start,
        q_gate_channel_slice.stop - q_gate_channel_slice.start,
        "q_gate" if attn_output_gate else "q",
        banks=banks,
    )
    if attn_output_gate:
        projected_q_gate = projected_q_gate.reshape(B, 1, local_query_head_count, 2 * head_dim)
        projected_q = projected_q_gate[..., :head_dim]
        gate = projected_q_gate[..., head_dim:]
    else:
        projected_q = projected_q_gate.reshape(B, 1, local_query_head_count, head_dim)
        gate = None

    projected_k = project_token(
        r_norm_tmp,
        w_k,
        w_scale_k,
        w_scale2_k,
        kv_channel_start,
        kv_channel_count,
        "k",
        banks=banks,
    )

    projected_v = project_token(
        r_norm_tmp,
        w_v,
        w_scale_v,
        w_scale2_v,
        kv_channel_start,
        kv_channel_count,
        "v",
        banks=banks,
    )

    # Explicit destinations let the two rotary halves write directly into one
    # contiguous tensor. Returning from rope_impl would record an AxConcat for
    # Q and K on every core.
    q = ax.dsl.create_tensor((B, 1, local_query_head_count, head_dim), dtype=x.dtype)
    k = ax.dsl.create_tensor((B, 1, local_kv_head_count, head_dim), dtype=x.dtype)
    projected_k = projected_k.reshape(B, 1, local_kv_head_count, head_dim)
    projected_v = projected_v.reshape(B, 1, local_kv_head_count, head_dim)

    # Normalize Q/K, then gather one shared pair of rotary rows for this token.
    normalized_q = rmsnorm(projected_q, load_tile(q_norm_w, banks=banks))
    normalized_k = rmsnorm(projected_k, load_tile(k_norm_w, banks=banks))

    if cache_location is None:
        store_tile(r=v_cache_out_cur[:, :, kv_head_slice, :], x=projected_v, banks=banks)
    else:
        store_paged_kv(
            projected_v.reshape(1, local_kv_head_count, head_dim),
            paged_kv_cache,
            cache_location,
            1,
            kv_head_start,
            banks=banks,
        )

    indptr = load_tile(indptr, banks=banks)
    indices = load_tile(indices, banks=banks)
    last_page_len = load_tile(last_page_len, banks=banks)
    position_ids = load_tile(position_ids, banks=banks)
    one_i32 = ax.dsl.from_numpy(np.ones(shape=(1,), dtype=np.int32))
    page_size_i32 = ax.dsl.from_numpy(np.array([PAGE_SIZE], dtype=np.int32))

    num_pages = indptr[1:2] - indptr[:1]
    cached_len = ax.dsl.maximum(
        (num_pages - one_i32) * page_size_i32 + last_page_len[:1],
        ax.dsl.zeros((1,), dtype=ax.int32),
    )
    rope_cos, rope_sin = gather_rope_cos_sin(
        cos,
        sin,
        position_ids,
        1,
        mrope_section,
        mrope_interleaved,
        has_mrope,
        cos_sin_cache,
        banks=banks,
    )
    rope_impl(normalized_q, rope_cos, rope_sin, r=q, rotary_dim=rotary_dim, banks=banks)
    rope_impl(normalized_k, rope_cos, rope_sin, r=k, rotary_dim=rotary_dim, banks=banks)

    if cache_location is None:
        store_tile(r=k_cache_out_cur[:, :, kv_head_slice, :], x=k, banks=banks)
    else:
        store_paged_kv(
            k.reshape(1, local_kv_head_count, head_dim),
            paged_kv_cache,
            cache_location,
            0,
            kv_head_start,
            banks=banks,
        )

    # History pages are the only dynamic-loop work. Current K/V are passed to
    # paged_attention and updated after the range finishes.
    r_attn = ax.dsl.create_tensor((B, 1, local_query_head_count, head_dim), dtype=x.dtype)
    # Flatten local Q heads for selecting each KV head's GQA query group.
    q_heads = q.reshape(local_query_head_count, head_dim)
    # Random MXFP4 projections have enough dynamic range to amplify BF16
    # online-softmax accumulation error in the output projection.
    attention_accumulator_dtype = (
        ax.fp32 if datatype.DType(w_o.dtype).typename in _PACKED_FP4_WEIGHT_TYPENAMES else x.dtype
    )
    paged_attention(
        q_heads,
        paged_kv_cache,
        indices,
        indptr[:1],
        indptr[1:2] - indptr[:1],
        k.reshape(local_kv_head_count, 1, head_dim),
        projected_v.reshape(local_kv_head_count, 1, head_dim),
        r_attn.reshape(1, local_query_head_count, head_dim),
        PAGE_SIZE,
        kv_head_start,
        attention_accumulator_dtype,
        valid_kv_len=cached_len,
        banks=banks,
    )

    if gate is not None:
        r_attn = ax.dsl.mul(
            r_attn,
            ax.dsl.sigmoid(gate, output_dtype=x.dtype),
            output_dtype=x.dtype,
        )
    r_attn = r_attn.reshape(B, query_projection_dim // core_num)

    # Project local attention heads and reduce them into each output shard.
    output_scale2 = load_tile(w_scale2_o, banks=banks)
    scalar_output_scale = load_tile(w_scale_o, banks=banks) if w_scale_o.shape_size == 1 else None
    shared_output_scale = None
    if (
        w_scale_o.ndim == 2
        and w_scale_o.shape[0] == w_o.shape[0]
        and w_scale_o.shape[-1] > 1
        and w_scale_o.shape_size * w_scale_o.dtype.num_bits <= 2 * 1024 * 1024 * 8
    ):
        shared_output_scale = load_shared_scale_rows(w_scale_o, banks=banks)
    ic_slice_factor = 2 if datatype.DType(w_o.dtype).typename in _PACKED_FP4_WEIGHT_TYPENAMES else 1
    assert r_attn.shape[-1] % ic_slice_factor == 0
    output_weight_storage_ic = r_attn.shape[-1] // ic_slice_factor
    output_matmul_oc_tile = min(hidden_dims_per_core, 256 if ic_slice_factor == 2 else 128)
    output_weight_load_oc_tile = min(hidden_dims_per_core, _OUTPUT_PROJECTION_WEIGHT_OC_LOAD_TILE)
    assert hidden_dims_per_core % output_matmul_oc_tile == 0
    if w_scale_o.shape_size > 1 and w_scale_o.shape[-1] != 1:
        w_o_ic = w_o.shape[-1] * ic_slice_factor
        scale_group_size = w_o_ic // w_scale_o.shape[-1]
        w_scale_ic_slice = slice(
            query_channel_slice.start // scale_group_size,
            query_channel_slice.stop // scale_group_size,
        )
    else:
        w_scale_ic_slice = slice(0, 1)

    r_fc_o_shocm = ax.dsl.create_tensor(
        (core_num, B, 1, hidden_dim),
        x_shocm.dtype,
        ax.MemoryType.SHARED_OCM,
        name="r_fc_o_shocm",
    )
    r_fc_o = r_fc_o_shocm[core_id : core_id + 1].reshape(B, hidden_dim)
    for bank_id in range(core_num):
        slice_emb_dim = slice(bank_id * hidden_dims_per_core, (bank_id + 1) * hidden_dims_per_core)
        output_local_oc_starts = tuple(range(0, hidden_dims_per_core, output_matmul_oc_tile))

        output_scales = []
        for local_oc_start in output_local_oc_starts:
            global_oc_start = slice_emb_dim.start + local_oc_start
            global_oc_slice = slice(global_oc_start, global_oc_start + output_matmul_oc_tile)
            output_scale = scalar_output_scale
            if output_scale is None:
                output_scale = (
                    shared_output_scale[global_oc_slice, w_scale_ic_slice]
                    if shared_output_scale is not None
                    else load_tile(w_scale_o[global_oc_slice, w_scale_ic_slice], banks=banks)
                )
            output_scales.append(output_scale)

        output_weights = []
        for load_oc_start in range(0, hidden_dims_per_core, output_weight_load_oc_tile):
            load_oc_count = min(output_weight_load_oc_tile, hidden_dims_per_core - load_oc_start)
            assert load_oc_count % output_matmul_oc_tile == 0
            global_oc_start = slice_emb_dim.start + load_oc_start
            global_oc_slice = slice(global_oc_start, global_oc_start + load_oc_count)
            loaded_output_weight = ax.dsl.create_tensor(
                (load_oc_count, output_weight_storage_ic),
                dtype=w_o.dtype,
                name=f"o_weight_bank{bank_id}_oc{load_oc_start}",
            )
            for local_ic_start in range(0, r_attn.shape[-1], _OUTPUT_PROJECTION_WEIGHT_IC_LOAD_TILE):
                local_ic_stop = min(
                    local_ic_start + _OUTPUT_PROJECTION_WEIGHT_IC_LOAD_TILE,
                    r_attn.shape[-1],
                )
                local_ic_slice = slice(local_ic_start // ic_slice_factor, local_ic_stop // ic_slice_factor)
                global_ic_slice = slice(
                    (query_channel_slice.start + local_ic_start) // ic_slice_factor,
                    (query_channel_slice.start + local_ic_stop) // ic_slice_factor,
                )
                load_tile(
                    w_o[global_oc_slice, global_ic_slice],
                    r=loaded_output_weight[:, local_ic_slice],
                    banks=banks,
                )
            for tile_oc_start in range(0, load_oc_count, output_matmul_oc_tile):
                output_weights.append(loaded_output_weight[tile_oc_start : tile_oc_start + output_matmul_oc_tile, :])

        for tile_index, local_oc_start in enumerate(output_local_oc_starts):
            global_oc_start = slice_emb_dim.start + local_oc_start
            global_oc_slice = slice(global_oc_start, global_oc_start + output_matmul_oc_tile)
            ax.dsl.matmul(
                r_attn,
                output_weights[tile_index],
                w_scale=output_scales[tile_index],
                w_scale2=output_scale2,
                transposed=True,
                output_dtype=ax.bf16,
                r=r_fc_o[:, global_oc_slice],
            )

    axd.pub(r_fc_o_shocm)
    axd.sub(r_fc_o_shocm)

    # The final BF16 add writes its owned output shard directly when it stays
    # on chip. Preserve the existing ordered rounding of the partial sums.
    destination = output[:, :, hidden_slice] if output.memory_type != ax.MemoryType.DDR else None
    bank_sum = ax.dsl.add(
        x_shocm[:, :, hidden_slice],
        r_fc_o_shocm[0:1, :, :, hidden_slice].reshape(B, 1, hidden_dims_per_core),
    )
    for reduce_core_id in range(1, core_num):
        bank_sum = ax.dsl.add(
            bank_sum,
            r_fc_o_shocm[reduce_core_id : reduce_core_id + 1, :, :, hidden_slice].reshape(B, 1, hidden_dims_per_core),
            r=destination if reduce_core_id == core_num - 1 else None,
        )
    if output.memory_type == ax.MemoryType.DDR:
        store_tile(r=output[:, :, hidden_slice], x=bank_sum, banks=banks)

    return output, k_cache_out, v_cache_out


def paged_attention_history(
    q,
    paged_kv_cache,
    paged_kv_indices,
    page_start,
    page_count,
    page_size,
    kv_head_index,
    valid_kv_len,
    real_query_head_count,
    page_offset,
    banks=None,
):
    """Update one core's alternating pages, carrying only the softmax state.

    Q is BF16 [Qpad,D]. Return max [Qreal,1] BF16, normalizer [1,Qreal]
    FP32 and accumulator [1,Qreal,D] BF16; page IDs/counts remain tensors.
    """
    query_head_count, head_dim = q.shape
    slot_positions = ax.dsl.from_numpy(np.arange(page_size, dtype=np.int32))
    page_size_scalar = ax.dsl.from_numpy(np.array([page_size], dtype=np.int32))
    assert page_size % _ATTENTION_OC_TILE == 0
    assert head_dim % _ATTENTION_OC_TILE == 0

    max_state = ax.dsl.full((query_head_count, 1), dtype=q.dtype, fill_value=-3e38)
    normalizer = ax.dsl.zeros((1, query_head_count), dtype=ax.fp32)
    accumulator = ax.dsl.zeros((1, query_head_count, head_dim), dtype=q.dtype)
    page_index = ax.dsl.zeros((1,), dtype=ax.int32)
    range_start = ax.dsl.full((1,), dtype=ax.int32, fill_value=page_offset)
    range_step = ax.dsl.full((1,), dtype=ax.int32, fill_value=2)

    def history_page_body(
        page_index,
        max_state,
        normalizer,
        accumulator,
    ):
        physical_page = gather_rows(paged_kv_indices, page_start + page_index, dim=0, banks=banks)
        k_partitions, v_partitions = _gather_history_page_tiles(
            paged_kv_cache,
            physical_page,
            page_size,
            kv_head_index,
            head_dim,
            value_tile=32,
            banks=banks,
        )
        state_scores = ax.dsl.create_tensor(
            (query_head_count, 1 + page_size + 1),
            dtype=q.dtype,
        )
        tail_mask = _history_tail_mask(
            page_index,
            slot_positions,
            page_size_scalar,
            valid_kv_len,
            page_size,
            q.dtype,
        )
        paged_attn_update(
            q,
            k_partitions,
            v_partitions,
            max_state,
            normalizer,
            accumulator,
            state_scores,
            tail_mask=tail_mask,
        )
        return (
            max_state,
            normalizer,
            accumulator,
        )

    # Page bounds and the page/head ownership offset remain runtime loop inputs.
    history_results = ax.dsl.range(
        page_index,
        range_start,
        page_count,
        range_step,
        history_page_body,
        max_state,
        normalizer,
        accumulator,
        unroll_factor=1,
    )
    max_state, normalizer, accumulator = history_results[-3:]
    if real_query_head_count != query_head_count:
        return (
            max_state[:real_query_head_count],
            normalizer[:, :real_query_head_count],
            accumulator[:, :real_query_head_count],
        )
    return max_state, normalizer, accumulator


def finish_paged_attention(q, current_k, current_v, max_state, normalizer, accumulator, output):
    """Merge one current K/V row into the history online-softmax state."""
    query_head_count, head_dim = q.shape
    score_products = ax.dsl.mul(q, current_k, output_dtype=ax.fp32)
    current_score = ax.dsl.sum(score_products, dim=-1, keepdim=True, output_dtype=q.dtype)
    updated_max = ax.dsl.maximum(max_state, current_score)
    shifted_previous_max = ax.dsl.sub(max_state, updated_max)
    shifted_current_score = ax.dsl.sub(current_score, updated_max)

    previous_ratio = ax.dsl.exp(shifted_previous_max, output_dtype=ax.fp32).reshape(1, query_head_count)
    current_probability = ax.dsl.exp(shifted_current_score)
    current_probability_fp32 = ax.dsl.exp(shifted_current_score, output_dtype=ax.fp32).reshape(1, query_head_count)
    scaled_previous_sum = ax.dsl.mul(previous_ratio, normalizer, output_dtype=normalizer.dtype)
    ax.dsl.add(
        scaled_previous_sum,
        current_probability_fp32,
        r=normalizer,
        output_dtype=normalizer.dtype,
    )

    weighted_current_value = ax.dsl.mul(
        current_probability.reshape(1, query_head_count, 1),
        current_v.reshape(1, 1, head_dim),
        output_dtype=accumulator.dtype,
    )
    scaled_previous_output = ax.dsl.mul(
        previous_ratio.reshape(1, query_head_count, 1),
        accumulator,
        output_dtype=ax.fp32,
    )
    ax.dsl.add(
        scaled_previous_output,
        weighted_current_value,
        r=accumulator,
        output_dtype=accumulator.dtype,
    )
    ax.dsl.div(accumulator, normalizer.reshape(1, query_head_count, 1), r=output)


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def decode_paged_kv_shared_kv_kernel(
    x,  # [B, 1, hidden_dim]
    x_norm_w,  # [1, hidden_dim]
    w_q,  # ungated [Hq * D, M]; gated packed [Hq * 2D, M]
    w_scale_q,
    w_scale2_q,
    w_zeropoint_q,
    w_k,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_k,
    w_scale2_k,
    w_zeropoint_k,
    w_v,  # [num_heads_kv * head_dim, hidden_dim]
    w_scale_v,
    w_scale2_v,
    w_zeropoint_v,
    q_norm_w,  # [1, head_dim]
    k_norm_w,  # [1, head_dim]
    cos,  # [max_seq_len, rotary_dim // 2]
    sin,  # [max_seq_len, rotary_dim // 2]
    position_ids,  # [1] or [3, 1] int32
    w_o,  # [hidden_dim, num_heads_qo * head_dim]
    w_scale_o,
    w_scale2_o,
    w_zeropoint_o,
    k_cache_out,  # [batch_size, max_seq_len, num_heads_kv, head_dim]
    v_cache_out,  # [batch_size, max_seq_len, num_heads_kv, head_dim]
    output,
    paged_kv_cache,  # [MAX_NUM_PAGES, 2, PAGE_SIZE, kv_head_num, head_dims]
    indptr,  # [BATCH+1], e.g. [0, 2, 5, 6] int32
    indices,  # [PAGES USED BY BATCH], e.g. [5, 7, 2, 9, 11, 14] int32
    last_page_len,  # [BATCH], e.g. [3, 1, 4] int32
    PAGE_SIZE,
    kv_offset=0,
    attn_output_gate=False,
    partial_rotary_factor=1.0,
    mrope_section=(16, 24, 24),
    mrope_interleaved=False,
    has_mrope=False,
    residual=None,
    post_residual_addition=None,
    cache_location=None,
    cos_sin_cache=None,
    banks=None,
):
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)

    B, seq_len, hidden_dim = x.shape
    assert B == 1
    assert seq_len == 1
    q_gate_row_factor = 2 if attn_output_gate else 1
    assert w_q.shape[0] % (q_gate_row_factor * q_norm_w.shape[-1]) == 0
    query_projection_dim = w_q.shape[0] // q_gate_row_factor
    kv_projection_dim = w_k.shape[0]
    head_dim = q_norm_w.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)
    assert hidden_dim % core_num == 0
    assert _PAGE_UNROLL_FACTOR > 0
    assert 0 < partial_rotary_factor <= 1.0
    assert rotary_dim % 2 == 0
    if cos_sin_cache is None:
        assert cos.shape[-1] == sin.shape[-1] == rotary_dim // 2
    else:
        assert cos_sin_cache.shape[-1] == rotary_dim
    if cache_location is None:
        assert kv_offset + 1 <= k_cache_out.shape[1]
        k_cache_out_cur = k_cache_out[:, kv_offset : kv_offset + 1, :, :]
        v_cache_out_cur = v_cache_out[:, kv_offset : kv_offset + 1, :, :]

    query_head_count = query_projection_dim // head_dim
    kv_head_count = kv_projection_dim // head_dim
    query_heads_per_kv = query_head_count // kv_head_count
    assert kv_head_count == 2
    assert query_head_count % core_num == 0
    local_query_head_count = query_head_count // core_num
    assert query_heads_per_kv == local_query_head_count * 2
    local_kv_head_count = 1
    kv_head_start = core_id // 2

    # Publish loaded hidden shards before the shared input normalization.
    hidden_dims_per_core = hidden_dim // core_num
    hidden_slice = slice(core_id * hidden_dims_per_core, (core_id + 1) * hidden_dims_per_core)
    if x.memory_type != ax.MemoryType.DDR and residual is None and post_residual_addition is None:
        x_shocm = x
    else:
        x_shocm = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM)
        load_tile(r=x_shocm[:, :, hidden_slice], x=x[:, :, hidden_slice], banks=banks)
        if residual is not None:
            ax.dsl.add(
                x_shocm[:, :, hidden_slice],
                load_tile(residual[:, :, hidden_slice], banks=banks),
                r=x_shocm[:, :, hidden_slice],
            )
        if post_residual_addition is not None:
            ax.dsl.add(
                x_shocm[:, :, hidden_slice],
                load_tile(post_residual_addition[:, :, hidden_slice], banks=banks),
                r=x_shocm[:, :, hidden_slice],
            )
        axd.pub(x_shocm)
        axd.sub(x_shocm)
    # Core 0 produces the shared normalized token consumed by all projections.
    r_norm_tmp = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM)
    if core_id == 0:
        x_norm_ocm = load_tile(x_norm_w, banks=banks)
        squared = ax.dsl.mul(x_shocm, x_shocm, output_dtype=ax.fp32)
        square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
        hidden_size = ax.dsl.full((1,), dtype=ax.fp32, fill_value=hidden_dim)
        square_mean = ax.dsl.div(square_sum, hidden_size)
        epsilon = ax.dsl.full((1,), dtype=ax.fp32, fill_value=1e-6)
        denominator = ax.dsl.sqrt(ax.dsl.add(square_mean, epsilon))
        normalized = ax.dsl.div(x_shocm, denominator, output_dtype=ax.fp32)
        ax.dsl.mul(normalized, x_norm_ocm, output_dtype=x.dtype, r=r_norm_tmp)
    axd.pub(r_norm_tmp)
    axd.sub(r_norm_tmp)

    # Adjacent cores share one KV head while keeping disjoint query heads.
    query_head_start = core_id * local_query_head_count
    query_head_stop = query_head_start + local_query_head_count
    query_channel_slice = slice(query_head_start * head_dim, query_head_stop * head_dim)
    q_gate_channel_slice = slice(
        query_head_start * q_gate_row_factor * head_dim,
        query_head_stop * q_gate_row_factor * head_dim,
    )
    kv_channel_start = kv_head_start * head_dim
    kv_head_slice = slice(kv_head_start, kv_head_start + local_kv_head_count)
    projected_q_gate = project_token(
        r_norm_tmp,
        w_q,
        w_scale_q,
        w_scale2_q,
        q_gate_channel_slice.start,
        q_gate_channel_slice.stop - q_gate_channel_slice.start,
        "q_gate" if attn_output_gate else "q",
        banks=banks,
    )
    if attn_output_gate:
        projected_q_gate = projected_q_gate.reshape(B, 1, local_query_head_count, 2 * head_dim)
        projected_q = projected_q_gate[..., :head_dim]
        gate = projected_q_gate[..., head_dim:]
    else:
        projected_q = projected_q_gate.reshape(B, 1, local_query_head_count, head_dim)
        gate = None

    # Pair-shared K/V rows: [B,1,1,D] BF16, assembled from disjoint C halves.
    shared_projected_k = [
        ax.dsl.create_tensor(
            (B, 1, 1, head_dim),
            x.dtype,
            ax.MemoryType.SHARED_OCM,
            name=f"shared_projected_k_h{kv_head}",
        )
        for kv_head in range(kv_head_count)
    ]
    shared_projected_v = [
        ax.dsl.create_tensor(
            (B, 1, 1, head_dim),
            x.dtype,
            ax.MemoryType.SHARED_OCM,
            name=f"shared_projected_v_h{kv_head}",
        )
        for kv_head in range(kv_head_count)
    ]
    pair_rank = core_id % 2

    # Both cores load disjoint C halves of their shared KV head. This balances
    # Q/K/V weight bytes across all four cores without duplicating K/V weights.
    half_dim = head_dim // 2
    half_start = pair_rank * half_dim
    for weight, scale, scale2, shared, name in (
        (w_k, w_scale_k, w_scale2_k, shared_projected_k, "k"),
        (w_v, w_scale_v, w_scale2_v, shared_projected_v, "v"),
    ):
        project_token(
            r_norm_tmp,
            weight,
            scale,
            scale2,
            kv_channel_start + half_start,
            half_dim,
            name,
            banks=banks,
            r=shared[kv_head_start][:, :, :, half_start : half_start + half_dim],
        )
    for tensor in (*shared_projected_k, *shared_projected_v):
        axd.pub(tensor)
        axd.sub(tensor)

    projected_k = shared_projected_k[kv_head_start]
    projected_v = shared_projected_v[kv_head_start]

    # RoPE writes each core's query rows directly into their shared destination.
    # The complete Q tensor is also the gather parent for small GQA groups.
    shared_q = ax.dsl.create_tensor(
        (query_head_count, head_dim), x.dtype, ax.MemoryType.SHARED_OCM, name="decode_all_q"
    )
    q = shared_q[query_head_start:query_head_stop, :].reshape(B, 1, local_query_head_count, head_dim)
    k = ax.dsl.create_tensor((B, 1, local_kv_head_count, head_dim), dtype=x.dtype)
    # Normalize Q/K, then gather one shared pair of rotary rows for this token.
    normalized_q = rmsnorm(projected_q, load_tile(q_norm_w, banks=banks))
    normalized_k = rmsnorm(projected_k, load_tile(k_norm_w, banks=banks))

    if core_id % 2 == 0:
        if cache_location is None:
            store_tile(r=v_cache_out_cur[:, :, kv_head_slice, :], x=projected_v, banks=banks)
        else:
            store_paged_kv(
                projected_v.reshape(1, local_kv_head_count, head_dim),
                paged_kv_cache,
                cache_location,
                1,
                kv_head_start,
                banks=banks,
            )

    indptr = load_tile(indptr, banks=banks)
    indices = load_tile(indices, banks=banks)
    last_page_len = load_tile(last_page_len, banks=banks)
    position_ids = load_tile(position_ids, banks=banks)
    one_i32 = ax.dsl.from_numpy(np.ones(shape=(1,), dtype=np.int32))
    page_size_i32 = ax.dsl.from_numpy(np.array([PAGE_SIZE], dtype=np.int32))

    num_pages = indptr[1:2] - indptr[:1]
    cached_len = ax.dsl.maximum(
        (num_pages - one_i32) * page_size_i32 + last_page_len[:1],
        ax.dsl.zeros((1,), dtype=ax.int32),
    )
    rope_cos, rope_sin = gather_rope_cos_sin(
        cos,
        sin,
        position_ids,
        1,
        mrope_section,
        mrope_interleaved,
        has_mrope,
        cos_sin_cache,
        banks=banks,
    )
    rope_impl(
        normalized_q,
        rope_cos,
        rope_sin,
        r=q,
        rotary_dim=rotary_dim,
        banks=banks,
    )
    rope_impl(normalized_k, rope_cos, rope_sin, r=k, rotary_dim=rotary_dim, banks=banks)

    if core_id % 2 == 0:
        if cache_location is None:
            store_tile(r=k_cache_out_cur[:, :, kv_head_slice, :], x=k, banks=banks)
        else:
            store_paged_kv(
                k.reshape(1, local_kv_head_count, head_dim),
                paged_kv_cache,
                cache_location,
                0,
                kv_head_start,
                banks=banks,
            )

    # Adjacent cores split history pages and retain all queries for their KV
    # head (eight rows, one BF16 M block for Qwen3.5). Each page is loaded once.
    axd.pub(shared_q)
    axd.sub(shared_q)
    head_start = kv_head_start * query_heads_per_kv
    selected_q = _select_decode_query_rows(shared_q, head_start, query_heads_per_kv, banks=banks)
    # Snapshot shared-query reads before the cores enter page loops with different bounds.
    pair_q = ax.dsl.create_tensor(selected_q.shape, x.dtype, name="decode_kv_pair_q")
    ax.dsl.move(selected_q, r=pair_q)
    max_state, normalizer, accumulator = paged_attention_history(
        pair_q,
        paged_kv_cache,
        indices,
        indptr[:1],
        indptr[1:2] - indptr[:1],
        PAGE_SIZE,
        kv_head_start,
        valid_kv_len=cached_len,
        real_query_head_count=query_heads_per_kv,
        page_offset=pair_rank,
        banks=banks,
    )
    # Merge the two online-softmax states after both page loops. Every core
    # crosses the same Range boundary, so publications stay in one BODY.
    shared_max = ax.dsl.create_tensor(
        (core_num, query_heads_per_kv),
        x.dtype,
        ax.MemoryType.SHARED_OCM,
        name="decode_pair_max",
    )
    shared_sum = ax.dsl.create_tensor(
        (core_num, query_heads_per_kv),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name="decode_pair_sum",
    )
    shared_acc = ax.dsl.create_tensor(
        (core_num, query_heads_per_kv, head_dim),
        x.dtype,
        ax.MemoryType.SHARED_OCM,
        name="decode_pair_acc",
    )
    ax.dsl.move(max_state.reshape(1, query_heads_per_kv), r=shared_max[core_id : core_id + 1, :])
    ax.dsl.move(normalizer, r=shared_sum[core_id : core_id + 1, :])
    ax.dsl.move(accumulator, r=shared_acc[core_id : core_id + 1, :, :])
    for tensor in (shared_max, shared_sum, shared_acc):
        axd.pub(tensor)
        axd.sub(tensor)
    local_start = pair_rank * local_query_head_count
    local_stop = local_start + local_query_head_count
    core0 = core_id - pair_rank
    m0 = shared_max[core0 : core0 + 1, local_start:local_stop].reshape(local_query_head_count, 1)
    m1 = shared_max[core0 + 1 : core0 + 2, local_start:local_stop].reshape(local_query_head_count, 1)
    maximum = ax.dsl.maximum(m0, m1)
    ratio0 = ax.dsl.exp(ax.dsl.sub(m0, maximum), output_dtype=ax.fp32).reshape(1, local_query_head_count)
    ratio1 = ax.dsl.exp(ax.dsl.sub(m1, maximum), output_dtype=ax.fp32).reshape(1, local_query_head_count)
    normalizer = ax.dsl.add(
        ax.dsl.mul(
            ratio0,
            shared_sum[core0 : core0 + 1, local_start:local_stop],
            output_dtype=ax.fp32,
        ),
        ax.dsl.mul(
            ratio1,
            shared_sum[core0 + 1 : core0 + 2, local_start:local_stop],
            output_dtype=ax.fp32,
        ),
        output_dtype=ax.fp32,
    )
    accumulator = ax.dsl.add(
        ax.dsl.mul(
            ratio0.reshape(1, local_query_head_count, 1),
            shared_acc[core0 : core0 + 1, local_start:local_stop, :],
            output_dtype=ax.fp32,
        ),
        ax.dsl.mul(
            ratio1.reshape(1, local_query_head_count, 1),
            shared_acc[core0 + 1 : core0 + 2, local_start:local_stop, :],
            output_dtype=ax.fp32,
        ),
        output_dtype=x.dtype,
    )
    shared_attn = ax.dsl.create_tensor(
        (B, 1, query_projection_dim), x.dtype, ax.MemoryType.SHARED_OCM, name="decode_all_attn"
    )
    r_attn = shared_attn[:, :, query_channel_slice].reshape(1, local_query_head_count, head_dim)
    finish_paged_attention(
        q.reshape(local_query_head_count, head_dim),
        k.reshape(1, head_dim),
        projected_v.reshape(1, head_dim),
        maximum,
        normalizer,
        accumulator,
        r_attn,
    )
    if gate is not None:
        r_attn = ax.dsl.mul(
            r_attn,
            ax.dsl.sigmoid(gate.reshape(1, local_query_head_count, head_dim), output_dtype=x.dtype),
            output_dtype=x.dtype,
            r=r_attn,
        )

    # All-gather the small attention rows once, then compute disjoint final
    # output channels. Full-IC weight rows need no partial-output reduction.
    axd.pub(shared_attn)
    axd.sub(shared_attn)
    projected_output = project_token(
        shared_attn,
        w_o,
        w_scale_o,
        w_scale2_o,
        hidden_slice.start,
        hidden_dims_per_core,
        "o",
        banks=banks,
    )
    if output.memory_type == ax.MemoryType.DDR:
        bank_sum = ax.dsl.add(x_shocm[:, :, hidden_slice], projected_output, output_dtype=x.dtype)
        store_tile(bank_sum, r=output[:, :, hidden_slice], banks=banks)
    else:
        ax.dsl.add(
            x_shocm[:, :, hidden_slice],
            projected_output,
            output_dtype=x.dtype,
            r=output[:, :, hidden_slice],
        )

    return output, k_cache_out, v_cache_out


ATTENTION_WEIGHT_NAMES = (
    "w_norm_attn",
    "w_q",
    "w_scale_q",
    "w_scale2_q",
    "w_zeropoint_q",
    "w_k",
    "w_scale_k",
    "w_scale2_k",
    "w_zeropoint_k",
    "w_v",
    "w_scale_v",
    "w_scale2_v",
    "w_zeropoint_v",
    "w_norm_q",
    "w_norm_k",
    "cos",
    "sin",
    "w_o",
    "w_scale_o",
    "w_scale2_o",
    "w_zeropoint_o",
)


def flash_attention(
    x,
    weights,
    position_ids,
    output,
    cache,
    indptr,
    indices,
    last_page_len,
    *,
    kv_page_out=None,
    kv_write_page_cursor=None,
    k_cache_out=None,
    v_cache_out=None,
    core_num=int(os.getenv("SGLANG_AXERA_KERNEL_CORE_NUM", "4")),
    attn_output_gate=False,
    partial_rotary_factor=1.0,
    mrope_section=(16, 24, 24),
    mrope_interleaved=False,
    has_mrope=False,
    residual=None,
    post_residual_addition=None,
    cache_location=None,
):
    """Launch one request; all weights and outputs are caller-owned.

    ``x`` has shape [1, tokens, hidden]: one token, 128, or a multiple of 256. The kernel forms
    z = x + residual + post_residual_addition, omitting None inputs, then
    computes Attention(RMSNorm(z)) + z. Already merged inputs use residual=None.
    Projection scales have shape [output_channels, groups], including unit scales.
    Q norm includes attention scaling. RoPE uses either contiguous cos/sin
    tables or cos_sin_cache=[positions,rotary_dim] with cos=sin=None;
    int32 positions contain only this call's valid query tokens, with shape
    [tokens] or [3, tokens] for M-RoPE. Prefill pads positions inside the kernel.
    Cache/page metadata obey the gather floor of 32 rows. Prefill
    writes a full KV page. With cache_location=[page, offset], cores scatter
    K/V directly into the retained cache input. Otherwise callers supply output
    page/slot views. Unwritten history remains intact.
    No allocation, conversion, padding, copy, or output slicing occurs here.
    """
    if x.shape[0] != 1 or x.shape[1] <= 0 or (x.shape[1] not in (1, 128) and x.shape[1] % 256):
        raise ValueError("attention requires one token, 128 tokens, or complete 256-token blocks")
    if output.shape != x.shape:
        raise ValueError("attention output must match the complete input block")
    if (
        position_ids.dtype != torch.int32
        or position_ids.ndim != (2 if has_mrope else 1)
        or (has_mrope and position_ids.shape[0] != 3)
        or not 0 < position_ids.shape[-1] <= x.shape[1]
    ):
        raise ValueError("attention positions must be int32 query positions with shape [tokens] or [3, tokens]")
    if x.shape[1] > 256 and (cache_location is None or position_ids.shape[-1] != x.shape[1]):
        raise ValueError("multi-page prefill requires a cache location and one position per input token")
    if not isinstance(core_num, int) or core_num < 1:
        raise ValueError("core_num must be positive")
    cos_sin_cache = weights.get("cos_sin_cache")
    input_names = ATTENTION_WEIGHT_NAMES
    if cos_sin_cache is not None:
        input_names = (
            *ATTENTION_WEIGHT_NAMES[:15],
            "cos_sin_cache",
            *ATTENTION_WEIGHT_NAMES[17:],
        )
    for name in input_names:
        value = weights[name]
        if not isinstance(value, torch.Tensor) or value.device != x.device:
            raise TypeError(f"{name} must be a runtime tensor on {x.device}")
    args = [
        x,
        *(weights[name] for name in ATTENTION_WEIGHT_NAMES[:17]),
        position_ids,
        *(weights[name] for name in ATTENTION_WEIGHT_NAMES[17:]),
    ]
    options = [
        attn_output_gate,
        partial_rotary_factor,
        mrope_section,
        mrope_interleaved,
        has_mrope,
        residual,
        post_residual_addition,
        cache_location,
        cos_sin_cache,
    ]
    if x.shape[1] > 1:
        if cache_location is None and (kv_page_out is None or kv_write_page_cursor is None):
            raise ValueError("prefill requires a KV page output and write cursor")
        kernel = flash_attn_with_kvcache
        kernel_args = (
            *args,
            kv_page_out,
            output,
            cache,
            indptr,
            indices,
            kv_write_page_cursor,
            *options,
        )
    else:
        if cache_location is None and (k_cache_out is None or v_cache_out is None):
            raise ValueError("decode requires K/V output slots")
        kernel = decode_paged_kv_shared_kv_kernel if cache.shape[3] == 2 else decode_paged_kv_kernel
        kernel_args = (
            *args,
            k_cache_out,
            v_cache_out,
            output,
            cache,
            indptr,
            indices,
            last_page_len,
            BASELINE_PAGE_SIZE,
            0,
            *options,
        )
    kernel[core_num, core_num](*kernel_args)
    return output
