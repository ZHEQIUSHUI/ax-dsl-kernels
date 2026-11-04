"""C128/HCA attention graph helpers."""

from __future__ import annotations

import numpy as np

import axnn.dsl as ax
from axnn.dsl import parallel as axd

from .common import *  # noqa: F401,F403


def hca_compressor(
    hidden,
    kv_weight,
    kv_weight_scale,
    gate_weight,
    gate_weight_scale,
    position_bias,
    norm_weight,
    cos,
    sin,
    meta,
    *,
    compress_rate,
    rotary_dim,
    rms_eps,
):
    """Compress one HCA prefill block with four-core output-channel ownership."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len = meta.batch, meta.seq_len
    width = kv_weight.shape[0]
    assert seq_len == compress_rate
    blocks = 1
    channel_slice = token_slice(core_id, core_num, width)
    local_width = channel_slice.stop - channel_slice.start
    # content: [B,Nc,R,Dc/4], hidden.dtype. HCA pooling is channel-independent,
    # so each core projects and pools one disjoint output-channel slice.
    local_kv_weight_scale = runtime_weight_scale_slice(
        kv_weight,
        kv_weight_scale,
        channel_slice,
        slice(0, hidden.shape[-1]),
    )
    content = runtime_linear(
        hidden,
        kv_weight[channel_slice, :],
        local_kv_weight_scale,
        output_dtype=hidden.dtype,
    ).reshape(batch, blocks, compress_rate, local_width)
    # gate: [B,Nc,R,Dc/4], hidden.dtype before position bias.
    local_gate_weight_scale = runtime_weight_scale_slice(
        gate_weight,
        gate_weight_scale,
        channel_slice,
        slice(0, hidden.shape[-1]),
    )
    gate = runtime_linear(
        hidden,
        gate_weight[channel_slice, :],
        local_gate_weight_scale,
        output_dtype=hidden.dtype,
    ).reshape(batch, blocks, compress_rate, local_width)
    # gate: [B,Nc,R,Dc/4], FP32.
    gate = ax.dsl.add(
        gate,
        ax.dsl.load(position_bias[:, channel_slice]),
        output_dtype=ax.fp32,
    )
    # probability/weighted_content: [B,Nc,R,Dc/4], FP32.
    probability = ax.dsl.softmax(gate, dim=2, output_dtype=ax.fp32)
    # weighted_content: [B,Nc,R,Dc/4], FP32.
    weighted_content = ax.dsl.mul(content, probability, output_dtype=ax.fp32)
    # pooled_shared: [B,Nc,Dc], FP32 SHARED_OCM. Each core publishes one
    # channel slice; the only cross-channel operation is the following RMSNorm.
    pooled_shared = ax.dsl.create_tensor(
        (batch, blocks, width),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name="hca_pooled",
    )
    # pooled_shared channel slice: [B,Nc,Dc/4], FP32.
    ax.dsl.sum(
        weighted_content,
        dim=2,
        output_dtype=ax.fp32,
        r=pooled_shared[:, :, channel_slice],
    )
    # compressed_shared: [B,Nc,Dc], hidden.dtype SHARED_OCM.
    compressed_shared = ax.dsl.create_tensor(
        (batch, blocks, width),
        hidden.dtype,
        ax.MemoryType.SHARED_OCM,
        name="hca_compressed",
    )
    # Synchronization only; no tensor output.
    axd.sync_all()
    if core_id == 0:
        # Core 0 performs the comparatively small cross-channel normalization
        # and RoPE after the four projection/pooling shards are visible.
        pooled = rmsnorm(
            pooled_shared,
            ax.dsl.load(norm_weight),
            eps=rms_eps,
            output_dtype=hidden.dtype,
        )
        tail_rope(
            pooled,
            cos[:, :1, :],
            sin[:, :1, :],
            rotary_dim,
            r=compressed_shared,
        )
    return compressed_shared


def hca_decode_compressor_update(
    hidden,
    positions_casual,
    content_state,
    gate_state,
    kv_weight,
    kv_weight_scale,
    gate_weight,
    gate_weight_scale,
    position_bias,
    norm_weight,
    compress_cos,
    compress_sin,
    meta,
    *,
    compress_rate,
    rotary_dim,
    rms_eps,
):
    """Append one token to the persistent HCA block and pool its current candidate."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len = meta.batch, meta.seq_len
    width = kv_weight.shape[0]
    assert batch == seq_len == 1
    assert content_state.shape == (batch, compress_rate, width)
    assert gate_state.shape == content_state.shape

    channel_slice = token_slice(core_id, core_num, width)
    local_width = channel_slice.stop - channel_slice.start
    local_kv_weight_scale = runtime_weight_scale_slice(
        kv_weight,
        kv_weight_scale,
        channel_slice,
        slice(0, hidden.shape[-1]),
    )
    content = runtime_linear(
        hidden,
        kv_weight[channel_slice, :],
        local_kv_weight_scale,
        output_dtype=hidden.dtype,
    ).reshape(1, local_width)
    local_gate_weight_scale = runtime_weight_scale_slice(
        gate_weight,
        gate_weight_scale,
        channel_slice,
        slice(0, hidden.shape[-1]),
    )
    gate = runtime_linear(
        hidden,
        gate_weight[channel_slice, :],
        local_gate_weight_scale,
        output_dtype=hidden.dtype,
    ).reshape(1, local_width)

    # position: [1], INT32.
    position = ax.dsl.load(positions_casual).reshape(1)
    rate_i32 = constant(np.array([compress_rate], dtype=np.int32), "hca_decode_rate")
    # block_index: [1], INT32.
    block_index = ax.dsl.div(
        position,
        rate_i32,
        output_dtype=ax.int32,
        output_rounding=1,
    )
    # block_start: [1], INT32.
    block_start = ax.dsl.mul(block_index, rate_i32, output_dtype=ax.int32)
    # phase: [1], INT32 in [0,R).
    phase = ax.dsl.sub(position, block_start, output_dtype=ax.int32)
    # local_bias: [1,Dc/4], position_bias.dtype.
    local_bias = ax.dsl.index_select(
        ax.dsl.load(position_bias[:, channel_slice]),
        phase,
        dim=0,
    )
    # gate: [1,Dc/4], FP32.
    gate = ax.dsl.add(gate, local_bias, output_dtype=ax.fp32)

    local_content_state = content_state[:, :, channel_slice].reshape(compress_rate, local_width)
    local_gate_state = gate_state[:, :, channel_slice].reshape(compress_rate, local_width)
    # local_content_state selected row: [1,Dc/4], content.dtype.
    ax.dsl.select_scatter(content, phase, dim=0, r=local_content_state)
    # local_gate_state selected row: [1,Dc/4], gate.dtype.
    ax.dsl.select_scatter(gate, phase, dim=0, r=local_gate_state)

    # content_block: [1,R,Dc/4], content_state.dtype.
    content_block = ax.dsl.load(local_content_state).reshape(1, compress_rate, local_width)
    # gate_block: [1,R,Dc/4], gate_state.dtype.
    gate_block = ax.dsl.load(local_gate_state).reshape(1, compress_rate, local_width)
    # probability: [1,R,Dc/4], FP32.
    probability = ax.dsl.softmax(gate_block, dim=1, output_dtype=ax.fp32)
    # weighted_content: [1,R,Dc/4], FP32.
    weighted_content = ax.dsl.mul(content_block, probability, output_dtype=ax.fp32)
    # pooled_shared: [B,1,Dc], FP32 SHARED_OCM.
    pooled_shared = ax.dsl.create_tensor(
        (batch, 1, width),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name="hca_decode_pooled",
    )
    # pooled_shared channel slice: [B,1,Dc/4], FP32.
    ax.dsl.sum(
        weighted_content,
        dim=1,
        keepdim=True,
        output_dtype=ax.fp32,
        r=pooled_shared[:, :, channel_slice],
    )
    # compressed_shared: [B,1,Dc], hidden.dtype SHARED_OCM.
    compressed_shared = ax.dsl.create_tensor(
        (batch, 1, width),
        hidden.dtype,
        ax.MemoryType.SHARED_OCM,
        name="hca_decode_compressed",
    )
    # Synchronization only; no tensor output.
    axd.sync_all()
    if core_id == 0:
        pooled = rmsnorm(
            pooled_shared,
            ax.dsl.load(norm_weight),
            eps=rms_eps,
            output_dtype=hidden.dtype,
        )
        # compress_cos_local: [1,1,Dr/2], compress_cos.dtype.
        compress_cos_local = ax.dsl.load(compress_cos)
        # compress_sin_local: [1,1,Dr/2], compress_sin.dtype.
        compress_sin_local = ax.dsl.load(compress_sin)
        tail_rope(
            pooled,
            compress_cos_local,
            compress_sin_local,
            rotary_dim,
            r=compressed_shared,
        )
    return compressed_shared


def write_hca_short_state(
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
    """Persist raw C128 content and gate projections for an incomplete block.

    C128 state uses one 128-token ring per request.  The state pool stores the
    un-biased compressor projections as ``[content, gate]``; APE, pooling,
    normalization, RoPE, and the compressed-cache write are performed only
    when a complete block is available.  Each physical core owns a disjoint
    output-channel slice and core 0 performs the indexed row scatter after the
    slice barrier.
    """
    projection_width = kv_weight.shape[0]
    hidden = normalized.shape[-1]
    channel_slice = token_slice(meta.core_id, meta.core_num, projection_width)
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


def head_owned_online_attention_tile(
    query,
    source_kv,
    additive_mask,
    scale,
    max_state,
    normalizer_state,
    accumulator_state,
    *,
    query_seq_tile=ATTENTION_QUERY_SEQ_TILE,
):
    """Merge one common source tile into head-owned online-softmax state."""
    batch, seq_len, heads, head_dim = query.shape
    source_len = source_kv.shape[1]
    # Return fresh state tensors instead of writing through views of the range
    # operands.  AxRange carries its returned tensors directly; relying on the
    # slice-write rewrite for a large q128 multi-head state is not stable.
    # next_max_state: max_state.shape/dtype private OCM.
    next_max_state = ax.dsl.create_tensor(max_state.shape, max_state.dtype)
    # next_normalizer_state: normalizer_state.shape/dtype private OCM.
    next_normalizer_state = ax.dsl.create_tensor(normalizer_state.shape, normalizer_state.dtype)
    # next_accumulator_state: accumulator_state.shape/dtype private OCM.
    next_accumulator_state = ax.dsl.create_tensor(accumulator_state.shape, accumulator_state.dtype)
    # source_head_first: [B,1,T_source,Dh], query.dtype. Every local head
    # reuses the same HCA compressed or SWA source rows.
    source_head_first = source_kv.reshape(batch, 1, source_len, head_dim)
    for query_start in range(0, seq_len, query_seq_tile):
        query_stop = min(query_start + query_seq_tile, seq_len)
        query_slice = slice(query_start, query_stop)
        query_tokens = query_stop - query_start
        # mask_head_first: [B,1,T_query,T_source], FP32.
        mask_head_first = additive_mask[:, query_slice, :].reshape(batch, 1, query_tokens, source_len)
        # The head loop is deliberately static: source is the dynamic outer
        # tile, while every range trip updates all head-owned state slices.
        for head_id in range(heads):
            head_slice = slice(head_id, head_id + 1)
            # query_head_first: [B,1,T_query,Dh], query.dtype.
            query_head_first = ax.dsl.transpose(
                query[:, query_slice, head_slice, :],
                perm=(0, 2, 1, 3),
            )
            # scores: [B,1,T_query,T_source], FP32.
            scores = ax.dsl.matmul(
                query_head_first,
                source_head_first,
                transposed=True,
                output_dtype=ax.fp32,
            )
            # scores: [B,1,T_query,T_source], FP32 after attention scaling.
            scores = ax.dsl.mul(scores, scale, output_dtype=ax.fp32)
            # scores: [B,1,T_query,T_source], FP32 after the additive mask.
            scores = ax.dsl.add(scores, mask_head_first, output_dtype=ax.fp32)

            old_max = max_state[:, head_slice, query_slice, :]
            old_normalizer = normalizer_state[:, head_slice, query_slice, :]
            old_accumulator = accumulator_state[:, head_slice, query_slice, :]
            # tile_max/next_max/old_scale: [B,1,T_query,1], FP32.
            tile_max = ax.dsl.max(scores, dim=-1, keepdim=True)
            # next_max: [B,1,T_query,1], FP32.
            next_max = ax.dsl.maximum(old_max, tile_max, output_dtype=ax.fp32)
            # old_shift: [B,1,T_query,1], FP32.
            old_shift = ax.dsl.sub(old_max, next_max, output_dtype=ax.fp32)
            # old_scale: [B,1,T_query,1], FP32.
            old_scale = ax.dsl.exp(old_shift, output_dtype=ax.fp32)
            # probability: [B,1,T_query,T_source], FP32 unnormalized exponentials.
            shifted_scores = ax.dsl.sub(scores, next_max, output_dtype=ax.fp32)
            # probability: [B,1,T_query,T_source], FP32.
            probability = ax.dsl.exp(shifted_scores, output_dtype=ax.fp32)
            # next_normalizer: [B,1,T_query,1], FP32.
            tile_sum = ax.dsl.sum(probability, dim=-1, keepdim=True, output_dtype=ax.fp32)
            # scaled_normalizer: [B,1,T_query,1], FP32.
            scaled_normalizer = ax.dsl.mul(old_normalizer, old_scale, output_dtype=ax.fp32)
            # next_normalizer_state slice: [B,1,T_query,1], FP32.
            ax.dsl.add(
                scaled_normalizer,
                tile_sum,
                output_dtype=ax.fp32,
                r=next_normalizer_state[:, head_slice, query_slice, :],
            )

            # AX8860 PV MatMul requires BF16 probabilities. The accumulator's
            # dtype is selected by the caller: FP32 for small states, BF16 for
            # production q128 x h8 so one range carry stays within 1 MiB.
            # value_probability: [B,1,T_query,T_source], query.dtype.
            value_probability = ax.dsl.cast(probability, dtype=query.dtype)
            # tile_output: [B,1,T_query,Dh], accumulator_state.dtype.
            tile_output = ax.dsl.matmul(
                value_probability,
                source_head_first,
                output_dtype=accumulator_state.dtype,
            )
            # scaled_accumulator: [B,1,T_query,Dh], accumulator_state.dtype.
            scaled_accumulator = ax.dsl.mul(
                old_accumulator,
                old_scale,
                output_dtype=accumulator_state.dtype,
            )
            # next_accumulator_state slice: [B,1,T_query,Dh], accumulator_state.dtype.
            ax.dsl.add(
                scaled_accumulator,
                tile_output,
                output_dtype=accumulator_state.dtype,
                r=next_accumulator_state[:, head_slice, query_slice, :],
            )
            # next_max_state slice: [B,1,T_query,1], FP32.
            ax.dsl.move(next_max, r=next_max_state[:, head_slice, query_slice, :])
    return next_max_state, next_normalizer_state, next_accumulator_state


def hca_paged_history_online_states(
    query,
    c128_kv_cache,
    c128_page_indices,
    c128_topk_lengths_clamp1,
    positions_casual,
    scale,
    max_state,
    normalizer_state,
    accumulator_state,
    *,
    compress_rate,
    source_chunk_size,
    source_query_index,
    query_seq_tile=ATTENTION_QUERY_SEQ_TILE,
):
    """Iterate common-prefix C128 rows once and update every head-owned state."""
    batch, seq_len, _, head_dim = query.shape
    assert batch == 1
    index_capacity = c128_page_indices.shape[1]
    padded_index_capacity = (index_capacity + source_chunk_size - 1) // source_chunk_size * source_chunk_size
    index_padding = padded_index_capacity - index_capacity

    # HCA's per-query C128 selections are nested prefixes. Load the final row
    # once per head wave, then use each query's runtime length as its mask.
    # history_indices: [T_index], INT32.
    history_indices = ax.dsl.load(c128_page_indices[source_query_index : source_query_index + 1, :]).reshape(
        index_capacity
    )
    if index_padding:
        # history_indices: [T_index_aligned], INT32 with -1 padding.
        history_indices = ax.dsl.pad(
            history_indices,
            padding=(0, index_padding),
            value=-1,
        )
    # SGLang clamps this metadata to one so the range always executes at
    # least once. It is not the causal C128 length: before the first complete
    # C128 block the sole page index is -1 and every history row must remain
    # masked. Derive the semantic length from the absolute token position.
    # positions: [S], INT32 absolute token positions.
    positions = ax.dsl.load(positions_casual).reshape(seq_len)
    one_i32 = constant(np.array([1], dtype=np.int32), "hca_history_one")
    compress_rate_i32 = constant(
        np.array([compress_rate], dtype=np.int32),
        "hca_compress_rate",
    )
    # position_ends: [S], INT32.
    position_ends = ax.dsl.add(positions, one_i32, output_dtype=ax.int32)
    # history_lengths: [S], INT32 semantic completed-C128 counts.
    history_lengths = ax.dsl.div(
        position_ends,
        compress_rate_i32,
        output_dtype=ax.int32,
        output_rounding=1,
    )

    chunk_size = constant(
        np.array([source_chunk_size], dtype=np.int32),
        "hca_history_chunk_size",
    )
    chunk_offsets = constant(
        np.arange(source_chunk_size, dtype=np.int32),
        "hca_history_chunk_offsets",
    )
    # range_start: [1], INT32.
    range_start = ax.dsl.zeros((1,), ax.int32)
    # zero_i32: [1], INT32.
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    # negative_bias: [1], FP32.
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)

    def source_loop(
        tile_index,
        loop_query,
        loop_cache,
        loop_indices,
        loop_lengths,
        loop_scale,
        loop_chunk_size,
        loop_chunk_offsets,
        loop_zero,
        loop_one,
        loop_negative_bias,
        loop_max,
        loop_normalizer,
        loop_accumulator,
    ):
        # source_offset/logical_positions: [1]/[T_source], INT32.
        source_offset = ax.dsl.mul(tile_index, loop_chunk_size, output_dtype=ax.int32)
        # logical_positions: [T_source], INT32.
        logical_positions = ax.dsl.add(
            loop_chunk_offsets,
            source_offset,
            output_dtype=ax.int32,
        )
        # physical_indices: [T_source], INT32.
        physical_indices = ax.dsl.index_select(loop_indices, logical_positions, dim=0)
        # safe_indices: [T_source], INT32.
        safe_indices = ax.dsl.maximum(
            physical_indices,
            loop_zero,
            output_dtype=ax.int32,
        )

        # history_source: [B,T_source,Dh], query.dtype private OCM; production
        # [1,128,512] BF16. A cooperative
        # SHOCM gather with barriers inside this range passes small CRUNs but
        # deadlocks the production head body, so every head-owning core loads
        # the common source independently until the collective path is fixed.
        history_source = ax.dsl.create_tensor(
            (batch, source_chunk_size, head_dim),
            query.dtype,
            ax.MemoryType.OCM,
            name="hca_history_source_tile",
        )
        # history_source: [B,T_source,Dh], query.dtype.
        ax.dsl.index_select(
            loop_cache,
            safe_indices,
            dim=0,
            r=history_source.reshape(source_chunk_size, head_dim),
        )

        # history_mask: [B,S,T_source], FP32. Padded page-index rows are
        # redirected to cache row zero and remain inactive by runtime length.
        remaining = ax.dsl.sub(
            loop_lengths.reshape(seq_len, 1),
            logical_positions.reshape(1, source_chunk_size),
            output_dtype=ax.int32,
        )
        active = positive_i32(remaining, loop_zero, loop_one)
        # inactive: [S,T_source], INT32.
        inactive = ax.dsl.sub(loop_one, active, output_dtype=ax.int32)
        # history_mask: [B,S,T_source], FP32.
        history_mask = ax.dsl.mul(
            inactive,
            loop_negative_bias,
            output_dtype=ax.fp32,
        ).reshape(batch, seq_len, source_chunk_size)
        next_max, next_normalizer, next_accumulator = head_owned_online_attention_tile(
            loop_query,
            history_source,
            history_mask,
            loop_scale,
            loop_max,
            loop_normalizer,
            loop_accumulator,
            query_seq_tile=query_seq_tile,
        )
        return (
            loop_query,
            loop_cache,
            loop_indices,
            loop_lengths,
            loop_scale,
            loop_chunk_size,
            loop_chunk_offsets,
            loop_zero,
            loop_one,
            loop_negative_bias,
            next_max,
            next_normalizer,
            next_accumulator,
        )

    # A cache that fits one source tile has exactly one semantic iteration.
    # Recording it as static dataflow also avoids carrying the large online
    # state through AX8860's dynamic dispatcher for the common 4K context.
    if index_capacity <= source_chunk_size:
        return source_loop(
            range_start,
            query,
            c128_kv_cache.reshape(-1, head_dim),
            history_indices,
            history_lengths,
            scale,
            chunk_size,
            chunk_offsets,
            zero_i32,
            one_i32,
            negative_bias,
            max_state,
            normalizer_state,
            accumulator_state,
        )[-3:]

    # loop_lengths: [S], INT32 clamped runtime C128 lengths. Only a dynamic
    # multi-tile traversal needs the runtime loop bound.
    loop_lengths = ax.dsl.load(c128_topk_lengths_clamp1).reshape(seq_len)
    # max_loop_length: [1], INT32.
    max_loop_length = ax.dsl.max(loop_lengths, dim=0, keepdim=True)
    # range_stop: [1], INT32 number of source tiles.
    range_stop = ax.dsl.div(
        max_loop_length,
        chunk_size,
        output_dtype=ax.int32,
        output_rounding=2,
    )
    # range_index/range_step: [1], INT32.
    range_index = ax.dsl.zeros((1,), ax.int32)
    range_step = constant(np.array([1], dtype=np.int32), "hca_history_step")
    # results[-3:]: max_state.shape FP32, normalizer_state.shape FP32, and
    # accumulator_state.shape/accumulator_state.dtype.
    results = ax.dsl.range(
        range_index,
        range_start,
        range_stop,
        range_step,
        source_loop,
        query,
        c128_kv_cache.reshape(-1, head_dim),
        history_indices,
        history_lengths,
        scale,
        chunk_size,
        chunk_offsets,
        zero_i32,
        one_i32,
        negative_bias,
        max_state,
        normalizer_state,
        accumulator_state,
        unroll_factor=1,
    )
    return results[-3:]


def hca_attention_paged(
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
    *,
    compress_rate=128,
    window=128,
    history_source_chunk_size=128,
):
    """Run head-owned dynamic C128 attention followed by one static SWA tile."""
    core_id = meta.core_id
    batch, seq_len, heads_per_core, head_dim = query.shape
    sink_slice = sinks_chip[core_id * heads_per_core : (core_id + 1) * heads_per_core]
    # scale: [1], FP32.
    scale = ax.dsl.full((1,), ax.fp32, fill_value=head_dim**-0.5)
    sink = ax.dsl.load(sink_slice).reshape(1, heads_per_core, 1, 1)
    # context: query.shape/dtype private OCM.
    context = ax.dsl.create_tensor(
        query.shape,
        query.dtype,
        ax.MemoryType.OCM,
        name="hca_attention_context",
    )
    # Keep the complete query tile (128 rows in production) and every locally
    # owned head in one online-softmax state. Only source tiles are dynamic.
    # max_state: [B,Hq_core,S,1], FP32.
    max_state = ax.dsl.expand(
        sink,
        shape=(batch, heads_per_core, seq_len, 1),
    )
    # normalizer_state: [B,Hq_core,S,1], FP32.
    normalizer_state = ax.dsl.full(
        (batch, heads_per_core, seq_len, 1),
        ax.fp32,
        fill_value=1.0,
    )
    # accumulator_state: [B,Hq_core,S,Dh], query.dtype.
    accumulator_state = ax.dsl.zeros(
        (batch, heads_per_core, seq_len, head_dim),
        query.dtype,
    )
    max_state, normalizer_state, accumulator_state = hca_paged_history_online_states(
        query,
        c128_kv_cache,
        c128_page_indices,
        c128_topk_lengths_clamp1,
        positions_casual,
        scale,
        max_state,
        normalizer_state,
        accumulator_state,
        compress_rate=compress_rate,
        source_chunk_size=history_source_chunk_size,
        source_query_index=seq_len - 1,
        query_seq_tile=seq_len,
    )
    swa_source, swa_mask = paged_swa_source(
        prefix,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
        name_prefix="hca_swa",
        mask_query_slice=slice(0, seq_len),
        source_alignment=history_source_chunk_size,
        cooperative=False,
    )
    max_state, normalizer_state, accumulator_state = head_owned_online_attention_tile(
        query,
        swa_source,
        swa_mask,
        scale,
        max_state,
        normalizer_state,
        accumulator_state,
        query_seq_tile=seq_len,
    )
    # normalized_head_first: [B,Hq_core,S,Dh], query.dtype.
    normalized_head_first = ax.dsl.div(
        accumulator_state,
        normalizer_state,
        output_dtype=query.dtype,
    )
    # context: [B,S,Hq_core,Dh], query.dtype.
    ax.dsl.transpose(
        normalized_head_first,
        perm=(0, 2, 1, 3),
        r=context,
    )
    return context
