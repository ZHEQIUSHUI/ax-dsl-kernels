"""Ratio-4 CSA and Lightning Indexer graph helpers."""

from __future__ import annotations

import numpy as np

import axnn.dsl as ax
from axnn.dsl import parallel as axd

from .common import *  # noqa: F401,F403


def stride4_compressor(
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
    rotary_dim,
    rms_eps,
    name_prefix,
    previous_content_state,
    previous_gate_state,
    next_content_state,
    next_gate_state,
):
    """Compress the query-owned stride-4 blocks for one core."""
    batch, seq_len = meta.batch, meta.seq_len
    width = kv_weight.shape[0] // 2
    blocks = seq_len // 4
    assert blocks % meta.core_num == 0
    block_slice = token_slice(meta.core_id, meta.core_num, blocks)
    local_blocks = block_slice.stop - block_slice.start
    assert local_blocks >= 2

    # Every local output needs the right projection of its current block and the
    # left projection of its predecessor.  Nonzero cores recompute one boundary
    # block so the compressors remain private and need no intermediate barrier.
    projection_block_start = max(0, block_slice.start - 1)
    projection_block_stop = block_slice.stop
    projection_blocks = projection_block_stop - projection_block_start
    projectiontoken_slice = slice(projection_block_start * 4, projection_block_stop * 4)
    # projected: [B,N_project,4,2*Dc], hidden.dtype (BF16 in production).
    projected = runtime_linear(
        hidden[:, projectiontoken_slice, :],
        kv_weight,
        kv_weight_scale,
        output_dtype=hidden.dtype,
    ).reshape(batch, projection_blocks, 4, 2 * width)
    # gate: [B,N_project,4,2*Dc], hidden.dtype before position bias.
    gate = runtime_linear(
        hidden[:, projectiontoken_slice, :],
        gate_weight,
        gate_weight_scale,
        output_dtype=hidden.dtype,
    ).reshape(batch, projection_blocks, 4, 2 * width)
    # gate: [B,N_project,4,2*Dc], FP32.
    gate = ax.dsl.add(
        gate,
        ax.dsl.load(position_bias),
        output_dtype=ax.fp32,
    )

    if block_slice.start == 0:
        # previous_content_head: [B,1,4,Dc], hidden.dtype.
        previous_content_head = ax.dsl.load(previous_content_state).reshape(batch, 1, 4, width)
        # previous_gate_head: [B,1,4,Dc], FP32.
        previous_gate_head = ax.dsl.load(previous_gate_state).reshape(batch, 1, 4, width)
        # previous_content: [B,N_local,4,Dc], hidden.dtype.
        previous_content = ax.dsl.concat(
            previous_content_head,
            projected[:, : local_blocks - 1, :, :width],
            dim=1,
        )
        # previous_gate: [B,N_local,4,Dc], FP32.
        previous_gate = ax.dsl.concat(
            previous_gate_head,
            gate[:, : local_blocks - 1, :, :width],
            dim=1,
        )
        current_content = projected[:, :local_blocks, :, width:]
        current_gate = gate[:, :local_blocks, :, width:]
    else:
        # The first projected block is the predecessor duplicated at this core
        # boundary; the remaining blocks are this core's current-query owners.
        previous_content = projected[:, :local_blocks, :, :width]
        previous_gate = gate[:, :local_blocks, :, :width]
        current_content = projected[:, 1 : local_blocks + 1, :, width:]
        current_gate = gate[:, 1 : local_blocks + 1, :, width:]

    # window: [B,N_local,8,Dc], hidden.dtype.
    window = ax.dsl.concat(previous_content, current_content, dim=2)
    # gate_window: [B,N_local,8,Dc], FP32.
    gate_window = ax.dsl.concat(previous_gate, current_gate, dim=2)
    # probability: [B,N_local,8,Dc], FP32.
    probability = ax.dsl.softmax(gate_window, dim=2, output_dtype=ax.fp32)
    # weighted_window: [B,N_local,8,Dc], FP32.
    weighted_window = ax.dsl.mul(window, probability, output_dtype=ax.fp32)
    # pooled: [B,N_local,Dc], FP32.
    pooled = ax.dsl.sum(weighted_window, dim=2, output_dtype=ax.fp32)
    pooled = rmsnorm(
        pooled,
        ax.dsl.load(norm_weight),
        eps=rms_eps,
        output_dtype=hidden.dtype,
    )
    block_positions = constant(
        np.arange(block_slice.start * 4, block_slice.stop * 4, 4, dtype=np.uint32),
        f"{name_prefix}_block_positions",
    )
    # block_cos: [B,N_local,Dr/2], cos.dtype (BF16 in production).
    block_cos = ax.dsl.index_select(cos, block_positions, dim=1)
    # block_sin: [B,N_local,Dr/2], sin.dtype (BF16 in production).
    block_sin = ax.dsl.index_select(sin, block_positions, dim=1)
    # compressed: [B,N_local,Dc], hidden.dtype.
    compressed = tail_rope(pooled, block_cos, block_sin, rotary_dim)
    if block_slice.stop == blocks:
        # The next chunk's first C4 entry overlaps this chunk's final block.
        # Publish its left projection and gate while they are still live. The
        # caller synchronizes and keeps core 0 as the sole DDR state writer.
        # next_content_state: previous_content_state.shape/dtype.
        ax.dsl.move(
            projected[:, projection_blocks - 1 : projection_blocks, :, :width].reshape(*previous_content_state.shape),
            r=next_content_state,
        )
        # next_gate_state: previous_gate_state.shape/dtype.
        ax.dsl.move(
            gate[:, projection_blocks - 1 : projection_blocks, :, :width].reshape(*previous_gate_state.shape),
            r=next_gate_state,
        )
    return compressed, block_slice


def normalized_hadamard(x, name):
    """Record the orthonormal Walsh-Hadamard rotation used by the Indexer."""
    width = x.shape[-1]
    rows = int(np.prod(x.shape[:-1]))
    current = x.reshape(rows, width)
    half_width = 1
    stage = 0
    while half_width < width:
        group_count = width // (2 * half_width)
        grouped = current.reshape(rows, group_count, 2, half_width)
        left = grouped[:, :, 0:1, :].reshape(rows, group_count, half_width)
        right = grouped[:, :, 1:2, :].reshape(rows, group_count, half_width)
        next_value = ax.dsl.create_tensor(
            (rows, group_count, 2, half_width),
            x.dtype,
            name=f"{name}_stage{stage}",
        )
        ax.dsl.add(
            left,
            right,
            output_dtype=x.dtype,
            r=next_value[:, :, 0:1, :].reshape(*left.shape),
        )
        ax.dsl.sub(
            left,
            right,
            output_dtype=x.dtype,
            r=next_value[:, :, 1:2, :].reshape(*left.shape),
        )
        current = next_value.reshape(rows, width)
        half_width *= 2
        stage += 1

    scale = ax.dsl.full((1,), ax.fp32, fill_value=float(width**-0.5))
    return ax.dsl.mul(current, scale, output_dtype=x.dtype).reshape(*x.shape)


def quantize_index_int8_rows(x, name):
    """Dynamically quantize last-axis rows to signed INT8 and FP16 scales."""
    rows = int(np.prod(x.shape[:-1]))
    width = x.shape[-1]
    x_fp32 = ax.dsl.cast(x.reshape(rows, width), dtype=ax.fp32)
    negative_one = ax.dsl.full((1,), ax.fp32, fill_value=-1.0)
    abs_x = ax.dsl.maximum(
        x_fp32,
        ax.dsl.mul(x_fp32, negative_one, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    max_abs = ax.dsl.max(abs_x, dim=-1, keepdim=True)
    epsilon = ax.dsl.full(max_abs.shape, ax.fp32, fill_value=1e-12)
    max_abs = ax.dsl.maximum(max_abs, epsilon, output_dtype=ax.fp32)
    quant_max = ax.dsl.full(max_abs.shape, ax.fp32, fill_value=127.0)
    scale_fp32 = ax.dsl.div(max_abs, quant_max, output_dtype=ax.fp32)
    scaled = ax.dsl.div(x_fp32, scale_fp32, output_dtype=ax.fp32)
    lower = ax.dsl.full(scaled.shape, ax.fp32, fill_value=-128.0)
    upper = ax.dsl.full(scaled.shape, ax.fp32, fill_value=127.0)
    scaled = ax.dsl.minimum(
        ax.dsl.maximum(scaled, lower, output_dtype=ax.fp32),
        upper,
        output_dtype=ax.fp32,
    )
    quantized = ax.dsl.create_tensor((rows, width), ax.int8, name=f"{name}_int8")
    ax.dsl.cast(scaled, dtype=ax.int8, r=quantized)
    scale = ax.dsl.cast(scale_fp32, dtype=ax.fp16)
    return (
        quantized.reshape(*x.shape),
        scale.reshape(*(x.shape[:-1] + (1,))),
    )


def stride4_state_closure(
    state_cache,
    state_read_loc,
    position_bias,
    norm_weight,
    cos,
    sin,
    *,
    rotary_dim,
    rms_eps,
    name_prefix,
):
    """Close one ratio-4 overlap group from eight caller-resolved state rows."""
    projection_width = state_cache.shape[-1] // 2
    width = projection_width // 2
    raw_rows = ax.dsl.create_tensor(
        (8, 2 * projection_width),
        state_cache.dtype,
        ax.MemoryType.OCM,
        name=f"{name_prefix}_state_rows",
    )
    ax.dsl.index_select(
        state_cache.reshape(-1, 2 * projection_width),
        ax.dsl.load(state_read_loc),
        dim=0,
        r=raw_rows,
    )
    content_rows = raw_rows[:, :projection_width]
    position_bias_value = ax.dsl.load(position_bias).reshape(4, projection_width)
    # The eight-row overlap window is two consecutive four-token projection
    # groups. Both halves use the same checkpoint APE before selecting the
    # predecessor's left projection and the current group's right projection.
    position_bias_rows = ax.dsl.concat(
        position_bias_value,
        position_bias_value,
        dim=0,
    )
    score_rows = ax.dsl.add(
        raw_rows[:, projection_width:],
        position_bias_rows,
        output_dtype=ax.fp32,
    )
    pooled_content = ax.dsl.concat(
        content_rows[:4, :width],
        content_rows[4:, width:],
        dim=0,
    )
    pooled_score = ax.dsl.concat(
        score_rows[:4, :width],
        score_rows[4:, width:],
        dim=0,
    )
    probability = ax.dsl.softmax(pooled_score, dim=0, output_dtype=ax.fp32)
    pooled = ax.dsl.sum(
        ax.dsl.mul(pooled_content, probability, output_dtype=ax.fp32),
        dim=0,
        keepdim=True,
        output_dtype=ax.fp32,
    )
    normalized = rmsnorm(
        pooled,
        ax.dsl.load(norm_weight),
        eps=rms_eps,
        output_dtype=state_cache.dtype,
    ).reshape(1, 1, width)
    # Keep the small compressed RoPE tables in local memory before the pairwise
    # multiplies.  Continuation receives these tensors as DDR kernel inputs;
    # letting AxMul consume them directly can exhaust the backend's DDR I/O EU
    # budget while compiling the second (sine) branch.
    cos_local = ax.dsl.load(cos)
    sin_local = ax.dsl.load(sin)
    return tail_rope(normalized, cos_local, sin_local, rotary_dim).reshape(1, width)


def csa_paged_history(swa_kv_cache, swa_page_indices, swa_topk_lengths, meta, *, window):
    """Publish right-aligned CSA history and padding; the caller supplies the collective."""
    batch = meta.batch
    head_dim = swa_kv_cache.shape[-1]
    history_capacity = window - 1
    history_slice = token_slice(meta.core_id, meta.core_num, history_capacity)
    local_history = history_slice.stop - history_slice.start
    assert batch == 1

    # history_source: [B,W-1,Dh], cache dtype SHARED_OCM. Each core publishes
    # a disjoint history-row slice gathered from the paged DDR cache.
    history_source = ax.dsl.create_tensor(
        (batch, history_capacity, head_dim),
        swa_kv_cache.dtype,
        ax.MemoryType.SHARED_OCM,
        name="csa_swa_history",
    )
    # history_mask: [W-1], FP32 SHARED_OCM. Only early-context padding is
    # masked; selected history/current rows are causal and in-window by construction.
    history_mask = ax.dsl.create_tensor(
        (history_capacity,),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name="csa_swa_history_mask",
    )
    assert local_history > 0
    # zero_i32: [1], INT32.
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    one_i32 = constant(np.array([1], dtype=np.int32), "csa_swa_history_one")
    # The first query length includes its current token, so the preceding
    # history has one fewer valid entry.
    # first_swa_length: [1], INT32.
    first_swa_length = ax.dsl.load(swa_topk_lengths[:1]).reshape(1)
    # history_length: [1], INT32.
    history_length = ax.dsl.sub(first_swa_length, one_i32, output_dtype=ax.int32)

    history_capacity_i32 = constant(
        np.array([history_capacity], dtype=np.int32),
        "csa_swa_history_capacity",
    )
    # The query-major page row stores valid entries at the beginning. Right-align
    # its history prefix so every statically sliced query window gets the newest
    # history tail, with any early-context padding kept at the front.
    # history_padding: [1], INT32.
    history_padding = ax.dsl.sub(
        history_capacity_i32,
        history_length,
        output_dtype=ax.int32,
    )
    history_offsets = constant(
        np.arange(history_slice.start, history_slice.stop, dtype=np.int32),
        "csa_swa_history_offsets",
    )
    # source_offsets: [W_core], INT32.
    source_offsets = ax.dsl.sub(
        history_offsets,
        history_padding,
        output_dtype=ax.int32,
    )
    # safe_source_offsets: [W_core], INT32.
    safe_source_offsets = ax.dsl.maximum(source_offsets, zero_i32, output_dtype=ax.int32)
    # local_indices: [W_core], INT32. Invalid leading padding is redirected
    # to the first page-index entry and excluded by history_mask.
    local_indices = ax.dsl.index_select(
        swa_page_indices[0:1, :].reshape(swa_page_indices.shape[1]),
        safe_source_offsets,
        dim=0,
    )
    # safe_indices: [W_core], INT32.
    safe_indices = ax.dsl.maximum(local_indices, zero_i32, output_dtype=ax.int32)
    # history_source local slice: [B,W_core,Dh], swa_kv_cache.dtype.
    ax.dsl.index_select(
        swa_kv_cache.reshape(-1, head_dim),
        safe_indices,
        dim=0,
        r=history_source[:, history_slice, :].reshape(local_history, head_dim),
    )

    # source_offsets >= 0 denotes a valid right-aligned history row.
    # active_distance: [W_core], INT32.
    active_distance = ax.dsl.add(source_offsets, one_i32, output_dtype=ax.int32)
    active = positive_i32(active_distance, zero_i32, one_i32)
    # inactive: [W_core], INT32.
    inactive = ax.dsl.sub(one_i32, active, output_dtype=ax.int32)
    # negative_bias: [1], FP32.
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)
    # history_mask local slice: [W_core], FP32.
    ax.dsl.mul(
        inactive,
        negative_bias,
        output_dtype=ax.fp32,
        r=history_mask[history_slice],
    )
    return history_source, history_mask


def csa_static_query_attention(
    query_group,
    source_ocm,
    source_mask,
    sink_group,
    *,
    c4_capacity,
    r,
):
    """Compute one query and all chip-local heads against the fixed CSA source OCM."""
    _, query_tokens, heads, head_dim = query_group.shape
    assert query_tokens == 1
    source_capacity = source_ocm.shape[0]
    # scale: [1], FP32.
    scale = ax.dsl.full((1,), ax.fp32, fill_value=head_dim**-0.5)
    query_rows = query_group.reshape(heads, head_dim)
    # c4_scores/swa_scores: [Hq_chip,K_top]/[Hq_chip,W], FP32; production
    # [32,512]/[32,128]. Independent fixed-capacity Conv waves expose more QK
    # scheduling freedom.
    c4_scores = ax.dsl.matmul(
        query_rows,
        source_ocm[:c4_capacity, :],
        transposed=True,
        output_dtype=ax.fp32,
    )
    # swa_scores: [Hq_chip,W], FP32.
    swa_scores = ax.dsl.matmul(
        query_rows,
        source_ocm[c4_capacity:, :],
        transposed=True,
        output_dtype=ax.fp32,
    )
    # scores: [Hq_chip,T_source], FP32.
    scores = ax.dsl.concat(c4_scores, swa_scores, dim=-1)
    # scores: [Hq_chip,T_source], FP32 after scale.
    scores = ax.dsl.mul(scores, scale, output_dtype=ax.fp32)
    # scores: [Hq_chip,T_source], FP32 after query-specific source mask.
    scores = ax.dsl.add(
        scores,
        source_mask.reshape(1, source_capacity),
        output_dtype=ax.fp32,
    )
    sink_values = sink_group.reshape(heads, 1)
    # scores_with_sink: [Hq_chip,T_source+1], FP32.
    scores_with_sink = ax.dsl.concat(scores, sink_values, dim=-1)
    # probability: [Hq_chip,T_source], query_group.dtype; the sink has no value row.
    probability = ax.dsl.softmax(
        scores_with_sink,
        dim=-1,
        output_dtype=query_group.dtype,
    )[:, :source_capacity]
    # source_transposed: [Dh,T_source], source_ocm.dtype. Materializing the
    # runtime source layout once lets both PV MatMuls use the native
    # transposed-weight contract instead of lowering an internal transpose.
    source_transposed = ax.dsl.transpose(source_ocm, perm=(1, 0))
    # c4_partial/swa_partial: [Hq_chip,Dh], FP32. Splitting the value reduction
    # keeps the Conv IC dimensions aligned to 512 and the exact SWA window.
    c4_partial = ax.dsl.matmul(
        probability[:, :c4_capacity],
        source_transposed[:, :c4_capacity],
        transposed=True,
        output_dtype=ax.fp32,
    )
    # swa_partial: [Hq_chip,Dh], FP32.
    swa_partial = ax.dsl.matmul(
        probability[:, c4_capacity:],
        source_transposed[:, c4_capacity:],
        transposed=True,
        output_dtype=ax.fp32,
    )
    # Shared context query slice: [B,1,Hq_chip,Dh], query_group.dtype. The
    # final PV accumulation writes its BF16 boundary directly, avoiding a
    # separate FP32 context tensor and cast pass.
    ax.dsl.add(
        c4_partial,
        swa_partial,
        output_dtype=query_group.dtype,
        r=r.reshape(heads, head_dim),
    )
    return r


def csa_lightning_topk(
    prefix,
    index_k_cache,
    c4_page_table,
    positions_casual,
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    meta,
    *,
    index_topk,
    rotary_dim,
    index_scale_cache=None,
):
    """Compute query-owned Lightning Indexer Top-K in bounded C4 source blocks."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len, _ = prefix.q_residual.shape
    index_dim = index_k_cache.shape[-1]
    index_heads = index_q_weight.shape[0] // index_dim
    c4_page_size = index_k_cache.shape[1]
    index_capacity = c4_page_table.shape[0] * c4_page_size
    # Long contexts advance by 4,096 C4 rows per trip. A smaller compile-time
    # cache uses its exact capacity so the one static trip neither computes nor
    # admits padded rows beyond the page table.
    source_block = min(CSA_INDEX_SOURCE_BLOCK, index_capacity)
    assert batch == 1
    quantized_index = index_scale_cache is not None
    if quantized_index:
        assert source_block % c4_page_size == 0
    assert index_topk <= index_capacity
    local_token_slice = token_slice(core_id, core_num, seq_len)
    token_count = local_token_slice.stop - local_token_slice.start
    assert token_count > 0

    # score_scale: [1], FP32. Applying both normalization constants to the
    # small head-weight tensor avoids scaling every score tile.
    score_scale = ax.dsl.full((1,), ax.fp32, fill_value=(index_dim * index_heads) ** -0.5)
    # Four heads bound each FP32 score tile to 2 MiB at the 32-query,
    # 4,096-source production block, leaving room for ReLU output and Top-K state.
    index_head_tile = min(4, index_heads)
    assert index_heads % index_head_tile == 0

    if core_num == 1:
        # Decode has one logical query owner on every physical core. Preserve
        # its private complete-head projection so the runtime page range has
        # no cross-core dependency.
        # head_weight: [T_core,Hi], FP32.
        head_weight = runtime_linear(
            prefix.normalized[:, local_token_slice, :],
            index_weight,
            index_weight_scale,
            output_dtype=ax.fp32,
        ).reshape(token_count, index_heads)
        # head_weight: [T_core,Hi], FP32 after global Indexer normalization.
        head_weight = ax.dsl.mul(head_weight, score_scale, output_dtype=ax.fp32)
        # head_weight: [Hi,T_core], FP32.
        head_weight = ax.dsl.transpose(head_weight, perm=(1, 0))

        index_query_groups = []
        for head_start in range(0, index_heads, index_head_tile):
            head_stop = head_start + index_head_tile
            index_rows = slice(head_start * index_dim, head_stop * index_dim)
            index_q_scale = runtime_weight_scale_slice(
                index_q_weight,
                index_q_weight_scale,
                index_rows,
                slice(0, index_q_weight.shape[1]),
            )
            # index_q_group: [B,T_core,H_tile,Di], prefix.q_residual.dtype.
            index_q_group = runtime_linear(
                prefix.q_residual[:, local_token_slice, :],
                index_q_weight[index_rows, :],
                index_q_scale,
                output_dtype=prefix.q_residual.dtype,
            ).reshape(batch, token_count, index_head_tile, index_dim)
            # index_q_group: [B,T_core,H_tile,Di], prefix.q_residual.dtype after RoPE.
            index_q_group = tail_rope(
                index_q_group,
                prefix.cos[:, local_token_slice, :],
                prefix.sin[:, local_token_slice, :],
                min(rotary_dim, index_dim),
            )
            if quantized_index:
                index_q_group = normalized_hadamard(
                    index_q_group,
                    f"csa_index_query_hadamard_{head_start}",
                )
            # index_q_group: [H_tile,T_core,Di], prefix.q_residual.dtype.
            index_q_group = ax.dsl.transpose(
                index_q_group.reshape(token_count, index_head_tile, index_dim),
                perm=(1, 0, 2),
            )
            index_query_groups.append(index_q_group)
        # index_query: [Hi,T_core,Di], prefix.q_residual.dtype.
        index_query = (
            index_query_groups[0] if len(index_query_groups) == 1 else ax.dsl.concat(*index_query_groups, dim=0)
        )
    else:
        # Prefill shards the constant projections by head so every Index-Q
        # weight row is consumed by exactly one core. The temporary head-major
        # SHOCM tensors transpose that ownership once before the page range
        # returns to independent query slices.
        assert index_heads % core_num == 0
        local_heads = index_heads // core_num
        local_head_start = core_id * local_heads
        local_head_stop = local_head_start + local_heads
        local_head_slice = slice(local_head_start, local_head_stop)

        # shared_head_weight: [Hi,S], FP32 SHARED_OCM.
        shared_head_weight = ax.dsl.create_tensor(
            (index_heads, seq_len),
            ax.fp32,
            ax.MemoryType.SHARED_OCM,
            name="csa_index_head_weight",
        )
        # shared_index_query: [Hi,S,Di], prefix.q_residual.dtype SHARED_OCM.
        shared_index_query = ax.dsl.create_tensor(
            (index_heads, seq_len, index_dim),
            prefix.q_residual.dtype,
            ax.MemoryType.SHARED_OCM,
            name="csa_index_query",
        )

        local_index_weight_scale = runtime_weight_scale_slice(
            index_weight,
            index_weight_scale,
            local_head_slice,
            slice(0, index_weight.shape[1]),
        )
        # local_head_weight: [S,H_local], FP32.
        local_head_weight = runtime_linear(
            prefix.normalized,
            index_weight[local_head_slice, :],
            local_index_weight_scale,
            output_dtype=ax.fp32,
        ).reshape(seq_len, local_heads)
        # local_head_weight: [S,H_local], FP32 after global Indexer normalization.
        local_head_weight = ax.dsl.mul(local_head_weight, score_scale, output_dtype=ax.fp32)
        # shared head-weight slice: [H_local,S], FP32.
        ax.dsl.transpose(
            local_head_weight,
            perm=(1, 0),
            r=shared_head_weight[local_head_slice, :],
        )

        projection_head_tile = min(index_head_tile, local_heads)
        assert local_heads % projection_head_tile == 0
        for head_start in range(local_head_start, local_head_stop, projection_head_tile):
            head_stop = head_start + projection_head_tile
            head_slice = slice(head_start, head_stop)
            index_rows = slice(head_start * index_dim, head_stop * index_dim)
            index_q_scale = runtime_weight_scale_slice(
                index_q_weight,
                index_q_weight_scale,
                index_rows,
                slice(0, index_q_weight.shape[1]),
            )
            # index_q_group: [B,S,H_proj,Di], prefix.q_residual.dtype.
            index_q_group = runtime_linear(
                prefix.q_residual,
                index_q_weight[index_rows, :],
                index_q_scale,
                output_dtype=prefix.q_residual.dtype,
            ).reshape(batch, seq_len, projection_head_tile, index_dim)
            # index_q_group: [B,S,H_proj,Di], prefix.q_residual.dtype after RoPE.
            index_q_group = tail_rope(
                index_q_group,
                prefix.cos,
                prefix.sin,
                min(rotary_dim, index_dim),
            )
            if quantized_index:
                index_q_group = normalized_hadamard(
                    index_q_group,
                    f"csa_index_query_hadamard_{head_start}",
                )
            # shared Index-Q head slice: [H_proj,S,Di], prefix.q_residual.dtype.
            ax.dsl.transpose(
                index_q_group.reshape(seq_len, projection_head_tile, index_dim),
                perm=(1, 0, 2),
                r=shared_index_query[head_slice, :, :],
            )

        # Publish all four disjoint head shards before query owners read them.
        axd.sync_all()
        # head_weight: [Hi,T_core], FP32 private OCM.
        head_weight = ax.dsl.create_tensor(
            (index_heads, token_count),
            ax.fp32,
            ax.MemoryType.OCM,
            name="csa_local_index_head_weight",
        )
        # head_weight: [Hi,T_core], FP32.
        ax.dsl.move(shared_head_weight[:, local_token_slice], r=head_weight)
        # index_query: [Hi,T_core,Di], prefix.q_residual.dtype private OCM.
        index_query = ax.dsl.create_tensor(
            (index_heads, token_count, index_dim),
            prefix.q_residual.dtype,
            ax.MemoryType.OCM,
            name="csa_local_index_query",
        )
        # index_query: [Hi,T_core,Di], prefix.q_residual.dtype.
        ax.dsl.move(shared_index_query[:, local_token_slice, :], r=index_query)
        # All readers have released the temporary SHOCM before the independent
        # dynamic ranges begin and later shared allocations may reuse it.
        axd.sync_all()

    # positions: [T_core], INT32.
    positions = ax.dsl.load(positions_casual[local_token_slice])
    one_i32 = constant(np.array([1], dtype=np.int32), "csa_index_one")
    four_i32 = constant(np.array([4], dtype=np.int32), "csa_index_four")
    three_i32 = constant(np.array([3], dtype=np.int32), "csa_index_three")
    # position_ends: [T_core], INT32.
    position_ends = ax.dsl.add(positions, one_i32, output_dtype=ax.int32)
    # c4_length_numerators: [T_core], INT32. For integer x >= 0,
    # floor(x / 4) == ceil((x - 3) / 4). AX8860 AxDiv uses an approximate
    # reciprocal, so the ceil form avoids undercounting exact multiples before
    # float-to-integer conversion.
    c4_length_numerators = ax.dsl.sub(position_ends, three_i32, output_dtype=ax.int32)
    # c4_lengths: [T_core], INT32 causal completed C4 counts.
    c4_lengths = ax.dsl.div(
        c4_length_numerators,
        four_i32,
        output_dtype=ax.int32,
        output_rounding=2,
    )

    page_size_i32 = constant(np.array([c4_page_size], dtype=np.int32), "csa_c4_page_size")
    source_block_i32 = constant(np.array([source_block], dtype=np.int32), "csa_index_source_block_size")
    pages_per_block = source_block // c4_page_size
    pages_per_block_i32 = constant(
        np.array([pages_per_block], dtype=np.int32),
        "csa_index_pages_per_block",
    )
    block_offsets = constant(np.arange(source_block, dtype=np.int32), "csa_index_block_offsets")
    page_offsets = constant(
        np.arange(source_block, dtype=np.int32) // c4_page_size,
        "csa_index_page_offsets",
    )
    row_offsets = constant(
        np.arange(source_block, dtype=np.int32) % c4_page_size,
        "csa_index_row_offsets",
    )
    last_page_index = constant(
        np.array([c4_page_table.shape[0] - 1], dtype=np.int32),
        "csa_index_last_page",
    )
    # zero_i32: [1], INT32.
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    # negative_bias: [1], FP32.
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)
    # zero_fp32: [1], FP32.
    zero_fp32 = ax.dsl.zeros((1,), ax.fp32)
    # top_scores: [T_core,K_top], FP32.
    top_scores = ax.dsl.full((token_count, index_topk), ax.fp32, fill_value=-1.0e30)
    # physical_top_ids: [T_core,K_top], INT32.
    physical_top_ids = ax.dsl.zeros((token_count, index_topk), ax.int32)
    # range_start: [1], INT32.
    range_start = ax.dsl.zeros((1,), ax.int32)
    range_step = constant(np.array([1], dtype=np.int32), "csa_index_step")
    # max_c4_length: [1], INT32.
    max_c4_length = ax.dsl.max(c4_lengths, dim=0, keepdim=True)
    # range_stop: [1], INT32 number of C4 pages.
    range_stop = ax.dsl.div(
        max_c4_length,
        source_block_i32,
        output_dtype=ax.int32,
        output_rounding=2,
    )

    def source_loop(
        page_index,
        loop_query,
        loop_head_weight,
        loop_cache,
        loop_scale_cache,
        loop_page_table,
        loop_lengths,
        loop_page_size,
        loop_source_block_size,
        loop_pages_per_block,
        loop_block_offsets,
        loop_page_offsets,
        loop_row_offsets,
        loop_last_page_index,
        loop_zero_i32,
        loop_one_i32,
        loop_zero_fp32,
        loop_negative_bias,
        loop_top_scores,
        loop_top_ids,
    ):
        # logical_base: [1], INT32.
        logical_base = ax.dsl.mul(page_index, loop_source_block_size, output_dtype=ax.int32)
        # logical_positions: [T_source], INT32.
        logical_positions = ax.dsl.add(logical_base, loop_block_offsets, output_dtype=ax.int32)
        # logical_page_base: [1], INT32.
        logical_page_base = ax.dsl.mul(page_index, loop_pages_per_block, output_dtype=ax.int32)
        # logical_pages: [T_source], INT32. The final source block may extend
        # beyond the compile-time page-table capacity. Redirect those masked
        # rows to the last real page so the gather itself remains in bounds.
        logical_pages = ax.dsl.add(logical_page_base, loop_page_offsets, output_dtype=ax.int32)
        # safe_logical_pages: [T_source], INT32.
        safe_logical_pages = ax.dsl.minimum(
            logical_pages,
            loop_last_page_index,
            output_dtype=ax.int32,
        )
        # physical_pages: [T_source], INT32.
        physical_pages = ax.dsl.index_select(loop_page_table, safe_logical_pages, dim=0)
        # physical_base: [T_source], INT32.
        physical_base = ax.dsl.mul(physical_pages, loop_page_size, output_dtype=ax.int32)
        # physical_ids: [T_source], INT32.
        physical_ids = ax.dsl.add(physical_base, loop_row_offsets, output_dtype=ax.int32)
        # index_source: [T_source,Di], index_k_cache.dtype private OCM.
        index_source = ax.dsl.create_tensor(
            (source_block, index_dim),
            index_k_cache.dtype,
            ax.MemoryType.OCM,
            name="csa_index_source_block",
        )
        # index_source: [T_source,Di], index_k_cache.dtype.
        ax.dsl.index_select(
            loop_cache,
            physical_ids,
            dim=0,
            r=index_source,
        )
        index_source_value = index_source
        if quantized_index:
            index_source_scale = ax.dsl.create_tensor(
                (source_block, 1),
                index_scale_cache.dtype,
                ax.MemoryType.OCM,
                name="csa_index_source_scale_block",
            )
            ax.dsl.index_select(
                loop_scale_cache,
                physical_ids,
                dim=0,
                r=index_source_scale,
            )
            index_source_value = ax.dsl.mul(
                ax.dsl.cast(index_source, dtype=prefix.q_residual.dtype),
                index_source_scale,
                output_dtype=prefix.q_residual.dtype,
            )

        score_partials = []
        for head_start in range(0, index_heads, index_head_tile):
            head_stop = head_start + index_head_tile
            # head_score: [H_tile,T_core,T_source], FP32.
            head_score = ax.dsl.matmul(
                loop_query[head_start:head_stop, :, :].reshape(index_head_tile * token_count, index_dim),
                index_source_value,
                transposed=True,
                output_dtype=ax.fp32,
            ).reshape(index_head_tile, token_count, source_block)
            # head_score: [H_tile,T_core,T_source], FP32 after ReLU.
            head_score = ax.dsl.maximum(head_score, loop_zero_fp32, output_dtype=ax.fp32)
            # weighted_head_score: [H_tile,T_core,T_source], FP32.
            weighted_head_score = ax.dsl.mul(
                head_score,
                loop_head_weight[head_start:head_stop, :].reshape(index_head_tile, token_count, 1),
                output_dtype=ax.fp32,
            )
            # score_partial: [T_core,T_source], FP32.
            score_partial = ax.dsl.sum(weighted_head_score, dim=0, output_dtype=ax.fp32)
            score_partials.append(score_partial)

        block_scores = score_partials[0]
        for score_partial in score_partials[1:]:
            # block_scores: [T_core,T_source], FP32.
            block_scores = ax.dsl.add(block_scores, score_partial, output_dtype=ax.fp32)
        # remaining: [T_core,T_source], INT32.
        remaining = ax.dsl.sub(
            loop_lengths.reshape(token_count, 1),
            logical_positions.reshape(1, source_block),
            output_dtype=ax.int32,
        )
        active = positive_i32(remaining, loop_zero_i32, loop_one_i32)
        # inactive: [T_core,T_source], INT32.
        inactive = ax.dsl.sub(loop_one_i32, active, output_dtype=ax.int32)
        # causal_bias: [T_core,T_source], FP32.
        causal_bias = ax.dsl.mul(inactive, loop_negative_bias, output_dtype=ax.fp32)
        # block_scores: [T_core,T_source], FP32 with causal masking.
        block_scores = ax.dsl.add(
            block_scores,
            causal_bias,
            output_dtype=ax.fp32,
        )

        # candidate_scores: [T_core,K_top+T_source], FP32.
        candidate_scores = ax.dsl.concat(loop_top_scores, block_scores, dim=-1)
        # block_ids: [T_core,T_source], INT32.
        block_ids = ax.dsl.expand(
            physical_ids.reshape(1, source_block),
            shape=(token_count, source_block),
        )
        # candidate_ids: [T_core,K_top+T_source], INT32.
        candidate_ids = ax.dsl.concat(loop_top_ids, block_ids, dim=-1)
        # next_top_scores/merge_positions: [T_core,K_top], FP32/INT32.
        next_top_scores, merge_positions = ax.dsl.topk(candidate_scores, k=index_topk, dim=-1)
        # next_top_ids: [T_core,K_top], INT32.
        next_top_ids = ax.dsl.gather(candidate_ids, merge_positions, dim=-1)
        return (
            loop_query,
            loop_head_weight,
            loop_cache,
            loop_scale_cache,
            loop_page_table,
            loop_lengths,
            loop_page_size,
            loop_source_block_size,
            loop_pages_per_block,
            loop_block_offsets,
            loop_page_offsets,
            loop_row_offsets,
            loop_last_page_index,
            loop_zero_i32,
            loop_one_i32,
            loop_zero_fp32,
            loop_negative_bias,
            next_top_scores,
            next_top_ids,
        )

    common_range_operands = (
        c4_page_table,
        c4_lengths,
        page_size_i32,
        source_block_i32,
        pages_per_block_i32,
        block_offsets,
        page_offsets,
        row_offsets,
        last_page_index,
        zero_i32,
        one_i32,
        zero_fp32,
        negative_bias,
        top_scores,
        physical_top_ids,
    )
    if quantized_index:
        range_operands = (
            index_query,
            head_weight,
            index_k_cache.reshape(-1, index_dim),
            index_scale_cache.reshape(-1, 1),
            *common_range_operands,
        )
        range_body = source_loop
    else:
        range_operands = (
            index_query,
            head_weight,
            index_k_cache.reshape(-1, index_dim),
            *common_range_operands,
        )

        def source_loop_without_scale(
            page_index,
            loop_query,
            loop_head_weight,
            loop_cache,
            *loop_operands,
        ):
            results = source_loop(
                page_index,
                loop_query,
                loop_head_weight,
                loop_cache,
                loop_cache,
                *loop_operands,
            )
            return results[:3] + results[4:]

        range_body = source_loop_without_scale
    if index_capacity <= source_block:
        # A cache that fits one source block has exactly one possible trip. Record the
        # same body statically to avoid paying dynamic-control overhead.
        results = range_body(range_start, *range_operands)
    else:
        # range_index: [1], INT32.
        range_index = ax.dsl.zeros((1,), ax.int32)
        # results[-2:]: [T_core,K_top] FP32 scores and INT32 physical ids.
        results = ax.dsl.range(
            range_index,
            range_start,
            range_stop,
            range_step,
            range_body,
            *range_operands,
            unroll_factor=1,
        )
    physical_top_ids = results[-1]
    topk_i32 = constant(np.array([index_topk], dtype=np.int32), "csa_index_topk")
    # topk_lengths: [T_core], INT32.
    topk_lengths = ax.dsl.minimum(c4_lengths, topk_i32, output_dtype=ax.int32)
    return physical_top_ids, topk_lengths


def csa_attention_paged(
    prefix,
    sinks_chip,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c4_kv_cache,
    index_k_cache,
    c4_page_table,
    positions_casual,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    index_q_weight,
    index_q_weight_scale,
    index_weight,
    index_weight_scale,
    meta,
    *,
    index_topk=512,
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
):
    """Run query-owned paged CSA with the fused Lightning Indexer."""
    batch, seq_len, _ = prefix.q_residual.shape
    head_dim = c4_kv_cache.shape[-1]
    assert batch == 1
    single_index_page = c4_page_table.shape[0] == 1
    if single_index_page:
        # Without AxRange, keeping the shared main query live matches the
        # static scheduler's natural producer/consumer order.
        query = project_shared_attention_query(
            prefix.q_residual,
            prefix.cos,
            prefix.sin,
            q_b_weight_chip,
            q_b_weight_chip_scale,
            meta,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rms_eps=rms_eps,
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
    if not single_index_page:
        # Keeping the multi-megabyte SHOCM query live across AxRange causes the
        # backend to reuse its storage nondeterministically. Publish it only
        # after every dynamic Indexer range has finished.
        query = project_shared_attention_query(
            prefix.q_residual,
            prefix.cos,
            prefix.sin,
            q_b_weight_chip,
            q_b_weight_chip_scale,
            meta,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rms_eps=rms_eps,
        )
    return csa_attention_with_topk(
        prefix,
        query,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c4_kv_cache,
        physical_top_ids,
        topk_lengths,
        meta,
        index_topk=index_topk,
        window=window,
    )


def csa_attention_from_topk(
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
    *,
    index_topk=512,
    window=128,
    rotary_dim=64,
    rms_eps=1e-6,
):
    """Run CSA attention from standalone query-major Lightning Top-K output."""
    head_dim = c4_kv_cache.shape[-1]
    query = project_shared_attention_query(
        prefix.q_residual,
        prefix.cos,
        prefix.sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
    )
    local_token_slice = token_slice(meta.core_id, meta.core_num, meta.seq_len)
    token_count = local_token_slice.stop - local_token_slice.start
    # local_physical_top_ids: [T_core,K_top], INT32 private OCM.
    local_physical_top_ids = ax.dsl.load(physical_top_ids[:, local_token_slice, :]).reshape(token_count, index_topk)
    # local_topk_lengths: [T_core], INT32 private OCM.
    local_topk_lengths = ax.dsl.load(topk_lengths[:, local_token_slice]).reshape(token_count)
    return csa_attention_with_topk(
        prefix,
        query,
        sinks_chip,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        c4_kv_cache,
        local_physical_top_ids,
        local_topk_lengths,
        meta,
        index_topk=index_topk,
        window=window,
    )


def csa_attention_with_topk(
    prefix,
    query,
    sinks_chip,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    c4_kv_cache,
    physical_top_ids,
    topk_lengths,
    meta,
    *,
    index_topk,
    window,
):
    """Assemble selected C4/SWA sources and evaluate query-owned CSA attention."""
    batch, seq_len, heads, head_dim = query.shape
    local_token_slice = token_slice(meta.core_id, meta.core_num, seq_len)
    token_count = local_token_slice.stop - local_token_slice.start
    # Each core publishes its disjoint SWA history slice; the following
    # collective makes the complete history visible to all query owners.
    history_source, history_mask = csa_paged_history(
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
    )
    # Synchronization only; no tensor output.
    axd.sync_all()
    source_capacity = index_topk + window
    # zero_i32/one_i32: [1], INT32.
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    one_i32 = constant(np.array([1], dtype=np.int32), "csa_static_one")
    c4_offsets = constant(np.arange(index_topk, dtype=np.int32), "csa_static_c4_offsets")
    # negative_bias: [1], FP32.
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)
    # current_mask: [S], FP32.
    current_mask = ax.dsl.zeros((seq_len,), ax.fp32)
    sink = ax.dsl.load(sinks_chip).reshape(1, heads, 1, 1)
    flat_c4_cache = c4_kv_cache.reshape(-1, head_dim)
    # context_rows: [B,T_core,Hq_chip,Dh], query.dtype. Keeping the
    # per-core result private removes the SHARED_OCM query/context alias.
    context_rows = ax.dsl.create_tensor(
        (batch, token_count, heads, head_dim),
        query.dtype,
        ax.MemoryType.OCM,
        name="csa_context_rows",
    )

    for query_index in range(local_token_slice.start, local_token_slice.stop):
        local_query_index = query_index - local_token_slice.start
        # source_ocm/source_mask are independent logical tensors per query. The
        # compiler may reuse or double-buffer their physical OCM by liveness.
        # source_ocm: [K_top+W,Dh], query.dtype private OCM.
        source_ocm = ax.dsl.create_tensor(
            (source_capacity, head_dim),
            query.dtype,
            ax.MemoryType.OCM,
            name=f"csa_source_q{query_index}",
        )
        # source_mask: [K_top+W], FP32 private OCM.
        source_mask = ax.dsl.create_tensor(
            (source_capacity,),
            ax.fp32,
            ax.MemoryType.OCM,
            name=f"csa_source_mask_q{query_index}",
        )
        current_count = min(query_index + 1, window)
        history_count = window - current_count
        history_start = window - 1 - history_count
        current_start_in_chunk = query_index - current_count + 1
        history_rows = history_source[:, history_start : window - 1, :]
        current_rows = prefix.kv[:, current_start_in_chunk : query_index + 1, :]
        if history_count:
            # source_ocm SWA slice: [W,Dh], query.dtype, assembled from
            # the query-specific history tail and visible current tokens.
            ax.dsl.concat(
                history_rows,
                current_rows,
                dim=1,
                r=source_ocm[index_topk:source_capacity, :].reshape(batch, window, head_dim),
            )
        else:
            # source_ocm SWA slice: [W,Dh], query.dtype; the window is
            # entirely inside the current chunk.
            ax.dsl.move(
                current_rows.reshape(window, head_dim),
                r=source_ocm[index_topk:source_capacity, :],
            )
        # physical_indices: [index_topk], INT32 from Lightning selection.
        physical_indices = physical_top_ids[local_query_index : local_query_index + 1, :].reshape(index_topk)
        # source_ocm C4 slice: [index_topk,Dh], query.dtype. Current C4
        # rows have already been written to the same DDR cache as history.
        ax.dsl.index_select(
            flat_c4_cache,
            physical_indices,
            dim=0,
            r=source_ocm[:index_topk, :],
        )
        # c4_length: [1], INT32.
        c4_length = topk_lengths[local_query_index : local_query_index + 1]
        # within_length/inactive: [index_topk], INT32.
        remaining = ax.dsl.sub(c4_length, c4_offsets, output_dtype=ax.int32)
        within_length = positive_i32(remaining, zero_i32, one_i32)
        # inactive: [K_top], INT32.
        inactive = ax.dsl.sub(one_i32, within_length, output_dtype=ax.int32)
        # source_mask C4 slice: [K_capacity], FP32.
        ax.dsl.mul(
            inactive,
            negative_bias,
            output_dtype=ax.fp32,
            r=source_mask[:index_topk],
        )
        history_mask_rows = history_mask[history_start : window - 1]
        current_mask_rows = current_mask[current_start_in_chunk : query_index + 1]
        if history_count:
            # source_mask SWA slice: [W], FP32.
            ax.dsl.concat(
                history_mask_rows,
                current_mask_rows,
                dim=0,
                r=source_mask[index_topk:],
            )
        else:
            # source_mask SWA slice: [W], FP32.
            ax.dsl.move(
                current_mask_rows.reshape(window),
                r=source_mask[index_topk:],
            )

        query_group = query[:, query_index : query_index + 1, :, :]
        csa_static_query_attention(
            query_group,
            source_ocm,
            source_mask,
            sink,
            c4_capacity=index_topk,
            r=context_rows[:, local_query_index : local_query_index + 1, :, :],
        )
    # Chip-visible context token slice: [B,T_core,Hq_chip,Dh], query.dtype.
    ax.dsl.move(context_rows, r=query[:, local_token_slice, :, :])
    # Synchronization only; no tensor output.
    axd.sync_all()
    return query
