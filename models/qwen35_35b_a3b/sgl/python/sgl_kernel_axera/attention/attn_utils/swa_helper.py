"""Sliding-window attention graph helper."""

from __future__ import annotations

import axnn.dsl as ax

from .common import *  # noqa: F401,F403


def swa_attention_paged(
    prefix,
    query,
    sinks_chip,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    meta,
    *,
    window=128,
    source_chunk_size=32,
):
    """Run each core's query-head shard over a shared paged source."""
    core_id = meta.core_id
    seq_len, heads, head_dim = query.shape[1:]
    source, source_mask = paged_swa_source(
        prefix,
        swa_kv_cache,
        swa_page_indices,
        swa_topk_lengths,
        meta,
        window=window,
        name_prefix="swa",
        mask_query_slice=slice(0, seq_len),
        source_alignment=source_chunk_size,
    )
    head_start = core_id * heads
    sink_values = sinks_chip[head_start : head_start + heads]
    sink = ax.dsl.load(sink_values).reshape(heads, 1)
    # scale: [1], FP32.
    scale = ax.dsl.full((1,), ax.fp32, fill_value=head_dim**-0.5)

    for query_index in range(seq_len):
        # The right-aligned shared source makes the exact query window a
        # contiguous compile-time view, so no per-query source copy is needed.
        query_source = source[:, query_index : query_index + window, :].reshape(window, head_dim)
        query_mask = source_mask[:, query_index : query_index + 1, query_index : query_index + window].reshape(
            1, window
        )
        query_rows = query[:, query_index : query_index + 1, :, :].reshape(heads, head_dim)
        # scores: [Hq_core,W], FP32.
        scores = ax.dsl.matmul(
            query_rows,
            query_source,
            transposed=True,
            output_dtype=ax.fp32,
        )
        # scores: [Hq_chip,W], FP32 after attention scaling.
        scores = ax.dsl.mul(scores, scale, output_dtype=ax.fp32)
        # scores: [Hq_chip,W], FP32 after early-history masking.
        scores = ax.dsl.add(scores, query_mask, output_dtype=ax.fp32)
        # scores_with_sink: [Hq_core,W+1], FP32.
        scores_with_sink = ax.dsl.concat(scores, sink, dim=-1)
        # probability_with_sink: [Hq_core,W+1], query.dtype; result view is [Hq_core,W].
        probability = ax.dsl.softmax(
            scores_with_sink,
            dim=-1,
            output_dtype=query.dtype,
        )[:, :window]
        # Materialize one context row before replacing its consumed query row;
        # this avoids an input/output alias inside MatMul while keeping the
        # production TP1 private peak close to the 2 MiB query allocation.
        context_row = ax.dsl.matmul(
            probability,
            query_source,
            output_dtype=query.dtype,
        )
        ax.dsl.move(
            context_row,
            r=query[:, query_index : query_index + 1, :, :].reshape(heads, head_dim),
        )
    return query
