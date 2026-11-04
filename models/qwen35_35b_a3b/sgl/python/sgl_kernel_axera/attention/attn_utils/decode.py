"""Shared helpers for DeepSeek V4 decode kernels."""

import axnn.dsl as ax


def decode_static_attention(query, source_rows, sinks, additive_mask, *, name_prefix):
    """Compute one decode query's local heads against one fixed shared source."""
    batch, query_tokens, heads, head_dim = query.shape
    source_len = source_rows.shape[0]
    assert batch == query_tokens == 1
    # scale: [1], FP32.
    scale = ax.dsl.full((1,), ax.fp32, fill_value=head_dim**-0.5)
    sink = ax.dsl.load(sinks).reshape(heads, 1)
    query_rows = query.reshape(heads, head_dim)
    # scores: [Hq_core,T_source], FP32.
    scores = ax.dsl.matmul(
        query_rows,
        source_rows,
        transposed=True,
        output_dtype=ax.fp32,
    )
    # scores: [Hq_core,T_source], FP32 after attention scaling.
    scores = ax.dsl.mul(scores, scale, output_dtype=ax.fp32)
    # scores: [Hq_core,T_source], FP32 after the padding/selection mask.
    scores = ax.dsl.add(
        scores,
        additive_mask.reshape(1, source_len),
        output_dtype=ax.fp32,
    )
    # scores_with_sink: [Hq_core,T_source+1], FP32.
    scores_with_sink = ax.dsl.concat(scores, sink, dim=-1)
    # probability_with_sink: [Hq_core,T_source+1], query.dtype.
    probability = ax.dsl.softmax(
        scores_with_sink,
        dim=-1,
        output_dtype=query.dtype,
    )[:, :source_len]
    # context: query.shape, query.dtype private OCM.
    context = ax.dsl.create_tensor(query.shape, query.dtype, ax.MemoryType.OCM, name=f"{name_prefix}_context")
    # context rows: [Hq_core,Dh], query.dtype.
    ax.dsl.matmul(
        probability,
        source_rows,
        output_dtype=query.dtype,
        r=context.reshape(heads, head_dim),
    )
    return context
