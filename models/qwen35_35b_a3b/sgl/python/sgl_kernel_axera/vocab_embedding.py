import math
import os

import torch

import axnn.dsl as ax


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def vocab_embedding_kernel(input_flat, weight, output_flat):
    # Split channels so even one decode token uses every core. Each core
    # reads the same indices and owns disjoint columns of the DDR output.
    channels = math.ceil(weight.shape[1] / ax.num_programs(0))
    start = ax.program_id(0) * channels
    end = min(start + channels, weight.shape[1])
    if start >= end:
        return output_flat
    flat_indices = ax.dsl.load(input_flat)
    ax.dsl.index_select(weight[:, start:end], flat_indices, dim=0, r=output_flat[:, start:end])
    return output_flat


def vocab_embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor = None,
    core_num: int = int(os.getenv("SGLANG_AXERA_KERNEL_CORE_NUM", "4")),
) -> torch.Tensor:
    """torch.nn.Embedding equivalent: maps input indices to embedding vectors.

    Args:
        input:     LongTensor of shape [batch_size, seq_len] containing token indices.
        weight:    FloatTensor of shape [num_embeddings, embedding_dim].
        output:    Optional pre-allocated output tensor of shape
                   [batch_size, seq_len, embedding_dim].
        core_num:  Number of cores sharing the embedding channels; defaults to four.

    Returns:
        Tensor of shape [batch_size, seq_len, embedding_dim].
    """

    orig_shape = tuple(input.shape)

    if input.dim() != 2:
        input = input.reshape(-1, input.shape[-1])

    batch_size, seq_len = input.shape[0], input.shape[1]
    embedding_dim = weight.shape[1]

    # AxGather hardware requires 32-bit indices.
    if input.dtype != torch.int32:
        input = input.to(torch.int32)

    if output is None:
        output = torch.empty(
            (batch_size, seq_len, embedding_dim),
            device=weight.device,
            dtype=weight.dtype,
        )

    # the result is written directly to the (DDR) output, then reshaped back.
    input_flat = input.reshape(batch_size * seq_len)
    output_flat = output.reshape(batch_size * seq_len, embedding_dim)

    cache_tokens = max(1, int(os.environ.get("SGLANG_AXERA_VOCAB_TOKENS", "256")))
    token_count = input_flat.shape[0]
    for chunk_start in range(0, token_count, cache_tokens):
        chunk_end = min(chunk_start + cache_tokens, token_count)
        chunk_input = input_flat[chunk_start:chunk_end]
        chunk_output = output_flat[chunk_start:chunk_end]
        chunk_tokens = chunk_end - chunk_start

        if chunk_tokens == cache_tokens:
            kernel_input = chunk_input
            kernel_output = chunk_output
        else:
            kernel_input = input_flat.new_empty((cache_tokens,))
            kernel_input[:chunk_tokens] = chunk_input
            kernel_input[chunk_tokens:] = chunk_input[-1]
            kernel_output = torch.empty(
                (cache_tokens, embedding_dim),
                device=output_flat.device,
                dtype=output_flat.dtype,
            )

        vocab_embedding_kernel[core_num, core_num](kernel_input, weight, kernel_output)
        if chunk_tokens < cache_tokens:
            chunk_output.copy_(kernel_output[:chunk_tokens])

    output = output.reshape(batch_size, seq_len, embedding_dim)

    return output.reshape(*orig_shape, embedding_dim)
