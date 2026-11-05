"""Chunked attention plus fixed-size prefill MoE, and fused block decode."""

import os
import numpy as np
import axnn.dsl as ax

from sgl_kernel_axera.models.attention import attention_chunks, run_attention
from sgl_kernel_axera.models.qwen36_35b_a3b.decode import prepare
from sgl_kernel_axera.moe.m_qwen3 import moe_kernel_expert_parallel_v2_scf

WORKSPACE_TOKENS = 4096
CAPTURE_DECODE = True
CACHE_NAME = "qwen36_decode"


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def clear_prefill_padding(activation, token_count):
    """Clear only padded rows without specializing on the live token count."""
    core, cores = ax.program_id(0), ax.num_programs(0)
    rows, hidden = activation.shape
    width = hidden // cores
    target = activation[:, core * width : (core + 1) * width]
    count = ax.dsl.load(token_count)
    lanes = ax.dsl.from_numpy(np.arange(256, dtype=np.int32))
    last = ax.dsl.full((1,), ax.int32, rows - 1)
    zeros = ax.dsl.zeros((256, width), activation.dtype)

    def body(start, output):
        # Repeated final lanes write the same zero to the final padding row.
        indices = ax.dsl.minimum(start + lanes, last)
        ax.dsl.select_scatter(zeros, indices, dim=0, r=output)
        return (output,)

    ax.dsl.range(
        ax.dsl.zeros((1,), ax.int32),
        count,
        ax.dsl.full((1,), ax.int32, rows),
        ax.dsl.full((1,), ax.int32, 256),
        body,
        target,
    )
    return ()


def forward(model, metadata, num_tokens):
    runtime = model.axera_state.runtime
    workspace = runtime.workspace
    page = metadata["page"]
    if metadata["kernel_tokens"] == 1:
        kernel, weights = runtime.kernel
        kernel[4, 4](
            workspace.hidden[:1],
            workspace.attention[:1],
            metadata["positions"],
            page["indptr"],
            page["indices"],
            page["last_page_len"],
            metadata["cache_location"],
            workspace.hidden[:1],
            **weights,
        )
        return
    chunks = attention_chunks(workspace, metadata)
    mlp_input = workspace.attention[:WORKSPACE_TOKENS]
    mlp_output = workspace.hidden[:WORKSPACE_TOKENS]
    for kv_buffer, weights, mlp_weights in runtime.layers:
        run_attention(kv_buffer, weights, chunks, page["indices"], tuple(model.config.rope_parameters["mrope_section"]))
        # GQA may write padded rows of its last chunk. The fixed 4096-row
        # MoE input must contain zeros beyond the real request, on every layer.
        if num_tokens < WORKSPACE_TOKENS:
            clear_prefill_padding[4, 4](mlp_input, metadata["num_tokens"])
        moe_kernel_expert_parallel_v2_scf[4, 4](mlp_input, output_ax=mlp_output, **mlp_weights)
