"""Conventional host forward with one attention and one MLP launch per layer."""

from sgl_kernel_axera.mlp import mlp_kernel
from sgl_kernel_axera.moe.m_qwen3 import moe_kernel_expert_parallel_v2_scf
from sgl_kernel_axera.models.attention import attention_chunks, run_attention

WORKSPACE_TOKENS = 256
CAPTURE_DECODE = False
CACHE_NAME = "standard"
prepare = None


def forward(model, metadata, num_tokens):
    runtime = model.axera_state.runtime
    is_moe = model.config.model_type == "qwen3_5_moe_text"
    workspace = runtime.workspace
    chunks = attention_chunks(workspace, metadata)
    rows = metadata["kernel_tokens"] if not is_moe or num_tokens > 256 else num_tokens
    mlp_input, mlp_output = workspace.attention[:rows], workspace.hidden[:rows]
    for kv_buffer, weights, mlp_weights in runtime.layers:
        run_attention(
            kv_buffer,
            weights,
            chunks,
            metadata["page"]["indices"],
            tuple(model.config.rope_parameters["mrope_section"]),
        )
        if is_moe:
            moe_kernel_expert_parallel_v2_scf[4, 4](mlp_input, output_ax=mlp_output, **mlp_weights)
        else:
            mlp_kernel[4, 4](
                mlp_input.view(1, rows, -1),
                residual=None,
                output=mlp_output.view(1, rows, -1),
                tile_size=[min(rows, 256)],
                **mlp_weights,
            )
