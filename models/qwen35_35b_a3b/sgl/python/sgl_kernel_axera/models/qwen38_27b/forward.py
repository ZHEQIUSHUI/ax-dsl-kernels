"""Qwen3.8 dense layers in one AXModel with runtime weights and state."""

import os
from pathlib import Path

import torch
import axnn.dsl as ax
from axnn.dsl import parallel as axd
from sgl_kernel_axera.models.weights import block_arguments, block_stages, block_layer

from sgl_kernel_axera.attention.flash_attn import (
    ATTENTION_WEIGHT_NAMES,
    decode_paged_kv_kernel,
    decode_paged_kv_shared_kv_kernel,
    flash_attn_with_kvcache,
)
from sgl_kernel_axera.attention.gdn import GDNKernel
from sgl_kernel_axera.models.qwen38_27b.mlp import mlp_kernel

WORKSPACE_TOKENS = 256
CAPTURE_DECODE = True
CACHE_NAME = "qwen38_forward"


def forward_qwen38_blocks(
    x,
    position_ids,
    indptr,
    indices,
    last_page_len,
    cache_location,
    output,
    layout,
    mrope_section,
    valid_tokens=None,
    **weights,
):
    """Record a runtime loop of three GDN+MLP stages and one GQA+MLP stage.

    x/output are [T,H] BF16, with T=1 for decode or T=256 for prefill.
    Every core carries its [T,H/cores] shard in private OCM. Completed stage
    activations are published in SHOCM; recurrent state and KV stay in DDR.
    """
    input_banks = {value.name: value for value in weights.values()}
    stages = block_stages(layout, weights)
    assert x.shape[0] in (1, 256) and x.shape == output.shape
    # Positions cover the physical rows: [T] or [3,T] INT32.
    rows, hidden_dim = x.shape
    assert (valid_tokens is not None) == (rows == 256)
    blocks = stages[0][2]["w_norm"].shape[0]
    core = ax.program_id(0)
    cores = ax.num_programs(0)
    assert cores == 4, "Qwen3.8 fused forward requires four cores"
    width = hidden_dim // cores
    owned = slice(core * width, (core + 1) * width)
    # DDR [T,H/cores] BF16 -> private OCM [T,H/cores] BF16.
    hidden = ax.dsl.load(x[:, owned])
    # Runtime range bounds: three [1] INT32 scalars.
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, 1)
    stop = ax.dsl.full((1,), ax.int32, blocks)

    def body(block_index, hidden):
        # Snapshot the private loop carry in shared memory before reusing that
        # private allocation for GDN Q/K projections and MLP residual shards.
        current = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name="block_input")
        ax.dsl.move(hidden, r=current[:, owned])
        axd.pub(current)
        axd.sub(current)

        # Four fixed phases share a dynamic block loop. Each consumer receives
        # explicit full-bank/index operands and ordinary static tensor slices.
        for phase, stage in enumerate(stages):
            (kv, weights, mlp_weights), banks = block_layer(stage, block_index, input_banks)
            # Small quantization and MLP normalization scales keep Gather.
            for name in tuple(banks):
                if not name.endswith(("_in", "_out", "_kv", "_norm")) and "scale" not in name:
                    banks["dynamic_ld", name] = True
            banks[("weight_slice_extent",)] = one
            activation = ax.dsl.create_tensor(
                x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name=f"attention_handoff_{phase}"
            )
            if kv is None:
                GDNKernel.func(
                    current.reshape(1, rows, hidden_dim),
                    None,
                    output=None,
                    residual_out=activation.reshape(1, rows, hidden_dim),
                    projection_buffer=hidden,
                    valid_tokens=valid_tokens,
                    banks=banks,
                    **weights,
                )
            else:
                args = (
                    current.reshape(1, rows, hidden_dim),
                    *(weights[name] for name in ATTENTION_WEIGHT_NAMES[:17]),
                    position_ids,
                    *(weights[name] for name in ATTENTION_WEIGHT_NAMES[17:]),
                )
                options = dict(
                    banks=banks,
                    cos_sin_cache=weights.get("cos_sin_cache"),
                    attn_output_gate=True,
                    partial_rotary_factor=weights["partial_rotary_factor"],
                    mrope_section=mrope_section,
                    mrope_interleaved=True,
                    has_mrope=position_ids.ndim == 2,
                    cache_location=cache_location,
                )
                if rows == 1:
                    decode = decode_paged_kv_shared_kv_kernel if kv.shape[3] == 2 else decode_paged_kv_kernel
                    decode.func(
                        *args,
                        None,
                        None,
                        activation.reshape(1, rows, hidden_dim),
                        kv,
                        indptr,
                        indices,
                        last_page_len,
                        256,
                        **options,
                    )
                else:
                    flash_attn_with_kvcache.func(
                        *args,
                        None,
                        activation.reshape(1, rows, hidden_dim),
                        kv,
                        indptr,
                        indices,
                        None,
                        **options,
                    )
            axd.pub(activation)
            axd.sub(activation)
            # MLP snapshots its residual into hidden before reusing activation
            # for its ordered BF16 down-projection reduction and output.
            mlp_kernel.func(
                activation.reshape(1, rows, hidden_dim),
                residual=None,
                output=activation.reshape(1, rows, hidden_dim),
                tile_size=[rows],
                residual_buffer=hidden,
                banks=banks,
                **mlp_weights,
            )
            axd.pub(activation)
            axd.sub(activation)
            current = activation

        # Preserve the completed block as a private value across the backedge.
        ax.dsl.move(current[:, owned], r=hidden)
        return (hidden,)

    (result,) = ax.dsl.range(ax.dsl.zeros((1,), ax.int32), zero, stop, one, body, hidden, unroll_factor=1)
    # Private [T,H/cores] BF16 -> the owned shard of DDR output [T,H] BF16.
    ax.dsl.store(result, r=output[:, owned])
    return output


@torch.no_grad()
def prepare(layers, *, mrope_section, cache_path, run_mode=None, weight_banks=()):
    """Bind block-indexed runtime banks once and build the fixed kernel variants.

    Persistent state pools and checkpoint weight banks keep their storage.
    Fused projection banks are contiguous kernel inputs; their QKV/gate/up
    slices are created inside the DSL graph. Other runtime tensors are copied
    once. Layer dictionaries adopt the bank views so prefill and decode share state.
    """
    assert len(layers) % 4 == 0
    assert all((kv is not None) == (i % 4 == 3) for i, (kv, _, _) in enumerate(layers))
    layout, weights = block_arguments(layers, "mlp", weight_banks)
    cache_path = Path(cache_path)
    # Decode and fixed-width prefill each retain one loaded model. Real token
    # counts and history page counts are runtime inputs to the prefill graph.
    kernels = {
        rows: ax.builder(
            run_mode=ax.RunMode(run_mode or os.getenv("RUN_MODE", "CRUN")),
            const_inputs=["layout", "mrope_section"],
            enable_cache=True,
            unique_axmodel=True,
            cache_path=cache_path,
            dump_path=cache_path / "artifacts",
        )(forward_qwen38_blocks)
        for rows in (1, 256)
    }
    return kernels, dict(layout=layout, mrope_section=tuple(mrope_section), **weights)


def forward(model, metadata, num_tokens):
    runtime = model.axera_state.runtime
    workspace = runtime.workspace
    kernels, weights = runtime.kernel
    page = metadata["page"]
    chunked = metadata["kernel_tokens"] > 256
    rows = min(256, metadata["kernel_tokens"])
    for index, start in enumerate(range(0, metadata["kernel_tokens"], rows)):
        end = start + rows
        hidden = workspace.hidden[start:end]
        valid_tokens = metadata["valid_tokens"][index] if chunked else metadata["valid_tokens"]
        kernel = kernels[rows]
        kernel[4, 4](
            hidden,
            metadata["positions"][index] if chunked else metadata["positions"],
            page["indptr"][index] if chunked else page["indptr"],
            page["indices"],
            page["last_page_len"][index] if chunked else page["last_page_len"],
            metadata["cache_location"][index] if chunked else metadata["cache_location"],
            hidden,
            valid_tokens=valid_tokens,
            **weights,
        )
