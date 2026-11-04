"""All Qwen3.5-35B-A3B Transformer decode layers in one runtime-weight AXModel."""

import os
from pathlib import Path

import torch
import axnn.dsl as ax
from axnn.dsl import parallel as axd
from sgl_kernel_axera.models.weights import block_arguments, block_stages, block_layer

from sgl_kernel_axera.attention.gdn import GDNKernel
from sgl_kernel_axera.attention.flash_attn import (
    ATTENTION_WEIGHT_NAMES,
    decode_paged_kv_shared_kv_kernel,
    decode_paged_kv_kernel,
)
from sgl_kernel_axera.moe.m_qwen3 import moe_kernel_expert_parallel_v2_scf


def decode_qwen35_blocks(
    x,
    attention,
    position_ids,
    indptr,
    indices,
    last_page_len,
    cache_location,
    output,
    layout,
    mrope_section,
    **weights,
):
    """Keep block activations on chip while runtime-indexing weights and state.

    Each iteration records three GDN+MoE stages and one GQA+MoE stage. Each
    core carries its [1,H/cores] BF16 hidden shard in private OCM; complete
    [1,H] BF16 stage results are published in SHOCM. State/KV stay in DDR.
    """
    input_banks = {value.name: value for value in weights.values()}
    stages = block_stages(layout, weights)
    blocks = stages[0][2]["input_norm_w"].shape[0]
    core, cores = ax.program_id(0), ax.num_programs(0)
    width = x.shape[-1] // cores
    owned = slice(core * width, (core + 1) * width)
    assert cores == 4, "Qwen3.5 fused decode requires four cores"
    hidden = ax.dsl.load(x[:, owned])
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, 1)
    stop = ax.dsl.full((1,), ax.int32, blocks)

    def body(block_index, hidden):
        current = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name="block_input")
        ax.dsl.move(hidden, r=current[:, owned])
        axd.pub(current)
        axd.sub(current)

        # Four fixed phases share a dynamic block loop. Each consumer receives
        # explicit full-bank/index operands and ordinary static tensor slices.
        for phase, stage in enumerate(stages):
            (kv, weights, moe_weights), banks = block_layer(stage, block_index, input_banks)
            activation = ax.dsl.create_tensor(
                x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name=f"attention_handoff_{phase}"
            )
            if kv is None:
                GDNKernel.func(
                    current.reshape(1, 1, -1),
                    None,
                    output=None,
                    residual_out=activation.reshape(1, 1, -1),
                    banks=banks,
                    **weights,
                )
            else:
                decode = decode_paged_kv_shared_kv_kernel if kv.shape[3] == 2 else decode_paged_kv_kernel
                decode.func(
                    current.reshape(1, 1, -1),
                    *(weights[name] for name in ATTENTION_WEIGHT_NAMES[:17]),
                    position_ids,
                    *(weights[name] for name in ATTENTION_WEIGHT_NAMES[17:]),
                    None,
                    None,
                    activation.reshape(1, 1, -1),
                    kv,
                    indptr,
                    indices,
                    last_page_len,
                    256,
                    attn_output_gate=True,
                    partial_rotary_factor=weights["partial_rotary_factor"],
                    mrope_section=mrope_section,
                    mrope_interleaved=True,
                    has_mrope=position_ids.ndim == 2,
                    cache_location=cache_location,
                    banks=banks,
                    cos_sin_cache=weights.get("cos_sin_cache"),
                )
            axd.pub(activation)
            axd.sub(activation)
            # Separate stage outputs preserve the shared input version while
            # the asymmetric router/shared-expert consumers are still reading.
            current = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name=f"moe_handoff_{phase}")
            moe_kernel_expert_parallel_v2_scf.func(activation, output_ax=current, banks=banks, **moe_weights)
            axd.pub(current)
            axd.sub(current)
        ax.dsl.move(current[:, owned], r=hidden)
        return (hidden,)

    (result,) = ax.dsl.range(ax.dsl.zeros((1,), ax.int32), zero, stop, one, body, hidden, unroll_factor=1)
    ax.dsl.store(result, r=output[:, owned])
    return output


@torch.no_grad()
def prepare(layers, *, mrope_section, cache_path, run_mode=None, weight_banks=()):
    """Bank runtime tensors by four-layer block and record one reusable body.

    GDN/KV state pools and preallocated weight banks retain their storage.
    Fused projection slices are recorded inside the DSL graph while external
    kernel inputs remain contiguous. Layer dictionaries share those views with
    the host prefill mechanism.
    """
    assert len(layers) % 4 == 0
    assert all((kv is not None) == (i % 4 == 3) for i, (kv, _, _) in enumerate(layers))
    layout, weights = block_arguments(layers, "moe", weight_banks)
    cache_path = Path(cache_path)
    kernel = ax.builder(
        run_mode=ax.RunMode(run_mode or os.getenv("RUN_MODE", "CRUN")),
        const_inputs=["layout", "mrope_section"],
        enable_cache=True,
        unique_axmodel=True,
        cache_path=cache_path,
        dump_path=cache_path / "artifacts",
    )(decode_qwen35_blocks)
    return kernel, dict(layout=layout, mrope_section=tuple(mrope_section), **weights)
