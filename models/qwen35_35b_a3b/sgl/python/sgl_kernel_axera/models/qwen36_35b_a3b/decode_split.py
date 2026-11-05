"""All Qwen3.5-35B-A3B decode layers in one AXModel with one KV input per attention layer (long contexts).

decode.py's dynamic block loop indexes one KV bank [blocks, pages, 2, 256, kv heads, head dim]; at 256k cells that bank
is 5 GiB, above the 2 GiB limit of one kernel IO. Here the blocks are unrolled (the weight banks still indexed by the
block, a constant) and attention layer j of the model reads its own input b3_kv<j> [pages, 2, 256, kv heads, head dim].
"""

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


def decode_qwen35_blocks_split(
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
    input_banks = {value.name: value for value in weights.values()}
    stages = block_stages(layout, weights)
    blocks = stages[0][2]["input_norm_w"].shape[0]
    core, cores = ax.program_id(0), ax.num_programs(0)
    width = x.shape[-1] // cores
    owned = slice(core * width, (core + 1) * width)
    assert cores == 4, "Qwen3.5 fused decode requires four cores"
    hidden = ax.dsl.load(x[:, owned])

    for block in range(blocks):
        block_index = ax.dsl.full((1,), ax.int32, block)
        current = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name=f"block_input_{block}")
        ax.dsl.move(hidden, r=current[:, owned])
        axd.pub(current)
        axd.sub(current)
        for phase, stage in enumerate(stages):
            (_, attn_w, moe_weights), banks = block_layer(stage, block_index, input_banks)
            activation = ax.dsl.create_tensor(
                x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name=f"attention_handoff_{block}_{phase}"
            )
            if phase < 3:
                GDNKernel.func(
                    current.reshape(1, 1, -1),
                    None,
                    output=None,
                    residual_out=activation.reshape(1, 1, -1),
                    banks=banks,
                    **attn_w,
                )
            else:
                kv = weights[f"b3_kv{block}"]
                decode = decode_paged_kv_shared_kv_kernel if kv.shape[3] == 2 else decode_paged_kv_kernel
                decode.func(
                    current.reshape(1, 1, -1),
                    *(attn_w[name] for name in ATTENTION_WEIGHT_NAMES[:17]),
                    position_ids,
                    *(attn_w[name] for name in ATTENTION_WEIGHT_NAMES[17:]),
                    None,
                    None,
                    activation.reshape(1, 1, -1),
                    kv,
                    indptr,
                    indices,
                    last_page_len,
                    256,
                    attn_output_gate=True,
                    partial_rotary_factor=attn_w["partial_rotary_factor"],
                    mrope_section=mrope_section,
                    mrope_interleaved=True,
                    has_mrope=position_ids.ndim == 2,
                    cache_location=cache_location,
                    banks=banks,
                    cos_sin_cache=attn_w.get("cos_sin_cache"),
                )
            axd.pub(activation)
            axd.sub(activation)
            current = ax.dsl.create_tensor(x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name=f"moe_handoff_{block}_{phase}")
            moe_kernel_expert_parallel_v2_scf.func(activation, output_ax=current, banks=banks, **moe_weights)
            axd.pub(current)
            axd.sub(current)
        ax.dsl.move(current[:, owned], r=hidden)
    ax.dsl.store(hidden, r=output[:, owned])
    return output


@torch.no_grad()
def prepare_split(layers, *, mrope_section, cache_path, run_mode=None, weight_banks=()):
    """decode.prepare with the shared KV pool replaced by one input per attention layer (b3_kv<j>)"""
    assert len(layers) % 4 == 0
    assert all((kv is not None) == (i % 4 == 3) for i, (kv, _, _) in enumerate(layers))
    layout, weights = block_arguments(layers, "moe", weight_banks)
    kv_name = layout[3][0][0]
    pool = weights.pop(kv_name)
    for j in range(pool.shape[0]):
        weights[f"b3_kv{j}"] = pool[j]
    layout = tuple((None if phase == 3 else kv, a, m) for phase, (kv, a, m) in enumerate(layout))
    cache_path = Path(cache_path)
    kernel = ax.builder(
        run_mode=ax.RunMode(run_mode or os.getenv("RUN_MODE", "CRUN")),
        const_inputs=["layout", "mrope_section"],
        enable_cache=True,
        unique_axmodel=True,
        cache_path=cache_path,
        dump_path=cache_path / "artifacts",
    )(decode_qwen35_blocks_split)
    return kernel, dict(layout=layout, mrope_section=tuple(mrope_section), **weights)
