"""Fused Transformer decode matches the established standalone kernels."""

import os

import pytest
import torch
import axnn.dsl as ax
from axnn.dsl.ctx import ctx
from sgl_kernel_axera.models.qwen36_35b_a3b.decode import prepare
from test_qwen35_runtime import clone_layers, layer_case, standalone
from runtime_cases import bank_runtime_weights


@pytest.mark.parametrize("cores,mrope", [(4, False), (4, True)])
def test_mixed_decode_continuation_matches_separate_kernels(tmp_path, cores, mrope):
    torch.manual_seed(360)
    torch.set_num_threads(4)
    ax.init(chip_num=1)
    # The transition after GQA exercises writing hidden shards for a following GDN.
    kinds = ("gdn", "gdn", "gdn", "gqa") * 2
    layers = [layer_case(kind) for kind in kinds]
    if mrope:
        # Production shares contiguous recurrent/KV pools across all layers.
        # The dynamic body must select 3*block+phase without changing reset IO.
        linear = [weights for kv, weights, _ in layers if kv is None]
        for name in ("state", "q_conv_state", "k_conv_state", "v_conv_state"):
            pool = torch.stack([weights[name + "_in"] for weights in linear])
            for i, weights in enumerate(linear):
                weights[name + "_in"] = weights[name + "_out"] = pool[i]
        kv_pool = torch.stack([kv for kv, _, _ in layers if kv is not None])
        for block in range(len(kv_pool)):
            _, weights, moe = layers[4 * block + 3]
            layers[4 * block + 3] = kv_pool[block], weights, moe
        for name in ("cos", "sin"):
            layers[7][1][name] = layers[3][1][name]
    reference_layers = clone_layers(layers)
    if mrope:
        cache = torch.cat((layers[3][1]["cos"], layers[3][1]["sin"]), dim=-1)
        for block in (3, 7):
            layers[block][1].update(cos=None, sin=None, cos_sin_cache=cache)
    banks, pointers = bank_runtime_weights(layers)
    kernel, weights = prepare(layers, mrope_section=(2, 3, 3), cache_path=tmp_path, weight_banks=banks)
    assert all(value.is_contiguous() for value in weights.values() if torch.is_tensor(value))
    assert all(values[name].data_ptr() == pointer for values, name, pointer in pointers)
    if mrope:
        assert weights["b3_attn_cos_sin_cache"].shape[0] == 1
        assert weights["b3_attn_cos_sin_cache"].data_ptr() == cache.data_ptr()
    x = (torch.randn(1, 256) * 0.05).bfloat16()
    # The serving path reuses workspace.hidden as both the first input and final output.
    attention = torch.empty_like(x)
    output = x if mrope else torch.empty_like(x)
    ref_attention, ref_output = torch.empty_like(x), torch.empty_like(x)
    positions = torch.zeros((3, 1) if mrope else (1,), dtype=torch.int32)
    metadata = (
        positions,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.full((32,), 3, dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([3, 1], dtype=torch.int32),
    )
    for step, prefix in enumerate((1, 2, 257)):
        if step == 2:
            # Changing runtime weights must reuse the same compiled model.
            layers[0][2]["input_norm_w"].mul_(0.9)
            reference_layers[0][2]["input_norm_w"].mul_(0.9)
        positions.fill_(prefix)
        # Keep capacity fixed while a runtime prefix adds a noncontiguous KV page.
        metadata[1][1] = 2 if prefix > 256 else 1
        metadata[2][1] = 7
        metadata[-2].fill_(prefix % 256)
        metadata[-1][0] = 7 if prefix > 256 else 3
        metadata[-1][1] = prefix % 256
        expected = standalone(x, ref_attention, ref_output, reference_layers, metadata, cores).clone()
        kernel[cores, cores](x, attention, *metadata, output, **weights)
        torch.testing.assert_close(output.float(), expected.float(), rtol=0.03, atol=0.015)
        for (kv, attn, _), (ref_kv, ref_attn, _) in zip(layers, reference_layers):
            if kv is None:
                for name in (
                    "state_in",
                    "q_conv_state_in",
                    "k_conv_state_in",
                    "v_conv_state_in",
                ):
                    torch.testing.assert_close(
                        attn[name].float(),
                        ref_attn[name].float(),
                        rtol=0.03,
                        atol=0.003,
                    )
            else:
                torch.testing.assert_close(kv.float(), ref_kv.float(), rtol=0.03, atol=0.003)
        x.copy_(output)
    assert len(weights) < 256  # Block-indexed weights plus in-place GDN/KV state.
    assert kernel.const_inputs == ["layout", "mrope_section"]
    chip = ctx.sub_ctxs[0][-1]
    assert all(torch.is_tensor(chip.args[name]) for name in chip.input_names)
    if os.getenv("RUN_MODE") == "CRUN":
        assert len(list(tmp_path.glob("*.compiled.axmodel"))) == 1
