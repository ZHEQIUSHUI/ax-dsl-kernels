"""Qwen3.5 runtime NVFP4 experts and checkpoint conversion contracts."""


import pytest
import torch
import torch.nn.functional as F

import axnn.dsl as ax
from sgl_kernel_axera.quantization import quantize_nvfp4
from test_nvfp4_lm_head import dequantize
from sgl_kernel_axera.moe.m_qwen3 import moe_kernel_expert_parallel_v2_scf
from sgl_kernel_axera.attention.gdn import GDNKernel, runtime_weights, split_conv_state
from sgl_kernel_axera.attention.flash_attn import (
    flash_attention,
    decode_paged_kv_shared_kv_kernel,
)
from test_gdn_attn import random_case, random_nvfp4_weights


@pytest.mark.parametrize("heads,gated", [(4, False), (32, True)])
def test_gqa_decode_runtime_page_counts(heads, gated, tmp_path, monkeypatch):
    """Reuse the decode model across noncontiguous pages and an empty page shard."""
    torch.manual_seed(359)
    torch.set_num_threads(4)
    hidden, head_dim, kv_heads = 256, 64, 2
    weights, decoded = {}, {}
    for name, shape in (
        ("q", (heads * head_dim * (2 if gated else 1), hidden)),
        ("k", (kv_heads * head_dim, hidden)),
        ("v", (kv_heads * head_dim, hidden)),
        ("o", (hidden, heads * head_dim)),
    ):
        packed = quantize_nvfp4(torch.randn(shape, dtype=torch.bfloat16) * 0.02)
        weights.update(
            {
                f"w_{name}": packed[0],
                f"w_scale_{name}": packed[1],
                f"w_scale2_{name}": packed[2],
                f"w_zeropoint_{name}": torch.zeros(1),
            }
        )
        decoded[name] = dequantize(*packed)
    norm = (torch.randn(hidden) * 0.1 + 1).bfloat16()
    q_norm, k_norm = ((torch.randn(head_dim) * 0.1 + 1).bfloat16() for _ in range(2))
    angles = torch.arange(1024)[:, None] / (1e7 ** (torch.arange(0, head_dim, 2).float() / head_dim))[None, :]
    weights.update(
        w_norm_attn=norm,
        w_norm_q=q_norm.float() / head_dim**0.5,
        w_norm_k=k_norm,
        cos=angles.cos(),
        sin=angles.sin(),
    )
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    monkeypatch.setattr(decode_paged_kv_shared_kv_kernel, "cache_path", cache_path)
    monkeypatch.setattr(decode_paged_kv_shared_kv_kernel, "dump_path", tmp_path)
    for history in (0, 1, 256, 514):
        x = (torch.randn(1, 1, hidden) * 0.05).bfloat16()
        n = F.rms_norm(x.float(), (hidden,), norm.float(), eps=1e-6).bfloat16()
        q = F.linear(n.float(), decoded["q"]).bfloat16().reshape(1, 1, heads, -1)
        q, gate = q.chunk(2, -1) if gated else (q, None)
        k = F.linear(n.float(), decoded["k"]).bfloat16().reshape(1, 1, kv_heads, head_dim)
        v = F.linear(n.float(), decoded["v"]).bfloat16().reshape(1, 1, kv_heads, head_dim)
        q = F.rms_norm(q.float(), (head_dim,), q_norm.float(), eps=1e-6).bfloat16()
        k = F.rms_norm(k.float(), (head_dim,), k_norm.float(), eps=1e-6).bfloat16()
        cosine = weights["cos"][history].reshape(1, 1, 1, -1)
        sine = weights["sin"][history].reshape(1, 1, 1, -1)
        for value in (q, k):
            first, second = value.chunk(2, -1)
            first, second = first.float(), second.float()
            value[..., : head_dim // 2] = first * cosine - second * sine
            value[..., head_dim // 2 :] = second * cosine + first * sine
        cache = (torch.randn(32, 2, 256, kv_heads, head_dim) * 0.05).bfloat16()
        indices = torch.tensor([7, 2, 11] + [0] * 29, dtype=torch.int32)
        page_count = history // 256 + 1
        rows = indices[:page_count].long()
        history_k = cache[rows, 0].reshape(1, -1, kv_heads, head_dim)[:, :history]
        history_v = cache[rows, 1].reshape(1, -1, kv_heads, head_dim)[:, :history]
        attention = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            torch.cat((history_k, k), 1).transpose(1, 2),
            torch.cat((history_v, v), 1).transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)
        if gated:
            attention = attention * gate.sigmoid()
        expected = x + F.linear(attention.reshape(1, 1, -1).float(), decoded["o"]).bfloat16()
        output = torch.empty_like(x)
        cache_before = cache.clone()
        flash_attention(
            x,
            weights,
            torch.tensor([history], dtype=torch.int32),
            output,
            cache,
            torch.tensor([0, page_count], dtype=torch.int32),
            indices,
            torch.tensor([history % 256] + [0] * 31, dtype=torch.int32),
            cache_location=torch.tensor([int(rows[-1]), history % 256], dtype=torch.int32),
            core_num=4,
            attn_output_gate=gated,
        )
        torch.testing.assert_close(output.float(), expected.float(), rtol=0.06, atol=0.003)
        assert (output.float() - expected.float()).norm() / expected.float().norm() < 0.02
        torch.testing.assert_close(
            cache[rows[-1], 0, history % 256].float(),
            k[0, 0].float(),
            rtol=0.03,
            atol=0.003,
        )
        torch.testing.assert_close(
            cache[rows[-1], 1, history % 256].float(),
            v[0, 0].float(),
            rtol=0.03,
            atol=0.003,
        )
        cache_before[rows[-1], :, history % 256] = cache[rows[-1], :, history % 256]
        torch.testing.assert_close(cache, cache_before, rtol=0, atol=0)
    # Empty and nonempty history share one graph; counts, positions and page IDs are runtime data.
    if decode_paged_kv_shared_kv_kernel.run_mode == ax.RunMode.CRUN:
        assert len(list((tmp_path / "cache").glob("*.compiled.axmodel"))) == 1


def moe_weights(hidden=256, experts=16, intermediate=256):
    def packed(shape):
        weight = (torch.randn(shape) * 0.025).bfloat16()
        payload, scale, global_scale = quantize_nvfp4(weight.reshape(-1, shape[-1]))
        return payload.reshape(*shape[:-1], shape[-1] // 2), scale.reshape(*shape[:-1], shape[-1] // 16), global_scale

    values = dict(
        input_norm_w=(1 + torch.randn(hidden) * 0.05).bfloat16(),
        add_residual=True,
        shared_expert_gate=torch.randn(hidden, 1).bfloat16() * 0.03,
        w_route=torch.randn(experts, hidden).bfloat16() * 0.1,
        w_scale_route=torch.ones(experts, 1),
        w_scale2_route=torch.ones(1),
        top_k=8,
        num_experts=experts,
        norm_topk_prob=True,
        fused_gate_up=True,
    )
    gate_up = packed((experts, 2 * intermediate, hidden))
    down = packed((experts, hidden, intermediate))
    gate_up = (*gate_up[:2], gate_up[2].expand(experts, 2).contiguous())
    down = (*down[:2], down[2].expand(experts).contiguous())
    for name, weight in [("gate", gate_up), ("up", gate_up), ("down", down)]:
        for prefix, value in zip(("w_", "w_scale_", "w_scale2_"), weight):
            values[prefix + name] = value
    for name, shape in [
        ("gate", (intermediate, hidden)),
        ("up", (intermediate, hidden)),
        ("down", (hidden, intermediate)),
    ]:
        for prefix, value in zip(("w_", "w_scale_", "w_scale2_"), packed(shape)):
            values[prefix + "shared_expert_mlp_" + name] = value
    return values


def layer_case(kind, hidden=256):
    if kind == "gdn":
        case = random_case(1, hidden=hidden, key_heads=4, value_heads=8, key_dim=16, value_dim=16)
        _, _, weights, state, conv = case
        weights, _ = random_nvfp4_weights(weights, seed=int(torch.randint(1000, (1,))))
        weights = runtime_weights(weights)
        weights["w_pre"] = None
        states = (state, *split_conv_state(conv, 64, 128))
        for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"), states):
            value = value.contiguous()
            weights[name + "_in"] = value
            weights[name + "_out"] = value
        return None, weights, moe_weights(hidden)
    weights = {}
    heads, kv_heads, head_dim = 8, 2, 64
    for name, shape in [
        ("q", (heads * head_dim * 2, hidden)),
        ("k", (kv_heads * head_dim, hidden)),
        ("v", (kv_heads * head_dim, hidden)),
        ("o", (hidden, heads * head_dim)),
    ]:
        for prefix, value in zip(
            ("w_", "w_scale_", "w_scale2_"), quantize_nvfp4((torch.randn(shape) * 0.025).bfloat16())
        ):
            weights[prefix + name] = value
        weights["w_zeropoint_" + name] = torch.randn(1) * 0.01
    angles = torch.arange(1024)[:, None] / (1e7 ** (torch.arange(0, 16, 2).float() / 16))[None, :]
    weights.update(
        w_norm_attn=(1 + torch.randn(hidden) * 0.05).bfloat16(),
        w_norm_q=(1 + torch.randn(head_dim) * 0.05) / head_dim**0.5,
        w_norm_k=(1 + torch.randn(head_dim) * 0.05).bfloat16(),
        cos=angles.cos(),
        sin=angles.sin(),
        partial_rotary_factor=0.25,
    )
    cache = (torch.randn(32, 2, 256, kv_heads, head_dim) * 0.03).bfloat16()
    return cache, weights, moe_weights(hidden)


def clone_layers(layers):
    result = []
    for kv, attn, moe in layers:
        # Preserve aliases between GDN state inputs/outputs and fused MoE gate/up.
        copies = {}

        def clone(value):
            if not torch.is_tensor(value):
                return value
            if id(value) not in copies:
                copies[id(value)] = value.clone()
            return copies[id(value)]

        result.append((clone(kv), {k: clone(v) for k, v in attn.items()}, {k: clone(v) for k, v in moe.items()}))
    return result


def standalone(x, attention, output, layers, metadata, cores, mrope_section=(2, 3, 3)):
    current = x
    positions, indptr, indices, lengths, location = metadata
    for kv, weights, moe in layers:
        if kv is None:
            GDNKernel[cores, cores](
                current.view(1, 1, -1), None, output=None, residual_out=attention.view(1, 1, -1), **weights
            )
        else:
            flash_attention(
                current.view(1, 1, -1),
                weights,
                positions,
                attention.view(1, 1, -1),
                kv,
                indptr,
                indices,
                lengths,
                core_num=cores,
                cache_location=location,
                attn_output_gate=True,
                partial_rotary_factor=0.25,
                mrope_section=mrope_section,
                mrope_interleaved=True,
                has_mrope=positions.ndim == 2,
            )
        moe_kernel_expert_parallel_v2_scf[cores, cores](attention, output_ax=output, **moe)
        current = output
    return output
