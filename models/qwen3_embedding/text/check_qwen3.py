"""PYRUN numerical check of the config-driven Qwen3 text kernels.

The checkpoint projections are MXFP8-quantized, then dequantized for both the
NPU reference run and the PyTorch Qwen3 decoder reference.  This isolates
kernel arithmetic and layout from quantization error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import axnn.dsl as ax

from qwen3_kernel import (
    DECODE_ROWS,
    PREFILL_ROWS,
    dequantize_mxfp8,
    load_spec,
    make_cache,
    make_rope_tables,
    prepare,
    quantize_mxfp8,
)


HERE = Path(__file__).resolve().parent


class Shards:
    def __init__(self, root: Path):
        self.root = root
        index = root / "model.safetensors.index.json"
        if index.is_file():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            from safetensors import safe_open

            with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
                self.map = {name: "model.safetensors" for name in handle.keys()}
        self.handles = {}

    def get(self, name):
        from safetensors import safe_open

        shard = self.map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                str(self.root / shard), framework="pt", device="cpu"
            )
        return self.handles[shard].get_tensor(name)

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


def layer_prefix(name: str, index: int):
    if name.startswith("asr_"):
        return f"thinker.model.layers.{index}."
    if name.startswith("tts_"):
        return f"talker.model.layers.{index}."
    return f"model.layers.{index}."


def norm_name(name: str):
    if name.startswith("asr_"):
        return "thinker.model.norm.weight"
    if name.startswith("tts_"):
        return "talker.model.norm.weight"
    return "model.norm.weight"


def projection(shards: Shards, name: str):
    weight = shards.get(name).float()
    codes, scale = quantize_mxfp8(weight)
    return codes, scale, dequantize_mxfp8(codes, scale).to(torch.bfloat16)


def kernel_layer(spec, shards: Shards, index: int, cos, sin, cache):
    prefix = layer_prefix(spec.name, index)
    values = {
        "w_norm_attn": shards.get(prefix + "input_layernorm.weight").to(torch.bfloat16),
        "w_norm_q": shards.get(prefix + "self_attn.q_norm.weight").float()
        / spec.head_dim**0.5,
        "w_norm_k": shards.get(prefix + "self_attn.k_norm.weight").to(torch.bfloat16),
        "cos": cos,
        "sin": sin,
        "partial_rotary_factor": 1.0,
        "mrope_interleaved": spec.mrope_interleaved,
    }
    dequantized = {}
    for short, hf_name in (
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
    ):
        codes, scale, deq = projection(shards, prefix + hf_name)
        dequantized[short] = deq
        values[f"w_{short}"] = codes
        values[f"w_scale_{short}"] = scale
        values[f"w_scale2_{short}"] = torch.ones(1, dtype=torch.float32)
        values[f"w_zeropoint_{short}"] = torch.zeros(1, dtype=torch.bfloat16)

    mlp = {
        "w_norm": shards.get(prefix + "post_attention_layernorm.weight").to(torch.bfloat16),
        "input_permutation": torch.zeros(1, dtype=torch.int32),
    }
    for number, hf_name in (
        (1, "mlp.gate_proj.weight"),
        (2, "mlp.up_proj.weight"),
        (3, "mlp.down_proj.weight"),
    ):
        codes, scale, deq = projection(shards, prefix + hf_name)
        dequantized[f"w{number}"] = deq
        mlp[f"w{number}"] = codes
        mlp[f"w_scale{number}"] = scale
        mlp[f"w_scale2_{number}"] = torch.ones(1, dtype=torch.float32)
        mlp[f"w_zp_scale{number}"] = torch.zeros(1, dtype=torch.bfloat16)
    return (cache[index], values, mlp), dequantized


def hf_config(spec, torch_dtype=torch.bfloat16):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_layers,
        num_attention_heads=spec.num_q_heads,
        num_key_value_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        hidden_act="silu",
        max_position_embeddings=spec.context,
        initializer_range=0.02,
        rms_norm_eps=spec.rms_norm_eps,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=spec.rope_theta,
        attention_bias=False,
        attention_dropout=0.0,
        torch_dtype=torch_dtype,
    )
    config._attn_implementation = "eager"
    return config


def hf_layer_from_dequant(config, spec, index, shards, dequantized):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    prefix = layer_prefix(spec.name, index)
    layer = Qwen3DecoderLayer(config, index).to(torch.bfloat16).eval()
    state = {
        "input_layernorm.weight": shards.get(prefix + "input_layernorm.weight"),
        "post_attention_layernorm.weight": shards.get(prefix + "post_attention_layernorm.weight"),
        "self_attn.q_norm.weight": shards.get(prefix + "self_attn.q_norm.weight"),
        "self_attn.k_norm.weight": shards.get(prefix + "self_attn.k_norm.weight"),
        "self_attn.q_proj.weight": dequantized["q"],
        "self_attn.k_proj.weight": dequantized["k"],
        "self_attn.v_proj.weight": dequantized["v"],
        "self_attn.o_proj.weight": dequantized["o"],
        "mlp.gate_proj.weight": dequantized["w1"],
        "mlp.up_proj.weight": dequantized["w2"],
        "mlp.down_proj.weight": dequantized["w3"],
    }
    layer.load_state_dict(state, strict=True)
    return layer.eval()


def make_inputs(spec, shards: Shards, rows: int):
    generator = torch.Generator().manual_seed(20261031)
    if spec.name.startswith("tts_"):
        text = shards.get("talker.model.text_embedding.weight")
        text_ids = torch.randint(0, text.shape[0], (rows,), generator=generator)
        text_hidden = text[text_ids].float()
        fc1 = shards.get("talker.text_projection.linear_fc1.weight").float()
        b1 = shards.get("talker.text_projection.linear_fc1.bias").float()
        fc2 = shards.get("talker.text_projection.linear_fc2.weight").float()
        b2 = shards.get("talker.text_projection.linear_fc2.bias").float()
        projected = F.silu(F.linear(text_hidden, fc1, b1))
        projected = F.linear(projected, fc2, b2)
        codec = shards.get("talker.model.codec_embedding.weight")
        codec_ids = torch.randint(0, codec.shape[0], (rows,), generator=generator)
        return (projected + codec[codec_ids].float()).to(torch.bfloat16)
    names = {
        "embedding_0.6b": ("model.embed_tokens.weight", "embed_tokens.weight"),
        "asr_0.6b": ("thinker.model.embed_tokens.weight", "thinker.model.language_model.embed_tokens.weight"),
        "asr_1.7b": ("thinker.model.embed_tokens.weight", "thinker.model.language_model.embed_tokens.weight"),
    }[spec.name]
    name = next((value for value in names if value in shards.map), None)
    if name is None:
        raise KeyError(f"no embedding tensor among {names}")
    embedding = shards.get(name)
    ids = torch.randint(0, embedding.shape[0], (rows,), generator=generator)
    return embedding[ids].to(torch.bfloat16)


def positions(spec, start: int, rows: int, *, varied_mrope: bool):
    base = np.arange(start, start + rows, dtype=np.int32)
    if not spec.has_mrope:
        return torch.from_numpy(base)
    if varied_mrope:
        h = start // 2 + np.arange(rows, dtype=np.int32) // 2
        w = start // 4 + np.arange(rows, dtype=np.int32) // 4
    else:
        h = base
        w = base
    return torch.from_numpy(np.stack((base, h, w), axis=0))


def position_embeddings(config, spec, hidden, pos):
    if not spec.has_mrope:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        return Qwen3RotaryEmbedding(config)(hidden[None], pos[None])
    half = spec.head_dim // 2
    inv_freq = 1.0 / (
        spec.rope_theta
        ** (torch.arange(0, spec.head_dim, 2, dtype=torch.float32) / spec.head_dim)
    )
    freqs = inv_freq[None, :, None] * pos.float()[:, None, :]
    freqs = freqs.transpose(1, 2)
    mixed = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = spec.mrope_section[dim] * 3
        mixed[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
    emb = torch.cat((mixed, mixed), dim=-1)
    return emb.cos().to(torch.bfloat16)[None], emb.sin().to(torch.bfloat16)[None]


def run_hf_layers(hf_layers, hidden, pos, spec, config, *, collect_kv=False):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    tokens = hidden.shape[0]
    mask = torch.full((1, 1, tokens, tokens), float("-inf"), dtype=torch.bfloat16).triu(1)
    rope_cos_sin = position_embeddings(config, spec, hidden, pos)
    current = hidden
    keys, values = [], []
    with torch.no_grad():
        for layer in hf_layers:
            if collect_kv:
                normed = layer.input_layernorm(current)
                attention = layer.self_attn
                k = attention.k_proj(normed).view(1, tokens, spec.num_kv_heads, spec.head_dim)
                k = attention.k_norm(k).transpose(1, 2)
                v = attention.v_proj(normed).view(1, tokens, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
                _, k = apply_rotary_pos_emb(k, k, *rope_cos_sin)
                keys.append(k.transpose(1, 2).reshape(tokens, spec.num_kv_heads, spec.head_dim))
                values.append(v.transpose(1, 2).reshape(tokens, spec.num_kv_heads, spec.head_dim))
            result = layer(
                current[None],
                attention_mask=mask,
                position_ids=pos[None],
                position_embeddings=rope_cos_sin,
            )
            current = result[0] if isinstance(result, tuple) else result
            current = current[0]
    return current, keys, values


def final_norm(hidden, weight):
    x = hidden.float()
    denom = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    normalized = (x * denom).to(hidden.dtype)
    return normalized * weight


def stats(actual, expected):
    actual = actual.float()
    expected = expected.float()
    cosine = F.cosine_similarity(actual, expected, dim=-1)
    ratio = actual.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-20)
    error = (actual - expected).abs()
    return {
        "cos_mean": cosine.mean().item(),
        "cos_min": cosine.min().item(),
        "cos_p01": cosine.quantile(0.01).item(),
        "norm_ratio_mean": ratio.mean().item(),
        "norm_ratio_min": ratio.min().item(),
        "norm_ratio_max": ratio.max().item(),
        "abs_mean": error.mean().item(),
        "abs_max": error.max().item(),
    }


def print_stats(label, value):
    print(
        f"{label}: cos mean={value['cos_mean']:.7f} min={value['cos_min']:.7f} "
        f"p01={value['cos_p01']:.7f}; norm ratio mean={value['norm_ratio_mean']:.6f} "
        f"range=[{value['norm_ratio_min']:.6f},{value['norm_ratio_max']:.6f}]; "
        f"abs mean={value['abs_mean']:.6g} max={value['abs_max']:.6g}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--mode", choices=("history", "full"), default="history")
    parser.add_argument("--hf", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--varied-mrope", action="store_true")
    parser.add_argument("--init", choices=("embedding", "random"), default="embedding")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.layers % 4:
        raise ValueError("layers must be a multiple of four")

    torch.set_num_threads(args.threads or min(64, os.cpu_count() or 1))
    ax.init(chip_num=1)
    spec = load_spec(args.model, config_path=args.config)
    roots = {
        "embedding_0.6b": Path("/data/junda/work/qsmall/embedding/hf/Qwen3-0.6B"),
        "asr_0.6b": "Qwen3-ASR-0.6B",
        "asr_1.7b": "Qwen3-ASR-1.7B",
        "tts_0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
        "tts_1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    }[args.model]
    if not isinstance(roots, Path):
        roots = HERE / "hf" / roots
    root = Path(args.hf) if args.hf else roots
    if not (root / "config.json").is_file():
        raise FileNotFoundError(root)
    print(f"model={args.model} HF={root} layers={args.layers} mode={args.mode}", flush=True)

    config = hf_config(spec)
    shards = Shards(root)
    cos, sin = make_rope_tables(spec)
    cache = make_cache(spec, args.layers)
    state, dequantized = [], []
    for index in range(args.layers):
        layer, deq = kernel_layer(spec, shards, index, cos, sin, cache)
        state.append(layer)
        dequantized.append(deq)
    final_weight = shards.get(norm_name(spec.name)).to(torch.bfloat16)
    out_dir = Path(args.out or HERE / "out" / f"check_{args.model}_{args.mode}{args.layers}")
    kernels, weights = prepare(
        spec,
        state,
        final_norm=final_weight,
        cache_path=out_dir,
        run_mode="PYRUN",
    )

    if args.init == "embedding":
        inputs = make_inputs(spec, shards, 2 * PREFILL_ROWS)
    else:
        generator = torch.Generator().manual_seed(20261031 + args.layers)
        inputs = (torch.randn((2 * PREFILL_ROWS, spec.hidden_size), generator=generator) * 0.02).bfloat16()
    x0, x1 = inputs[:PREFILL_ROWS], inputs[PREFILL_ROWS:]
    print(
        f"input={args.init} x0 norm mean={x0.float().norm(dim=-1).mean():.6f}",
        flush=True,
    )
    pos0 = positions(spec, 0, PREFILL_ROWS, varied_mrope=args.varied_mrope)
    pos1 = positions(spec, PREFILL_ROWS, PREFILL_ROWS, varied_mrope=args.varied_mrope)
    posd = positions(spec, PREFILL_ROWS, DECODE_ROWS, varied_mrope=args.varied_mrope)
    indices = torch.zeros(spec.pages, dtype=torch.int32)
    out0 = torch.empty_like(x0)
    print("kernel call 1: prefix 0", flush=True)
    kernels[PREFILL_ROWS][4, 4](
        x0,
        pos0,
        torch.tensor([0, 0], dtype=torch.int32),
        indices,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        out0,
        valid_tokens=torch.tensor([PREFILL_ROWS], dtype=torch.int32),
        **weights,
    )
    cache_after_first = cache.clone()

    hf_layers = [
        hf_layer_from_dequant(config, spec, index, shards, deq)
        for index, deq in enumerate(dequantized)
    ]
    print("reference call 1", flush=True)
    ref0, ref_k0, ref_v0 = run_hf_layers(
        hf_layers,
        x0,
        pos0,
        spec,
        config,
        collect_kv=(args.mode == "history"),
    )
    ref0 = final_norm(ref0, final_weight)
    result = {
        "model": spec.as_dict(),
        "hf": str(root),
        "layers": args.layers,
        "mode": args.mode,
        "varied_mrope": args.varied_mrope,
        "call1": stats(out0, ref0),
    }
    print_stats("call1 output", result["call1"])

    if args.mode == "history":
        outd = torch.empty((DECODE_ROWS, spec.hidden_size), dtype=torch.bfloat16)
        print("kernel decode T=1 after call 1", flush=True)
        kernels[DECODE_ROWS][4, 4](
            x1[:1],
            posd,
            torch.tensor([0, 1], dtype=torch.int32),
            indices,
            torch.tensor([PREFILL_ROWS], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int32),
            outd,
            **weights,
        )
        out1 = torch.empty_like(x1)
        print("kernel call 2: prefix 256, one history page", flush=True)
        kernels[PREFILL_ROWS][4, 4](
            x1,
            pos1,
            torch.tensor([0, 1], dtype=torch.int32),
            indices,
            torch.tensor([PREFILL_ROWS], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int32),
            out1,
            valid_tokens=torch.tensor([PREFILL_ROWS], dtype=torch.int32),
            **weights,
        )
        print("reference call 2 (prefix 512)", flush=True)
        ref_all, ref_k, ref_v = run_hf_layers(
            hf_layers,
            torch.cat((x0, x1), dim=0),
            positions(spec, 0, 2 * PREFILL_ROWS, varied_mrope=args.varied_mrope),
            spec,
            config,
            collect_kv=True,
        )
        ref_all = final_norm(ref_all, final_weight)
        result["call2"] = stats(out1, ref_all[PREFILL_ROWS:])
        result["decode"] = stats(outd, ref_all[PREFILL_ROWS : PREFILL_ROWS + 1])
        result["decode_vs_call2"] = stats(outd, out1[:1])
        print_stats("call2 output", result["call2"])
        print_stats("decode output", result["decode"])
        print_stats("decode vs call2 row0", result["decode_vs_call2"])

        k0 = torch.stack([key[:PREFILL_ROWS] for key in ref_k0], 0)
        v0 = torch.stack([value[:PREFILL_ROWS] for value in ref_v0], 0)
        k1 = torch.stack([key[PREFILL_ROWS:] for key in ref_k], 0)
        v1 = torch.stack([value[PREFILL_ROWS:] for value in ref_v], 0)
        after0 = cache_after_first[:, 0]
        after1 = cache[:, 1]
        result["kv_page0_k"] = stats(after0[:, 0].reshape(-1, spec.head_dim), k0.reshape(-1, spec.head_dim))
        result["kv_page0_v"] = stats(after0[:, 1].reshape(-1, spec.head_dim), v0.reshape(-1, spec.head_dim))
        result["kv_page1_k"] = stats(after1[:, 0].reshape(-1, spec.head_dim), k1.reshape(-1, spec.head_dim))
        result["kv_page1_v"] = stats(after1[:, 1].reshape(-1, spec.head_dim), v1.reshape(-1, spec.head_dim))
        for name in ("kv_page0_k", "kv_page0_v", "kv_page1_k", "kv_page1_v"):
            print_stats(name, result[name])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_dir / 'result.json'}", flush=True)
    shards.close()


if __name__ == "__main__":
    main()
