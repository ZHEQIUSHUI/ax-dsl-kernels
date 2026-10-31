"""Build random-weight real-shape Qwen3 prefill/decode kernels.

Examples:
    RUN_MODE=PYRUN python build_qwen3.py out --model embedding_0.6b --layers 4
    RUN_MODE=CRUN  python build_qwen3.py out --model asr_1.7b --layers 28
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import axnn.dsl as ax

from qwen3_kernel import (
    DECODE_ROWS,
    PREFILL_ROWS,
    make_cache,
    make_rope_tables,
    prepare,
    quantize_mxfp8,
)


WEIGHT_RNG = torch.Generator().manual_seed(20261031)


def dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.uint32: "uint32",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.float8_e8m0fnu: "float8_e8m0fnu",
    }.get(dtype, str(dtype).removeprefix("torch."))


def mxfp8_random(n: int, k: int):
    weight = torch.randn((n, k), generator=WEIGHT_RNG, dtype=torch.float32) * 0.02
    return quantize_mxfp8(weight)


def attention_layer(spec, cos, sin, cache):
    weights = {
        "w_norm_attn": torch.ones(spec.hidden_size, dtype=torch.bfloat16),
        "w_norm_q": torch.full((spec.head_dim,), spec.head_dim**-0.5, dtype=torch.float32),
        "w_norm_k": torch.ones(spec.head_dim, dtype=torch.bfloat16),
        "cos": cos,
        "sin": sin,
        "partial_rotary_factor": 1.0,
        "mrope_interleaved": spec.mrope_interleaved,
    }
    for name, shape in (
        ("q", (spec.num_q_heads * spec.head_dim, spec.hidden_size)),
        ("k", (spec.num_kv_heads * spec.head_dim, spec.hidden_size)),
        ("v", (spec.num_kv_heads * spec.head_dim, spec.hidden_size)),
        ("o", (spec.hidden_size, spec.num_q_heads * spec.head_dim)),
    ):
        codes, scale = mxfp8_random(*shape)
        weights[f"w_{name}"] = codes
        weights[f"w_scale_{name}"] = scale
        weights[f"w_scale2_{name}"] = torch.ones(1, dtype=torch.float32)
        weights[f"w_zeropoint_{name}"] = torch.zeros(1, dtype=torch.bfloat16)
    return cache, weights, mlp_layer(spec)


def mlp_layer(spec):
    weights = {
        "w_norm": torch.ones(spec.hidden_size, dtype=torch.bfloat16),
        "input_permutation": torch.zeros(1, dtype=torch.int32),
    }
    for index, shape in enumerate(
        (
            (spec.intermediate_size, spec.hidden_size),
            (spec.intermediate_size, spec.hidden_size),
            (spec.hidden_size, spec.intermediate_size),
        ),
        1,
    ):
        codes, scale = mxfp8_random(*shape)
        weights[f"w{index}"] = codes
        weights[f"w_scale{index}"] = scale
        weights[f"w_scale2_{index}"] = torch.ones(1, dtype=torch.float32)
        weights[f"w_zp_scale{index}"] = torch.zeros(1, dtype=torch.bfloat16)
    return weights


def make_state(spec, layers: int):
    cos, sin = make_rope_tables(spec)
    cache = make_cache(spec, layers)
    state = [attention_layer(spec, cos, sin, cache[index]) for index in range(layers)]
    return state, cache


def positions(spec, start: int, rows: int, dtype=torch.int32):
    value = np.arange(start, start + rows, dtype=np.int32)
    if not spec.has_mrope:
        return torch.from_numpy(value).to(dtype)
    t = value
    h = start // 2 + np.arange(rows, dtype=np.int32) // 2
    w = start // 4 + np.arange(rows, dtype=np.int32) // 4
    return torch.from_numpy(np.stack((t, h, w), axis=0)).to(dtype)


def describe_runtime(spec, weights):
    runtime = {}
    for name, value in weights.items():
        if name in ("layout", "mrope_section") or not torch.is_tensor(value):
            continue
        runtime[name] = {"shape": list(value.shape), "dtype": dtype_name(value.dtype)}
    variants = {}
    for rows in (PREFILL_ROWS, DECODE_ROWS):
        pos_shape = [3, rows] if spec.has_mrope else [rows]
        extra = {
            "x": {"shape": [rows, spec.hidden_size], "dtype": "bfloat16"},
            "position_ids": {"shape": pos_shape, "dtype": "int32"},
            "indptr": {"shape": [2], "dtype": "int32"},
            "indices": {"shape": [spec.pages], "dtype": "int32"},
            "last_page_len": {"shape": [1], "dtype": "int32"},
            "cache_location": {"shape": [2], "dtype": "int32"},
            "final_norm": {"shape": [spec.hidden_size], "dtype": "bfloat16"},
            "output": {"shape": [rows, spec.hidden_size], "dtype": "bfloat16"},
        }
        if rows == PREFILL_ROWS:
            extra["valid_tokens"] = {"shape": [1], "dtype": "int32"}
        variants[f"rows{rows}"] = {"inputs": {**extra, **runtime}}
    return {"model": spec.as_dict(), "variants": variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    from qwen3_kernel import load_spec

    spec = load_spec(args.model, config_path=args.config)
    layers = args.layers or spec.num_layers
    if layers % 4:
        raise ValueError("layers must be a multiple of four")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    ax.init(chip_num=1)

    state, cache = make_state(spec, layers)
    final_norm = torch.ones(spec.hidden_size, dtype=torch.bfloat16)
    kernels, weights = prepare(
        spec,
        state,
        final_norm=final_norm,
        cache_path=out,
        run_mode=os.getenv("RUN_MODE", "CRUN"),
    )
    manifest = describe_runtime(spec, weights)
    manifest["layers"] = layers
    manifest["cache"] = {"shape": list(cache.shape), "dtype": dtype_name(cache.dtype)}
    manifest["context"] = spec.context
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"model={args.model} layers={layers} H={spec.hidden_size} pages={spec.pages} "
        f"mrope={spec.mrope_section} inputs={len(weights)}",
        flush=True,
    )

    x0 = (torch.randn((PREFILL_ROWS, spec.hidden_size), generator=WEIGHT_RNG) * 0.02).bfloat16()
    x1 = (torch.randn((DECODE_ROWS, spec.hidden_size), generator=WEIGHT_RNG) * 0.02).bfloat16()
    out0 = torch.empty_like(x0)
    out1 = torch.empty_like(x1)
    indices = torch.zeros(spec.pages, dtype=torch.int32)
    print("prefill graph", flush=True)
    kernels[PREFILL_ROWS][4, 4](
        x0,
        positions(spec, 0, PREFILL_ROWS),
        torch.tensor([0, 0], dtype=torch.int32),
        indices,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        out0,
        valid_tokens=torch.tensor([PREFILL_ROWS], dtype=torch.int32),
        **weights,
    )
    print("decode graph", flush=True)
    kernels[DECODE_ROWS][4, 4](
        x1,
        positions(spec, PREFILL_ROWS, DECODE_ROWS),
        torch.tensor([0, 1], dtype=torch.int32),
        indices,
        torch.tensor([PREFILL_ROWS], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        out1,
        **weights,
    )
    print(
        f"done prefill_finite={bool(torch.isfinite(out0).all())} "
        f"decode_finite={bool(torch.isfinite(out1).all())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
