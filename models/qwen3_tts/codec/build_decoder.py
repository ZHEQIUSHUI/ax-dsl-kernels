"""Build the whole Qwen3-TTS-12Hz codec decoder as one AX8860 kernel.

    RUN_MODE=PYRUN python build_decoder.py OUT     # DSL smoke test
    RUN_MODE=CRUN  python build_decoder.py OUT     # compile (stops at the axmodel)

Random weights with the real packed shapes; separate from check_decoder.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

import axnn.dsl as ax
from decoder_kernel import WINDOW, SAMPLES, DAC_RES, decoder
from weight_layout import LAYOUT

ROOT = Path(__file__).resolve().parent


def bank_elems(group):
    return sum(math.prod(spec["shape"]) for spec in LAYOUT[group].values())


QN, UN, DN = bank_elems("quant_w"), bank_elems("up_w"), bank_elems("dac_w")


def inputs_spec():
    """(name, shape, dtype) in the exact kernel parameter order."""
    s = [
        ("codes", [WINDOW, 16], "int32"),
        ("quant_w", [QN], "bfloat16"),
        ("up_w", [UN], "bfloat16"),
        ("dac_w", [DN], "bfloat16"),
        ("pos", [1], "int32"),
        ("kv_state", [8, 2, 1024, 71], "bfloat16"),
        ("kv_next", [8, 2, 1024, 71], "bfloat16"),
        ("pre_state", [2, 512], "bfloat16"),
        ("pre_next", [2, 512], "bfloat16"),
        ("up0_state", [6, 1024], "bfloat16"),
        ("up0_next", [6, 1024], "bfloat16"),
        ("up1_state", [6, 1024], "bfloat16"),
        ("up1_next", [6, 1024], "bfloat16"),
        ("dac_entry_state", [6, 1024], "bfloat16"),
        ("dac_entry_next", [6, 1024], "bfloat16"),
        ("dac0_tail", [8, 768], "bfloat16"),
        ("dac0_tail_next", [8, 768], "bfloat16"),
        ("dac1_tail", [5, 384], "bfloat16"),
        ("dac1_tail_next", [5, 384], "bfloat16"),
        ("dac2_tail", [4, 192], "bfloat16"),
        ("dac2_tail_next", [4, 192], "bfloat16"),
        ("dac3_tail", [3, 96], "bfloat16"),
        ("dac3_tail_next", [3, 96], "bfloat16"),
    ]
    for b, c in DAC_RES:
        for r, d in enumerate((1, 3, 9)):
            s.append((f"dac{b}_res{r}_state", [6 * d, c], "bfloat16"))
            s.append((f"dac{b}_res{r}_next", [6 * d, c], "bfloat16"))
    s += [("post_state", [6, 96], "bfloat16"),
          ("post_next", [6, 96], "bfloat16"),
          ("wav", [SAMPLES], "float32")]
    return s


DT = {"int32": torch.int32, "bfloat16": torch.bfloat16, "float32": torch.float32}


def make_inputs(seed=20261031):
    g = torch.Generator().manual_seed(seed)
    args = []
    for name, shape, dt in inputs_spec():
        n = math.prod(shape)
        if dt == "int32":
            t = torch.randint(0, 2048, shape, generator=g, dtype=torch.int32) if name == "codes" else torch.zeros(shape, dtype=torch.int32)
        elif name.endswith("w"):
            t = (torch.randn(n, generator=g) * 0.02).to(torch.bfloat16).reshape(shape)
        else:
            t = torch.zeros(shape, dtype=torch.bfloat16)
        if name == "wav":
            t = torch.zeros(shape, dtype=torch.float32)
        args.append(t.contiguous())
    return args


def make_kernel(out, run_mode):
    return ax.builder(
        run_mode=ax.RunMode(run_mode),
        enable_cache=True,
        unique_axmodel=True,
        cache_path=out / "cache",
        dump_path=out / "artifacts",
    )(decoder)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_decoder")
    out.mkdir(parents=True, exist_ok=True)
    run_mode = os.getenv("RUN_MODE", "CRUN")
    ax.init(chip_num=1)
    spec = inputs_spec()
    (out / "inputs.json").write_text(json.dumps({
        "kernel": "decoder",
        "window": WINDOW,
        "samples_per_window": SAMPLES,
        "cores": 4,
        "weights": {"quant_w": QN, "up_w": UN, "dac_w": DN},
        "inputs": {name: {"shape": shape, "dtype": dt} for name, shape, dt in spec},
        "order": [name for name, _, _ in spec],
    }, indent=2) + "\n")
    args = make_inputs()
    print(f"run_mode={run_mode} inputs={len(args)} quant_w={QN} up_w={UN} dac_w={DN}", flush=True)
    kernel = make_kernel(out, run_mode)
    kernel[4](*args)
    print("ran", flush=True)
    found = sorted((out / "cache").glob("*.compiled.axmodel"))
    print("axmodels:", [str(p) for p in found], flush=True)


if __name__ == "__main__":
    main()
