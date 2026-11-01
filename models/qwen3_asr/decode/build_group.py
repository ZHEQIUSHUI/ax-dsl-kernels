"""Build the one-call Qwen3 dense decode group (28 layers, paged BF16 KV).

Random weights of the real shapes are used; the compiled AXModel binds the
GGUF's weights at load, exactly as the prefill kernel's build does.  The four
products are asr_0.6b, asr_1.7b, tts_0.6b and tts_1.7b.  Every variant
contains its MXFP8 output head (lm8): the ASR 151936-row text head, or the TTS
3072-row codec head.

Usage:
    RUN_MODE=CRUN python build_group.py out --model asr_0.6b
    RUN_MODE=PYRUN python build_group.py out --model tts_0.6b --layers 4
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOTS = {
    "embedding_0.6b": Path("/data/junda/work/qsmall/embedding/hf/Qwen3-0.6B"),
    "asr_0.6b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-ASR-0.6B"),
    "asr_1.7b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-ASR-1.7B"),
    "tts_0.6b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-TTS-12Hz-0.6B-Base"),
    "tts_1.7b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-TTS-12Hz-1.7B-Base"),
}


def _preparse():
    for i, a in enumerate(sys.argv):
        if a == "--model":
            return sys.argv[i + 1]
    return "asr_0.6b"


_NAME = _preparse()
os.environ.setdefault("MODEL", str(MODEL_ROOTS[_NAME]))
os.environ.setdefault("LMAX", "4096")
os.environ.setdefault("BC", "256")
os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("CORES", "4")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("DEC_SPLIT", "1")
os.environ.setdefault("QW_RANDOM", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import axnn.dsl as ax  # noqa: E402
import q35_group as Q  # noqa: E402
import q35_step as q  # noqa: E402
import q35_layer_mc as L  # noqa: E402


def dtype_name(dtype):
    return {torch.bfloat16: "bfloat16", torch.float32: "float32", torch.int32: "int32",
            torch.uint32: "uint32", torch.float8_e4m3fn: "float8_e4m3fn",
            # the compiled model exports E8M0 scales as U8 bytes (the board runtime's convention); the
            # torch-side tensor is float8_e8m0fnu with one byte per element
            torch.float8_e8m0fnu: "uint8"}.get(dtype, str(dtype).removeprefix("torch."))


def make_case(spec, packs, lm, layers, lmhead):
    H, HKV, D, BC, ROT = q.H, q.HKV, q.D, q.BC, q.ROT
    pages = q.LMAX // BC
    case = dict(
        x=torch.zeros((1, 1, H), dtype=torch.bfloat16),
        cs=torch.zeros((2, ROT // 2), dtype=torch.bfloat16),
        pn=torch.tensor([0, 1, 0], dtype=torch.int32),
        mask=torch.zeros((pages, BC), dtype=torch.bfloat16),
        kv=torch.zeros((layers, pages, 2, BC, HKV, D), dtype=torch.bfloat16),
        fnorm=torch.ones((H,), dtype=torch.bfloat16),
        hbuf=torch.zeros((1, 1, H), dtype=torch.bfloat16),
        out=torch.zeros((1, 1, H), dtype=torch.bfloat16),
    )
    case.update(packs)
    if lmhead:
        case["logits"] = torch.zeros((1, lm["w_lm"].shape[0]), dtype=torch.float32)
    return case


def describe(case, names):
    out = {}
    for n in names:
        t = case[n]
        out[n] = {"shape": list(t.shape), "dtype": dtype_name(t.dtype)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="asr_0.6b")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.layers % 4:
        raise ValueError("layers must be a multiple of four")
    root = MODEL_ROOTS[args.model]
    lmhead = args.model.startswith(("asr_", "tts_"))
    out_dir = Path(args.out).resolve() / f"build_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    run_mode = os.environ.get("RUN_MODE", "CRUN")
    ax.init(chip_num=1, core_num=4)
    W = q.Weights(str(root))
    print(f"model={args.model} layers={args.layers} H={q.H} F={q.F} lmhead={lmhead} "
          f"mrope={q.MROPE_SECTION} mode={run_mode}", flush=True)
    kernel, packs, lm, meta = Q.prepare_group(W, out_dir, layers=args.layers, lmhead=lmhead, run_mode=run_mode)
    build = Path(meta["build"])
    sys.path.insert(0, str(build))
    importlib.invalidate_caches()
    fn = importlib.import_module("group_kernel").group_kernel
    names = L.order(fn)
    case = make_case(None, packs, lm, args.layers, lmhead)
    io = describe(case, names)
    (out_dir / "inputs.json").write_text(json.dumps(
        {"model": args.model, "layers": args.layers, "lmhead": lmhead, "mrope_section": list(q.MROPE_SECTION),
         "mrope_interleaved": q.MROPE_INTERLEAVED, "context": q.LMAX, "cores": L.CORES,
         "kv": {"shape": list(case["kv"].shape), "dtype": "bfloat16"},
         "pn": "pn = [cell, nblk, krow]; krow = (cell // 256) * 2 * 256 + cell % 256",
         "notes": {"ps": "E8M0 scale bytes, exported as UINT8",
                   "hbuf": "scratch for core 0's last-row DDR materialization; no cross-core handoff",
                   "out": "final RMSNorm row", "logits": "output head: ASR lm_head or TTS codec_head"},
         "inputs": io}, indent=2) + "\n")
    print("inputs:", names, flush=True)

    if run_mode == "PYRUN":
        kernel[4](*[case[n] for n in names])
        print(f"PYRUN smoke ok, out finite={bool(torch.isfinite(case['out']).all())}", flush=True)
        return
    print("compiling", flush=True)
    path = kernel[4].compile(*[case[n] for n in names])
    tag = args.name or f"{q.KP}_group{args.layers}_attn_mx8{'_lm8' if lmhead else ''}_paged_c{q.LMAX}_npu{L.CORES}"
    dst = out_dir / f"{tag}.axmodel"
    shutil.copy(path, dst)
    Q.write_layout(str(dst)[:-len(".axmodel")] + ".layout", meta["sizes"], meta["elem"])
    print(f"compiled {dst} ({dst.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
