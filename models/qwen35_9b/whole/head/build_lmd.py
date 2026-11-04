"""The draft head: a 1-row MXFP4 output head over the draft vocabulary's rows.

The kernel is the main output head's linear at ``LM_C`` rows (the draft vocabulary,
65536) and ``LM_K`` columns (the model width).  The host copies those rows of the
already-converted MXFP4 output weight into the kernel's ``w`` / ``s`` inputs, so the
axmodel carries no weights.  The CRUN check here builds random MXFP4 weights of the
real shape, runs the kernel and compares against the dequantized torch projection.

    source /data/junda/work/axdsl_env.sh
    LM_C=65536 LM_K=4096 RUN_MODE=CRUN python build_lmd.py
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear

import board_io
from mx4_common import dequantize_mxfp4, quantize_mxfp4

C = int(os.environ.get("LM_C", "65536"))
K = int(os.environ.get("LM_K", "4096"))
CORES = int(os.environ.get("CORES", "4"))
RUN_MODE = os.environ.get("RUN_MODE", "CRUN")
TILE = max(16, min(1024, (256 * 1024) // (K // 2)))
ROOT = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "out")
BUILD = os.path.join(ROOT, "build", f"n{C}_k{K}")


def mul_mat_mx4(x, w, s, out):
    c = ax.program_id(0)
    rows = slice(c * (C // CORES), (c + 1) * (C // CORES))
    o = linear(ax.dsl.load(x), w[rows], s[rows], "lmd", output_dtype=ax.fp32, oc_tile_size=TILE)
    ax.dsl.store(o, r=out[:, rows])
    return out


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def main():
    assert C % CORES == 0 and K % 32 == 0 and (C // CORES) <= 65535 * 4
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    torch.manual_seed(1234)
    # real-shape random MXFP4 weights; the same dequantized matrix is the reference
    wf = (torch.randn(C, K, dtype=torch.float32) * 0.02)
    pk, sc = quantize_mxfp4(wf)
    del wf
    w = pk.view(torch.float4_e2m1fn_x2).reshape(C, K // 2)
    s = sc.view(torch.float8_e8m0fnu).reshape(C, K // 32)
    x = (torch.randn(1, K, dtype=torch.float32) * 0.02).bfloat16()
    ref = (x.float() @ dequantize_mxfp4(pk, sc).T).reshape(-1)
    dump = os.environ.get("DUMP_BOARD")
    if dump:
        from pathlib import Path
        d = Path(dump)
        d.mkdir(parents=True, exist_ok=True)
        for n, t in (("x", x), ("w", w), ("s", s)):
            (d / f"{n}.bin").write_bytes(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        torch.save({"out": ref.reshape(1, C).clone()}, d / "ref.pt")
        print(f"draft head board data written to {d}", flush=True)
    if os.environ.get("BOARD_ONLY"):
        return
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    k = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN", dump_path=BUILD)(mul_mat_mx4)
    out = torch.zeros(1, C, dtype=torch.float32)
    k[CORES](x, w, s, out)
    value = cos(out[0], ref)
    print(f"draft head n{C} k{K}: cos {value:.9f} in {time.time()-t0:.1f}s", flush=True)
    import json
    def desc(n, t):
        return {"name": n, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
    with open(os.path.join(OUT, f"mul_mat_mx4_n{C}_k{K}.inputs.json"), "w") as f:
        json.dump({"kernel": f"mul_mat_mx4_n{C}_k{K}", "run_mode": RUN_MODE,
                   "inputs": [desc("x", x), desc("w", w), desc("s", s)],
                   "outputs": [desc("out", out)]}, f, indent=2)
    if value < 0.999:
        raise SystemExit(1)
    if RUN_MODE == "CRUN":
        paths = sorted(glob.glob(os.path.join(BUILD, "cache", "mul_mat_mx4.*.compiled.axmodel")),
                       key=os.path.getmtime)
        if not paths:
            raise RuntimeError("no compiled axmodel")
        dst = board_io.kernel_file(OUT, f"mul_mat_mx4_n{C}_k{K}", CORES)
        shutil.copy(paths[-1], dst)
        board_io.fix(dst)
        print("wrote", dst, os.path.getsize(dst), flush=True)


if __name__ == "__main__":
    main()
