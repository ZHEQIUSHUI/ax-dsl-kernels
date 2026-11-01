"""Build the unified runtime-top-k code-predictor kernel.

    RUN_MODE=PYRUN python build_cp.py out_pyrun --model 0.6b --top-k 50
    RUN_MODE=CRUN  python build_cp.py out_crun  --model 0.6b --top-k 50

The same graph serves every top-k in 1..50; ``--top-k`` only selects the
simulator input and numerical check.  CRUN stops after the compiled axmodel is
written (the builder itself runs one frame on the simulator as a smoke test).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", choices=("0.6b", "1.7b"), required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--real", action="store_true", help="use the downloaded Qwen3-TTS checkpoint weights")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        raise SystemExit("top-k must be in 1..50")

    mode = os.environ.get("RUN_MODE", "CRUN").upper()
    os.environ["RUN_MODE"] = mode
    os.environ["CP_STATIC_PYRUN"] = "1" if mode == "PYRUN" else "0"

    import cp_data

    cp_data.configure(args.model, real=args.real)
    import torch
    import axnn.dsl as ax
    import cp_kernel as C
    import q3tts_cp_step as S

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    build = os.path.join(out, "k")
    os.makedirs(build, exist_ok=True)

    case, pk, sizes, es, raw, extra = cp_data.make_case(S)
    case["top_k"] = torch.tensor([args.top_k], dtype=torch.int32)
    with open(os.path.join(build, "cp_frame_kernel.py"), "w") as f:
        f.write(C.kernel_source())
    if build not in sys.path:
        sys.path.insert(0, build)
    os.chdir(build)
    fn = __import__("cp_frame_kernel").cp_frame

    kk = ax.builder(
        run_mode=ax.RunMode(mode),
        enable_cache=not args.perf,
        dump_dsl_perf=args.perf,
        dump_path=build,
    )(fn)
    ax.init(chip_num=1, core_num=C.CORES)

    for j in range(C.NL):
        S.q.RESIDENT[f"kc{j}"] = case[f"kc{j}"].clone()
        S.q.RESIDENT[f"vc{j}"] = case[f"vc{j}"].clone()
    kk[C.CORES](*[case[n] for n in S.L.order(fn)])
    S.q.RESIDENT.clear()

    torch.save(
        {
            "model": args.model,
            "real": args.real,
            "top_k": args.top_k,
            "codes": case["codes"],
            "lgs": case["lgs"],
            "e0": case["e0"],
            "e1": case["e1"],
            "rnd": case["rnd"],
        },
        os.path.join(out, "result.pt"),
    )
    print(
        f"model={args.model} top_k={args.top_k} mode={mode} "
        f"codes={case['codes'][0].tolist()} lgs_finite={bool(torch.isfinite(case['lgs']).all())}",
        flush=True,
    )

    if mode != "CRUN":
        return
    paths = glob.glob(os.path.join(build, "cache", "cp_frame.*.compiled.axmodel"))
    if not paths:
        raise SystemExit("compiled axmodel not found")
    path = max(paths, key=os.path.getmtime)
    name = f"qwen3tts_cp_frame{C.NL}{C.Kq.FMT}_h{C.HT}_krt_c{C.Lq.LMAX}"
    dst = S.write_kernel(path, out, name, sizes, es)
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
