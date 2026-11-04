#!/usr/bin/env python3
"""Compare AX8860 axrun outputs of the T=4 verify kernel with its CRUN outputs.

    compare_board_verify.py <dump_board_dir> <board_out_dir> [report.json]

<dump_board_dir> is the DUMP_BOARD directory written by check_verify_m4_crun.py
(it contains crun.pt and meta.json); <board_out_dir> contains axrun's
out_<name>.bin files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

DT = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int32": torch.int32}


def load_bin(path, shape, dtype):
    raw = np.fromfile(path, dtype=np.uint8).copy()
    t = torch.from_numpy(raw).view(DT[dtype])
    if t.numel() != int(np.prod(shape)):
        raise ValueError(f"{path}: {t.numel()} values, expected {shape}")
    return t.reshape(shape)


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def main():
    dump = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    report_path = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else dump / "board_compare.json"
    meta = json.load(open(dump / "meta.json"))["outputs"]
    ref = torch.load(dump / "crun.pt", map_location="cpu", weights_only=False)
    res = {}
    for name, info in meta.items():
        p = out / f"out_{name}.bin"
        got = load_bin(p, info["shape"], info["dtype"])
        res[name] = cos(got, ref[name])
    vals = torch.tensor(list(res.values()), dtype=torch.float64)
    report = {"outputs": res, "n": len(res), "min": float(vals.min()), "mean": float(vals.mean()),
              "pass": bool(vals.min() >= 0.999)}
    json.dump(report, open(report_path, "w"), indent=2)
    print(f"verify board: n={len(res)} min={report['min']:.7f} mean={report['mean']:.7f} "
          f"pass={report['pass']} report={report_path}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
