#!/usr/bin/env python3
"""Compare K/V row outputs against the causal MTP reference.

usage: cmp_kv_rows.py ref.pt new_out_dir [old_out_dir]

The board writes one bf16 file per output. The 1k kernel emits `kout` and
`vout` with [rows, HKV*D] rows; this checks those rows against the paged `kv`
tensor in ref.pt. If old_out_dir is given, compare the first old rows with the
new kernel on the same inputs.
"""
import glob
import os
import sys

import numpy as np
import torch

HKV, D = 4, 256
ROW = HKV * D


def load_rows(folder, name):
    paths = glob.glob(os.path.join(folder, f"*{name}*.bin"))
    if not paths:
        raise FileNotFoundError(f"{folder}: no {name} output")
    path = max(paths, key=os.path.getsize)
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size % ROW:
        raise ValueError(f"{path}: {raw.size} values is not a whole number of rows")
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(-1, ROW)


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def check(ref, folder):
    k = load_rows(folder, "kout")
    v = load_rows(folder, "vout")
    n = min(k.shape[0], ref["kv"].shape[0] * ref["kv"].shape[2])
    vals = []
    for p in range(n):
        page, cell = divmod(p, 256)
        vals.append(cos(k[p], ref["kv"][page, 0, cell]))
        vals.append(cos(v[p], ref["kv"][page, 1, cell]))
    t = torch.tensor(vals)
    print(f"{folder}: rows={n} min={t.min():.7f} mean={t.mean():.7f} max={t.max():.7f}")
    return k, v, t


def main():
    ref = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    k, v, t = check(ref, sys.argv[2])
    ok = float(t.min()) >= 0.9999
    if len(sys.argv) > 3:
        ok_, v_, t_ = check(ref, sys.argv[3])
        n = min(k.shape[0], ok_.shape[0])
        ck = torch.tensor([cos(k[i], ok_[i]) for i in range(n)])
        cv = torch.tensor([cos(v[i], v_[i]) for i in range(n)])
        print(f"new vs old: rows={n} K min={ck.min():.7f} mean={ck.mean():.7f} "
              f"V min={cv.min():.7f} mean={cv.mean():.7f}")
        ok = ok and float(ck.min()) >= 0.9999 and float(cv.min()) >= 0.9999
    print("pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
