#!/usr/bin/env python3
"""Make a bf16-input fixture for build_combined.py from an argmax fixture."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=("mxfp4", "mxfp8"), required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    a = ap.parse_args()

    old = json.load(open(os.path.join(a.src, "meta.json")))
    V, H, rows = int(old["V"]), int(old["H"]), int(old["rows"])
    x = torch.from_numpy(np.fromfile(os.path.join(a.src, "x.bin"), dtype=np.float32).copy()).reshape(rows, H)
    x = x.bfloat16().float()
    norm = torch.from_numpy(np.fromfile(os.path.join(a.src, "norm.bin"), dtype=np.float32).copy())
    w = torch.from_numpy(np.fromfile(os.path.join(a.src, "w.bin"), dtype=np.uint8).copy()).reshape(
            V, H // 2 if a.format == "mxfp4" else H)
    s = torch.from_numpy(np.fromfile(os.path.join(a.src, "s.bin"), dtype=np.uint8).copy()).reshape(V, H // 32)

    xf = x.float()
    xn = (torch.rsqrt(xf.square().mean(-1, keepdim=True) + a.eps) * xf * norm).bfloat16()
    if a.format == "mxfp4":
        logits = quant.torch_logits(xn, w, s, "mx4")
    else:
        logits = quant.torch_logits(xn, w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu), "mx8")
    ids = torch.argmax(logits, dim=-1).to(torch.int32)
    top = torch.gather(logits, 1, ids.long().reshape(rows, 1)).reshape(rows)
    top2 = torch.topk(logits, 2, dim=-1)

    os.makedirs(a.out, exist_ok=True)
    x.bfloat16().view(torch.uint16).numpy().tofile(os.path.join(a.out, "x.bin"))
    norm.numpy().tofile(os.path.join(a.out, "norm.bin"))
    w.view(torch.uint8).numpy().tofile(os.path.join(a.out, "w.bin"))
    s.view(torch.uint8).numpy().tofile(os.path.join(a.out, "s.bin"))
    torch.save({"ids": ids, "logits": top, "logits_all": logits,
                "top2": top2.values, "top2_ids": top2.indices},
               os.path.join(a.out, "ref_argmax.pt"))
    meta = dict(old)
    meta.update({"x_out_dtype": "bf16", "source_fixture": os.path.abspath(a.src)})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(json.dumps({"ids": ids.tolist(), "logits": top.tolist(),
                      "top1_top2": (top2.values[:, 0] - top2.values[:, 1]).tolist()}, indent=2))


if __name__ == "__main__":
    main()
