#!/usr/bin/env python3
"""A small deterministic MXFP4 fixture for the combined head's exact-tie rule.

The largest real projection row is copied to id 0, to another chunk and to
another core.  All four copies have bit-identical logits, so argmax must return
the lowest id (0).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=32)
    a = ap.parse_args()
    V, H = a.vocab, a.hidden
    assert V % 32 == 0 and H % 32 == 0
    g = torch.Generator().manual_seed(35)
    w = torch.randint(0, 256, (V, H // 2), generator=g, dtype=torch.int32).to(torch.uint8)
    s = torch.randint(120, 130, (V, H // 32), generator=g, dtype=torch.int32).to(torch.uint8)
    x = (torch.randn((1, H), generator=g) * 0.25).bfloat16()
    norm = torch.ones(H, dtype=torch.float32)

    xn = (torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * x.float() * norm).bfloat16()
    lg = quant.torch_logits(xn, w, s, "mx4")
    top = int(lg.argmax(-1)[0])
    # id 0: another chunk in core 0; id 100: a later chunk; id 300: core 1.
    for dst in (0, 100, 300):
        if dst != top:
            w[dst] = w[top]
            s[dst] = s[top]
    lg = quant.torch_logits(xn, w, s, "mx4")
    ids = torch.argmax(lg, dim=-1).to(torch.int32)
    vals = torch.topk(lg, 4)
    assert int(ids[0]) == 0, (top, ids.tolist(), vals.indices.tolist(), vals.values.tolist())
    assert bool((vals.values[0, 0] == vals.values[0]).all()), vals.values.tolist()

    os.makedirs(a.out, exist_ok=True)
    x.view(torch.uint16).numpy().tofile(os.path.join(a.out, "x.bin"))
    norm.numpy().tofile(os.path.join(a.out, "norm.bin"))
    w.view(torch.uint8).numpy().tofile(os.path.join(a.out, "w.bin"))
    s.view(torch.uint8).numpy().tofile(os.path.join(a.out, "s.bin"))
    torch.save({"ids": ids, "logits": lg.gather(1, ids.long().reshape(1, 1)).reshape(1),
                "logits_all": lg, "top2": vals.values[:, :2],
                "top2_ids": vals.indices[:, :2]}, os.path.join(a.out, "ref_argmax.pt"))
    json.dump({"name": "synthetic_tie", "format": "mxfp4", "V": V, "H": H,
               "rows": 1, "eps": 1e-6, "x_out_dtype": "bf16",
               "tie_ids": sorted(set([top, 0, 100, 300]))},
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(json.dumps({"source_top": top, "tie_ids": sorted(set([top, 0, 100, 300])),
                      "argmax": ids.tolist(), "top4_ids": vals.indices.tolist(),
                      "top4_values": vals.values.tolist()}, indent=2))


if __name__ == "__main__":
    main()
