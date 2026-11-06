#!/usr/bin/env python3
"""Prepare real CRUN inputs and a torch reference for lm_head.argmax.

The output head and final-norm tensors come from the real GGUF.  x is a real
last-layer hidden-state row captured from llama.cpp (`hraw.bin`); callers may
select the last ROWS rows of a prompt dump for a 1-row or MTP 4-row check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from quant import (load_gguf_tensor, quantize_mxfp4, quantize_mxfp8,  # noqa: E402
                   torch_logits)


def load_gguf_vector(path, name):
    from gguf import GGUFReader, quants
    r = GGUFReader(path)
    t = next(t for t in r.tensors if t.name == name)
    dense = quants.dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(dense)).reshape(-1).float()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--norm-tensor", default="output_norm.weight")
    ap.add_argument("--format", choices=("mxfp4", "mxfp8"), required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--rows", type=int, default=1, choices=(1, 4))
    ap.add_argument("--hraw", required=True,
                    help="[T,H] raw hidden dump; its last ROWS nonzero rows are used")
    ap.add_argument("--hraw-dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--x-out-dtype", choices=("bf16", "f32"), default="f32",
                    help="store x in the dtype the output-head kernel consumes")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    w = load_gguf_tensor(a.gguf, a.tensor)
    assert w.shape == (a.vocab, a.hidden), (w.shape, a.vocab, a.hidden)
    norm = load_gguf_vector(a.gguf, a.norm_tensor)
    assert norm.shape == (a.hidden,), (norm.shape, a.hidden)

    if a.format == "mxfp4":
        packed, scale = quantize_mxfp4(w, chunk=a.chunk)
        head = "mx4"
    else:
        packed, scale = quantize_mxfp8(w, chunk=a.chunk)
        head = "mx8"
    del w

    if a.hraw_dtype == "bf16":
        raw = np.fromfile(a.hraw, dtype=np.uint16)
        all_x = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(-1, a.hidden).float()
    else:
        raw = np.fromfile(a.hraw, dtype=np.float32)
        all_x = torch.from_numpy(raw.copy()).reshape(-1, a.hidden)
    rows = all_x.shape[0]
    nonzero = all_x.abs().sum(-1) > 0
    assert int(nonzero.sum()) >= a.rows, (rows, int(nonzero.sum()), a.rows)
    x = all_x[nonzero][-a.rows:].contiguous()
    if a.x_out_dtype == "bf16":
        # the fused decode head consumes the layer kernel's bf16 row directly
        x = x.bfloat16().float()

    xf = x.float()
    xn = (torch.rsqrt(xf.square().mean(-1, keepdim=True) + a.eps) * xf * norm).bfloat16()
    logits = torch_logits(xn, packed, scale, head, chunk=a.chunk)
    ids = torch.argmax(logits, dim=-1).to(torch.int32)
    top = torch.gather(logits, 1, ids.long().reshape(a.rows, 1)).reshape(a.rows)
    top2 = torch.topk(logits, 2, dim=-1)

    packed.view(torch.uint8).numpy().tofile(os.path.join(a.out, "w.bin"))
    scale.view(torch.uint8).numpy().tofile(os.path.join(a.out, "s.bin"))
    if a.x_out_dtype == "bf16":
        x.bfloat16().view(torch.uint16).numpy().tofile(os.path.join(a.out, "x.bin"))
    else:
        x.numpy().tofile(os.path.join(a.out, "x.bin"))
    norm.numpy().tofile(os.path.join(a.out, "norm.bin"))
    torch.save({"ids": ids, "logits": top, "logits_all": logits,
                "top2": top2.values, "top2_ids": top2.indices},
               os.path.join(a.out, "ref_argmax.pt"))
    meta = {
        "name": a.name, "gguf": a.gguf, "tensor": a.tensor,
        "norm_tensor": a.norm_tensor, "format": a.format,
        "V": a.vocab, "H": a.hidden, "rows": a.rows, "eps": a.eps,
        "x_out_dtype": a.x_out_dtype,
        "hraw": os.path.abspath(a.hraw), "hraw_rows": rows,
        "w_bytes": int(packed.numel()), "s_bytes": int(scale.numel()),
        "w_md5": md5(os.path.join(a.out, "w.bin")),
        "s_md5": md5(os.path.join(a.out, "s.bin")),
        "x_md5": md5(os.path.join(a.out, "x.bin")),
        "norm_md5": md5(os.path.join(a.out, "norm.bin")),
        "ref_md5": md5(os.path.join(a.out, "ref_argmax.pt")),
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"name": a.name, "ids": ids.tolist(),
                      "logits": [float(v) for v in top],
                      "top1_top2": (top2.values[:, 0] - top2.values[:, 1]).tolist()},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
