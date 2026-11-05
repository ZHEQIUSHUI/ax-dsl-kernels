#!/usr/bin/env python3
"""Create CRUN inputs and a torch reference from a real GGUF output head.

  python gen_real.py --name qwen35_2b --gguf <file.gguf> --tensor token_embd.weight \
      --format mxfp4 --hidden 2048 --vocab 248320 --rows 1 --out data/qwen35_2b

The packed bytes use the same MXFP4/MXFP8 rules as ggml-axera's adapter.  The
reference applies the same per-row repeat/frequency/presence transform as the
kernel, then keeps the exact top-K and the full-vocabulary vmax/sumexp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from quant import (load_gguf_tensor, quantize_mxfp4, quantize_mxfp8,  # noqa: E402
                   torch_logits)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def penalty_arrays(vocab):
    bias = torch.zeros(vocab, dtype=torch.float32)
    mul_pos = torch.ones(vocab, dtype=torch.float32)
    mul_neg = torch.ones(vocab, dtype=torch.float32)
    repeat = 1.5
    for tid in (0, 7, 31, 255, 1024, 65535, vocab // 2, vocab - 3):
        mul_pos[tid] = 1.0 / repeat
        mul_neg[tid] = repeat
        bias[tid] = -float((tid % 5) + 1)
    return bias, mul_pos, mul_neg, repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--format", choices=("mxfp4", "mxfp8"), required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--rows", type=int, default=1, choices=(1, 4))
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print(f"loading {a.tensor} from {a.gguf}", flush=True)
    w = load_gguf_tensor(a.gguf, a.tensor)
    assert w.shape == (a.vocab, a.hidden), (w.shape, a.vocab, a.hidden)
    print("weight", tuple(w.shape), w.dtype, f"{time.time() - t0:.1f}s", flush=True)

    if a.format == "mxfp4":
        packed, scale = quantize_mxfp4(w, chunk=a.chunk)
    else:
        packed, scale = quantize_mxfp8(w, chunk=a.chunk)
    del w
    print("packed", tuple(packed.shape), tuple(scale.shape), f"{time.time() - t0:.1f}s", flush=True)

    gen = torch.Generator().manual_seed(1234)
    x = (torch.randn((4, a.hidden), generator=gen) * 1.0).bfloat16()
    x[2] += 7.0
    x[3] -= 7.0
    x = x[:a.rows].contiguous()
    bias, mul_pos, mul_neg, repeat = penalty_arrays(a.vocab)
    temp = torch.tensor([a.temp], dtype=torch.float32)

    print("torch reference", flush=True)
    head = "mx4" if a.format == "mxfp4" else "mx8"
    logits = torch_logits(x, packed, scale, head, chunk=a.chunk)
    logits = logits * torch.where(logits > 0, mul_pos, mul_neg) + bias
    ref_v, ref_i = torch.topk(logits, a.topk, dim=-1)
    ref_max = logits.max(dim=-1, keepdim=True).values
    ref_sum = torch.exp((logits - ref_max) / temp[0]).sum(dim=-1, keepdim=True)

    packed.view(torch.uint8).numpy().tofile(os.path.join(a.out, "w.bin"))
    scale.view(torch.uint8).numpy().tofile(os.path.join(a.out, "s.bin"))
    x.view(torch.uint16).numpy().tofile(os.path.join(a.out, "x.bin"))
    bias.numpy().tofile(os.path.join(a.out, "bias.bin"))
    mul_pos.numpy().tofile(os.path.join(a.out, "mul_pos.bin"))
    mul_neg.numpy().tofile(os.path.join(a.out, "mul_neg.bin"))
    temp.numpy().tofile(os.path.join(a.out, "temp.bin"))
    torch.save({"ids": ref_i.to(torch.int32), "values": ref_v,
                "max": ref_max.reshape(a.rows), "sumexp": ref_sum.reshape(a.rows),
                "logits": logits}, os.path.join(a.out, "ref_topk.pt"))
    meta = {
        "name": a.name, "gguf": a.gguf, "tensor": a.tensor, "format": a.format,
        "V": a.vocab, "H": a.hidden, "rows": a.rows, "topk": a.topk,
        "repeat": repeat, "temp": a.temp,
        "w_bytes": int(packed.numel()), "s_bytes": int(scale.numel()),
        "w_md5": md5(os.path.join(a.out, "w.bin")),
        "s_md5": md5(os.path.join(a.out, "s.bin")),
        "x_md5": md5(os.path.join(a.out, "x.bin")),
        "ref_md5": md5(os.path.join(a.out, "ref_topk.pt")),
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"name": a.name, "top1": int(ref_i[0, 0]),
                      "max": [float(v) for v in ref_max.reshape(-1)],
                      "sumexp": [float(v) for v in ref_sum.reshape(-1)]},
                     indent=2), f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
