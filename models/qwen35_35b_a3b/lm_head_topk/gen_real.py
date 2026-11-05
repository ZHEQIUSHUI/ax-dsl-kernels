"""Real Qwen3.5-35B output.weight -> the fused kernel's MXFP4 test vectors.

The board GGUF stores output.weight as Q8_0 (one fp16 scale + 32 int8 values
per block).  We extracted just that tensor from the GGUF and dequantize it
exactly as ggml, then pack MXFP4 with the same algorithm as
ggml-axera's axera_quantize_mxfp4.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "qwen38_27b", "head"))
from mx4_common import quantize_mxfp4, torch_logits

Q8 = os.path.join(ROOT, "data", "qwen35_output_q8.bin")
OUT = os.path.join(ROOT, "data")
V, H = 248320, 2048
GROUP = 32


def dequant_q8_0(path):
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size == V * H // GROUP * 34, raw.size
    blocks = raw.reshape(V, H // GROUP, 34)
    d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return torch.from_numpy((q * d).reshape(V, H)).contiguous()


def main():
    t0 = time.time()
    print("dequantize Q8_0", flush=True)
    w = dequant_q8_0(Q8)
    print("dequantized", tuple(w.shape), f"{time.time() - t0:.1f}s", flush=True)

    print("quantize MXFP4", flush=True)
    pk, sc = quantize_mxfp4(w, chunk=4096)
    del w
    print("packed", tuple(pk.shape), tuple(sc.shape), f"{time.time() - t0:.1f}s", flush=True)

    torch.manual_seed(1234)
    x = (torch.randn(1, H) * 1.0).bfloat16()
    bias = torch.zeros(V, dtype=torch.float32)
    mul_pos = torch.ones(V, dtype=torch.float32)
    mul_neg = torch.ones(V, dtype=torch.float32)
    repeat = 1.5
    for tid in (0, 7, 31, 255, 1024, 65535, V // 2, V - 3):
        mul_pos[tid] = 1.0 / repeat
        mul_neg[tid] = repeat
        bias[tid] = -float((tid % 5) + 1)
    temp = torch.tensor([0.7], dtype=torch.float32)

    print("torch reference", flush=True)
    logits = torch_logits(x, pk, sc, chunk=4096)
    lrep = torch.where(logits > 0, logits * mul_pos, logits * mul_neg)
    logits = lrep + bias
    ref_v, ref_i = torch.topk(logits, 1024, dim=-1)
    gmax = logits.max(dim=-1, keepdim=True).values
    sumexp = torch.exp((logits - gmax) / temp[0]).sum(dim=-1, keepdim=True)

    pk.numpy().tofile(os.path.join(OUT, "w.bin"))
    sc.numpy().tofile(os.path.join(OUT, "s.bin"))
    x.view(torch.uint16).numpy().tofile(os.path.join(OUT, "x.bin"))
    bias.numpy().tofile(os.path.join(OUT, "bias.bin"))
    mul_pos.numpy().tofile(os.path.join(OUT, "mul_pos.bin"))
    mul_neg.numpy().tofile(os.path.join(OUT, "mul_neg.bin"))
    temp.numpy().tofile(os.path.join(OUT, "temp.bin"))
    torch.save({"ids": ref_i[0].to(torch.int32), "values": ref_v[0],
                "max": gmax.reshape(1), "sumexp": sumexp.reshape(1),
                "logits": logits[0]}, os.path.join(OUT, "ref_topk.pt"))
    json.dump({"source": "qwen3.5-35b-a3b-mxfp4_moe.gguf:output.weight",
               "q8_md5": "28ce749e556a3d67d65f806b19634da4",
               "V": V, "H": H, "repeat": repeat, "temp": float(temp[0])},
              open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps({"max": float(gmax), "sumexp": float(sumexp),
                      "top1": int(ref_i[0, 0])}), f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
