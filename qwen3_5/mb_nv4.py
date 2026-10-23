"""Simulator micro-benchmark of one core's share of a dense projection (2048 x 2048: a quarter of the GDN qkv weight)
in MXFP8, MXFP4 and NVFP4 (block 16, E4M3 scales, FP32 tensor scale2), and NVFP4 at half the rows (the per-call fixed
cost): simulator cycles and cos against torch on the dequantized weights. Usage: python mb_nv4.py <dump dir>"""
import os
import sys

import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import q35_moe as mo
import q35_mx_build as mb

K = 2048
DUMP = os.path.abspath(sys.argv[1])
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quant_nvfp4(w):
    """[N, K] -> packed FP4 [N, K / 2] (low nibble first), E4M3 scales [N, K / 16], FP32 scale2, dequantized [N, K]"""
    n, k = w.shape
    s2 = w.abs().max().float() / (6.0 * 448.0)
    g = w.float().reshape(n, k // 16, 16)
    sc = (g.abs().amax(-1, keepdim=True) / 6.0 / s2).to(torch.float8_e4m3fn)
    d = sc.float() * s2
    x = torch.where(d > 0, g / d, torch.zeros_like(g)).reshape(n, k)
    code = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1) | ((x < 0).long() << 3)
    dq = (E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0)).reshape(n, k // 16, 16) * d
    code = code.to(torch.uint8)
    return ((code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), sc.reshape(n, k // 16).contiguous(),
            np.array([s2.item()], dtype=np.float32), dq.reshape(n, k))


def proj(x, w, s, out):
    ax.dsl.store(linear(ax.dsl.load(x), w, s, "p", output_dtype=ax.fp32), r=out)
    return out


S2 = {}


def proj_nv(x, w, s, out):
    ax.dsl.store(linear(ax.dsl.load(x), w, s, "p", weight_scale2=S2["v"], output_dtype=ax.fp32), r=out)
    return out


def run(fn, tag, args):
    os.makedirs(f"{DUMP}/{tag}", exist_ok=True)
    os.chdir(f"{DUMP}/{tag}")
    b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/{tag}")(fn)
    b[1](*args)
    b[1](*args)
    return b.simulator_cycles


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def main():
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    x = torch.randn((1, 1, K)).bfloat16()
    for n in (2048, 1024):
        w = torch.randn((n, K)) * 0.02
        out = {f: torch.zeros((1, 1, n)) for f in ("mx8", "mx4", "nv4")}
        res = {}
        if n == 2048:
            q8, s8 = mb.quant_mxfp8(w) if hasattr(mb, "quant_mxfp8") else (None, None)
            if q8 is not None:
                res["mx8"] = run(proj, f"mx8_{n}", (x, q8, s8, out["mx8"]))
            pk, sc = mo.quant_mxfp4(w)
            res["mx4"] = run(proj, f"mx4_{n}", (x, pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu),
                                                 out["mx4"]))
        pk, sc, s2, dq = quant_nvfp4(w)
        S2["v"] = s2
        res["nv4"] = run(proj_nv, f"nv4_{n}", (x, pk.view(torch.float4_e2m1fn_x2), sc, out["nv4"]))
        ref = x.float().reshape(1, K) @ dq.T
        for f, c in res.items():
            logger.info(f"MB {f} [{n} x {K}]: {c:,} cycles" + (f", cos vs torch {cos(out[f], ref):.6f}" if f == "nv4" else ""))


if __name__ == "__main__":
    main()
