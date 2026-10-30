"""Simulator cycles of one core's prefill-shaped projection x [1, M, K] . W[N, K]^T (M = 128 rows) in MXFP8, MXFP4 and
BF16 (q35_mx_build's tiles, also wider output tiles), and the multiply-accumulates per cycle.
Usage: python mb_mm.py <dump dir>"""
import os
import sys

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import q35_moe as mo
import q35_mx_build as mb

M, K, N = int(os.environ.get("MM_M", 128)), 2048, 2048
DUMP = os.path.abspath(sys.argv[1])
CUR = {}


def proj(x, w, s, out):
    t = CUR["tile"]
    ax.dsl.store(linear(ax.dsl.load(x), w, s, "p", output_dtype=ax.bf16, oc_tile_size=t), r=out)
    return out


def proj_bf(x, w, out):
    ax.dsl.store(linear(ax.dsl.load(x), w, None, "p", output_dtype=ax.bf16, oc_tile_size=CUR["tile"]), r=out)
    return out


def run(fn, tag, args):
    os.makedirs(f"{DUMP}/{tag}", exist_ok=True)
    os.chdir(f"{DUMP}/{tag}")
    b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/{tag}")(fn)
    b[1](*args)
    b[1](*args)
    return b.simulator_cycles


def main():
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    x = torch.randn((1, M, K)).bfloat16()
    w = torch.randn((N, K)) * 0.02
    out = torch.zeros((1, M, N), dtype=torch.bfloat16)
    q8, s8 = mb.quant_mxfp8(w)
    q4, s4 = mo.quant_mxfp4(w)
    res = []
    for tag, fn, args, tile in (("mx8_t256", proj, (x, q8, s8, out), 256), ("mx8_t512", proj, (x, q8, s8, out), 512),
                                ("mx8_t1024", proj, (x, q8, s8, out), 1024),
                                ("mx4_t1024", proj, (x, q4.view(torch.float4_e2m1fn_x2), s4.view(torch.float8_e8m0fnu), out), 1024),
                                ("bf16_t128", proj_bf, (x, w.bfloat16(), out), 128)):
        CUR["tile"] = tile
        c = run(fn, tag, args)
        res.append((tag, c))
        logger.info(f"MB {tag} M={M} K={K} N={N}: {c:,} cycles, {M * K * N / max(c, 1):,.0f} MAC/cycle")


if __name__ == "__main__":
    main()
