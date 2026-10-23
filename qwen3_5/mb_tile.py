"""Simulator cycles of one core's MXFP8 prefill projection x [1, M, K] . W[N, K]^T over output / input channel tiles
(linear's oc_tile_size / ic_tile_size: IC tiles accumulate in the output), M = MM_M rows; the shapes of the 35B
GDN prefill per core. Usage: MM_M=256 python mb_tile.py <dump dir>"""
import os
import sys

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import q35_mx_build as mb

M = int(os.environ.get("MM_M", 256))
DUMP = os.path.abspath(sys.argv[1])
CUR = {}


def proj(x, w, s, out):
    ax.dsl.store(linear(ax.dsl.load(x), w, s, "p", output_dtype=ax.bf16, oc_tile_size=CUR["oc"],
                        ic_tile_size=CUR["ic"]), r=out)
    return out


def main():
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    for K, N, tiles in ((2048, 2048, ((256, None), (128, None), (512, 1024), (256, 1024), (512, 512))),
                        (4096, 512, ((128, None), (128, 2048), (256, 2048), (512, 2048), (512, 1024)))):
        x = torch.randn((1, M, K)).bfloat16()
        q8, s8 = mb.quant_mxfp8(torch.randn((N, K)) * 0.02)
        for oc, ic in tiles:
            CUR.update(oc=oc, ic=ic)
            tag = f"k{K}_n{N}_oc{oc}_ic{ic or K}"
            os.makedirs(f"{DUMP}/{tag}", exist_ok=True)
            os.chdir(f"{DUMP}/{tag}")
            try:
                b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/{tag}")(proj)
                out = torch.zeros((1, M, N), dtype=torch.bfloat16)
                b[1](x, q8, s8, out)
                b[1](x, q8, s8, out)
                c = b.simulator_cycles
                logger.info(f"MB {tag} M={M}: {c:,} cycles, {M * K * N / max(c, 1):,.0f} MAC/cycle")
            except Exception as e:  # tiles that do not fit
                logger.info(f"MB {tag} M={M}: failed ({type(e).__name__}: {str(e)[:80]})")


if __name__ == "__main__":
    main()
