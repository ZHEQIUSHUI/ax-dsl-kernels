"""Qwen3-ASR audio front-end on the NPU: the matrix products of its 3 x 3 stride-2 convolutions as ggml computes them
(im2col rows x kernel), the conv2d.2 / conv2d.3 layers of 480 channels being most of the encoder's work on the CPU.

  qwen3a_conv_o<OC>_k<K>_r<R>   out [OC, R] f32 = w [OC, K] . x [R, K]^T  (bf16 in: w the conv kernel as ggml reshapes
                                it, x the im2col rows; out in ggml's layout of that product, rows fastest)

R covers the im2col rows of one encoder call (mtmd feeds at most 8 one-second chunks: conv2d.2 800 rows per chunk,
conv2d.3 208); the host pads the rest. Bias and GELU stay on the CPU.

Written as qwen3a_conv_o<OC>_k<K>_r<R>_npu1.axmodel. Check on the simulator against torch on random data.
Usage: OC=480 K=4320 R=6400 python q3a_conv.py [kernel_dir]
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import board_io

OC, K, R = int(os.environ.get("OC", 480)), int(os.environ.get("K", 4320)), int(os.environ.get("R", 6400))
OB = int(os.environ.get("OB", 96))  # output channels per pass (the w rows held on chip)
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q3a_conv"))
assert OC % OB == 0


def row_tile(cols, rows):
    """x rows per linear tile: about 512 KB of bf16, dividing rows"""
    t = max(16, min(256, (512 * 1024) // (cols * 2)))
    while rows % t:
        t -= 1
    return t


def conv_gemm(w, x, out):
    for b in range(OC // OB):
        wb = ax.dsl.load(w[b * OB : (b + 1) * OB]).reshape(1, OB, K)
        y = linear(wb, x, None, f"conv{b}", output_dtype=ax.fp32, oc_tile_size=row_tile(K, R))  # [1, OB, R]
        ax.dsl.store(y.reshape(OB, R), r=out[b * OB : (b + 1) * OB])
    return out


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    w = (torch.randn((OC, K)) * 0.02).bfloat16()
    x = torch.randn((R, K)).bfloat16()
    out = torch.zeros((OC, R))
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=BUILD)(conv_gemm)[1](w, x, out)
    ref = w.float() @ x.float().T
    c = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    logger.info(f"conv gemm OC={OC} K={K} R={R}: cos vs torch {c:.6f}")
    if out_dir:
        (path,) = glob.glob(f"{BUILD}/cache/conv_gemm.*.compiled.axmodel")
        dst = board_io.kernel_file(out_dir, f"qwen3a_conv_o{OC}_k{K}_r{R}")
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
