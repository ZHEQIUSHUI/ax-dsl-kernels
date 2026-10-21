"""Matrix products of a ggml graph that has no template of its own (ggml-axera's generic executor: mtmd's generation
graphs, e.g. the Qwen3-TTS codec decoder), one kernel per shape:

  mm_bf16_m<M>_n<N>_k<K>   x [M, K], w [N, K] bf16 (both runtime inputs) -> out [M, N] f32 = x . w^T

in ggml's MUL_MAT terms out = src1 . src0^T (out ne0 = N = src0 rows): src0 a weight, or for a convolution the im2col
rows with the kernel as src1. The kernel holds MB rows of x on chip at a time and streams w past them.

Each kernel is run once on the simulator against torch (random data).
Usage: SHAPES="<M>x<N>x<K>,..." python mm_kernels.py out_dir
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear

BUILD = os.path.abspath(os.environ.get("DUMP", "out_mm"))
XB = int(os.environ.get("XB", 768 * 1024))  # bytes of x rows on chip per pass
YB = int(os.environ.get("YB", 1024 * 1024))  # bytes of the f32 result of one product


def tile(cols, rows):
    """w rows per linear tile: about 512 KB of bf16, dividing rows"""
    t = max(16, min(256, (512 * 1024) // (cols * 2)))
    while rows % t:
        t -= 1
    return t


def x_rows(m, k):
    """x rows per pass: the largest divisor of m within XB bytes (and 256 rows)"""
    b = max(1, min(m, 256, XB // (k * 2)))
    while m % b:
        b -= 1
    return b


def out_cols(n, mb):
    """out columns per product: N, else the largest divisor of N (a multiple of 16) whose [mb, NB] f32 fits YB"""
    if mb * n * 4 <= YB:
        return n
    for c in range(YB // (mb * 4) // 16 * 16, 15, -16):
        if n % c == 0:
            return c
    raise ValueError(f"no column block for N={n}")


def build(M, N, K, out_dir):
    MB = x_rows(M, K)
    NB = out_cols(N, MB)

    def mm(x, w, out):
        for m0 in range(0, M, MB):
            xb = ax.dsl.load(x[m0 : m0 + MB]).reshape(1, MB, K)
            for n0 in range(0, N, NB):
                y = linear(xb, w[n0 : n0 + NB], None, f"mm{m0}_{n0}", output_dtype=ax.fp32, oc_tile_size=tile(K, NB))
                ax.dsl.store(y.reshape(MB, NB), r=out[m0 : m0 + MB, n0 : n0 + NB])
        return out

    name = f"mm_bf16_m{M}_n{N}_k{K}"
    dump = f"{BUILD}/{name}"
    os.makedirs(dump, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dump)
    torch.manual_seed(0)
    x = torch.randn((M, K)).bfloat16()
    w = (torch.randn((N, K)) * 0.02).bfloat16()
    out = torch.zeros((M, N))
    mm.__name__ = "mm"
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(mm)[1](x, w, out)
    ref = x.float() @ w.float().T
    c = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    logger.info(f"{name} (x rows per pass {MB}, out columns {NB}): cos vs torch {c:.6f}")
    (path,) = glob.glob(f"{dump}/cache/mm.*.compiled.axmodel")
    shutil.copy(path, f"{out_dir}/{name}.axmodel")
    os.chdir(cwd)


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    for sh in filter(None, os.environ.get("SHAPES", "").split(",")):
        build(*(int(v) for v in sh.split("x")), out_dir)


if __name__ == "__main__":
    main()
