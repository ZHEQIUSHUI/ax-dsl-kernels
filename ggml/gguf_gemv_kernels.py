"""Precompile the decode GEMV kernels ggml-axera looks up, one per weight shape found in a GGUF.

Kernel "mul_mat_bf16_n<C>_k<K>.axmodel": x [1, K] bf16, w [C, K] bf16 (runtime input), out [1, C] f32. Every BF16
2-D weight shape [N, K] of the GGUF gets one, with C = N, or for N > MAX_N the largest multiple of 16 not above MAX_N
that divides N; the backend then calls it N / C times (e.g. token_embd as the tied output projection). The conv1d
weights are skipped (not matrix products). Each kernel is run once on the simulator against torch.

Usage: PYTHONPATH=<llama.cpp>/gguf-py python gguf_gemv_kernels.py model.gguf out_dir
SHAPES="<N>x<K>,..." adds shapes (e.g. a quantized GGUF's weights the backend converts to bf16); model.gguf "-" reads
no GGUF.
"""
import glob
import os
import shutil
import sys

import torch
from gguf import GGUFReader, GGMLQuantizationType
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear

MAX_N = int(os.environ.get("MAX_N", 32768))
SKIP = ("ssm_conv1d",)
BUILD = os.path.abspath(os.environ.get("DUMP", "out_gguf_gemv"))


def oc_tile(cols):
    """Output-channel tile so one bf16 weight tile stays around 512 KB of OCM."""
    return max(16, min(256, (512 * 1024) // (cols * 2)))


def chunk_rows(n):
    """Weight rows per kernel call: N itself if small enough, else the largest divisor of N that fits."""
    if n <= MAX_N:
        return n
    for c in range(MAX_N - MAX_N % 16, 15, -16):
        if n % c == 0:
            return c
    return 0


def mul_mat_bf16(x, w, out):
    o = linear(ax.dsl.load(x), w, None, "mm", output_dtype=ax.fp32, oc_tile_size=oc_tile(w.shape[1]))
    ax.dsl.store(o, r=out)
    return out


def main():
    reader = GGUFReader(sys.argv[1]) if sys.argv[1] != "-" else None
    out_dir = os.path.abspath(sys.argv[2])
    shapes = {}
    for sh in filter(None, os.environ.get("SHAPES", "").split(",")):
        n, k = (int(v) for v in sh.split("x"))
        shapes.setdefault((chunk_rows(n), k), []).append(f"SHAPES {sh}")
    for t in reader.tensors if reader else []:
        if t.tensor_type != GGMLQuantizationType.BF16 or len(t.shape) != 2 or any(s in t.name for s in SKIP):
            continue
        k, n = int(t.shape[0]), int(t.shape[1])  # ggml ne0 = K（输入维），ne1 = N
        c = chunk_rows(n)
        if c:
            shapes.setdefault((c, k), []).append(t.name if c == n else f"{t.name} ({n // c} chunks)")
        else:
            logger.warning(f"{t.name}: no chunk size for N={n}, stays on the CPU")
    logger.info(f"{len(shapes)} shapes: " + ", ".join(f"{n}x{k}({len(v)})" for (n, k), v in sorted(shapes.items())))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    for (n, k), names in sorted(shapes.items()):
        tag = f"n{n}_k{k}"
        os.makedirs(f"{BUILD}/{tag}", exist_ok=True)
        os.chdir(f"{BUILD}/{tag}")  # 每种形状单独的编译缓存目录
        kern = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{tag}")(mul_mat_bf16)
        x = torch.randn((1, k), dtype=torch.bfloat16)
        w = (torch.randn((n, k)) * k**-0.5).bfloat16()
        out = torch.zeros((1, n), dtype=torch.float32)
        kern[1](x, w, out)
        ref = x.float() @ w.float().T
        c = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
        (path,) = glob.glob(f"{BUILD}/{tag}/cache/mul_mat_bf16.*.compiled.axmodel")
        shutil.copy(path, f"{out_dir}/mul_mat_bf16_{tag}.axmodel")
        logger.info(f"{tag}: cos={c:.6f} ({', '.join(names[:2])}{'...' if len(names) > 2 else ''})")
        if c < 0.999:
            sys.exit(f"{tag}: kernel mismatch")


if __name__ == "__main__":
    main()
