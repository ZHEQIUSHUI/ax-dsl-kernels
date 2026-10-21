"""Board bandwidth of runtime-weight GEMV: bf16 vs MXFP8, by output-channel tile (and input-channel tile).

Compiles one single-core kernel per variant (x [1, K] bf16, w [N, K] runtime, out [1, N] f32) and copies it to
<out_dir>/<fmt>_n<N>_k<K>_oc<T>_ic<I>_npu1.axmodel; time them on the board with ax_run_model (random inputs).
Each variant runs once on the simulator against torch (dequantized weights for MX).

Usage: python mx_tile_bench.py out_dir
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger
from onnx import TensorProto

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
from axnn.frontend import graph_utils

graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXE8M0", TensorProto.UINT8)
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXFP8_E4M3", TensorProto.UINT8)

GROUP = 32
# (N, K) -> variants (format, oc tile, ic tile; 0 = whole K). Large K needs an ic tile to fit the OCM.
SHAPES = {
    (28672, 1024): [("bf16", 128, 0), ("bf16", 256, 0), ("mx8", 128, 0), ("mx8", 256, 0), ("mx8", 512, 0),
                    ("mx8", 1024, 0)],
    (1024, 28672): [("bf16", 256, 2048), ("bf16", 128, 4096), ("mx8", 256, 2048), ("mx8", 512, 2048),
                    ("mx8", 256, 4096), ("mx8", 1024, 1024)],
}
BUILD = os.path.abspath(os.environ.get("DUMP", "out_mx_tile"))
TILE = {}


def quant_mxfp8(w):
    g = w.float().reshape(w.shape[0], -1, GROUP)
    e = torch.ceil(torch.log2(g.abs().amax(-1, keepdim=True).clamp_min(2.0**-126) / 448.0)).clamp(-127, 127)
    q8 = (g / torch.exp2(e)).to(torch.float8_e4m3fn).reshape(w.shape)
    return q8.contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).view(torch.float8_e8m0fnu).contiguous()


def gemv_bf16(x, w, out):
    o = linear(ax.dsl.load(x), w, None, "mm", output_dtype=ax.fp32, oc_tile_size=TILE["oc"],
               ic_tile_size=TILE["ic"] or None)
    ax.dsl.store(o, r=out)
    return out


def gemv_mx8(x, w, s, out):
    o = linear(ax.dsl.load(x), w, s, "mm", output_dtype=ax.fp32, oc_tile_size=TILE["oc"],
               ic_tile_size=TILE["ic"] or None)
    ax.dsl.store(o, r=out)
    return out


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    only = os.environ.get("ONLY")  # e.g. "1024x28672"
    for (n, k), variants in SHAPES.items():
        if only and only != f"{n}x{k}":
            continue
        x = torch.randn((1, k), dtype=torch.bfloat16)
        w = (torch.randn((n, k)) * k**-0.5).bfloat16()
        w8, s8 = quant_mxfp8(w)
        wdq = w8.float() * torch.exp2(s8.view(torch.uint8).float() - 127).repeat_interleave(GROUP, dim=-1)
        for fmt, oc, ic in variants:
            tag = f"{fmt}_n{n}_k{k}_oc{oc}_ic{ic}"
            os.makedirs(f"{BUILD}/{tag}", exist_ok=True)
            os.chdir(f"{BUILD}/{tag}")
            TILE.update(oc=oc, ic=ic)
            fn = gemv_bf16 if fmt == "bf16" else gemv_mx8
            kern = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{tag}")(fn)
            out = torch.zeros((1, n), dtype=torch.float32)
            if fmt == "bf16":
                kern[1](x, w, out)
                ref = x.float() @ w.float().T
            else:
                kern[1](x, w8, s8, out)
                ref = x.float() @ wdq.T
            c = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
            (path,) = glob.glob(f"{BUILD}/{tag}/cache/{fn.__name__}.*.compiled.axmodel")
            shutil.copy(path, f"{out_dir}/{tag}_npu1.axmodel")
            logger.info(f"{tag}: cos={c:.6f}")


if __name__ == "__main__":
    main()
