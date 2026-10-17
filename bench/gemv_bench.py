"""Phase-0 bench: single-core decode GEMV, MXFP4 weight as constant (FC) vs runtime input (MatMul).

Env:
  WMODE = rt | const      weight passed as runtime input or baked as constant
  K, N  = in/out features (default Qwen3.5-9B gate: 4096 x 12288)
  TN    = output-channel tile per step (default 256)
  M     = tokens (default 1 = decode)
"""

import os
import sys

import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.backend.pytests.casegen import C
from axnn.backend.pytests.fullyconnected_test import transform_float
from axnn.opset import datatype
from axnn.opset.reference.fullyconnected import pyfn as fc_pyfn
from onnx import TensorProto
from axnn.frontend import graph_utils

# 导出器缺 MX 运行时输入的 dtype 映射：按 uint8 字节流导出（仅本脚本内生效，不改仓库）
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXE8M0", TensorProto.UINT8)
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXFP4_X2", TensorProto.UINT8)

MODE = os.environ.get("WMODE", "rt")
K = int(os.environ.get("K", 4096))
N = int(os.environ.get("N", 12288))
TN = int(os.environ.get("TN", 256))
M = int(os.environ.get("M", 1))
CORES = int(os.environ.get("CORES", 1))
GROUP = 32
DUMP = os.path.abspath(f"out_{MODE}_c{CORES}_m{M}_k{K}_n{N}_t{TN}")


def gemv_rt(x, w, s, out):
    xt = ax.dsl.load(x)
    for n0 in range(0, N, TN):
        wt = ax.dsl.load(w[n0 : n0 + TN, :])
        st = ax.dsl.load(s[n0 : n0 + TN, :])
        o = ax.dsl.matmul(xt, wt, w_scale=st, transposed=True, output_dtype=ax.bf16)
        ax.dsl.store(o, r=out[:, n0 : n0 + TN])
    return out


def gemv_rt_mc(x, w, s, out):
    # 多核：每个核算 N 的一段，核间无依赖，不做 sync
    n_per = N // CORES
    base = ax.program_id(0) * n_per
    xt = ax.dsl.load(x)
    for n0 in range(base, base + n_per, TN):
        wt = ax.dsl.load(w[n0 : n0 + TN, :])
        st = ax.dsl.load(s[n0 : n0 + TN, :])
        o = ax.dsl.matmul(xt, wt, w_scale=st, transposed=True, output_dtype=ax.bf16)
        ax.dsl.store(o, r=out[:, n0 : n0 + TN])
    return out


def gemv_const(x, w, s, out):
    xt = ax.dsl.load(x)
    for n0 in range(0, N, TN):
        o = ax.dsl.matmul(
            xt,
            w[n0 : n0 + TN, :],
            w_scale=s[n0 : n0 + TN, :],
            transposed=True,
            output_dtype=ax.bf16,
            const_inputs=["y"],
        )
        ax.dsl.store(o, r=out[:, n0 : n0 + TN])
    return out


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(DUMP)  # 编译缓存（含 axmodel）写到当前目录的 cache/
    ax.init(chip_num=1, core_num=CORES)
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16)
    out = torch.empty((M, N), dtype=torch.bfloat16)

    if MODE == "rt":
        g = torch.Generator().manual_seed(1)
        codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.uint8)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
        enc = torch.randint(118, 124, (N, K // GROUP), generator=g, dtype=torch.uint8)
        scale = enc.view(torch.float8_e8m0fnu)
        fn = gemv_rt_mc if CORES > 1 else gemv_rt
        kernel = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=DUMP)(fn)
        kernel[CORES](x, packed, scale, out)
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        wf = lut[codes.long()] * (2.0 ** (enc.float() - 127)).repeat_interleave(GROUP, dim=-1)
        ref = x.float() @ wf.T
    else:
        case = C(
            if_shape=(M, K), ic=K, oc=N, if_dtype=datatype.bf16, w_dtype=datatype.mxfp4,
            of_dtype=datatype.bf16, block_size=0,
        )
        attrs, _ = transform_float(case)
        attrs["const_inputs"]["b"] = np.zeros(attrs["const_inputs"]["b"].shape, dtype=attrs["const_inputs"]["b"].dtype)
        w = attrs["const_inputs"]["w"]
        s = attrs["const_inputs"]["w_scale"]
        kernel = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=DUMP, const_inputs=["w", "s"])(
            gemv_const
        )
        kernel[1](x, w, s, out)
        from axnn.dsl import util

        ref = util.np2torch(fc_pyfn(attrs, {"x": util.torch2np(x)})["r"])

    c = cos(ref, out)
    logger.info(f"RESULT mode={MODE} M={M} K={K} N={N} TN={TN} cosine={c:.6f} dump={DUMP}")
    if c < 0.99:
        sys.exit(3)


if __name__ == "__main__":
    main()
