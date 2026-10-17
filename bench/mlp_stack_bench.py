"""Phase-0 bench: L stacked MLP layers (gate/up/silu*mul/down + residual) in ONE call, single core.

Weights are runtime MXFP4 inputs, stacked over layers, like a ggml backend would bind them.
Compare t(L layers in one call) against L * t(1 layer per call) to measure call-overhead amortization.

Env: L (layers, default 24), H (hidden, 1024), F (ffn, 3584), TN (tile, 256)
"""

import os
import sys

import torch
from loguru import logger
from onnx import TensorProto

import axnn.dsl as ax
from axnn.frontend import graph_utils

# 导出器缺 MX 运行时输入的 dtype 映射：按 uint8 字节流导出（仅本脚本内生效，不改仓库）
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXE8M0", TensorProto.UINT8)
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXFP4_X2", TensorProto.UINT8)

L = int(os.environ.get("L", 24))
H = int(os.environ.get("H", 1024))
F = int(os.environ.get("F", 3584))
TN = int(os.environ.get("TN", 256))
GROUP = 32
DUMP = os.path.abspath(f"out_mlp_l{L}_h{H}_f{F}_t{TN}")


def mlp_stack(x, wg, sg, wu, su, wd, sd, out):
    cur = ax.dsl.load(x)  # [1, H] bf16
    for layer in range(L):
        g = ax.dsl.create_tensor((1, F), ax.bf16, ax.MemoryType.OCM, name=f"g{layer}")
        u = ax.dsl.create_tensor((1, F), ax.bf16, ax.MemoryType.OCM, name=f"u{layer}")
        for n0 in range(0, F, TN):
            rows = slice(layer * F + n0, layer * F + n0 + TN)
            ax.dsl.matmul(cur, ax.dsl.load(wg[rows, :]), w_scale=ax.dsl.load(sg[rows, :]), transposed=True,
                          output_dtype=ax.bf16, r=g[:, n0 : n0 + TN])
            ax.dsl.matmul(cur, ax.dsl.load(wu[rows, :]), w_scale=ax.dsl.load(su[rows, :]), transposed=True,
                          output_dtype=ax.bf16, r=u[:, n0 : n0 + TN])
        h = ax.dsl.mul(ax.dsl.silu(g, output_dtype=ax.fp32), u, output_dtype=ax.bf16)
        nxt = ax.dsl.create_tensor((1, H), ax.bf16, ax.MemoryType.OCM, name=f"x{layer + 1}")
        for n0 in range(0, H, TN):
            rows = slice(layer * H + n0, layer * H + n0 + TN)
            d = ax.dsl.matmul(h, ax.dsl.load(wd[rows, :]), w_scale=ax.dsl.load(sd[rows, :]), transposed=True,
                              output_dtype=ax.bf16)
            ax.dsl.add(cur[:, n0 : n0 + TN], d, output_dtype=ax.bf16, r=nxt[:, n0 : n0 + TN])
        cur = nxt
    ax.dsl.store(cur, r=out)
    return out


LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def make_weight(rows, cols, g):
    codes = torch.randint(0, 16, (rows, cols), generator=g, dtype=torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    enc = torch.randint(116, 121, (rows, cols // GROUP), generator=g, dtype=torch.uint8)
    deq = LUT[codes.long()] * (2.0 ** (enc.float() - 127)).repeat_interleave(GROUP, dim=-1)
    return packed, enc.view(torch.float8_e8m0fnu), deq


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(DUMP)  # 编译缓存（含 axmodel）写到当前目录的 cache/
    ax.init(chip_num=1, core_num=1)
    g = torch.Generator().manual_seed(7)
    x = torch.randn((1, H), dtype=torch.bfloat16)
    wg, sg, dg = make_weight(L * F, H, g)
    wu, su, du = make_weight(L * F, H, g)
    wd, sd, dd = make_weight(L * H, F, g)
    out = torch.empty((1, H), dtype=torch.bfloat16)

    kernel = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=DUMP)(mlp_stack)
    kernel[1](x, wg, sg, wu, su, wd, sd, out)

    ref = x.float()
    for layer in range(L):
        gg = ref @ dg[layer * F : (layer + 1) * F].T
        uu = ref @ du[layer * F : (layer + 1) * F].T
        hh = torch.nn.functional.silu(gg) * uu
        ref = ref + hh @ dd[layer * H : (layer + 1) * H].T
    c = torch.nn.functional.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0).item()
    weight_mb = (wg.numel() + wu.numel() + wd.numel() + sg.numel() + su.numel() + sd.numel()) / 1e6
    logger.info(f"RESULT L={L} H={H} F={F} TN={TN} weights={weight_mb:.1f}MB cosine={c:.6f} dump={DUMP}")
    if c < 0.98:
        sys.exit(3)


if __name__ == "__main__":
    main()
