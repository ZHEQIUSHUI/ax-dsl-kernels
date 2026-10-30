"""Qwen3-ASR audio front-end on four cores: each 3 x 3 stride-2 convolution's product with its bias and GELU (erf) in
one kernel, its output bf16 for the next layer's im2col rows.

  qwen3a_fe_o<OC>_k<K>_r<R>   out [OC, R] bf16 = gelu(w [OC, K] . x [R, K]^T + b [OC])  (x the im2col rows the host
                              writes as bf16, w the conv kernel in ggml's order [OC][IC][KH][KW], K padded to a
                              multiple of 16 with zero columns)

Each core takes OC / CORES output channels (no synchronization); per block of OB channels the kernel's rows stay on
chip and the x rows pass in tiles of RT, each one product over the whole K (no accumulation), the bias added in f32.
R is the rows of one encoder call (n one-second chunks: conv2d.1 3200 n rows, conv2d.2 800 n, conv2d.3 208 n).

KIND=im2col: conv2d.2 with its im2col on chip, for n chunks (NCH):

  qwen3a_fe2_o<OC>_n<n>       out [OC, n * 800] bf16 = gelu(conv(x) + b): x conv2d.1's output [IC, n * 3200] bf16 in rows
                              (chunk, h of 64, column parity, w / 2 of 25) (the host orders conv2d.1's im2col rows so),
                              w [OC][3][3][IC] bf16 (the taps' matrices side by side), out rows (chunk, h of 32, w of 25)

per tile of TO output rows of 25 steps the 9 taps' [IC, 25] windows (the even and odd columns of an input row, the odd
ones shifted by one for the left tap) stacked on chip [9 IC, TO * 25], one product over the whole K.

Written as qwen3a_fe_o<OC>_k<K>_r<R>_npu<CORES>.axmodel / qwen3a_fe2_o<OC>_n<n>_npu<CORES>.axmodel. Checked on the
simulator against torch on random data.
Usage: OC=480 K=4320 R=6400 CORES=4 python q3a_fe.py [kernel_dir]; KIND=im2col NCH=8 python q3a_fe.py [kernel_dir]
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger

import axnn.dsl as ax
import board_io

OC, K, R = int(os.environ.get("OC", 480)), int(os.environ.get("K", 4320)), int(os.environ.get("R", 6400))
CORES = int(os.environ.get("CORES", 4))
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q3a_fe"))
OCC = OC // CORES  # output channels per core
WB = int(os.environ.get("WB", 1024 * 1024))  # bytes of one channel block's kernel rows on chip
XB = int(os.environ.get("XB", 1536 * 1024))  # bytes of one x tile and its output on chip
assert OC % CORES == 0 and K % 16 == 0


def divisor_at_most(n, m, multiple=1):
    """the largest divisor of n not above m (a multiple of `multiple` where one exists)"""
    for c in range(min(n, m), 0, -1):
        if n % c == 0 and (c % multiple == 0 or c == n):
            return c
    return 1


OB = divisor_at_most(OCC, max(1, WB // (K * 2)))
RT = divisor_at_most(R, max(16, XB // (K * 2 + OB * 6)), 16)


def fe_conv(x, w, b, out):
    c = ax.program_id(0)
    for o0 in range(0, OCC, OB):
        oc = c * OCC + o0
        wb = ax.dsl.load(w[oc : oc + OB])  # [OB, K]
        bb = ax.dsl.load(b[oc : oc + OB]).reshape(OB, 1)
        for r0 in range(0, R, RT):
            xt = ax.dsl.load(x[r0 : r0 + RT])  # [RT, K]
            acc = ax.dsl.matmul(wb, xt, transposed=True, output_dtype=ax.fp32)  # [OB, RT]
            y = ax.dsl.gelu(ax.dsl.add(acc, bb, output_dtype=ax.fp32), output_dtype=ax.bf16)
            ax.dsl.store(y, r=out[oc : oc + OB, r0 : r0 + RT])
    return out


KIND = os.environ.get("KIND", "gemm")
NCH = int(os.environ.get("NCH", 8))
IC2, IH, IW, OH2, OW2 = 480, 64, 50, 32, 25  # conv2d.2: 480 x 64 x 50 -> 32 x 25
TO = int(os.environ.get("TO", 4))  # output rows of 25 steps per tile


def fe_conv2(x, w, b, out):
    c = ax.program_id(0)
    zero = ax.dsl.zeros(shape=(IC2, OW2), dtype=ax.bf16)
    for o0 in range(0, OCC, OB2):
        oc = c * OCC + o0
        wb = ax.dsl.load(w[oc : oc + OB2])  # [OB2, 9 IC]
        bb = ax.dsl.load(b[oc : oc + OB2]).reshape(OB2, 1)
        for ch in range(NCH):
            for t0 in range(0, OH2, TO):
                xk = ax.dsl.create_tensor((9 * IC2, TO * OW2), ax.bf16, ax.MemoryType.OCM, name=f"xk{o0}_{ch}_{t0}")
                rows = {}
                for t in range(TO):
                    for kh in range(3):
                        h = 2 * (t0 + t) + kh - 1
                        if 0 <= h < IH and h not in rows:
                            base = (ch * IH + h) * IW
                            rows[h] = ax.dsl.load(x[:, base : base + IW])  # [IC, 50]: even | odd columns
                        for kw in range(3):
                            dst = xk[(kh * 3 + kw) * IC2 : (kh * 3 + kw + 1) * IC2, t * OW2 : (t + 1) * OW2]
                            if not 0 <= h < IH:
                                ax.dsl.move(r=dst, x=zero)
                            elif kw == 1:  # w = 2 ow: the even columns
                                ax.dsl.move(r=dst, x=rows[h][:, 0:OW2])
                            elif kw == 2:  # w = 2 ow + 1: the odd columns
                                ax.dsl.move(r=dst, x=rows[h][:, OW2 : 2 * OW2])
                            else:  # w = 2 ow - 1: the odd columns one step later, zero at ow = 0
                                ax.dsl.move(r=dst[:, 0:1], x=zero[:, 0:1])
                                ax.dsl.move(r=dst[:, 1:OW2], x=rows[h][:, OW2 : 2 * OW2 - 1])
                acc = ax.dsl.matmul(wb, xk, output_dtype=ax.fp32)  # [OB2, TO * 25]
                y = ax.dsl.gelu(ax.dsl.add(acc, bb, output_dtype=ax.fp32), output_dtype=ax.bf16)
                r0 = (ch * OH2 + t0) * OW2
                ax.dsl.store(y, r=out[oc : oc + OB2, r0 : r0 + TO * OW2])
    return out


OB2 = divisor_at_most(OCC, max(1, WB // (9 * IC2 * 2)))


def main_im2col():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    name = f"qwen3a_fe2_o{OC}_n{NCH}"
    dump = f"{BUILD}/{name}"
    os.makedirs(dump, exist_ok=True)
    os.chdir(dump)
    ax.init(chip_num=1, core_num=CORES)
    torch.manual_seed(0)
    xm = torch.randn((IC2, NCH, IH, IW)).bfloat16()  # conv2d.1's output map
    wt = (torch.randn((OC, IC2, 3, 3)) * (9 * IC2) ** -0.5).bfloat16()
    b = torch.randn(OC) * 0.1
    # the kernel's layouts: columns (chunk, h, parity, w / 2); taps [OC][kh][kw][IC]
    xph = torch.cat([xm[..., 0::2], xm[..., 1::2]], -1).reshape(IC2, NCH * IH * IW).contiguous()
    wk = wt.permute(0, 2, 3, 1).reshape(OC, 9 * IC2).contiguous()
    out = torch.zeros((OC, NCH * OH2 * OW2), dtype=torch.bfloat16)
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(fe_conv2)[CORES](xph, wk, b, out)
    ref = torch.nn.functional.conv2d(xm.float().permute(1, 0, 2, 3), wt.float(), b, stride=2, padding=1)  # [n, OC, 32, 25]
    ref = torch.nn.functional.gelu(ref).permute(1, 0, 2, 3).reshape(OC, -1)
    c = torch.nn.functional.cosine_similarity(out.float().reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    logger.info(f"{name} (channels per block {OB2}, rows per tile {TO * OW2}): cos vs torch {c:.6f}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        (path,) = glob.glob(f"{dump}/cache/fe_conv2.*.compiled.axmodel")
        dst = board_io.kernel_file(out_dir, name, CORES)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


def main():
    if KIND == "im2col":
        return main_im2col()
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    name = f"qwen3a_fe_o{OC}_k{K}_r{R}"
    dump = f"{BUILD}/{name}"
    os.makedirs(dump, exist_ok=True)
    os.chdir(dump)
    ax.init(chip_num=1, core_num=CORES)
    torch.manual_seed(0)
    w = (torch.randn((OC, K)) * K**-0.5).bfloat16()
    x = torch.randn((R, K)).bfloat16()
    b = torch.randn(OC) * 0.1
    out = torch.zeros((OC, R), dtype=torch.bfloat16)
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(fe_conv)[CORES](x, w, b, out)
    ref = torch.nn.functional.gelu(w.float() @ x.float().T + b[:, None])
    c = torch.nn.functional.cosine_similarity(out.float().reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    logger.info(f"{name} (channels per block {OB}, rows per tile {RT}): cos vs torch {c:.6f}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        (path,) = glob.glob(f"{dump}/cache/fe_conv.*.compiled.axmodel")
        dst = board_io.kernel_file(out_dir, name, CORES)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
