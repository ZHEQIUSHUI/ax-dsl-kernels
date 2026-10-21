"""1-D convolutions of a ggml graph that has no template of their own (ggml-axera's generic executor: the Qwen3-TTS
codec decoder's causal convolutions), one kernel per shape, read in ggml's layout instead of im2col rows:

  conv1d_bf16_ic<IC>_oc<OC>_k<K>_d<D>_l<L>   x [IC, L] bf16 (channels, time: the input with its left context, no
                                              padding), w [K, OC, IC] bf16 (the taps' matrices; both runtime inputs)
                                              -> out [OC, L - (K - 1) * D] f32

out[o][t] = sum over i, k of w[o][i][k] * x[i][t + k * D], ggml's ggml_conv_1d(w, x, 1, 0, D) (whose im2col and
MUL_MAT the backend then skips), as K matrix products accumulated in f32: out[:, t] += w[k] . x[:, t + k * D] (AxConvV2
takes channels-last input and writes bf16 only). Blocks of OB output channels (their K matrices on chip) and LT output
steps (an input block with the (K - 1) * D steps after it, the taps its shifted windows).

Each kernel is run once on the simulator against torch (random data).
Usage: SHAPES="<IC>x<OC>x<K>x<D>x<L>,..." python conv1d_kernels.py out_dir
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger

import axnn.dsl as ax

BUILD = os.path.abspath(os.environ.get("DUMP", "out_conv1d"))
WB = int(os.environ.get("WB", 1024 * 1024))  # bytes of one output-channel block's weights on chip
XB = int(os.environ.get("XB", 2 * 1024 * 1024))  # bytes of one time block's input and output on chip


def divisor_at_most(n, m, multiple=1):
    """the largest divisor of n not above m (a multiple of `multiple` where one exists)"""
    for c in range(min(n, m), 0, -1):
        if n % c == 0 and (c % multiple == 0 or c == n):
            return c
    return 1


def build(IC, OC, K, D, L, out_dir):
    OW = L - (K - 1) * D
    halo = (K - 1) * D
    OB = divisor_at_most(OC, max(1, WB // (IC * K * 2)))
    LT = divisor_at_most(OW, max(16, XB // (IC * 2 + OB * 4)), 16)

    def conv1d(x, w, out):
        for o0 in range(0, OC, OB):
            wb = [ax.dsl.load(w[k : k + 1, o0 : o0 + OB]).reshape(OB, IC) for k in range(K)]  # K x [OB, IC]
            for t0 in range(0, OW, LT):
                xb = ax.dsl.load(x[:, t0 : t0 + LT + halo])  # [IC, LT + halo]
                acc = ax.dsl.create_tensor((OB, LT), ax.fp32, name=f"acc{o0}_{t0}")
                for k in range(K):
                    ax.dsl.matmul(wb[k], xb[:, k * D : k * D + LT], output_dtype=ax.fp32, output_add_assign=k > 0, r=acc)
                ax.dsl.store(acc, r=out[o0 : o0 + OB, t0 : t0 + LT])
        return out

    name = f"conv1d_bf16_ic{IC}_oc{OC}_k{K}_d{D}_l{L}"
    dump = f"{BUILD}/{name}"
    os.makedirs(dump, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dump)
    torch.manual_seed(0)
    x = torch.randn((IC, L)).bfloat16()
    w = (torch.randn((OC, IC, K)) * 0.05).bfloat16()
    wk = w.permute(2, 0, 1).contiguous()  # [K, OC, IC], as the backend repacks ggml's [OC][IC][K]
    out = torch.zeros((OC, OW))
    conv1d.__name__ = "conv1d"
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(conv1d)[1](x, wk, out)
    ref = torch.nn.functional.conv1d(x.float()[None], w.float(), dilation=D)[0]
    c = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    logger.info(f"{name} (output channels per block {OB}, steps per block {LT}): cos vs torch {c:.6f}")
    (path,) = glob.glob(f"{dump}/cache/conv1d.*.compiled.axmodel")
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
