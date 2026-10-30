"""1-D convolutions of a ggml graph that has no template of their own (ggml-axera's generic executor: the Qwen3-TTS
codec decoder's causal convolutions), one kernel per shape, read in ggml's layout instead of im2col rows:

  conv1d_bf16_ic<IC>_oc<OC>_k<K>_d<D>_l<L>   x [IC, L] bf16 (channels, time: the input with its left context, no
                                              padding), w [K, OC, IC] bf16 (the taps' matrices; both runtime inputs)
                                              -> out [OC, L - (K - 1) * D] f32

out[o][t] = sum over i, k of w[o][i][k] * x[i][t + k * D], ggml's ggml_conv_1d(w, x, 1, 0, D) (whose im2col and
MUL_MAT the backend then skips), as one matrix product over K * IC (AxConvV2 takes channels-last input and writes bf16
only): the K taps' matrices side by side on chip [OB, K * IC], the input block's K shifted windows stacked on chip
[K * IC, LT]. No product accumulates into another's output (the NPU's add-assign of f32 outputs gave results that
differed between runs on the board). Blocks of OB output channels and LT output steps (an input block with the
(K - 1) * D steps after it).

CORES=<n> splits each kernel over n cores without synchronization: each core takes OC / n output channels, else
(OW = L - (K - 1) * D steps) OW / n output steps; shapes that split neither way are skipped.

Each kernel (file <name>_npu<CORES>.axmodel) is run once on the simulator against torch (random data).
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
CORES = int(os.environ.get("CORES", 1))


def divisor_at_most(n, m, multiple=1):
    """the largest divisor of n not above m (a multiple of `multiple` where one exists)"""
    for c in range(min(n, m), 0, -1):
        if n % c == 0 and (c % multiple == 0 or c == n):
            return c
    return 1


def build(IC, OC, K, D, L, out_dir):
    OW = L - (K - 1) * D
    halo = (K - 1) * D
    if CORES == 1 or OC % CORES == 0:
        OCC, OWC = OC // CORES, OW  # output channels, steps per core
    elif OW % CORES == 0:
        OCC, OWC = OC, OW // CORES
    else:
        logger.warning(f"conv1d ic{IC} oc{OC} k{K} d{D} l{L}: splits over {CORES} cores neither way, skipped")
        return
    OB = divisor_at_most(OCC, max(1, WB // (IC * K * 2)))
    LT = divisor_at_most(OWC, max(16, XB // (IC * 2 * (K + 1 if K > 1 else 1) + OB * 4)), 16)

    def conv1d(x, w, out):
        c = ax.program_id(0)
        oc1, t1 = (c * OCC, 0) if OCC < OC else (0, c * OWC)
        for o0 in range(oc1, oc1 + OCC, OB):
            if K == 1:
                wb = ax.dsl.load(w[0:1, o0 : o0 + OB]).reshape(OB, IC)
            else:
                wb = ax.dsl.create_tensor((OB, K * IC), ax.bf16, ax.MemoryType.OCM, name=f"w{o0}")  # [OB, K * IC]
                for k in range(K):
                    ax.dsl.move(r=wb[:, k * IC : (k + 1) * IC], x=ax.dsl.load(w[k : k + 1, o0 : o0 + OB]).reshape(OB, IC))
            for t0 in range(t1, t1 + OWC, LT):
                xb = ax.dsl.load(x[:, t0 : t0 + LT + halo])  # [IC, LT + halo]
                xk = xb
                if K > 1:
                    xk = ax.dsl.create_tensor((K * IC, LT), ax.bf16, ax.MemoryType.OCM, name=f"x{o0}_{t0}")
                    for k in range(K):
                        ax.dsl.move(r=xk[k * IC : (k + 1) * IC], x=xb[:, k * D : k * D + LT])
                acc = ax.dsl.matmul(wb, xk, output_dtype=ax.fp32)
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
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(conv1d)[CORES](x, wk, out)
    ref = torch.nn.functional.conv1d(x.float()[None], w.float(), dilation=D)[0]
    c = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    logger.info(f"{name} (output channels per block {OB}, steps per block {LT}): cos vs torch {c:.6f}")
    (path,) = glob.glob(f"{dump}/cache/conv1d.*.compiled.axmodel")
    shutil.copy(path, f"{out_dir}/{name}_npu{CORES}.axmodel")
    os.chdir(cwd)


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    ax.init(chip_num=1, core_num=CORES)
    for sh in filter(None, os.environ.get("SHAPES", "").split(",")):
        build(*(int(v) for v in sh.split("x")), out_dir)


if __name__ == "__main__":
    main()
