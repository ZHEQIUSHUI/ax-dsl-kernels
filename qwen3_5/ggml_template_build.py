"""Build the Qwen3.5 layer kernels for the ggml-axera graph template (GGML_AXERA_TEMPLATE=qwen35).

Writes into <kernel_dir>: qwen35_gdn.axmodel, qwen35_attn.axmodel (q35_step.py's layer kernels; attention compiled for
LMAX/BC from env, default 4096/64) and qwen35.txt "LMAX BC ROT THETA". The backend feeds them the GGUF weights as they
are, except ssm_a -> log(-ssm_a) and the conv1d weight (time-flipped, bf16). Each kernel runs once on the simulator.

Usage: MODEL=<hf dir> LMAX=4096 BC=64 python ggml_template_build.py <kernel_dir>
"""
import glob
import os
import shutil
import sys

os.environ.setdefault("LMAX", "4096")
os.environ.setdefault("BC", "64")

import torch
from loguru import logger

import axnn.dsl as ax
import q35_step as q

OUT = os.path.abspath(sys.argv[1])
BUILD = os.path.abspath(os.environ.get("DUMP", "out_ggml_template"))


def main():
    os.makedirs(OUT, exist_ok=True)
    for sub in ("gdn", "attn"):
        os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    x = W("embed_tokens.weight")[1000:1001].reshape(1, 1, q.H).contiguous()

    os.chdir(f"{BUILD}/gdn")
    k_gdn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/gdn")(q.gdn_layer)
    case = dict(q.gdn_args(W, 0), x=x, state_in=torch.zeros((q.NV, q.DK, q.DV)),
                conv_in=torch.zeros((1, q.CONV - 1, 2 * q.KD + q.VD), dtype=torch.bfloat16),
                out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16), state_out=torch.zeros((q.NV, q.DK, q.DV)),
                conv_out=torch.zeros((1, q.CONV - 1, 2 * q.KD + q.VD), dtype=torch.bfloat16))
    k_gdn[1](*[case[n] for n in q.GDN_ORDER])
    logger.info(f"gdn_layer compiled, out finite={bool(case['out'].float().isfinite().all())}")

    os.chdir(f"{BUILD}/attn")
    k_attn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/attn")(q.attn_layer)
    c, s = q.rope_cos_sin(0)
    mask = torch.full((q.LMAX,), -3e38, dtype=torch.bfloat16)
    mask[0] = 0
    case = dict(q.attn_args(W, 3), x=x, cos=c, sin=s, kc=torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16),
                vc=torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16), pos=torch.tensor([0], dtype=torch.int32),
                nblk=torch.tensor([1], dtype=torch.int32), mask=mask.reshape(q.LMAX // q.BC, q.BC),
                out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16))
    k_attn[1](*[case[n] for n in q.ATTN_ORDER])
    logger.info(f"attn_layer LMAX={q.LMAX} BC={q.BC} compiled, out finite={bool(case['out'].float().isfinite().all())}")

    for sub, name in (("gdn", "gdn_layer"), ("attn", "attn_layer")):
        (path,) = glob.glob(f"{BUILD}/{sub}/cache/{name}.*.compiled.axmodel")
        shutil.copy(path, f"{OUT}/qwen35_{sub}.axmodel")
    open(f"{OUT}/qwen35.txt", "w").write(f"{q.LMAX} {q.BC} {q.ROT} {q.THETA}\n")
    logger.info(f"wrote {OUT}/qwen35_gdn.axmodel, qwen35_attn.axmodel, qwen35.txt")


if __name__ == "__main__":
    main()
