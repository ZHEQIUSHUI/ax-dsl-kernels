"""Build the Qwen3.5 layer kernels for the ggml-axera graph template (GGML_AXERA_TEMPLATE=qwen35).

Writes into <kernel_dir>: qwen35_gdn.axmodel, qwen35_attn.axmodel (q35_step.py's layer kernels; attention compiled for
LMAX/BC from env, default 4096/64). The backend reads LMAX and BC from the attention kernel's mask input and the RoPE
parameters from the llama.cpp graph; any llama.cpp context up to LMAX cells works. It feeds them the GGUF weights as they
are, except ssm_a -> log(-ssm_a) and the conv1d weight (time-flipped, bf16). Each kernel runs once on the simulator.

WFMT=mx8 builds the MXFP8 variants instead (GGML_AXERA_WEIGHTS=mxfp8; the backend quantizes the bf16 GGUF weights at
load): qwen35_gdn_mx8.axmodel, qwen35_attn_mx8.axmodel (q35_mx_build.py's kernels, a *_s scale input per projection)
and the output projection GEMV mul_mat_mx8_n<C>_k<K>.axmodel (x, w, s -> out f32), checked against dequantized bf16.

Usage: MODEL=<hf dir> LMAX=4096 BC=64 [WFMT=mx8] python ggml_template_build.py <kernel_dir>
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
from axnn.dsl.kernels.utils.linear import linear
import q35_step as q

WFMT = os.environ.get("WFMT", "bf16")
KVC = q.KVL == "cell"  # llama.cpp KV layout: only the attention kernels change, written as qwen35_attn*_kvc_c<LMAX>.axmodel
KV_SFX = f"_kvc_c{q.LMAX}" if KVC else ""


def kv_cache():
    if KVC:
        return torch.zeros((q.LMAX, q.HKV * q.D), dtype=torch.half), torch.zeros((q.LMAX, q.HKV * q.D), dtype=torch.half)
    return (torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16),
            torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16))
HEAD_C = int(os.environ.get("HEAD_C", 31040))  # output projection rows per GEMV call (248320 = 8 x 31040)

OUT = os.path.abspath(sys.argv[1])
BUILD = os.path.abspath(os.environ.get("DUMP", "out_ggml_template"))


def mul_mat_mx8(x, w, s, out):
    o = linear(ax.dsl.load(x), w, s, "mm", output_dtype=ax.fp32, oc_tile_size=256)
    ax.dsl.store(o, r=out)
    return out


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def build_mx8(W, x):
    import q35_mx_build as mb  # patches q.proj to the MX linear

    outs = {}
    os.chdir(f"{BUILD}/gdn")
    for tag, fn, names, args in [("mx", mb.gdn_layer_mx, mb.order(mb.gdn_layer_mx), None),
                                 ("deq", q.gdn_layer, q.GDN_ORDER, None)]:
        m, deq = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)
        k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/gdn")(fn)
        case = dict(m if tag == "mx" else deq, x=x, state_in=torch.zeros((q.NV, q.DK, q.DV)),
                    conv_in=torch.zeros((1, q.CONV - 1, 2 * q.KD + q.VD), dtype=torch.bfloat16),
                    out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16), state_out=torch.zeros((q.NV, q.DK, q.DV)),
                    conv_out=torch.zeros((1, q.CONV - 1, 2 * q.KD + q.VD), dtype=torch.bfloat16))
        mb.SCALES.clear()
        k[1](*[case[n] for n in names])
        outs[tag] = case["out"].float()
    logger.info(f"gdn_layer_mx: cos vs dequantized bf16 {cos(outs['mx'], outs['deq']):.6f}")

    os.chdir(f"{BUILD}/attn")
    c, s = q.rope_cos_sin(0)
    mask = torch.full((q.LMAX,), -3e38, dtype=torch.bfloat16)
    mask[0] = 0
    for tag, fn, names in [("mx", mb.attn_layer_mx, mb.order(mb.attn_layer_mx)), ("deq", q.attn_layer, q.ATTN_ORDER)]:
        m, deq = mb.mx_args(q.attn_args(W, 3), mb.ATTN_W)
        k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/attn")(fn)
        kc, vc = kv_cache()
        case = dict(m if tag == "mx" else deq, x=x, cos=c, sin=s, kc=kc, vc=vc, pos=torch.tensor([0], dtype=torch.int32),
                    nblk=torch.tensor([1], dtype=torch.int32), mask=mask.reshape(q.LMAX // q.BC, q.BC),
                    out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16))
        mb.SCALES.clear()
        k[1](*[case[n] for n in names])
        outs[tag] = case["out"].float()
    logger.info(f"attn_layer_mx LMAX={q.LMAX}: cos vs dequantized bf16 {cos(outs['mx'], outs['deq']):.6f}")

    os.makedirs(f"{BUILD}/head", exist_ok=True)
    os.chdir(f"{BUILD}/head")
    w = W("embed_tokens.weight")[:HEAD_C].contiguous()
    w8, s8 = mb.quant_mxfp8(w)
    out = torch.zeros((1, HEAD_C), dtype=torch.float32)
    xr = x.reshape(1, q.H)
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/head")(mul_mat_mx8)[1](xr, w8, s8, out)
    logger.info(f"mul_mat_mx8 n{HEAD_C}: cos vs dequantized {cos(out, xr.float() @ mb.dequant(w8, s8).T):.6f}")

    for sub, name, dst in (("gdn", "gdn_layer_mx", "qwen35_gdn_mx8"), ("attn", "attn_layer_mx", "qwen35_attn_mx8" + KV_SFX),
                           ("head", "mul_mat_mx8", f"mul_mat_mx8_n{HEAD_C}_k{q.H}")):
        if KVC and sub != "attn":
            continue
        (path,) = glob.glob(f"{BUILD}/{sub}/cache/{name}.*.compiled.axmodel")
        shutil.copy(path, f"{OUT}/{dst}.axmodel")
    logger.info(f"wrote {OUT}/qwen35_gdn_mx8.axmodel, qwen35_attn_mx8.axmodel, mul_mat_mx8_n{HEAD_C}_k{q.H}.axmodel")


def main():
    os.makedirs(OUT, exist_ok=True)
    for sub in ("gdn", "attn"):
        os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    x = W("embed_tokens.weight")[1000:1001].reshape(1, 1, q.H).contiguous()
    if WFMT == "mx8":
        build_mx8(W, x)
        return

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
    kc, vc = kv_cache()
    case = dict(q.attn_args(W, 3), x=x, cos=c, sin=s, kc=kc, vc=vc, pos=torch.tensor([0], dtype=torch.int32),
                nblk=torch.tensor([1], dtype=torch.int32), mask=mask.reshape(q.LMAX // q.BC, q.BC),
                out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16))
    k_attn[1](*[case[n] for n in q.ATTN_ORDER])
    logger.info(f"attn_layer LMAX={q.LMAX} BC={q.BC} compiled, out finite={bool(case['out'].float().isfinite().all())}")

    for sub, name in (("gdn", "gdn_layer"), ("attn", "attn_layer")):
        if KVC and sub == "gdn":
            continue
        (path,) = glob.glob(f"{BUILD}/{sub}/cache/{name}.*.compiled.axmodel")
        shutil.copy(path, f"{OUT}/qwen35_{sub}{KV_SFX if sub == 'attn' else ''}.axmodel")
    logger.info(f"wrote {OUT}/qwen35_gdn.axmodel, qwen35_attn.axmodel")


if __name__ == "__main__":
    main()
