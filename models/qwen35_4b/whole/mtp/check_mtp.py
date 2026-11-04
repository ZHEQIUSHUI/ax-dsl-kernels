#!/usr/bin/env python3
"""PYRUN/CRUN check of the Qwen3.5-4B MTP kernels against the PyTorch reference.

    source /data/junda/work/axdsl_env.sh
    MODEL=/data/junda/hf/Qwen3.5-4B QW_RANDOM=1 python check_mtp.py out_mtp_check
    ONLY=t1 python check_mtp.py out_mtp_check          # T=1 only
    ONLY=prompt NPROMPT=256 python check_mtp.py ...    # 256-row MTP prefill
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ.setdefault("KVL", "cell")
os.environ.setdefault("BC", "256")
os.environ.setdefault("CORES", "4")
os.environ.setdefault("LMAX", "16384")
os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("WFMT4", "1")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("KVX", "1")
os.environ.setdefault("DEC_SPLIT", "1")

import torch
from loguru import logger

import axnn.dsl as ax
import q35_layer_mc as L
import q35_step as q
from mtp_kernel import P, make_bundle, mtp_prompt, mtp_t1
from mtp_ref import H, MtpWeights, mtp_args, mtp_step
import resident_pyrun as rp

MODEL = os.environ.get("MODEL", "/data/junda/hf/Qwen3.5-4B")
RUN_MODE = os.environ.get("RUN_MODE", "PYRUN")
QW_RANDOM = os.environ.get("QW_RANDOM", "1") == "1"
ONLY = os.environ.get("ONLY", "t1,prompt")
CORES = int(os.environ.get("CORES", "4"))
NBLK, BC, HKV, D = L.NBLK, L.BC, L.HKV, L.D
ROT = L.ROT


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def make_mask(pos):
    col = torch.arange(BC).reshape(1, BC)
    return torch.where(col <= pos % BC, 0.0, -3e38).bfloat16() if pos < BC else \
        torch.where(torch.arange(BC).reshape(1, BC) <= BC - 1, 0.0, -3e38).bfloat16()


def mask_row(pos):
    """[NBLK, BC], as q35_group's decode attention expects."""
    col = torch.arange(L.LMAX).reshape(NBLK, BC)
    return torch.where(col <= pos, 0.0, -3e38).bfloat16()


def build(fn, sub, build_dir):
    os.makedirs(f"{build_dir}/{sub}", exist_ok=True)
    os.chdir(f"{build_dir}/{sub}")
    ax.init(chip_num=1, core_num=CORES)
    return ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN",
                      dump_path=f"{build_dir}/{sub}")(fn)


def check_t1(W, bundle, deq, build_dir):
    torch.manual_seed(7)
    pos = int(os.environ.get("POS", "0"))
    es = torch.randn((pos + 1, H), dtype=torch.float32).bfloat16()
    hs = torch.randn((pos + 1, H), dtype=torch.float32).bfloat16()
    kv_ref = torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16)
    ref = []
    for p in range(pos + 1):
        hn, y = mtp_step(es[p], hs[p], p, kv_ref, deq)
        ref.append((hn, y))
    kv_kernel = kv_ref.clone()
    c, s = q.rope_cos_sin(pos)
    case = dict(bundle.packs, e=es[pos].reshape(1, 1, H), h=hs[pos].reshape(1, 1, H),
                cs=torch.cat([c, s]), pn=torch.tensor([pos, pos // BC + 1, (pos // BC) * 2 * BC + pos % BC],
                                                      dtype=torch.int32),
                mask=mask_row(pos), kv=kv_kernel,
                hnext=torch.zeros((1, 1, H), dtype=torch.bfloat16),
                hraw=torch.zeros((1, 1, H), dtype=torch.bfloat16))
    if pos:
        rp.set_(kv=kv_ref.clone())
        q.RESIDENT["kv"] = kv_ref.clone()
    k = build(mtp_t1, "t1", build_dir)
    order = list(mtp_t1.__code__.co_varnames[: mtp_t1.__code__.co_argcount])
    k[CORES](*[case[n] for n in order])
    rp.clear()
    q.RESIDENT.clear()
    results = {}
    results["hnext"] = cos(case["hnext"][0, 0], ref[-1][0])
    results["hraw"] = cos(case["hraw"][0, 0], ref[-1][1])
    page, cell = divmod(pos, BC)
    results["kv_k"] = cos(kv_kernel[page, 0, cell], kv_ref[page, 0, cell])
    results["kv_v"] = cos(kv_kernel[page, 1, cell], kv_ref[page, 1, cell])
    logger.info("MTP T=1 " + " ".join(f"{k}={v:.7f}" for k, v in results.items()))
    return results


def check_prompt(W, bundle, deq, build_dir):
    torch.manual_seed(11)
    es = torch.randn((P, H), dtype=torch.float32).bfloat16()
    hs = torch.randn((P, H), dtype=torch.float32).bfloat16()
    kv_ref = torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16)
    hn_ref = torch.zeros((P, H), dtype=torch.float32)
    y_ref = torch.zeros((P, H), dtype=torch.float32)
    for p in range(P):
        hn_ref[p], y_ref[p] = mtp_step(es[p], hs[p], p, kv_ref, deq)
    cs = torch.zeros((P, 2, ROT // 2), dtype=torch.bfloat16)
    pn = torch.zeros((P, 3), dtype=torch.int32)
    mask = torch.zeros((NBLK, P, BC), dtype=torch.bfloat16)
    for p in range(P):
        c, s = q.rope_cos_sin(p)
        cs[p, 0], cs[p, 1] = c, s
        pn[p] = torch.tensor([p, p // BC + 1, (p // BC) * 2 * BC + p % BC], dtype=torch.int32)
        mask[:, p] = mask_row(p)
    case = dict(bundle.packs, e=es.reshape(1, P, H), h=hs.reshape(1, P, H), cs=cs, pn=pn, mask=mask,
                nblk=torch.tensor([(P - 1) // BC + 1], dtype=torch.int32),
                kv=torch.zeros_like(kv_ref), hnext=torch.zeros((1, P, H), dtype=torch.bfloat16),
                hraw=torch.zeros((1, P, H), dtype=torch.bfloat16))
    k = build(mtp_prompt, "prompt", build_dir)
    order = list(mtp_prompt.__code__.co_varnames[: mtp_prompt.__code__.co_argcount])
    k[CORES](*[case[n] for n in order])
    hc = torch.tensor([cos(case["hnext"][0, p], hn_ref[p]) for p in range(P)])
    yc = torch.tensor([cos(case["hraw"][0, p], y_ref[p]) for p in range(P)])
    kvc = torch.tensor([cos(case["kv"][p // BC, 0, p % BC], kv_ref[p // BC, 0, p % BC]) for p in range(P)])
    results = {"hnext_mean": float(hc.mean()), "hnext_min": float(hc.min()),
               "hraw_mean": float(yc.mean()), "hraw_min": float(yc.min()),
               "kv_mean": float(kvc.mean()), "kv_min": float(kvc.min())}
    logger.info("MTP prompt256 " + " ".join(f"{k}={v:.7f}" for k, v in results.items()))
    return results


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "out_mtp_check")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    W = MtpWeights(MODEL, random=QW_RANDOM)
    bundle, args, deq = make_bundle(W)
    res = {}
    if "t1" in ONLY:
        res["t1"] = check_t1(W, bundle, deq, out)
    if "prompt" in ONLY:
        res["prompt"] = check_prompt(W, bundle, deq, out)
    torch.save(res, Path(out) / "check_mtp.pt")


if __name__ == "__main__":
    main()

