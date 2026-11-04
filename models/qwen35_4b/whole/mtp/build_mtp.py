#!/usr/bin/env python3
"""Build the Qwen3.5-4B MTP T=1 and 256-row prompt AX kernels (real shapes, MXFP4 packs).

    source /data/junda/work/axdsl_env.sh
    MODEL=/data/junda/hf/Qwen3.5-4B QW_RANDOM=1 RUN_MODE=PYRUN python build_mtp.py out_mtp
    MODEL=/data/junda/hf/Qwen3.5-4B QW_RANDOM=1 RUN_MODE=CRUN  python build_mtp.py out_mtp

The build stops after each compiled axmodel appears. Random weights are the
default; QW_RANDOM=0 loads the real mtp.* tensors.
"""
from __future__ import annotations

import glob
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KVL", "cell")
os.environ.setdefault("BC", "256")
os.environ.setdefault("CORES", "4")
os.environ.setdefault("LMAX", "16384")
os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("WFMT4", "1")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("KVX", "1")
os.environ.setdefault("DEC_SPLIT", "1")

os.environ.setdefault("MODEL", "/data/junda/hf/Qwen3.5-4B")
import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
import q35_group as Q
import q35_layer_mc as L
import q35_step as q
from group_common import dump_json
from mtp_kernel import P, T1_INPUTS, T1_OUTPUTS, PROMPT_INPUTS, PROMPT_OUTPUTS, make_bundle, mtp_prompt, mtp_t1
from mtp_ref import H, MTP_ORDER, MtpWeights, ROT

MODEL = os.environ.get("MODEL", "/data/junda/hf/Qwen3.5-4B")
RUN_MODE = os.environ.get("RUN_MODE", "PYRUN")
QW_RANDOM = os.environ.get("QW_RANDOM", "1") == "1"
CORES = int(os.environ.get("CORES", "4"))
LMAX, BC, HKV, D = L.LMAX, L.BC, L.HKV, L.D
NBLK = LMAX // BC


def dtype_name(dt):
    return {
        torch.bfloat16: "bfloat16", torch.float32: "float32", torch.int32: "int32",
        torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e8m0fnu: "float8_e8m0fnu",
        torch.float4_e2m1fn_x2: "float4_e2m1fn_x2",
    }.get(dt, str(dt).removeprefix("torch."))


def desc(name, t: torch.Tensor):
    return {"name": name, "shape": list(t.shape), "dtype": dtype_name(t.dtype)}


def make_state_t1():
    e = torch.randn((1, 1, H), dtype=torch.float32).bfloat16()
    h = torch.randn((1, 1, H), dtype=torch.float32).bfloat16()
    pos = 3
    cs = torch.stack([q.rope_cos_sin(pos)[0], q.rope_cos_sin(pos)[1]]).reshape(2, ROT // 2)
    pn = torch.tensor([pos, pos // BC + 1, (pos // BC) * 2 * BC + pos % BC], dtype=torch.int32)
    mask = L.mask_rows(pos)
    kv = torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16)
    hnext = torch.zeros((1, 1, H), dtype=torch.bfloat16)
    hraw = torch.zeros((1, 1, H), dtype=torch.bfloat16)
    return dict(e=e, h=h, cs=cs, pn=pn, mask=mask, kv=kv, hnext=hnext, hraw=hraw)


def make_state_prompt():
    st = make_state_t1()
    e = torch.randn((1, P, H), dtype=torch.float32).bfloat16()
    h = torch.randn((1, P, H), dtype=torch.float32).bfloat16()
    cs = torch.zeros((P, 2, ROT // 2), dtype=torch.bfloat16)
    pn = torch.zeros((P, 3), dtype=torch.int32)
    mask = torch.zeros((NBLK, P, BC), dtype=torch.bfloat16)
    for p in range(P):
        c, s = q.rope_cos_sin(p)
        cs[p, 0], cs[p, 1] = c, s
        pn[p] = torch.tensor([p, p // BC + 1, (p // BC) * 2 * BC + p % BC], dtype=torch.int32)
        mask[:, p] = L.mask_rows(p).reshape(NBLK, BC)
    return dict(e=e, h=h, cs=cs, pn=pn, mask=mask, nblk=torch.tensor([1], dtype=torch.int32),
                kv=st["kv"].clone(), hnext=torch.zeros((1, P, H), dtype=torch.bfloat16),
                hraw=torch.zeros((1, P, H), dtype=torch.bfloat16))


def run_kernel(fn, sub, case, packs, inputs, outputs, out_dir, name, layout_lines, pdt):
    build = os.path.abspath(out_dir)
    os.makedirs(f"{build}/{sub}", exist_ok=True)
    os.chdir(f"{build}/{sub}")
    ax.init(chip_num=1, core_num=CORES)
    logger.info(f"building {name} ({RUN_MODE})")
    k = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN", dump_path=f"{build}/{sub}")(fn)
    args = dict(case, **packs)
    order = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    k[CORES](*[args[n] for n in order])
    if RUN_MODE == "CRUN":
        paths = glob.glob(f"{build}/{sub}/cache/{fn.__name__}.*.compiled.axmodel")
        if not paths:
            raise RuntimeError(f"no compiled axmodel under {build}/{sub}/cache")
        dst = Path(out_dir) / f"{name}.axmodel"
        shutil.copy(max(paths, key=os.path.getmtime), dst)
        logger.info(f"wrote {dst} ({dst.stat().st_size} bytes)")
    io = {
        "kernel": name,
        "run_mode": RUN_MODE,
        "inputs": [desc(n, args[n]) for n in inputs],
        "outputs": [desc(n, args[n]) for n in outputs],
        "weight_packs": [desc(n, packs[n]) for n in packs],
    }
    dump_json(Path(out_dir) / f"{name}.inputs.json", io)
    with open(Path(out_dir) / f"{name}.layout", "w") as f:
        for p in pdt:
            f.write(f"{p} - 0 {packs[p].numel() * packs[p].element_size()}\n")
        f.write("\n".join(layout_lines) + "\n")


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "out_mtp")
    os.makedirs(out_dir, exist_ok=True)
    W = MtpWeights(MODEL, random=QW_RANDOM)
    bundle, args, deq = make_bundle(W)
    packs = bundle.packs
    logger.info(f"MTP packs: " + ", ".join(f"{p} {t.numel() * t.element_size()}B" for p, t in packs.items()))
    only = os.environ.get("ONLY", "t1,prompt").split(",")
    if "t1" in only:
        run_kernel(mtp_t1, "t1", make_state_t1(), packs, T1_INPUTS, T1_OUTPUTS, out_dir,
                   "qwen35_4b_mtp_t1_mx4_kvc_c16384_npu4", bundle.lines, bundle.pdt)
    if "prompt" in only:
        run_kernel(mtp_prompt, "prompt", make_state_prompt(), packs, PROMPT_INPUTS, PROMPT_OUTPUTS, out_dir,
                   "qwen35_4b_mtp_prefill256_mx4_kvc_c16384_npu4", bundle.lines, bundle.pdt)


if __name__ == "__main__":
    main()

