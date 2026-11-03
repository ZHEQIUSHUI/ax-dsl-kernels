#!/usr/bin/env python3
"""Build the Qwen3.8-27B MTP M=8 kernel with the real 27B pack layout."""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/data/junda/work/q27m/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k, v in dict(KVL="cell", BC="256", CORES="4", LMAX="16384", WFMT="mx8", WFMT4="1",
                 BLOCK="1", KVX="1", DEC_SPLIT="1",
                 MODEL="/data/junda/hf/Qwen3.8-27B").items():
    os.environ.setdefault(k, v)

import torch
from loguru import logger

import axnn.dsl as ax
import q35_layer_mc as L
import q35_step as q
from group_common import dump_json
from mtp_kernel import make_bundle
if os.environ.get("MTP_KERNEL", "m8") == "if":
    from mtp_m8_if_kernel import M8_IF_INPUTS as M8_INPUTS, M8_IF_OUTPUTS as M8_OUTPUTS, T, mtp_m8_if as mtp_m8
else:
    from mtp_m8_kernel import M8_INPUTS, M8_OUTPUTS, T, mtp_m8
from mtp_ref import H, MtpWeights, ROT

MODEL = os.environ.get("MODEL", "/data/junda/hf/Qwen3.8-27B")
RUN_MODE = os.environ.get("RUN_MODE", "PYRUN")
QW_RANDOM = os.environ.get("QW_RANDOM", "1") == "1"
CORES = int(os.environ.get("CORES", "4"))
NBLK, BC, HKV, D = L.NBLK, L.BC, L.HKV, L.D


def dtype_name(dt):
    return {
        torch.bfloat16: "bfloat16", torch.float32: "float32", torch.int32: "int32",
        torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e8m0fnu: "float8_e8m0fnu",
        torch.float4_e2m1fn_x2: "float4_e2m1fn_x2",
    }.get(dt, str(dt).removeprefix("torch."))


def desc(name, t):
    return {"name": name, "shape": list(t.shape), "dtype": dtype_name(t.dtype)}


def make_state(c0=0):
    torch.manual_seed(23)
    e = torch.randn((1, T, H), dtype=torch.float32).bfloat16()
    h = torch.randn((1, T, H), dtype=torch.float32).bfloat16()
    cs = torch.zeros((T, 2, ROT // 2), dtype=torch.bfloat16)
    pn = torch.zeros((T, 3), dtype=torch.int32)
    mask = torch.zeros((T, NBLK, BC), dtype=torch.bfloat16)
    for t in range(T):
        p = c0 + t
        c, s = q.rope_cos_sin(p)
        cs[t, 0], cs[t, 1] = c, s
        pn[t] = torch.tensor([p, p // BC + 1, (p // BC) * 2 * BC + p % BC], dtype=torch.int32)
        mask[t] = L.mask_rows(p).reshape(NBLK, BC)
    return dict(e=e, h=h, cs=cs, pn=pn, mask=mask, ntok=torch.tensor([T], dtype=torch.int32),
                kv=torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16),
                hnext=torch.zeros((1, T, H), dtype=torch.bfloat16),
                hraw=torch.zeros((1, T, H), dtype=torch.bfloat16))


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "out_mtp_m8")
    os.makedirs(out_dir, exist_ok=True)
    W = MtpWeights(MODEL, random=QW_RANDOM)
    bundle, args, deq = make_bundle(W)
    packs = bundle.packs
    case = make_state(int(os.environ.get("C0", "0")))
    is_if = os.environ.get("MTP_KERNEL", "m8") == "if"
    name = os.environ.get("NAME", "qwen35_mtp_m8_mx4_kvc_c16384_npu4")
    sub = os.environ.get("SUB", "m8if" if is_if else "m8")
    build = os.path.abspath(out_dir)
    os.makedirs(f"{build}/{sub}", exist_ok=True)
    os.chdir(f"{build}/{sub}")
    ax.init(chip_num=1, core_num=CORES)
    logger.info(f"building {name} ({RUN_MODE})")
    k = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN", dump_path=f"{build}/{sub}")(mtp_m8)
    order = list(mtp_m8.__code__.co_varnames[: mtp_m8.__code__.co_argcount])
    kwargs = dict(case, **packs)
    k[CORES](*[kwargs[n] for n in order])
    if RUN_MODE == "CRUN":
        stem = mtp_m8.__name__
        paths = glob.glob(f"{build}/{sub}/cache/{stem}.*.compiled.axmodel")
        if not paths:
            raise RuntimeError("no compiled axmodel")
        dst = Path(out_dir) / f"{name}.axmodel"
        shutil.copy(max(paths, key=os.path.getmtime), dst)
        logger.info(f"wrote {dst} ({dst.stat().st_size} bytes)")
    io = {"kernel": name, "run_mode": RUN_MODE,
          "inputs": [desc(n, kwargs[n]) for n in M8_INPUTS],
          "outputs": [desc(n, kwargs[n]) for n in M8_OUTPUTS],
          "weight_packs": [desc(n, packs[n]) for n in packs]}
    dump_json(Path(out_dir) / f"{name}.inputs.json", io)
    with open(Path(out_dir) / f"{name}.layout", "w") as f:
        for p in bundle.pdt:
            f.write(f"{p} - 0 {packs[p].numel() * packs[p].element_size()}\n")
        f.write("\n".join(bundle.lines) + "\n")


if __name__ == "__main__":
    main()
