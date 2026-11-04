#!/usr/bin/env python3
"""Regenerate the Qwen3.5-4B verify4 T=1-group reference with REAL weights (PYRUN).

Produces board/verify/ref_real.pt with the same structure as
out_verify_check_g16/ref1.pt: outs1[4], snaps1[4], final1.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k, v in dict(G="16", FIRST="0", BANKS="1", KSTRIDE="1", KVL="cell", BC="256", CORES="4", LMAX="16384",
                 WFMT="mx8", WFMT4="1", BLOCK="1", KVX="1", DEC_SPLIT="1",
                 MODEL="/data/junda/hf/Qwen3.5-4B").items():
    os.environ.setdefault(k, v)
os.environ["QW_RANDOM"] = "0"

import torch
from loguru import logger

import axnn.dsl as ax
import q35_group as G
import q35_layer_mc as L
import q35_step as q
import resident_pyrun as rp
from group_common import compare_layout, make_group_bundle
T = 4

POS0 = int(os.environ.get("POS0", "254"))
FIRST = int(os.environ.get("FIRST", "0"))
SEED = int(os.environ.get("SEED", "3"))
OUT = Path(os.environ.get("OUT", "board/verify/ref_real.pt")).resolve()
MODEL = os.environ["MODEL"]
CORES = int(os.environ.get("CORES", "4"))
NBLK, BC, HKV_D = L.LMAX // L.BC, L.BC, L.HKV * L.D
REF_LAYOUT = os.environ.get("REF_LAYOUT", "")


def zeros_ios():
    ios = []
    for kind in G.KINDS:
        if kind == "gdn":
            ios.append((torch.zeros((L.NV, L.DK, L.DV)),
                        torch.zeros((1, L.CONV - 1, L.KD), dtype=torch.bfloat16),
                        torch.zeros((1, L.CONV - 1, L.KD), dtype=torch.bfloat16),
                        torch.zeros((1, L.CONV - 1, L.VD), dtype=torch.bfloat16)))
        else:
            ios.append((torch.zeros((NBLK, 2, BC, HKV_D), dtype=torch.bfloat16),))
    return ios


def names_ios(ios):
    out = {}
    for j, kind in enumerate(G.KINDS):
        ns = [f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"] if kind == "gdn" else [f"kv{j}"]
        out.update(dict(zip(ns, ios[j])))
    return out


def build_module(build, sub, src, name):
    d = f"{build}/{sub}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{name}.py", "w") as f:
        f.write(src)
    sys.path.insert(0, d)
    importlib.invalidate_caches()
    return importlib.import_module(name)


def main():
    os.makedirs(OUT.parent, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    t0 = time.time()
    q.VORDER = "tiled"
    W = q.Weights(MODEL)
    b = make_group_bundle(W, G.KINDS, FIRST)
    if REF_LAYOUT:
        same, diff = compare_layout(REF_LAYOUT, b.lines)
        logger.info(f"real bundle built in {time.time()-t0:.1f}s; layout identical={same} diff={diff}")
        if not same:
            raise SystemExit(f"reference bundle layout differs from the decode group at {diff}")
    G.PDT.clear()
    G.PDT.update(b.pdt)
    sys.modules["q35_group"] = G
    build = os.path.abspath(os.environ.get("BUILD", "board/verify_ref_build"))
    src1, _ = G.kernel_source()
    mod1 = build_module(build, "t1", src1, "group_t1")
    d = f"{build}/t1run"
    os.makedirs(d, exist_ok=True)
    os.chdir(d)
    ax.init(chip_num=1, core_num=CORES)
    k1 = ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False, dump_path=d)(mod1.group_kernel)
    order1 = list(mod1.group_kernel.__code__.co_varnames[: mod1.group_kernel.__code__.co_argcount])

    torch.manual_seed(SEED)
    xs = (torch.randn((1, T, L.H), dtype=torch.float32) * 0.02).bfloat16()
    cs, pn, mask = [], [], []
    for t in range(T):
        cell = POS0 + t
        c, s = q.rope_cos_sin(cell)
        cs.append(torch.cat([c, s]).reshape(2, L.ROT // 2))
        pn.append(torch.tensor([cell, cell // BC + 1, (cell // BC) * 2 * BC + cell % BC], dtype=torch.int32))
        mask.append(L.mask_rows(cell).reshape(NBLK, BC))

    cur = zeros_ios()
    outs1, snaps1 = [], []
    for t in range(T):
        case = dict(b.packs, **b.banks, x=xs[:, t:t + 1], cs=cs[t], pn=pn[t], mask=mask[t],
                    out=torch.zeros((1, 1, L.H), dtype=torch.bfloat16))
        case.update(names_ios(cur))
        seed = {k: v.clone() for k, v in names_ios(cur).items()}
        rp.set_(**seed)
        q.RESIDENT.clear()
        q.RESIDENT.update(seed)
        logger.info(f"T=1 real reference row {t} (pos {POS0 + t})")
        k1[CORES](*[case[n] for n in order1])
        rp.clear()
        q.RESIDENT.clear()
        outs1.append(case["out"][0, 0].clone())
        cur = [tuple(case[n] for n in ([f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"] if kind == "gdn" else [f"kv{j}"]))
               for j, kind in enumerate(G.KINDS)]
        snaps1.append({k: v.clone() for k, v in names_ios(cur).items()})
    final1 = {k: v.clone() for k, v in names_ios(cur).items()}
    torch.save({"outs1": outs1, "snaps1": snaps1, "final1": final1}, OUT)
    logger.info(f"saved {OUT} ({OUT.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
