#!/usr/bin/env python3
"""Dump real-weight verify4 (group 0, layers 0..15) inputs and references.

Reference = board/verify/ref_real.pt (real-weight PYRUN T=1 group run four
times, POS0=254, SEED=3, zero initial state/KV).  This script regenerates the
identical packs/x/cs/pn/mask and stores them as raw board inputs.

CRUN=0/1  also execute the cached compiled verify4 kernel in CRUN (T=4 only).
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/junda/work/q27m/src_ks")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for k, v in dict(G="16", FIRST="0", KVL="cell", BC="256", CORES="4", LMAX="16384",
                 WFMT="mx8", WFMT4="1", BLOCK="1", KVX="1", DEC_SPLIT="1",
                 MODEL="/data/junda/hf/Qwen3.8-27B").items():
    os.environ.setdefault(k, v)

import torch
from loguru import logger

import axnn.dsl as ax
import q35_group as G
import q35_layer_mc as L
import q35_step as q
from board_common import dump_json, write_bin
from group_common import make_group_bundle
from verify_group import T, make_source, snapshot_shapes

POS0 = int(os.environ.get("POS0", "254"))
SEED = int(os.environ.get("SEED", "3"))
ROOT = Path(os.environ.get("ROOT", "board/verify"))
MODEL = os.environ["MODEL"]
CORES = int(os.environ.get("CORES", "4"))
RUN_CRUN = os.environ.get("CRUN", "0") == "1"
REF1 = Path(os.environ.get("REF1", "board/verify/ref_real.pt"))
VERIFY_KEY = "67fdae53026f478023798cbcd622e2ba29f6bb6c2364a7a9387aa4a584d2fa06"
NBLK, BC, HKV, D = L.LMAX // L.BC, L.BC, L.HKV, L.D
HKV_D = HKV * D
CORES_N = int(os.environ.get("CORES", "4"))


def make_inputs(bundle, xs, cs, pn, mask):
    """Seed KV only where it cannot affect the valid prefix or be overwritten."""
    kv = {}
    gen = torch.Generator().manual_seed(12345)
    for j, kind in enumerate(G.KINDS):
        if kind != "attn":
            continue
        t = torch.zeros((NBLK, 2, BC, HKV_D), dtype=torch.bfloat16)
        # pages 2..63 are never read at POS0..POS0+3
        t[2:] = torch.randn((NBLK - 2, 2, BC, HKV_D), generator=gen).bfloat16()
        # page 1 cells 2..255 are masked for the positions used and not written
        t[1, :, 2:] = torch.randn((2, BC - 2, HKV_D), generator=gen).bfloat16()
        kv[f"kv{j}"] = t
    ios = {}
    gi = 0
    for j, kind in enumerate(G.KINDS):
        if kind == "gdn":
            ios[f"st{j}"] = torch.zeros((L.NV, L.DK, L.DV))
            ios[f"cq{j}"] = torch.zeros((1, L.CONV - 1, L.KD), dtype=torch.bfloat16)
            ios[f"ck{j}"] = torch.zeros((1, L.CONV - 1, L.KD), dtype=torch.bfloat16)
            ios[f"cv{j}"] = torch.zeros((1, L.CONV - 1, L.VD), dtype=torch.bfloat16)
            gi += 1
    return kv, ios


def main():
    os.makedirs(ROOT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    t0 = time.time()
    q.VORDER = "tiled"
    W = q.Weights(MODEL)
    b = make_group_bundle(W, G.KINDS, 0)
    G.PDT.clear()
    G.PDT.update(b.pdt)
    sys.modules["q35_group"] = G
    logger.info(f"real group bundle built in {time.time()-t0:.1f}s; packs {list(b.pdt)}")

    # Packs are T-independent: write them once.
    pdir = ROOT / "packs"
    psizes = {}
    for n in b.pdt:
        psizes[n] = write_bin(pdir / f"{n}.bin", b.packs[n])
    logger.info(f"packs written {sum(psizes.values())/1e9:.2f} GB in {time.time()-t0:.1f}s")

    ref1 = torch.load(REF1, map_location="cpu", weights_only=False)
    logger.info(f"loaded {REF1}: outs {len(ref1['outs1'])}, snaps {len(ref1['snaps1'])}")

    torch.manual_seed(SEED)
    xs_all = (torch.randn((1, T, L.H), dtype=torch.float32) * 0.02).bfloat16()
    cs_all, pn_all, mask_all = [], [], []
    for t in range(T):
        cell = POS0 + t
        c, s = q.rope_cos_sin(cell)
        cs_all.append(torch.cat([c, s]).reshape(2, L.ROT // 2))
        pn_all.append(torch.tensor([cell, cell // BC + 1, (cell // BC) * 2 * BC + cell % BC], dtype=torch.int32))
        mask_all.append(L.mask_rows(cell).reshape(NBLK, BC))

    kv, ios = make_inputs(b, xs_all, cs_all, pn_all, mask_all)
    # One shared board input directory; T-specific x/cs/pn/mask are swapped by the driver.
    indir = ROOT / "in"
    kvnames = []
    for j, kind in enumerate(G.KINDS):
        if kind == "attn":
            kvnames.append(f"kv{j}")
        else:
            kvnames += [f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"]
    in_sizes = {}
    for n in kvnames:
        t = kv[n] if n.startswith("kv") else ios[n]
        in_sizes[n] = write_bin(indir / f"{n}.bin", t)

    for tt in range(1, T + 1):
        d = ROOT / f"T{tt}"
        os.makedirs(d / "in", exist_ok=True)
        x = xs_all.clone()
        x[:, tt:] = 0
        cs = torch.stack(cs_all)
        pn = torch.stack(pn_all)
        mask = torch.stack(mask_all)
        # Rows >= tt have zero x but advance to distinct positions.  They
        # therefore cannot alter the valid prefix's KV rows or snapshots;
        # pointing them back at the last valid cell would overwrite it.
        sizes = {n: write_bin(d / "in" / f"{n}.bin", t) for n, t in
                 (("x", x), ("cs", cs), ("pn", pn), ("mask", mask), ("ntok", torch.tensor([tt], dtype=torch.int32)))}
        ref = {"out": torch.stack(ref1["outs1"][:tt]),
               "state": {f"st{j}": torch.stack([ref1["snaps1"][t][f"st{j}"] for t in range(tt)])
                         for j, kind in enumerate(G.KINDS) if kind == "gdn"},
               "cq": {f"cq{j}": torch.stack([ref1["snaps1"][t][f"cq{j}"] for t in range(tt)])
                      for j, kind in enumerate(G.KINDS) if kind == "gdn"},
               "ck": {f"ck{j}": torch.stack([ref1["snaps1"][t][f"ck{j}"] for t in range(tt)])
                      for j, kind in enumerate(G.KINDS) if kind == "gdn"},
               "cv": {f"cv{j}": torch.stack([ref1["snaps1"][t][f"cv{j}"] for t in range(tt)])
                      for j, kind in enumerate(G.KINDS) if kind == "gdn"},
               "kv_rows": {}, "ntok": tt}
        for j, kind in enumerate(G.KINDS):
            if kind == "attn":
                rows = []
                for t in range(tt):
                    cell = POS0 + t
                    rows.append(ref1["snaps1"][t][f"kv{j}"][cell // BC, :, cell % BC].clone())
                ref["kv_rows"][f"kv{j}"] = torch.stack(rows)
        if tt == T:
            ref["final"] = {n: ref1["final1"][n].clone() for n in kvnames}
        torch.save(ref, d / "ref.pt")
        dump_json(d / "io.json", {"case": f"verify_T{tt}", "pos0": POS0, "ntok": tt,
                                  "bytes": sizes, "written_rows": [POS0 + t for t in range(tt)],
                                  "packs": list(b.pdt)})
        logger.info(f"T{tt}: wrote {d}")

    dump_json(ROOT / "io.json", {"packs": list(b.pdt), "pack_bytes": psizes,
                                 "kv_state_bytes": in_sizes, "pos0": POS0, "seed": SEED,
                                 "sentinel_kv_pages": [2, NBLK - 1]})

    if not RUN_CRUN:
        return

    # T=4 forced-cache CRUN reference.
    src4, _ = make_source(b.pdt)
    d4 = ROOT / "T4"
    moddir = Path(os.path.abspath("out_verify_crun/t4"))
    sys.path.insert(0, str(moddir))
    importlib.invalidate_caches()
    mod4 = importlib.import_module("group_t4")
    order = list(mod4.verify_kernel.__code__.co_varnames[: mod4.verify_kernel.__code__.co_argcount])
    x = xs_all
    cs = torch.stack(cs_all)
    pn = torch.stack(pn_all)
    mask = torch.stack(mask_all)
    sh = snapshot_shapes()
    case = dict(b.packs, x=x, cs=cs, pn=pn, mask=mask,
                out=torch.zeros((1, T, L.H), dtype=torch.bfloat16),
                ntok=torch.tensor([T], dtype=torch.int32),
                ssnap=torch.zeros(sh[0]), cqsnap=torch.zeros(sh[1], dtype=torch.bfloat16),
                cksnap=torch.zeros(sh[2], dtype=torch.bfloat16),
                cvsnap=torch.zeros(sh[3], dtype=torch.bfloat16))
    case.update(kv)
    case.update(ios)
    os.chdir(str(moddir))
    ax.init(chip_num=1, core_num=CORES)
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=str(moddir))(mod4.verify_kernel)
    k.gen_cache_key = lambda: VERIFY_KEY
    logger.info("CRUN verify T=4 forced-cache run")
    t1 = time.time()
    k[CORES](*[case[n] for n in order])
    logger.info(f"CRUN verify wall {time.time()-t1:.1f}s")
    got = {n: case[n].clone() for n in order if n in (
        ["out", "ssnap", "cqsnap", "cksnap", "cvsnap"] + kvnames)}
    torch.save(got, d4 / "crun.pt")
    logger.info(f"CRUN verify outputs -> {d4 / 'crun.pt'}")


if __name__ == "__main__":
    main()
