#!/usr/bin/env python3
"""Check the M=4 verifier against the saved real-weight T=1-group reference.

    source /data/junda/work/axdsl_env.sh
    REF1=out_verify_check_g16/ref1.pt python check_verify_m4.py out_verify_m4_check

Rows 0..T-1 are the valid prefixes for ntok=1..4; later rows cannot affect
earlier state updates or KV rows.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/junda/work/q27m/src_ks")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for k, v in dict(G="16", FIRST="0", KVL="cell", BC="256", CORES="4", LMAX="16384", WFMT="mx8",
                 WFMT4="1", BLOCK="1", KVX="1", DEC_SPLIT="1", MODEL="/data/junda/hf/Qwen3.8-27B").items():
    os.environ.setdefault(k, v)

import torch
from loguru import logger

import axnn.dsl as ax
import q35_group as G
import q35_layer_mc as L
import q35_step as q
import resident_pyrun as rp
from group_common import make_group_bundle
from verify_group_m4 import T, make_source, snapshot_shapes

RUN_MODE = os.environ.get("RUN_MODE", "PYRUN")
CORES = int(os.environ.get("CORES", "4"))
POS0 = int(os.environ.get("POS0", "254"))
REF1 = Path(os.environ.get("REF1", "out_verify_check_g16/ref1.pt")).resolve()
NBLK, BC, HKV_D = L.LMAX // L.BC, L.BC, L.HKV * L.D


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


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
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "out_verify_m4_check")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    W = q.Weights(os.environ["MODEL"])
    b = make_group_bundle(W, G.KINDS, 0)
    G.PDT.clear()
    G.PDT.update(b.pdt)
    sys.modules["q35_group"] = G

    src4, _ = make_source(b.pdt)
    mod4 = build_module(out, "t4", src4, "group_t4")
    order4 = list(mod4.verify_kernel.__code__.co_varnames[: mod4.verify_kernel.__code__.co_argcount])
    os.makedirs(f"{out}/t4run", exist_ok=True)
    os.chdir(f"{out}/t4run")
    ax.init(chip_num=1, core_num=CORES)
    k4 = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=False, dump_path=f"{out}/t4run")(
        mod4.verify_kernel)

    torch.manual_seed(int(os.environ.get("SEED", "3")))
    xs = (torch.randn((1, T, L.H), dtype=torch.float32) * 0.02).bfloat16()
    cs, pn, mask = [], [], []
    for t in range(T):
        c, s = q.rope_cos_sin(POS0 + t)
        cs.append(torch.cat([c, s]).reshape(2, L.ROT // 2))
        cell = POS0 + t
        pn.append(torch.tensor([cell, cell // BC + 1, (cell // BC) * 2 * BC + cell % BC], dtype=torch.int32))
        mask.append(L.mask_rows(cell).reshape(NBLK, BC))

    ref = torch.load(REF1, map_location="cpu", weights_only=False)
    outs1, snaps1, final1 = ref["outs1"], ref["snaps1"], ref["final1"]

    ios4 = zeros_ios()
    sh = snapshot_shapes()
    case4 = dict(b.packs, x=xs, cs=torch.stack(cs), pn=torch.stack(pn), mask=torch.stack(mask),
                 out=torch.zeros((1, T, L.H), dtype=torch.bfloat16), ntok=torch.tensor([T], dtype=torch.int32),
                 ssnap=torch.zeros(sh[0]), cqsnap=torch.zeros(sh[1], dtype=torch.bfloat16),
                 cksnap=torch.zeros(sh[2], dtype=torch.bfloat16), cvsnap=torch.zeros(sh[3], dtype=torch.bfloat16))
    case4.update(names_ios(ios4))
    seed = {k: v.clone() for k, v in names_ios(ios4).items()}
    rp.set_(**seed)
    q.RESIDENT.clear()
    q.RESIDENT.update(seed)
    logger.info(f"M=4 verifier ({RUN_MODE})")
    k4[CORES](*[case4[n] for n in order4])
    rp.clear()
    q.RESIDENT.clear()

    res = {"out": [cos(case4["out"][0, t], outs1[t]) for t in range(T)], "kv": [], "state": [], "conv": [],
           "snap_state": [], "snap_conv": []}
    gi = 0
    for j, kind in enumerate(G.KINDS):
        if kind == "gdn":
            res["state"].append(cos(case4[f"st{j}"], final1[f"st{j}"]))
            for nm in ("cq", "ck", "cv"):
                res["conv"].append(cos(case4[f"{nm}{j}"], final1[f"{nm}{j}"]))
            res["snap_state"].append([cos(case4["ssnap"][t, gi], snaps1[t][f"st{j}"]) for t in range(T)])
            cq = [cos(case4["cqsnap"][t, gi], snaps1[t][f"cq{j}"]) for t in range(T)]
            ck = [cos(case4["cksnap"][t, gi], snaps1[t][f"ck{j}"]) for t in range(T)]
            cv_ = [cos(case4["cvsnap"][t, gi], snaps1[t][f"cv{j}"]) for t in range(T)]
            res["snap_conv"].append(list(zip(cq, ck, cv_)))
            gi += 1
        else:
            kv = case4[f"kv{j}"]
            ok = []
            for t in range(T):
                cell = POS0 + t
                ok.append(cos(kv[cell // BC, 0, cell % BC], final1[f"kv{j}"][cell // BC, 0, cell % BC]))
                ok.append(cos(kv[cell // BC, 1, cell % BC], final1[f"kv{j}"][cell // BC, 1, cell % BC]))
            res["kv"].append(min(ok))

    flat = [x for xs_ in (res["out"], res["state"]) for x in xs_]
    flat += res["conv"]
    flat += [x for row in res["snap_state"] for x in row]
    flat += [x for rows in res["snap_conv"] for row in rows for x in row]
    flat += res["kv"]
    res["min"] = min(flat)
    logger.info("VERIFY " + " ".join(f"{k}={v}" for k, v in res.items() if k not in ("snap_state", "snap_conv")))
    logger.info(f"snap_state min {min(x for row in res['snap_state'] for x in row):.7f} "
                f"snap_conv min {min(x for rows in res['snap_conv'] for row in rows for x in row):.7f}")
    torch.save(res, Path(out) / "check_verify_m4.pt")


if __name__ == "__main__":
    main()
