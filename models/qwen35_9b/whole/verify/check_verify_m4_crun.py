#!/usr/bin/env python3
"""Run the compiled Qwen3.5-9B M=4 verifier in CRUN for T=1..4 on real weights.

The graph is static T=4.  Shorter cases zero the later x/cs rows and repeat the
last pn/mask row, exactly as the board input preparation does.  Reference rows
come from ``board/verify/ref_real.pt`` (four sequential T=1 group calls).
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for k, v in dict(G="16", FIRST="0", BANKS="1", KVL="cell", BC="256", CORES="4", LMAX="16384",
                 WFMT="mx8", WFMT4="1", BLOCK="1", KVX="1", DEC_SPLIT="1",
                 MODEL="/data/junda/hf/Qwen3.5-9B").items():
    os.environ.setdefault(k, v)

import torch
from loguru import logger

import axnn.dsl as ax
import q35_group as G
import q35_layer_mc as L
import q35_step as q
from board_common import dump_json
from group_common import make_group_bundle
import resident_pyrun as rp
from verify_group_m4 import T, snapshot_shapes, layer_io_names

POS0 = int(os.environ.get("POS0", "254"))
FIRST = int(os.environ.get("FIRST", "0"))
SEED = int(os.environ.get("SEED", "3"))
CORES = int(os.environ.get("CORES", "4"))
RUN_MODE = os.environ.get("RUN_MODE", "CRUN")
REF1 = Path(os.environ.get("REF1", "board/verify/ref_real.pt")).resolve()
BUILD = Path(os.environ.get("BUILD", "out_verify_m4_crun/t4")).resolve()
OUT = Path(os.environ.get("OUT", "out_verify_m4_crun")).resolve()
NT = (1, 2, 3, 4)
NBLK, BC, HKV_D = L.LMAX // L.BC, L.BC, L.HKV * L.D


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def stats(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return {"n": len(vals), "min": float(t.min()), "mean": float(t.mean()), "max": float(t.max())}


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


def cache_key():
    if os.environ.get("CACHE_KEY"):
        return os.environ["CACHE_KEY"]
    paths = glob.glob(str(BUILD / "run/cache/verify_kernel.*.compiled.axmodel"))
    if not paths:
        raise RuntimeError(f"no compiled cache under {BUILD / 'run/cache'}")
    return Path(max(paths, key=os.path.getmtime)).name.split(".")[1]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    key = cache_key() if RUN_MODE == "CRUN" else None
    logger.info(f"compiled cache key {key}" if key else f"run mode {RUN_MODE}")
    W = q.Weights(os.environ["MODEL"])
    b = make_group_bundle(W, G.KINDS, FIRST)
    G.PDT.clear()
    G.PDT.update(b.pdt)
    sys.modules["q35_group"] = G
    logger.info(f"real bundle ready; packs {list(b.pdt)}")

    sys.path.insert(0, str(BUILD))
    importlib.invalidate_caches()
    mod = importlib.import_module("group_t4_m4")
    order = list(mod.verify_kernel.__code__.co_varnames[: mod.verify_kernel.__code__.co_argcount])
    os.makedirs(BUILD / "run", exist_ok=True)
    os.chdir(BUILD / "run")
    ax.init(chip_num=1, core_num=CORES)
    k = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN",
                   dump_path=str(BUILD / "run"))(mod.verify_kernel)
    if key:
        k.gen_cache_key = lambda: key

    ref = torch.load(REF1, map_location="cpu", weights_only=False)
    torch.manual_seed(SEED)
    xs_all = (torch.randn((1, T, L.H), dtype=torch.float32) * 0.02).bfloat16()
    cs_all, pn_all, mask_all = [], [], []
    for t in range(T):
        cell = POS0 + t
        c, s = q.rope_cos_sin(cell)
        cs_all.append(torch.cat([c, s]).reshape(2, L.ROT // 2))
        pn_all.append(torch.tensor([cell, cell // BC + 1, (cell // BC) * 2 * BC + cell % BC], dtype=torch.int32))
        mask_all.append(L.mask_rows(cell).reshape(NBLK, BC))

    sh = snapshot_shapes()
    ios = zeros_ios()
    case = dict(b.packs, **b.banks, x=xs_all, cs=torch.stack(cs_all), pn=torch.stack(pn_all),
                mask=torch.stack(mask_all),
                out=torch.zeros((1, T, L.H), dtype=torch.bfloat16),
                ntok=torch.tensor([T], dtype=torch.int32),
                ssnap=torch.zeros(sh[0]), cqsnap=torch.zeros(sh[1], dtype=torch.bfloat16),
                cksnap=torch.zeros(sh[2], dtype=torch.bfloat16),
                cvsnap=torch.zeros(sh[3], dtype=torch.bfloat16))
    case.update(names_ios(ios))
    if RUN_MODE == "PYRUN":
        seed = names_ios(ios)
        seed.update({n: case[n] for n in ("out", "ssnap", "cqsnap", "cksnap", "cvsnap")})
        rp.set_(**seed)
        q.RESIDENT.clear()
        q.RESIDENT.update(seed)
    logger.info(f"{RUN_MODE} verifier T={T} (also carries the T=1..3 prefixes)")
    t0 = time.time()
    k[CORES](*[case[n] for n in order])
    if RUN_MODE == "PYRUN":
        rp.clear()
        q.RESIDENT.clear()
    logger.info(f"{RUN_MODE} verifier T={T} wall {time.time()-t0:.1f}s")

    dump_dir = os.environ.get("DUMP_BOARD")
    if dump_dir:
        d = Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        for n in order:
            (d / f"{n}.bin").write_bytes(case[n].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        outs_all = ["out"] + [n for j, kind in enumerate(G.KINDS) for n in layer_io_names(kind, j)] + \
                   ["ssnap", "cqsnap", "cksnap", "cvsnap"]
        torch.save({n: case[n].detach().cpu().clone() for n in outs_all}, d / "crun.pt")
        dump_json(d / "meta.json", {"outputs": {n: {"shape": list(case[n].shape),
                                                    "dtype": str(case[n].dtype).removeprefix("torch."),
                                                    "bytes": case[n].numel() * case[n].element_size()}
                                                for n in outs_all}})
        logger.info(f"board inputs and CRUN outputs written to {d}")

    allres = {}
    for nt in NT:
        r = {"out": [], "kv": [], "state_snap": [], "conv_snap": [], "final_state": [], "final_conv": []}
        gi = 0
        for t in range(nt):
            r["out"].append(cos(case["out"][0, t], ref["outs1"][t]))
        for j, kind in enumerate(G.KINDS):
            if kind == "gdn":
                for t in range(nt):
                    r["state_snap"].append(cos(case["ssnap"][t, gi], ref["snaps1"][t][f"st{j}"]))
                    for nm in ("cq", "ck", "cv"):
                        r["conv_snap"].append(cos(case[f"{nm}snap"][t, gi], ref["snaps1"][t][f"{nm}{j}"]))
                if nt == T:
                    r["final_state"].append(cos(case[f"st{j}"], ref["final1"][f"st{j}"]))
                    for nm in ("cq", "ck", "cv"):
                        r["final_conv"].append(cos(case[f"{nm}{j}"], ref["final1"][f"{nm}{j}"]))
                gi += 1
            else:
                for t in range(nt):
                    cell = POS0 + t
                    pg, cl = divmod(cell, BC)
                    def kvrow(pl):  # the kernel's bf16 K / V row
                        return case[f"kv{j}"][pg, pl, cl].float()
                    r["kv"] += [cos(kvrow(0), ref["snaps1"][t][f"kv{j}"][pg, 0, cl]),
                                cos(kvrow(1), ref["snaps1"][t][f"kv{j}"][pg, 1, cl])]
        res = {n: stats(v) for n, v in r.items() if v}
        res["min"] = min(v["min"] for v in res.values())
        res["pass"] = res["min"] >= 0.999
        allres[f"T{nt}"] = res
        logger.info(f"T={nt} " + " ".join(f"{n}={v['min']:.7f}" for n, v in res.items() if isinstance(v, dict)))
        logger.info(f"T={nt} overall min {res['min']:.7f} pass={res['pass']}")

    overall = min(v["min"] for v in allres.values())
    allres["pass"] = all(v["pass"] for v in allres.values())
    allres["overall_min"] = overall
    dump_json(OUT / f"check_verify_m4_{RUN_MODE.lower()}.json", allres)
    logger.info(f"overall min {overall:.7f} pass={allres['pass']}")
    if not allres["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
