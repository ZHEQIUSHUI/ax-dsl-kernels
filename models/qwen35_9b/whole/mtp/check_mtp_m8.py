#!/usr/bin/env python3
"""Numeric check for the M=8 MTP kernel against row-by-row real-weight reference.

The reference is ``mtp_ref.mtp_step`` (the same MTP math and quantized weights
used by the existing mtp_t1 checks), run sequentially on the MTP's resident
paged KV.  With ``CHECK_T1=1`` the actual mtp_t1 kernel is also replayed row by
row and compared against the M=8 result.
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import time
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
import resident_pyrun as rp
from board_common import dump_json
from mtp_kernel import make_bundle, mtp_t1
if os.environ.get("MTP_KERNEL", "m8") == "if":
    from mtp_m8_if_kernel import T, mtp_m8_if as mtp_m8
else:
    from mtp_m8_kernel import T, mtp_m8
from mtp_ref import H, MtpWeights, mtp_step

RUN_MODE = os.environ.get("RUN_MODE", "PYRUN")
CORES = int(os.environ.get("CORES", "4"))
MODEL = os.environ["MODEL"]
NBLK, BC, HKV, D = L.NBLK, L.BC, L.HKV, L.D
CHECK_T1 = os.environ.get("CHECK_T1", "1") == "1"
OUT = Path(os.environ.get("OUT", "out_mtp_m8_check")).resolve()


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def stats(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return {"min": float(t.min()), "mean": float(t.mean()), "max": float(t.max())}


def make_case(es, hs, c0, ntok, kv_in):
    cs = torch.zeros((T, 2, L.ROT // 2), dtype=torch.bfloat16)
    pn = torch.zeros((T, 3), dtype=torch.int32)
    mask = torch.zeros((T, NBLK, BC), dtype=torch.bfloat16)
    for t in range(T):
        pos = c0 + t
        c, s = q.rope_cos_sin(pos)
        cs[t, 0], cs[t, 1] = c, s
        pn[t] = torch.tensor([pos, pos // BC + 1, (pos // BC) * 2 * BC + pos % BC], dtype=torch.int32)
        mask[t] = L.mask_rows(pos).reshape(NBLK, BC)
    return dict(e=es[c0 : c0 + T].reshape(1, T, H).contiguous(),
                h=hs[c0 : c0 + T].reshape(1, T, H).contiguous(),
                cs=cs, pn=pn, mask=mask,
                ntok=torch.tensor([ntok], dtype=torch.int32),
                kv=kv_in.clone(),
                hnext=torch.zeros((1, T, H), dtype=torch.bfloat16),
                hraw=torch.zeros((1, T, H), dtype=torch.bfloat16))


def seed_resident(case):
    rp.set_(kv=case["kv"].clone())
    q.RESIDENT["kv"] = case["kv"].clone()


def run_case(kernel, order, packs, case):
    seed_resident(case)
    kernel[CORES](*[dict(case, **packs)[n] for n in order])
    rp.clear()
    q.RESIDENT.clear()
    return case


def untouch_equal(got, initial, written):
    diff = (got != initial)
    for pos in written:
        pg, cl = divmod(pos, BC)
        diff[pg, :, cl, :] = False
    return bool(not diff.any())


def check_case(name, c0, ntok, es, hs, deq, packs, k8, order8, k1=None, order1=None):
    kv_in = torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16)
    refs = []
    for p in range(c0):
        mtp_step(es[p], hs[p], p, kv_in, deq)
    kv_start = kv_in.clone()
    for t in range(ntok):
        p = c0 + t
        refs.append(mtp_step(es[p], hs[p], p, kv_in, deq))
    kv_hist = kv_start.clone() if CHECK_T1 else None

    case8 = make_case(es, hs, c0, ntok, kv_in)
    t0 = time.time()
    run_case(k8, order8, packs, case8)
    dt8 = time.time() - t0

    r = {"hnext": [], "hraw": [], "kv_k": [], "kv_v": [], "t1_hnext": [], "t1_hraw": []}
    for t in range(ntok):
        r["hnext"].append(cos(case8["hnext"][0, t], refs[t][0]))
        r["hraw"].append(cos(case8["hraw"][0, t], refs[t][1]))
        pos = c0 + t
        pg, cl = divmod(pos, BC)
        r["kv_k"].append(cos(case8["kv"][pg, 0, cl], kv_in[pg, 0, cl]))
        r["kv_v"].append(cos(case8["kv"][pg, 1, cl], kv_in[pg, 1, cl]))

    if CHECK_T1 and k1 is not None:
        kv1 = kv_hist
        case1 = dict(e=es[c0 : c0 + T].reshape(1, T, H), h=hs[c0 : c0 + T].reshape(1, T, H),
                     cs=case8["cs"], pn=case8["pn"], mask=case8["mask"], kv=kv1,
                     hnext=torch.zeros((T, 1, H), dtype=torch.bfloat16),
                     hraw=torch.zeros((T, 1, H), dtype=torch.bfloat16))
        for t in range(ntok):
            c1 = dict(case1, **packs,
                      e=es[c0 + t].reshape(1, 1, H).contiguous(),
                      h=hs[c0 + t].reshape(1, 1, H).contiguous(),
                      cs=case8["cs"][t].contiguous(),
                      pn=case8["pn"][t].contiguous(),
                      mask=case8["mask"][t].contiguous())
            c1["hnext"].zero_()
            c1["hraw"].zero_()
            seed_resident(c1)
            k1[CORES](*[c1[n] for n in order1])
            rp.clear()
            q.RESIDENT.clear()
            r["t1_hnext"].append(cos(case8["hnext"][0, t], c1["hnext"][0, 0]))
            r["t1_hraw"].append(cos(case8["hraw"][0, t], c1["hraw"][0, 0]))

    written = list(range(c0, c0 + ntok))
    untouched = untouch_equal(case8["kv"], kv_start, written)
    out = {"case": name, "c0": c0, "ntok": ntok, "wall_s": dt8, "untouched_kv_exact": untouched}
    for key, vals in r.items():
        if vals:
            out[key] = stats(vals)
    out["min"] = min(v["min"] for k_, v in out.items() if isinstance(v, dict) and "min" in v)
    out["pass"] = out["min"] >= 0.999 and untouched
    logger.info(f"{name} ntok={ntok} c0={c0} min={out['min']:.7f} untouched={untouched} "
                + " ".join(f"{k}={v['min']:.7f}" for k, v in out.items() if isinstance(v, dict)))
    torch.save({"case8": case8, "ref": refs, "kv_in": kv_in, "written": written}, OUT / f"{name}.pt")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    W = MtpWeights(MODEL, random=False)
    bundle, args, deq = make_bundle(W)
    packs = bundle.packs
    torch.manual_seed(int(os.environ.get("SEED", "17")))
    maxpos = int(os.environ.get("MAXPOS", "280"))
    es = torch.randn((maxpos + T, H), dtype=torch.float32).bfloat16()
    hs = torch.randn((maxpos + T, H), dtype=torch.float32).bfloat16()

    case_text = os.environ.get("CASES", "0:1,0:2,0:4,0:8,5:1,5:2,5:4,5:8,253:8")
    cases = [tuple(map(int, x.split(":"))) for x in case_text.split(",") if x]

    build = OUT / "build"
    os.makedirs(build, exist_ok=True)
    os.chdir(build)
    ax.init(chip_num=1, core_num=CORES)
    k8 = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN", dump_path=str(build))(mtp_m8)
    m8_key = os.environ.get("M8_CACHE_KEY")
    if m8_key:
        k8.gen_cache_key = lambda: m8_key
    order8 = list(mtp_m8.__code__.co_varnames[: mtp_m8.__code__.co_argcount])
    k1 = None
    order1 = None
    if CHECK_T1:
        k1 = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN", dump_path=str(build))(mtp_t1)
        t1_key = os.environ.get("T1_CACHE_KEY")
        if t1_key:
            k1.gen_cache_key = lambda: t1_key
        order1 = list(mtp_t1.__code__.co_varnames[: mtp_t1.__code__.co_argcount])

    results = {}
    for c0, ntok in cases:
        name = f"c{c0}_n{ntok}"
        results[name] = check_case(name, c0, ntok, es, hs, deq, packs, k8, order8, k1, order1)
    overall = min(v["min"] for v in results.values())
    ok = all(v["pass"] for v in results.values())
    results["overall_min"] = overall
    results["pass"] = ok
    dump_json(OUT / "check_mtp_m8.json", results)
    logger.info(f"overall min {overall:.7f} pass={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
