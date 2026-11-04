#!/usr/bin/env python3
"""Dump real-weight Qwen3.5-4B MTP T=1 / prompt inputs and references for the board test.

CASE=t1_pos0|t1_pos5|prompt   output root is board/mtp/<case>
CRUN=0/1                     also execute the cached compiled kernel in CRUN
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for k, v in dict(KVL="cell", BC="256", CORES="4", LMAX="16384", WFMT="mx8", WFMT4="1",
                 BLOCK="1", KVX="1", DEC_SPLIT="1",
                 MODEL="/data/junda/hf/Qwen3.5-4B").items():
    os.environ.setdefault(k, v)

import torch
from loguru import logger

import axnn.dsl as ax
import q35_layer_mc as L
import q35_step as q
import resident_pyrun as rp
from board_common import cos, dump_json, raw_bytes, write_bin
from mtp_kernel import H, P, make_bundle, mtp_prompt, mtp_t1
from mtp_ref import MtpWeights, mtp_step

CASE = os.environ["CASE"]
RUN_CRUN = os.environ.get("CRUN", "0") == "1"
ROOT = (Path(os.environ.get("ROOT", "board/mtp")) / CASE).resolve()
MODEL = os.environ["MODEL"]
CORES = int(os.environ.get("CORES", "4"))
BUILD = Path(os.environ.get("MTP_BUILD", "/data/junda/work/qsmall/whole4b/build/4b/mtp")).resolve()


def cache_key(sub, fn):
    """The key of the most recent compiled cache of `fn` under BUILD/sub."""
    if os.environ.get("MTP_KEY"):
        return os.environ["MTP_KEY"]
    import glob
    ps = sorted(glob.glob(str(BUILD / sub / "cache" / f"{fn.__name__}.*.compiled.axmodel")), key=os.path.getmtime)
    if not ps:
        raise RuntimeError(f"no compiled cache under {BUILD / sub / 'cache'}")
    return Path(ps[-1]).name.split(".")[1]


KEYS = {"t1": None, "prompt": None}
PACKS = ["p4", "pw", "ps", "pf", "pb"]


def build_case(W, bundle, deq):
    """Return (inputs, reference dict, order, cache_key, cache_dir)."""
    NBLK, BC, HKV, D = L.NBLK, L.BC, L.HKV, L.D
    if CASE.startswith("t1"):
        pos = int(CASE.split("_pos")[1])
        torch.manual_seed(7)
        es = torch.randn((pos + 1, H), dtype=torch.float32).bfloat16()
        hs = torch.randn((pos + 1, H), dtype=torch.float32).bfloat16()
        kv = torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16)
        for p in range(pos + 1):
            hn, y = mtp_step(es[p], hs[p], p, kv, deq)
        c, s = q.rope_cos_sin(pos)
        case = dict(bundle.packs,
                    e=es[pos].reshape(1, 1, H).contiguous(),
                    h=hs[pos].reshape(1, 1, H).contiguous(),
                    cs=torch.cat([c, s]).contiguous(),
                    pn=torch.tensor([pos, pos // BC + 1, (pos // BC) * 2 * BC + pos % BC], dtype=torch.int32),
                    mask=L.mask_rows(pos),
                    kv=kv.clone(),
                    hnext=torch.zeros((1, 1, H), dtype=torch.bfloat16),
                    hraw=torch.zeros((1, 1, H), dtype=torch.bfloat16))
        ref = dict(hnext=hn.reshape(1, 1, H), hraw=y.reshape(1, 1, H), kv=kv,
                   kv_in=case["kv"].clone(), written=[pos], nvalid=1)
        order = list(mtp_t1.__code__.co_varnames[: mtp_t1.__code__.co_argcount])
        return case, ref, order, cache_key("t1", mtp_t1), BUILD / "t1"

    torch.manual_seed(11)
    es = torch.randn((P, H), dtype=torch.float32).bfloat16()
    hs = torch.randn((P, H), dtype=torch.float32).bfloat16()
    kv = torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16)
    hn = torch.zeros((P, H))
    yref = torch.zeros((P, H))
    for p in range(P):
        hn[p], yref[p] = mtp_step(es[p], hs[p], p, kv, deq)
    cs = torch.zeros((P, 2, L.ROT // 2), dtype=torch.bfloat16)
    pn = torch.zeros((P, 3), dtype=torch.int32)
    mask = torch.zeros((NBLK, P, BC), dtype=torch.bfloat16)
    for p in range(P):
        c, s = q.rope_cos_sin(p)
        cs[p, 0], cs[p, 1] = c, s
        pn[p] = torch.tensor([p, p // BC + 1, (p // BC) * 2 * BC + p % BC], dtype=torch.int32)
        mask[:, p] = L.mask_rows(p).reshape(NBLK, BC)
    case = dict(bundle.packs, e=es.reshape(1, P, H), h=hs.reshape(1, P, H), cs=cs, pn=pn, mask=mask,
                nblk=torch.tensor([(P - 1) // BC + 1], dtype=torch.int32),
                kv=torch.zeros_like(kv), hnext=torch.zeros((1, P, H), dtype=torch.bfloat16),
                hraw=torch.zeros((1, P, H), dtype=torch.bfloat16))
    ref = dict(hnext=hn.reshape(1, P, H), hraw=yref.reshape(1, P, H), kv=kv,
               kv_in=case["kv"].clone(), written=list(range(P)), nvalid=P)
    order = list(mtp_prompt.__code__.co_varnames[: mtp_prompt.__code__.co_argcount])
    return case, ref, order, cache_key("prompt", mtp_prompt), BUILD / "prompt"


def main():
    os.makedirs(ROOT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    t0 = time.time()
    W = MtpWeights(MODEL, random=False)
    bundle, args, deq = make_bundle(W)
    logger.info(f"[{CASE}] real bundle built in {time.time()-t0:.1f}s")
    case, ref, order, key, cache_dir = build_case(W, bundle, deq)
    logger.info(f"[{CASE}] reference built in {time.time()-t0:.1f}s")

    indir = ROOT / "in"
    sizes = {}
    for n in order:
        t = case[n]
        sizes[n] = write_bin(indir / f"{n}.bin", t)
    dump_json(ROOT / "io.json", {"case": CASE, "order": order, "bytes": sizes,
                                 "written": ref["written"], "nvalid": ref["nvalid"]})
    torch.save(ref, ROOT / "ref.pt")
    logger.info(f"[{CASE}] inputs+torch ref written under {ROOT} ({sum(sizes.values())/1e6:.1f} MB)")

    if not RUN_CRUN:
        return

    d = os.path.abspath(cache_dir)
    os.chdir(d)
    ax.init(chip_num=1, core_num=CORES)
    fn = mtp_t1 if CASE.startswith("t1") else mtp_prompt
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=d)(fn)
    k.gen_cache_key = lambda: key
    if CASE.startswith("t1") and ref["written"][0] > 0:
        rp.set_(kv=case["kv"].clone())
        q.RESIDENT["kv"] = case["kv"].clone()
    logger.info(f"[{CASE}] CRUN forced-cache run")
    t1 = time.time()
    k[CORES](*[case[n] for n in order])
    logger.info(f"[{CASE}] CRUN wall {time.time()-t1:.1f}s")
    rp.clear()
    q.RESIDENT.clear()
    got = dict(hnext=case["hnext"].clone(), hraw=case["hraw"].clone(), kv=case["kv"].clone())
    res = {"hnext": cos(got["hnext"], ref["hnext"]), "hraw": cos(got["hraw"], ref["hraw"])}
    if CASE.startswith("t1"):
        p = ref["written"][0]
        page, cell = divmod(p, L.BC)
        res["kv_k"] = cos(got["kv"][page, 0, cell], ref["kv"][page, 0, cell])
        res["kv_v"] = cos(got["kv"][page, 1, cell], ref["kv"][page, 1, cell])
    else:
        ck = [cos(got["kv"][p // L.BC, 0, p % L.BC], ref["kv"][p // L.BC, 0, p % L.BC]) for p in range(P)]
        cv = [cos(got["kv"][p // L.BC, 1, p % L.BC], ref["kv"][p // L.BC, 1, p % L.BC]) for p in range(P)]
        res.update(kv_k_mean=float(torch.tensor(ck).mean()), kv_k_min=min(ck),
                   kv_v_mean=float(torch.tensor(cv).mean()), kv_v_min=min(cv))
    torch.save(got, ROOT / "crun.pt")
    dump_json(ROOT / "crun.json", res)
    logger.info(f"[{CASE}] CRUN vs torch " + " ".join(f"{k}={v:.7f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
