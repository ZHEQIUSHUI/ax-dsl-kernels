#!/usr/bin/env python3
"""Compile the M=4 verify group with the be6410c packs.

    source /data/junda/work/axdsl_env.sh
    QW_RANDOM=1 RUN_MODE=PYRUN python build_verify_m4.py out_verify_m4
    RUN_MODE=CRUN python build_verify_m4.py out_verify_m4_crun
"""
from __future__ import annotations

import glob
import importlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/data/junda/work/q27m/src_ks")
for k, v in dict(G="16", FIRST="0", KVL="cell", BC="256", CORES="4", LMAX="16384", WFMT="mx8",
                 WFMT4="1", BLOCK="1", KVX="1", DEC_SPLIT="1", MODEL="/data/junda/hf/Qwen3.8-27B").items():
    os.environ.setdefault(k, v)

import torch
from loguru import logger

import axnn.dsl as ax
import q35_group as G
import q35_layer_mc as L
import q35_step as q
from group_common import compare_layout, dump_json, make_group_bundle
from verify_group_m4 import T, make_source, snapshot_shapes

MODEL = os.environ["MODEL"]
RUN_MODE = os.environ.get("RUN_MODE", "PYRUN")
CORES = int(os.environ.get("CORES", "4"))
LMAX, BC, HKV, D = L.LMAX, L.BC, L.HKV, L.D
NBLK = LMAX // BC
REF_LAYOUT = "/data/junda/work/q27m/k_g16/qwen35_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.layout"
NAME = "qwen35_group16_verify4_m4_gggagggagggaggga_mx4_kvc_c16384_npu4"


def dtype_name(dt):
    return {
        torch.bfloat16: "bfloat16", torch.float32: "float32", torch.int32: "int32",
        torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e8m0fnu: "float8_e8m0fnu",
        torch.float4_e2m1fn_x2: "float4_e2m1fn_x2",
    }.get(dt, str(dt).removeprefix("torch."))


def desc(t, name):
    return {"name": name, "shape": list(t.shape), "dtype": dtype_name(t.dtype)}


def state_ios():
    ios = []
    for kind in G.KINDS:
        if kind == "gdn":
            ios.append((torch.zeros((L.NV, L.DK, L.DV)),
                        torch.zeros((1, L.CONV - 1, L.KD), dtype=torch.bfloat16),
                        torch.zeros((1, L.CONV - 1, L.KD), dtype=torch.bfloat16),
                        torch.zeros((1, L.CONV - 1, L.VD), dtype=torch.bfloat16)))
        else:
            ios.append((torch.zeros((NBLK, 2, BC, HKV * D), dtype=torch.bfloat16),))
    return ios


def init_resident(ios):
    q.RESIDENT.clear()
    for j, kind in enumerate(G.KINDS):
        if kind == "gdn":
            for name, t in zip((f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"), ios[j]):
                q.RESIDENT[name] = t.clone()
        else:
            q.RESIDENT[f"kv{j}"] = ios[j][0].clone()


def build_module(build, sub, src, modname):
    d = f"{build}/{sub}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{modname}.py", "w") as f:
        f.write(src)
    sys.path.insert(0, d)
    importlib.invalidate_caches()
    return importlib.import_module(modname)


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "out_verify_m4")
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(MODEL)
    bundle = make_group_bundle(W, G.KINDS, 0)
    same, diff = compare_layout(REF_LAYOUT, bundle.lines)
    logger.info(f"layout identical to be6410c: {same}" + ("" if same else f" first diff {diff}"))
    with open(Path(out_dir) / "verify_group.layout", "w") as f:
        f.write("\n".join(bundle.lines) + "\n")
    G.PDT.clear()
    G.PDT.update(bundle.pdt)
    sys.modules["q35_group"] = G

    src4, _ = make_source(bundle.pdt)
    mod4 = build_module(out_dir, "t4", src4, "group_t4_m4")
    ios4 = state_ios()
    cs4, pn4, mask4 = [], [], []
    for t in range(T):
        ct, st = q.rope_cos_sin(t)
        cs4.append(torch.cat([ct, st]).reshape(2, L.ROT // 2))
        pn4.append(torch.tensor([t, t // BC + 1, (t // BC) * 2 * BC + t % BC], dtype=torch.int32))
        mask4.append(L.mask_rows(t).reshape(NBLK, BC))
    shapes = snapshot_shapes()
    case4 = dict(bundle.packs, **bundle.banks, x=torch.randn((1, T, L.H), dtype=torch.float32).bfloat16(),
                 cs=torch.stack(cs4), pn=torch.stack(pn4), mask=torch.stack(mask4),
                 out=torch.zeros((1, T, L.H), dtype=torch.bfloat16), ntok=torch.tensor([T], dtype=torch.int32),
                 ssnap=torch.zeros(shapes[0]), cqsnap=torch.zeros(shapes[1], dtype=torch.bfloat16),
                 cksnap=torch.zeros(shapes[2], dtype=torch.bfloat16),
                 cvsnap=torch.zeros(shapes[3], dtype=torch.bfloat16))
    for j, kind in enumerate(G.KINDS):
        for name, t in zip([f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"] if kind == "gdn" else [f"kv{j}"], ios4[j]):
            case4[name] = t
    init_resident(ios4)

    sub = f"{out_dir}/t4"
    os.makedirs(f"{sub}/run", exist_ok=True)
    os.chdir(f"{sub}/run")
    ax.init(chip_num=1, core_num=CORES)
    logger.info(f"building {NAME} ({RUN_MODE})")
    fn = mod4.verify_kernel
    k = ax.builder(run_mode=ax.RunMode(RUN_MODE), enable_cache=RUN_MODE == "CRUN", dump_path=f"{sub}/run")(fn)
    order = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    k[CORES](*[case4[n] for n in order])
    q.RESIDENT.clear()

    if RUN_MODE == "CRUN":
        paths = glob.glob(f"{sub}/run/cache/{fn.__name__}.*.compiled.axmodel")
        if not paths:
            raise RuntimeError(f"no compiled axmodel under {sub}/run/cache")
        dst = Path(out_dir) / f"{NAME}.axmodel"
        shutil.copy(max(paths, key=os.path.getmtime), dst)
        logger.info(f"wrote {dst} ({dst.stat().st_size} bytes)")
    with open(Path(out_dir) / f"{NAME}.layout", "w") as f:
        for p in bundle.pdt:
            f.write(f"{p} - 0 {case4[p].numel() * case4[p].element_size()}\n")
        f.write("\n".join(bundle.lines) + "\n")
    io = {"kernel": NAME, "run_mode": RUN_MODE,
          "inputs": [desc(case4[n], n) for n in order if case4[n] is not None], "order": order}
    dump_json(Path(out_dir) / f"{NAME}.inputs.json", io)


if __name__ == "__main__":
    main()
