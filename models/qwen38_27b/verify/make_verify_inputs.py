#!/usr/bin/env python3
"""Write the verify4 inputs.json/layout from the be6410c layout and kernel signature."""
import json, os, shutil
from pathlib import Path

OUT = Path(os.environ.get("OUT", "out_verify_crun"))
NAME = "qwen35_group16_verify4_gggagggagggaggga_mx4_kvc_c16384_npu4"
REF = Path("/data/junda/work/q27m/k_g16/qwen35_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.layout")
KINDS = "gggagggagggaggga"
T, H, NV, DK, DV, CONV, KD, VD, HKV, D, BC, NBLK = 4, 5120, 48, 128, 128, 4, 2048, 6144, 4, 256, 256, 64
n_gdn = KINDS.count("g")

lines = REF.read_text().splitlines()
pack_lines = lines[:9]
packs = []
for ln in pack_lines:
    name, _, _, nbytes = ln.split()
    nbytes = int(nbytes)
    dt, shape = {
        "p4": ("float4_e2m1fn_x2", [nbytes]),
        "pw": ("float8_e4m3fn", [nbytes]),
        "ps": ("float8_e8m0fnu", [nbytes]),
        "pf": ("float32", [nbytes // 4]),
        "pb": ("bfloat16", [nbytes // 2]),
    }[name.split("_")[0]]
    packs.append({"name": name, "shape": shape, "dtype": dt})

outs = []
for j, kind in enumerate(KINDS):
    outs += [f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"] if kind == "gdn" else [f"kv{j}"]

order = (["x"] + [p["name"] for p in packs] + ["cs", "pn", "mask", "out"] + outs
         + ["ssnap", "cqsnap", "cksnap", "cvsnap", "ntok"])
shape = {
    "x": ([1, T, H], "bfloat16"),
    "cs": ([T, 2, 32], "bfloat16"),
    "pn": ([T, 3], "int32"),
    "mask": ([T, NBLK, BC], "bfloat16"),
    "out": ([1, T, H], "bfloat16"),
    "ssnap": ([T, n_gdn, NV, DK, DV], "float32"),
    "cqsnap": ([T, n_gdn, 1, CONV - 1, KD], "bfloat16"),
    "cksnap": ([T, n_gdn, 1, CONV - 1, KD], "bfloat16"),
    "cvsnap": ([T, n_gdn, 1, CONV - 1, VD], "bfloat16"),
    "ntok": ([1], "int32"),
}
for j, kind in enumerate(KINDS):
    if kind == "gdn":
        shape[f"st{j}"] = ([NV, DK, DV], "float32")
        for n, w in (("cq", KD), ("ck", KD), ("cv", VD)):
            shape[f"{n}{j}"] = ([1, CONV - 1, w], "bfloat16")
    else:
        shape[f"kv{j}"] = ([NBLK, 2, BC, HKV * D], "bfloat16")
entries = []
for n in order:
    sh, dt = shape.get(n, (None, None))
    if sh is None:
        p = next(p for p in packs if p["name"] == n)
        sh, dt = p["shape"], p["dtype"]
    entries.append({"name": n, "shape": sh, "dtype": dt})
out_names = ["out"] + outs + ["ssnap", "cqsnap", "cksnap", "cvsnap"]
io = {"kernel": NAME, "run_mode": "CRUN", "inputs": entries,
      "outputs": [e for e in entries if e["name"] in out_names], "order": order}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{NAME}.inputs.json").write_text(json.dumps(io, indent=2))
shutil.copyfile(REF, OUT / f"{NAME}.layout")
print("wrote", OUT / f"{NAME}.inputs.json", "and layout;", len(entries), "args")
