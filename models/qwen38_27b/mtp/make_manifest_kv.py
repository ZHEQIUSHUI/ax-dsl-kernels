#!/usr/bin/env python3
"""Add a compact mtp.promptkv role to a released manifest.

usage: make_manifest_kv.py <src manifest> <dst manifest> <axmodel> <layout> <inputs.json>
"""
import copy
import json
import math
import sys


def main():
    src, dst, model, layout, inputs = sys.argv[1:6]
    m = json.load(open(src))
    io = json.load(open(inputs))
    base = m["kernels"]["mtp.prompt"]

    packs = {}
    members = []
    for line in open(layout):
        f = line.split()
        if not f:
            continue
        if f[1] == "-":
            packs[f[0]] = int(f[3])
            continue
        name, pack, off, nbytes = f[0], f[1], int(f[2]), int(f[3])
        if name not in m["params"]["mtp"]:
            raise KeyError(f"unknown mtp parameter {name}")
        members.append({
            "name": "L0." + name, "pack": pack, "offset": off, "bytes": nbytes,
            "layer": 0, "param": name, "kind": "mtp",
        })

    by_name = {d["name"]: copy.deepcopy(d) for d in base["in"]}
    dtype_code = {"float4_e2m1fn_x2": 1, "float32": 3, "bfloat16": 9, "float8_e4m3fn": 11,
                  "float8_e8m0fnu": 11}
    dtype_bytes = {"float4_e2m1fn_x2": 0.5, "float32": 4, "bfloat16": 2, "float8_e4m3fn": 1,
                   "float8_e8m0fnu": 1}
    for d in io["inputs"]:
        if d["name"] in by_name and d["name"] in ("e", "h", "cs"):
            by_name[d["name"]].update(dtype=dtype_code[d["dtype"]], shape=d["shape"],
                                      bytes=int(math.prod(d["shape"]) * dtype_bytes[d["dtype"]]))
    sizes = {d["name"]: d for d in io["weight_packs"]}
    for name in ("p4", "pw", "ps", "pf", "pb"):
        d = by_name[name]
        shape = sizes[name]["shape"]
        d["shape"] = shape
        d["bytes"] = packs[name]
    keep = ("e", "h", "p4", "pw", "ps", "pf", "pb", "cs")
    kv = copy.deepcopy(base)
    kv["file"] = model
    kv["in"] = [by_name[n] for n in keep]
    kv["out"] = [{"name": d["name"], "bytes": int(math.prod(d["shape"]) * dtype_bytes[d["dtype"]]),
                  "dtype": dtype_code[d["dtype"]], "shape": d["shape"]} for d in io["outputs"]]
    kv["packs"] = packs
    kv["members"] = members
    m["kernels"]["mtp.promptkv"] = kv
    json.dump(m, open(dst, "w"), indent=1)
    print(f"{dst}: mtp.promptkv -> {model} ({len(kv['in'])} in, packs {packs})")


if __name__ == "__main__":
    main()
