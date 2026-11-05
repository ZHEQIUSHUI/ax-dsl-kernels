#!/usr/bin/env python3
"""Assemble dist/: the fused lm_head.topk kernel and the manifest role.

  python make_dist.py <model.axmodel> [--base-manifest <manifest.json>] [--out <dir>]

The role entry is read from the compiled axmodel's IO table (board_io.py has already declared the MX payloads as u8).
With a base manifest (the model directory's current one) the output is a complete drop-in manifest.json; otherwise
only the role fragment (manifest.role.json) is written.
"""
import argparse
import json
import os
import shutil

import onnx

ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "..", "dist"))

# ONNX element type -> AX_ENGINE_DATA_TYPE_T
DTYPE = {1: 3, 2: 1, 6: 6, 16: 9}
SIZE = {1: 4, 2: 1, 6: 4, 16: 2}
NAME_DTYPE = {"w": 15, "s": 1}   # the MXFP4 payloads, as the release head declares them


def io_of(model, is_input):
    out = []
    for v in (model.graph.input if is_input else model.graph.output):
        tt = v.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        n = 1
        for d in dims:
            n *= d
        dt = NAME_DTYPE.get(v.name, DTYPE[tt.elem_type])
        size = SIZE.get(tt.elem_type, 1)
        out.append({"name": v.name, "bytes": n * size, "dtype": dt, "shape": dims})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("axmodel")
    ap.add_argument("--base-manifest")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.axmodel)
    shutil.copy(a.axmodel, os.path.join(a.out, name))
    m = onnx.load(a.axmodel, load_external_data=False)
    role = {"file": name, "in": io_of(m, True), "out": io_of(m, False)}
    with open(os.path.join(a.out, "manifest.role.json"), "w") as f:
        json.dump({"lm_head.topk": role}, f, indent=2)
    if a.base_manifest:
        base = json.load(open(a.base_manifest))
        # the release kernels are symlinked next to the manifest, not copied
        base.setdefault("kernels", {})["lm_head.topk"] = role
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump(base, f, indent=2)
    print(json.dumps({"kernel": name, "K": next(o["bytes"] for o in role["out"] if o["name"] == "values") // 4}, indent=2))


if __name__ == "__main__":
    main()
