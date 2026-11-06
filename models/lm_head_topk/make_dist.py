#!/usr/bin/env python3
"""Assemble a model's dist entry: fused lm_head.topk kernel plus manifest role.

  python make_dist.py <model.axmodel> --format mxfp4|mxfp8 \
      [--role lm_head.topk] [--base-manifest <manifest.json>] [--out <dir>] \
      [--role-name manifest.role.json]
"""
import argparse
import json
import os
import shutil

import onnx

ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "..", "dist"))

DTYPE = {1: 3, 2: 1, 6: 6, 16: 9}
SIZE = {1: 4, 2: 1, 6: 4, 16: 2}


def io_of(model, is_input, fmt):
    out = []
    for v in (model.graph.input if is_input else model.graph.output):
        tt = v.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        n = 1
        for d in dims:
            n *= d
        if v.name == "w":
            dt = 15 if fmt == "mxfp4" else 11
            size = 1
        elif v.name == "s":
            dt = 1
            size = 1
        else:
            dt = DTYPE[tt.elem_type]
            size = SIZE[tt.elem_type]
        out.append({"name": v.name, "bytes": n * size, "dtype": dt, "shape": dims})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("axmodel")
    ap.add_argument("--format", choices=("mxfp4", "mxfp8"), required=True)
    ap.add_argument("--role", default="lm_head.topk")
    ap.add_argument("--role-name", default="manifest.role.json")
    ap.add_argument("--base-manifest")
    ap.add_argument("--manifest-name", default="manifest.json")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.axmodel)
    shutil.copy(a.axmodel, os.path.join(a.out, name))
    m = onnx.load(a.axmodel, load_external_data=False)
    role = {"file": name, "in": io_of(m, True, a.format),
            "out": io_of(m, False, a.format)}
    with open(os.path.join(a.out, a.role_name), "w") as f:
        json.dump({a.role: role}, f, indent=2)
    if a.base_manifest:
        base = json.load(open(a.base_manifest))
        base.setdefault("kernels", {})[a.role] = role
        with open(os.path.join(a.out, a.manifest_name), "w") as f:
            json.dump(base, f, indent=2)
    print(json.dumps({"kernel": name, "role": a.role,
                      "format": a.format, "rows": role["in"][0]["shape"][0]},
                     indent=2))


if __name__ == "__main__":
    main()
