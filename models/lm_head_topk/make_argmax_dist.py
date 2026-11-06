#!/usr/bin/env python3
"""Assemble a model's dist entry: fused lm_head.argmax kernel plus manifest role.

  python make_argmax_dist.py <model.axmodel> [--m4-axmodel <m4.axmodel>] --format mxfp4|mxfp8 \
      [--base-manifest <manifest.json>] [--out <dir>] \
      [--role-name manifest.argmax.json] [--manifest-name manifest.argmax.full.json]
"""
import argparse
import json
import os
import shutil

import onnx

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
    ap.add_argument("--m4-axmodel")
    ap.add_argument("--format", choices=("mxfp4", "mxfp8"), required=True)
    ap.add_argument("--role", default="lm_head.argmax")
    ap.add_argument("--m4-role", default="")
    ap.add_argument("--remove-role", action="append", default=[],
                    help="drop a role from the base manifest before adding the combined head")
    ap.add_argument("--role-name", default="manifest.argmax.json")
    ap.add_argument("--base-manifest")
    ap.add_argument("--manifest-name", default="manifest.argmax.full.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.axmodel)
    dst = os.path.join(a.out, name)
    if os.path.abspath(a.axmodel) != os.path.abspath(dst):
        shutil.copy(a.axmodel, dst)
    m = onnx.load(a.axmodel, load_external_data=False)
    role = {"file": name, "in": io_of(m, True, a.format),
            "out": io_of(m, False, a.format)}
    roles = {a.role: role}
    if a.m4_axmodel:
        m4name = os.path.basename(a.m4_axmodel)
        m4dst = os.path.join(a.out, m4name)
        if os.path.abspath(a.m4_axmodel) != os.path.abspath(m4dst):
            shutil.copy(a.m4_axmodel, m4dst)
        m4 = onnx.load(a.m4_axmodel, load_external_data=False)
        m4_role = a.m4_role or a.role + ".m4"
        roles[m4_role] = {"file": m4name, "in": io_of(m4, True, a.format),
                          "out": io_of(m4, False, a.format)}
    with open(os.path.join(a.out, a.role_name), "w") as f:
        json.dump(roles, f, indent=2)
    if a.base_manifest:
        base = json.load(open(a.base_manifest))
        for r in a.remove_role:
            base.get("kernels", {}).pop(r, None)
        base.setdefault("kernels", {}).update(roles)
        with open(os.path.join(a.out, a.manifest_name), "w") as f:
            json.dump(base, f, indent=2)
    print(json.dumps({"kernels": {k: v["file"] for k, v in roles.items()},
                      "format": a.format,
                      "rows": [v["in"][0]["shape"][0] for v in roles.values()]}, indent=2))


if __name__ == "__main__":
    main()
