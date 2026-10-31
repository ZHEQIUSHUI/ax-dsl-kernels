"""Collect the compiled prefill/decode AXModel ABIs into inputs.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto


DTYPES = {
    TensorProto.FLOAT: "float32",
    TensorProto.INT32: "int32",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.FLOAT8E4M3FN: "float8_e4m3fn",
    TensorProto.FLOAT8E8M0: "float8_e8m0fnu",
}
ROWS = {"prefill256": 256, "prefill1": 1}


def shape(value):
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def dtype(value):
    kind = value.type.tensor_type.elem_type
    return DTYPES.get(kind, f"onnx_elem_type_{kind}")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    result = {}
    for variant_dir in sorted((root / "out").glob("build_*")):
        model = variant_dir.name.removeprefix("build_")
        result[model] = {}
        for subdir, rows in ROWS.items():
            paths = sorted((variant_dir / subdir).glob("*.compiled.axmodel"))
            if not paths:
                result[model][f"rows{rows}"] = {"axmodel": None}
                continue
            path = paths[-1]
            graph = onnx.load(str(path))
            result[model][f"rows{rows}"] = {
                "axmodel": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
                "inputs": {
                    value.name: {"shape": shape(value), "dtype": dtype(value)}
                    for value in graph.graph.input
                },
                "outputs": {
                    value.name: {"shape": shape(value), "dtype": dtype(value)}
                    for value in graph.graph.output
                },
            }
    (root / "inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {root / 'inputs.json'} for {len(result)} models")


if __name__ == "__main__":
    main()
