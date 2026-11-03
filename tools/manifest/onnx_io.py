#!/usr/bin/env python3
"""IO tables of axmodel files as iojson's JSON, read from their ONNX metadata without a board.

onnx_io.py <x.axmodel>... > io.json

Run board_io.py on kernels whose MX payloads are declared FP4 / E8M0 first (the board runtime sees those as UINT8).
This is for constructing a manifest before the board's iojson is available; the manifest's IO check compares names and
byte counts, which are the storage sizes ONNX declares.
"""
import json
import os
import sys

import onnx
from onnx import TensorProto

# ONNX element type -> AX_ENGINE_DATA_TYPE_T (the board's tables) and bytes per element
DT = {
    TensorProto.FLOAT: (3, 4),
    TensorProto.BFLOAT16: (9, 2),
    TensorProto.FLOAT16: (10, 2),
    TensorProto.INT32: (6, 4),
    TensorProto.INT64: (6, 8),
    TensorProto.UINT8: (1, 1),
    TensorProto.FLOAT8E4M3FN: (11, 1),
    TensorProto.FLOAT8E5M2: (12, 1),
    TensorProto.FLOAT4E2M1: (15, 0.5),
    TensorProto.FLOAT8E8M0: (1, 1),
}


def entry(vi):
    tt = vi.type.tensor_type
    shape = [int(d.dim_value) for d in tt.shape.dim]
    ax, size = DT[tt.elem_type]
    n = 1
    for d in shape:
        n *= d
    return [vi.name, int(n * size), ax, shape]


def main():
    out = {}
    for p in sys.argv[1:]:
        m = onnx.load(p, load_external_data=False)
        out[os.path.basename(p)] = {"in": [entry(v) for v in m.graph.input],
                                    "out": [entry(v) for v in m.graph.output]}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
