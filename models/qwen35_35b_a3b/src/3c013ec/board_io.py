"""Declare a compiled axmodel's MX inputs / outputs the way the AX8860 board runtime (engine 3.0.001) accepts them.

The current builder writes MX payloads with their ONNX storage types: E8M0 scales as FLOAT8E8M0 (24) and FP4 weights as
FLOAT4E2M1 (23) in logical coordinates (two elements per byte). The board runtime rejects both types (0x80060084), and
an FP4 tensor declared as uint8 in logical coordinates fails to run (twice its bytes). Here both become UINT8, an FP4
tensor's last dimension halved to its bytes; the bytes the kernel reads do not change.

Usage: python board_io.py model.axmodel [...]   (in place), or board_io.fix(path) from a build script.

kernel_file(out_dir, name, cores): the file a kernel is written to, <name>_npu<cores>.axmodel (the NPU cores it was
compiled for, 1 included; ggml-axera looks a kernel up by its name and GGML_AXERA_CORES).
"""
import sys

import onnx
from onnx import TensorProto

FP4, E8M0 = 23, 24  # TensorProto.FLOAT4E2M1, TensorProto.FLOAT8E8M0


def fix(path):
    m = onnx.load(path, load_external_data=False)
    changed = []
    g = m.graph
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == E8M0:
            tt.elem_type = TensorProto.UINT8
            changed.append(f"{vi.name}: E8M0 -> u8")
        elif tt.elem_type == FP4:
            tt.elem_type = TensorProto.UINT8
            d = tt.shape.dim[-1]
            assert d.dim_value % 2 == 0, (vi.name, d.dim_value)
            d.dim_value //= 2
            changed.append(f"{vi.name}: FP4 -> u8, last dim {d.dim_value * 2} -> {d.dim_value}")
    if changed:
        onnx.save(m, path)
    return changed


def kernel_file(out_dir, name, cores=1):
    return f"{out_dir}/{name}_npu{cores}.axmodel"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, fix(p))
