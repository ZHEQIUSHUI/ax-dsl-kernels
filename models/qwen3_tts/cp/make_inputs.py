"""Write the compiled kernel ABI in kernel argument order.

Usage:
    python make_inputs.py 0.6b OUT/0.6b.axmodel 1.7b OUT/1.7b.axmodel

The packs and their order come from the same frozen builder used by
``build_cp.py``.  ONNX's metadata cannot distinguish E8M0 scales from UINT8 in
this toolchain, so dtypes are taken from the actual runtime tensors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch


DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int32: "int32",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e8m0fnu: "float8_e8m0fnu",
    torch.float16: "float16",
}


def describe(model: str, axmodel: str):
    import cp_data

    cp_data.configure(model)
    import cp_kernel as C
    import q3tts_cp_step as S

    case, _, _, _, _, _ = cp_data.make_case(S)
    case["top_k"] = torch.tensor([50], dtype=torch.int32)
    namespace = {}
    exec(C.kernel_source(), namespace)
    fn = namespace["cp_frame"]
    names = list(S.L.order(fn))
    output_names = ["codes", "lgs"] + [
        f"{prefix}{j}" for j in range(S.NL) for prefix in ("kc", "vc")
    ]

    inputs = []
    for name in names:
        if name in output_names:
            continue
        tensor = case[name]
        inputs.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": DTYPE.get(tensor.dtype, str(tensor.dtype)),
            }
        )

    outputs = [
        {"name": "codes", "shape": list(case["codes"].shape), "dtype": "int32"},
        {"name": "lgs", "shape": list(case["lgs"].shape), "dtype": "float32"},
    ]
    for j in range(S.NL):
        for prefix in ("kc", "vc"):
            outputs.append(
                {
                    "name": f"{prefix}{j}",
                    "shape": list(case[f"{prefix}{j}"].shape),
                    "dtype": "float16",
                }
            )

    input_names = [item["name"] for item in inputs]
    return {
        "model": model,
        "axmodel": os.path.abspath(axmodel),
        "size_bytes": os.path.getsize(axmodel),
        "hidden": C.HT,
        "layers": C.NL,
        "positions": C.NP,
        "vocab": C.V,
        "top_k_max": C.TOPK_MAX,
        "kernel_order": input_names,
        "inputs": {item["name"]: {"shape": item["shape"], "dtype": item["dtype"]} for item in inputs},
        "outputs": {item["name"]: {"shape": item["shape"], "dtype": item["dtype"]} for item in outputs},
    }


def main(argv):
    if len(argv) == 4 and argv[1] == "--one":
        print(json.dumps(describe(argv[2], argv[3])))
        return
    if len(argv) < 3 or len(argv) % 2 == 0:
        raise SystemExit("usage: make_inputs.py MODEL AXMODEL [MODEL AXMODEL ...]")
    result = {}
    for i in range(1, len(argv), 2):
        model, path = argv[i], argv[i + 1]
        # cp_kernel/q3tts_cp_step freeze shapes at import time.  Use one fresh
        # interpreter per model so 0.6B's HT=1024 cannot leak into 1.7B.
        child = subprocess.run(
            [sys.executable, __file__, "--one", model, path],
            check=True,
            capture_output=True,
            text=True,
        )
        result[model] = json.loads(child.stdout)
    with open("inputs.json", "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("wrote inputs.json")


if __name__ == "__main__":
    main(sys.argv)
