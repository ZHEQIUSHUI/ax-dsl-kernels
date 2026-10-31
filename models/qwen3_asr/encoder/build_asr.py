"""Build the fused Qwen3-ASR audio-tower kernels (random weights, real shapes).

    python build_asr.py                 # both models, n = 1,2,4,8, compile all
    python build_asr.py --smoke         # one variant, L=1, CRUN compile
    python build_asr.py --one --model hf/... --n 1 --cache DIR

The CRUN call compiles the AXModel and then runs the (long) simulator.  The
parent waits for the compiled AXModel to appear and stops the child, as the
task asks.  build/check are deliberately separate files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "axmodels")
CACHE = os.path.join(HERE, "cache", "build")

MODELS = {
    "0.6b": os.path.join(HERE, "hf", "Qwen3-ASR-0.6B"),
    "1.7b": os.path.join(HERE, "hf", "Qwen3-ASR-1.7B"),
}
NS = [1, 2, 4, 8]


def variant_name(tag, H, n):
    return f"qwen3a_fused_{tag}_n{n}_h{H}"


def expected_inputs(cfg, n, layers=None):
    H, F, O = cfg.d_model, cfg.encoder_ffn_dim, cfg.output_dim
    L = cfg.encoder_layers if layers is None else layers
    P = 64 if n <= 4 else 128
    dims = [
        ("mel", (n * 100, 128), "bfloat16"),
        ("w1", (480, 16), "bfloat16"), ("b1", (480,), "bfloat16"),
        ("w2", (480, 4320), "bfloat16"), ("b2", (480,), "bfloat16"),
        ("w3", (480, 4320), "bfloat16"), ("b3", (480,), "bfloat16"),
        ("conv_out_w", (H, 7680), "bfloat16"),
        ("ln1w", (L, H), "bfloat16"), ("ln1b", (L, H), "bfloat16"),
        ("qkvw", (L, 3 * H, H), "bfloat16"), ("qkvb", (L, 3 * H), "bfloat16"),
        ("ow", (L, H, H), "bfloat16"), ("ob", (L, H), "bfloat16"),
        ("ln2w", (L, H), "bfloat16"), ("ln2b", (L, H), "bfloat16"),
        ("upw", (L, F, H), "bfloat16"), ("upb", (L, F), "bfloat16"),
        ("downw", (L, H, F), "bfloat16"), ("downb", (L, H), "bfloat16"),
        ("postw", (H,), "bfloat16"), ("postb", (H,), "bfloat16"),
        ("p1w", (H, H), "bfloat16"), ("p1b", (H,), "bfloat16"),
        ("p2w", (O, H), "bfloat16"), ("p2b", (O,), "bfloat16"),
    ]
    return dims, [("out", (P, O), "float32")]


def run_one(model_dir, n, cache_dir, layers=None):
    import torch
    import axnn.dsl as ax
    from asr_ref import audio_config
    from asr_kernel import build_kernel, random_weights, run_forward

    cfg = audio_config(model_dir)
    os.makedirs(cache_dir, exist_ok=True)
    ax.init(chip_num=1, core_num=4)
    print(f"[one] {model_dir} n={n} layers={layers or cfg.encoder_layers}", flush=True)
    kernel, _ = build_kernel(cfg, n, layers=layers, run_mode="CRUN",
                             cache_path=cache_dir, dump_path=cache_dir)
    w = random_weights(cfg, layers=layers)
    mel = torch.randn(n * 100, 128)
    t = time.time()
    run_forward(kernel, w, mel, n, cfg.output_dim)
    print(f"[one] finished in {time.time() - t:.1f}s", flush=True)


def compile_variant(model_dir, tag, n, layers, log_path):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"missing {cfg_path}")
    with open(cfg_path) as fh:
        cfg = json.load(fh)["thinker_config"]["audio_config"]
    H = cfg["d_model"]
    name = variant_name(tag, H, n)
    cache_dir = os.path.join(CACHE, name)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--one", "--model", model_dir,
           "--n", str(n), "--cache", cache_dir]
    if layers is not None:
        cmd += ["--layers", str(layers)]
    print(f"[build] {name} -> {cache_dir}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
        deadline = time.time() + 6 * 3600
        found = None
        last = None
        stable_since = None
        try:
            while time.time() < deadline:
                hits = glob.glob(os.path.join(cache_dir, "*.compiled.axmodel"))
                if hits:
                    path = max(hits, key=os.path.getmtime)
                    size = os.path.getsize(path)
                    if size == last and size > 0:
                        if stable_since is None:
                            stable_since = time.time()
                        elif time.time() - stable_since > 1.0:
                            found = path
                            break
                    else:
                        last, stable_since = size, None
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if found is None:
            tail = open(log_path).read()[-2000:]
            raise SystemExit(f"[build] no compiled axmodel for {name}\n{tail}")
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, name + "_npu4.axmodel")
    shutil.copy(found, dst)
    print(f"[build] wrote {dst} ({os.path.getsize(dst)} bytes)", flush=True)
    return dst, name


def axmodel_io(path):
    import onnx
    from onnx import TensorProto

    dt = {TensorProto.FLOAT: "float32", TensorProto.INT32: "int32",
          TensorProto.BFLOAT16: "bfloat16"}
    m = onnx.load(path)
    def one(v):
        return {"shape": [int(d.dim_value) for d in v.type.tensor_type.shape.dim],
                "dtype": dt.get(v.type.tensor_type.elem_type, str(v.type.tensor_type.elem_type))}
    return ({v.name: one(v) for v in m.graph.input}, {v.name: one(v) for v in m.graph.output})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--tags", default="0.6b,1.7b")
    ap.add_argument("--ns", default="1,2,4,8")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if args.one:
        run_one(args.model, args.n, args.cache, layers=args.layers)
        return

    tags = args.tags.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    if args.smoke:
        tags, ns, args.layers = ["0.6b"], [1], 1
    inputs_json = os.path.join(HERE, "inputs.json")
    variants = []
    if os.path.exists(inputs_json):
        try:
            variants = json.load(open(inputs_json)).get("variants", [])
        except Exception:
            variants = []
    existing = {v["name"]: v for v in variants}
    for tag in tags:
        model_dir = MODELS[tag]
        for n in ns:
            from asr_ref import audio_config
            cfg = audio_config(model_dir)
            H = cfg.d_model
            name = variant_name(tag, H, n)
            pre = os.path.join(OUT, name + "_npu4.axmodel")
            if args.skip_existing and os.path.exists(pre) and name in existing:
                print(f"[build] keep {pre}", flush=True)
                existing[name]["axmodel"] = os.path.relpath(pre, HERE)
                continue
            log = os.path.join(HERE, "output", f"build_{tag}_n{n}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            dst, name = compile_variant(model_dir, tag, n, args.layers, log)
            ins, outs = expected_inputs(cfg, n, args.layers)
            existing[name] = ({
                "name": name,
                "model": os.path.basename(model_dir),
                "n_chunks": n,
                "axmodel": os.path.relpath(dst, HERE),
                "inputs": [{"name": a, "shape": list(b), "dtype": c} for a, b, c in ins],
                "outputs": [{"name": a, "shape": list(b), "dtype": c} for a, b, c in outs],
            })
    variants = sorted(existing.values(), key=lambda v: (v["model"], v["n_chunks"]))
    # Keep the exact kernel order in every variant; validate against the compiled ABI.
    for v in variants:
        ai, ao = axmodel_io(os.path.join(HERE, v["axmodel"]))
        got = [x["name"] for x in v["inputs"]]
        if set(got) != set(ai):
            raise SystemExit(f"ABI name mismatch for {v['name']}: {sorted(set(got) ^ set(ai))}")
        for x in v["inputs"]:
            if tuple(ai[x["name"]]["shape"]) != tuple(x["shape"]):
                raise SystemExit(f"ABI shape mismatch {v['name']} {x['name']}: {ai[x['name']]['shape']} != {x['shape']}")
    with open(inputs_json, "w") as fh:
        json.dump({"kernel_order": True, "variants": variants}, fh, indent=2)
        fh.write("\n")
    print(f"[build] wrote inputs.json ({len(variants)} variants)")


if __name__ == "__main__":
    main()
