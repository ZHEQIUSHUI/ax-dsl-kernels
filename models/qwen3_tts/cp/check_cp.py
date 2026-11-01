"""Run and check one Qwen3-TTS code-predictor frame.

The old (frozen, statically unrolled) and new (one runtime position loop)
kernels are selected with ``--impl``.  A separate process is intentional:
the old kernel fixes top-k at module import time.

Examples:
  source /data/junda/work/sgl_env.sh
  python check_cp.py --impl new --model 0.6b --top-k 50 --out out_check/new50.pt
  python check_cp.py --impl old --model 0.6b --top-k 50 --out out_check/old50.pt
  python check_cp.py --impl new --model 0.6b --top-k 1  --out out_check/new1.pt

``--run-mode PYRUN`` evaluates the DSL graph in numpy; ``CRUN`` compiles and
simulates it.  The script also evaluates a torch reference over the same
dequantized MXFP8 packs and reports per-position logit and per-layer KV
cosines, and samples the kernel's own logits independently.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
REF_SRC = os.environ.get(
    "CP_REF_SRC", os.path.dirname(os.path.abspath(__file__))
)
if REF_SRC not in sys.path:
    sys.path.insert(0, REF_SRC)
os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("KVL", "cell")
os.environ.setdefault("LMAX", "16")
os.environ.setdefault("BC", "16")
os.environ.setdefault("KVX", "1")
os.environ.setdefault("DEC_SPLIT", "0")
os.environ.setdefault("LMHEAD", "0")
os.environ.setdefault("FIRST", "0")


def torch_sample(logits, draw: float, top_k: int) -> int:
    """ggml top-k: descending/stable, swap the first two when k > 1."""
    values, indices = torch.sort(logits.reshape(-1), descending=True, stable=True)
    values, indices = values[:top_k], indices[:top_k]
    if top_k > 1:
        values = torch.cat([values[1:2], values[0:1], values[2:]])
        indices = torch.cat([indices[1:2], indices[0:1], indices[2:]])
    cdf = torch.softmax(values.float(), -1).cumsum(-1)
    crossed = (cdf > draw).nonzero()
    return int(indices[int(crossed[0, 0]) if len(crossed) else top_k - 1])


def cosine(a, b) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.reshape(-1).double(), b.reshape(-1).double(), dim=0
        )
    )


def reference_frame(S, raw, extra, heads, embd, case, top_k):
    """Teacher-forced torch reference at every position, all five layers."""
    HT, V, NP = S.HT, S.V, 16
    kref = [[torch.zeros((S.L.LMAX, S.L.HKV * S.L.D)) for _ in range(2)] for _ in range(S.NL)]
    logits, sampled = [], []
    row = case["e0"].float().reshape(HT)
    for pos in range(NP):
        x = row
        if S.PROJ:
            x = (extra["w_pin"].float() @ x + extra["b_pin"]).bfloat16().float()
        x = S.ref_layers(x, pos, raw, kref)
        if pos == 0:
            row = case["e1"].float().reshape(HT)
            continue
        lg = heads[pos - 1].float() @ S.rms(x, extra["norm"])
        code = torch_sample(lg, float(case["rnd"][0, pos - 1]), top_k)
        logits.append(lg)
        sampled.append(code)
        # Teacher-force the recurrence with the kernel's code.  This isolates
        # numerical error from a top-k draw that lands on a CDF boundary.
        row = embd[pos - 1, int(case["codes"][0, pos])].float().reshape(HT)
    return torch.stack(logits), torch.tensor(sampled, dtype=torch.int32), kref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("new", "old"), required=True)
    ap.add_argument("--model", choices=("0.6b", "1.7b"), required=True)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-mode", choices=("PYRUN", "CRUN"), default="PYRUN")
    ap.add_argument("--cache-dir", help="reuse a compiled graph from this cache directory")
    ap.add_argument("--real", action="store_true", help="use the downloaded Qwen3-TTS checkpoint weights")
    args = ap.parse_args()
    if not 1 <= args.top_k <= 50:
        raise SystemExit("top-k must be in 1..50")
    if args.impl == "old":
        os.environ["TOPK"] = str(args.top_k)

    out_arg = os.path.abspath(args.out)
    os.environ["RUN_MODE"] = args.run_mode
    os.environ["CP_STATIC_PYRUN"] = "1" if args.run_mode == "PYRUN" else "0"

    import cp_data

    cp_data.configure(args.model, real=args.real)
    import axnn.dsl as ax
    import q3tts_cp_step as S

    case, _, _, _, raw, extra = cp_data.make_case(S, args.frame)
    if args.impl == "new":
        import cp_kernel as C

        case["top_k"] = torch.tensor([args.top_k], dtype=torch.int32)
        source = C.kernel_source()
        outdir = os.path.join(out_arg + ".build", "k")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "cp_frame_kernel.py"), "w") as f:
            f.write(source)
        sys.path.insert(0, outdir)
        os.chdir(outdir)
        fn = __import__("cp_frame_kernel").cp_frame
        names = list(S.L.order(fn))
        cores = C.CORES
    else:
        import q3tts_cp_frame as O

        outdir = os.path.join(out_arg + ".build", "k")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "cp_frame_kernel.py"), "w") as f:
            f.write(O.kernel_source())
        sys.path.insert(0, outdir)
        os.chdir(outdir)
        fn = __import__("cp_frame_kernel").cp_frame
        names = list(S.L.order(fn))
        cores = S.CORES

    mode = ax.RunMode(args.run_mode)
    kk = ax.builder(
        run_mode=mode,
        enable_cache=bool(args.cache_dir),
        cache_path=os.path.abspath(args.cache_dir) if args.cache_dir else None,
        dump_path=outdir,
    )(fn)
    ax.init(chip_num=1, core_num=cores)
    # The simulator zeroes output buffers.  The board keeps resident KV in
    # place, so prefill the in-place kc*/vc* outputs for both implementations.
    for j in range(S.NL):
        S.q.RESIDENT[f"kc{j}"] = case[f"kc{j}"].clone()
        S.q.RESIDENT[f"vc{j}"] = case[f"vc{j}"].clone()
    kk[cores](*[case[n] for n in names])
    S.q.RESIDENT.clear()

    heads = case["heads"].reshape(16 - 1, S.V, S.H)
    embd = case["embd"].reshape(16 - 1, S.V, S.HT)
    ref_lgs, ref_codes, ref_kv = reference_frame(
        S, raw, extra, heads, embd, case, args.top_k
    )
    got_lgs = case["lgs"].float()
    got_codes = case["codes"][0, 1:].int()
    row_cos = torch.tensor([cosine(got_lgs[p], ref_lgs[p]) for p in range(15)])
    kv_cos = torch.tensor(
        [
            cosine(case[f"kc{j}"], ref_kv[j][0])
            for j in range(S.NL)
        ]
        + [
            cosine(case[f"vc{j}"], ref_kv[j][1])
            for j in range(S.NL)
        ]
    )
    own_codes = torch.tensor(
        [
            torch_sample(got_lgs[p], float(case["rnd"][0, p]), args.top_k)
            for p in range(15)
        ],
        dtype=torch.int32,
    )

    out = out_arg
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "impl": args.impl,
        "model": args.model,
        "top_k": args.top_k,
        "run_mode": args.run_mode,
        "real": args.real,
        "codes": case["codes"].clone(),
        "lgs": case["lgs"].clone(),
        "e0": case["e0"].clone(),
        "e1": case["e1"].clone(),
        "rnd": case["rnd"].clone(),
        "expected_codes_from_own_logits": own_codes,
        "reference_codes": ref_codes,
        "reference_lgs": ref_lgs,
        "lgs_cos": row_cos,
        "kv_cos": kv_cos,
    }
    for j in range(S.NL):
        payload[f"kc{j}"] = case[f"kc{j}"].clone()
        payload[f"vc{j}"] = case[f"vc{j}"].clone()
    torch.save(payload, out)

    same_own = bool((got_codes == own_codes).all())
    same_ref = bool((got_codes == ref_codes).all())
    print(
        f"{args.impl} {args.model} k={args.top_k} {args.run_mode}: "
        f"lgs_cos mean={row_cos.mean():.7f} min={row_cos.min():.7f}; "
        f"kv_cos mean={kv_cos.mean():.7f} min={kv_cos.min():.7f}; "
        f"codes_vs_own={same_own} codes_vs_ref={same_ref}",
        flush=True,
    )
    print(f"codes={case['codes'][0].tolist()}", flush=True)
    if not same_own:
        print(
            f"own={own_codes.tolist()} ref={ref_codes.tolist()}", flush=True
        )


if __name__ == "__main__":
    main()
