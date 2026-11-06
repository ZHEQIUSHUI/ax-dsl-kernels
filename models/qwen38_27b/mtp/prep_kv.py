#!/usr/bin/env python3
"""Write the real-weight mtp.promptkv board fixture and reference.

usage: MODEL=/data/junda/hf/Qwen3.8-27B python prep_kv.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/data/junda/work/q27m/src')

for k, v in dict(KVL="cell", BC="256", CORES="4", LMAX="16384", WFMT="mx8", WFMT4="1",
                 BLOCK="1", KVX="1", DEC_SPLIT="1", MODEL="/data/junda/hf/Qwen3.8-27B").items():
    os.environ.setdefault(k, v)

import torch

import q35_layer_mc as L
import q35_step as q
from mtp_kernel import H, P, EF32, HF32, make_bundle, make_kv_bundle
from mtp_ref import MtpWeights, mtp_step


def raw_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    elif t.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.uint8, torch.int8):
        t = t.view(torch.uint8)
    return t.numpy().tobytes()


def write_bin(path: Path, t: torch.Tensor) -> int:
    b = raw_bytes(t)
    path.write_bytes(b)
    return len(b)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "board_kv").resolve()
    indir = out / "in"
    indir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))

    W = MtpWeights(os.environ["MODEL"], random=False)
    bundle, _, deq = make_kv_bundle(W)

    torch.manual_seed(11)
    es = torch.randn((P, H), dtype=torch.float32)
    if not EF32:
        es = es.bfloat16()
    h_f32 = torch.randn((P, H), dtype=torch.float32)
    hs = h_f32 if HF32 else h_f32.bfloat16()
    kv_ref = torch.zeros((L.NBLK, 2, L.BC, L.HKV * L.D), dtype=torch.bfloat16)
    for p in range(P):
        mtp_step(es[p].bfloat16(), hs[p].bfloat16(), p, kv_ref, deq)
    cs = torch.zeros((P, 2, L.ROT // 2), dtype=torch.bfloat16)
    pn = torch.zeros((P, 3), dtype=torch.int32)
    mask = torch.zeros((L.NBLK, P, L.BC), dtype=torch.bfloat16)
    for p in range(P):
        c, s = q.rope_cos_sin(p)
        cs[p, 0], cs[p, 1] = c, s
        pn[p] = torch.tensor([p, p // L.BC + 1, (p // L.BC) * 2 * L.BC + p % L.BC], dtype=torch.int32)
        mask[:, p] = L.mask_rows(p).reshape(L.NBLK, L.BC)

    inputs = dict(bundle.packs, e=es.reshape(1, P, H), h=hs.reshape(1, P, H), cs=cs, pn=pn)
    order = [n for n in ("e", "h", "p4", "pw", "ps", "pf", "pb", "cs") if n in inputs]
    sizes = {n: write_bin(indir / f"{n}.bin", inputs[n]) for n in order}
    kout = torch.zeros((P, L.HKV * L.D), dtype=torch.bfloat16)
    vout = torch.zeros((P, L.HKV * L.D), dtype=torch.bfloat16)
    torch.save(dict(kout=kout, vout=vout, kv=kv_ref), out / "ref.pt")
    (out / "io.json").write_text(json.dumps(dict(order=order, bytes=sizes), indent=2) + "\n")
    def desc(name, t):
        dt = {torch.bfloat16: "bfloat16", torch.float32: "float32",
              torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e8m0fnu: "float8_e8m0fnu",
              torch.float4_e2m1fn_x2: "float4_e2m1fn_x2"}.get(t.dtype, str(t.dtype).removeprefix("torch."))
        return {"name": name, "shape": list(t.shape), "dtype": dt}
    io = {
        "kernel": f"qwen35_mtp_promptkv{P}{'f32' if HF32 else ''}{'e32' if EF32 else ''}_mx4_kvc_c{L.LMAX}_npu4",
        "run_mode": "CRUN",
        "inputs": [desc(n, inputs[n]) for n in order],
        "outputs": [desc("kout", kout), desc("vout", vout)],
        "weight_packs": [desc(n, bundle.packs[n]) for n in bundle.packs],
    }
    (out / "inputs.json").write_text(json.dumps(io, indent=2) + "\n")
    with open(out / "layout", "w") as f:
        for p in bundle.pdt:
            f.write(f"{p} - 0 {bundle.packs[p].numel() * bundle.packs[p].element_size()}\n")
        f.write("\n".join(bundle.lines) + "\n")
    if os.environ.get("FULL") == "1":
        full_bundle, _, _ = make_bundle(W)
        (out / "full_in").mkdir(parents=True, exist_ok=True)
        full = dict(full_bundle.packs, e=es.reshape(1, P, H), h=hs.reshape(1, P, H), cs=cs, pn=pn, mask=mask,
                    nblk=torch.tensor([(P - 1) // L.BC + 1], dtype=torch.int32))
        full_order = [n for n in ("e", "h", "p4", "pw", "ps", "pf", "pb", "cs", "pn", "mask", "nblk") if n in full]
        full_sizes = {n: write_bin(out / "full_in" / f"{n}.bin", full[n]) for n in full_order}
        (out / "full_io.json").write_text(json.dumps(dict(order=full_order, bytes=full_sizes), indent=2) + "\n")
        print(f"{out}/full_in: {len(full_order)} inputs, {sum(full_sizes.values()) / 1e6:.1f} MB")
    print(f"{out}: {len(order)} inputs, {sum(sizes.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
