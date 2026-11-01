"""PYRUN/CRUN check of the codec decoder against PyTorch chunked/full decode."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import axnn.dsl as ax
from decoder_kernel import decoder, WINDOW, SAMPLES, DAC_RES

ART = Path("artifacts")


def state_shapes():
    kv = (8, 2, 1024, 71)
    pre = (2, 512)
    up0 = (6, 1024)
    up1 = (6, 1024)
    dent = (6, 1024)
    tails = [(8, 768), (5, 384), (4, 192), (3, 96)]
    res = []
    for b, c in DAC_RES:
        for d in (1, 3, 9):
            res.append((6 * d, c))
    post = (6, 96)
    return kv, pre, up0, up1, dent, tails, res, post


def zeros_states(bf16=True):
    kv, pre, up0, up1, dent, tails, res, post = state_shapes()
    dt = torch.bfloat16 if bf16 else torch.float32
    def z(s):
        return torch.zeros(s, dtype=dt)
    tail_pairs = [(z(t), z(t)) for t in tails]
    res_pairs = [(z(s), z(s)) for s in res]
    return dict(kv=(z(kv), z(kv)), pre=(z(pre), z(pre)), up0=(z(up0), z(up0)),
                up1=(z(up1), z(up1)), dent=(z(dent), z(dent)),
                tails=tail_pairs, res=res_pairs, post=(z(post), z(post)))


def make_probe(run_mode="PYRUN", cache_path=None, dump_path=None):
    return ax.builder(run_mode=ax.RunMode(run_mode), enable_cache=True,
                      unique_axmodel=True, cache_path=cache_path,
                      dump_path=dump_path)(decoder)


def run_window(probe, codes_win, weights, st, window_index):
    qw, uw, dw = weights
    kv_i, kv_o = st["kv"]
    pre_i, pre_o = st["pre"]
    up0_i, up0_o = st["up0"]
    up1_i, up1_o = st["up1"]
    dent_i, dent_o = st["dent"]
    (t0i, t0o), (t1i, t1o), (t2i, t2o), (t3i, t3o) = st["tails"]
    (r00i, r00o), (r01i, r01o), (r02i, r02o), \
    (r10i, r10o), (r11i, r11o), (r12i, r12o), \
    (r20i, r20o), (r21i, r21o), (r22i, r22o), \
    (r30i, r30o), (r31i, r31o), (r32i, r32o) = st["res"]
    post_i, post_o = st["post"]
    wav = torch.zeros(SAMPLES, dtype=torch.float32)
    pos = torch.tensor([window_index * WINDOW], dtype=torch.int32)
    probe[4](codes_win.to(torch.int32).contiguous(), qw, uw, dw, pos,
             kv_i, kv_o, pre_i, pre_o, up0_i, up0_o, up1_i, up1_o, dent_i, dent_o,
             t0i, t0o, t1i, t1o, t2i, t2o, t3i, t3o,
             r00i, r00o, r01i, r01o, r02i, r02o,
             r10i, r10o, r11i, r11o, r12i, r12o,
             r20i, r20o, r21i, r21o, r22i, r22o,
             r30i, r30o, r31i, r31o, r32i, r32o,
             post_i, post_o, wav)
    # swap in the new states
    st["kv"] = (kv_o, torch.empty_like(kv_o))
    st["pre"] = (pre_o, torch.empty_like(pre_o))
    st["up0"] = (up0_o, torch.empty_like(up0_o))
    st["up1"] = (up1_o, torch.empty_like(up1_o))
    st["dent"] = (dent_o, torch.empty_like(dent_o))
    st["tails"] = [(t0o, torch.empty_like(t0o)), (t1o, torch.empty_like(t1o)),
                   (t2o, torch.empty_like(t2o)), (t3o, torch.empty_like(t3o))]
    st["res"] = [(o, torch.empty_like(o)) for _, o in st["res"]]
    st["post"] = (post_o, torch.empty_like(post_o))
    return wav


def report(name, got, ref):
    got = got.float(); ref = ref.float()
    cos = F.cosine_similarity(got, ref, dim=0).item()
    noise = (got - ref).square().sum().clamp_min(1e-30)
    snr = (10 * torch.log10(ref.square().sum() / noise)).item()
    print(f"{name}: n={got.numel()} cos={cos:.9f} snr={snr:.3f}dB absmax={(got-ref).abs().max():.5f}")
    return cos, snr


def load_bank(path):
    bits = np.fromfile(path, dtype="<u2")
    return torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32)).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--run-mode", choices=("PYRUN", "CRUN"), default="PYRUN")
    ap.add_argument("--cache-dir", type=Path, default=Path("out_decoder_gguf/cache"))
    ap.add_argument("--dump-dir", type=Path, default=Path("out_decoder_gguf/artifacts"))
    ap.add_argument("--save-prefix", type=Path)
    gguf_banks = ART / "gguf_banks"
    gguf_ref = gguf_banks / "reference_gguf.pt"
    ap.add_argument("--weights-dir", type=Path,
                    default=gguf_banks if (gguf_banks / "quant_w.bin").exists() else ART)
    ap.add_argument("--reference", type=Path,
                    default=gguf_ref if gguf_ref.exists() else ART / "reference.pt")
    args = ap.parse_args()
    ax.init(chip_num=1)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    ref = torch.load(args.reference, weights_only=False)
    codes = ref["codes"]
    if (args.weights_dir / "quant_w.bin").exists():
        weights = tuple(load_bank(args.weights_dir / f"{g}.bin") for g in ("quant_w", "up_w", "dac_w"))
        print(f"weights={args.weights_dir} (raw bf16)")
    else:
        weights = (torch.load(args.weights_dir / "quant_w.pt", weights_only=True),
                   torch.load(args.weights_dir / "up_w.pt", weights_only=True),
                   torch.load(args.weights_dir / "dac_w.pt", weights_only=True))
        print(f"weights={args.weights_dir} (.pt)")
    st = zeros_states(not args.fp32)
    probe = make_probe(args.run_mode, args.cache_dir, args.dump_dir)
    for w in range(args.windows):
        win = codes[w * WINDOW : (w + 1) * WINDOW]
        if win.shape[0] < WINDOW:
            break
        got = run_window(probe, win, weights, st, w)
        if args.save_prefix is not None:
            path = Path(f"{args.save_prefix}_window{w}.bin")
            got.detach().float().cpu().numpy().astype("<f4").tofile(path)
            print(f"saved {path}")
        lo, hi = w * SAMPLES, (w + 1) * SAMPLES
        report(f"window{w} vs chunked", got, ref["chunked"][lo:hi])
        report(f"window{w} vs full", got, ref["decoded"][lo:hi])


if __name__ == "__main__":
    main()
