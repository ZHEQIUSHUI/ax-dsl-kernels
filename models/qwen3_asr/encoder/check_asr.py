"""PYRUN of the fused kernel vs the official HF audio tower on real speech.

    python check_asr.py --tag 0.6b --n 1 --n 8
    python check_asr.py --tag both

Weights: the checkpoint's BF16 tensors, dequantized to FP32 for the reference
and fed unchanged as BF16 runtime inputs to the kernel.  The report is per-row
cosine on the 13*n valid projector rows.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import axnn.dsl as ax
from asr_ref import (audio_config, build_encoder, extract_mel, reference, row_cos)
from asr_kernel import build_kernel, prepare_weights, run_forward

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "0.6b": os.path.join(HERE, "hf", "Qwen3-ASR-0.6B"),
    "1.7b": os.path.join(HERE, "hf", "Qwen3-ASR-1.7B"),
}
WAV = os.path.join(HERE, "audio", "asr_en.wav")
BANK_NAMES = ("ln1w", "ln1b", "qkvw", "qkvb", "ow", "ob", "ln2w", "ln2b", "upw", "upb", "downw", "downb")


def slice_layers(w, layers):
    return {k: (v[:layers].contiguous() if k in BANK_NAMES else v) for k, v in w.items()}


def compare(tag, model_dir, n, layer_counts, save, quant_mel=False):
    cfg = audio_config(model_dir)
    H, O, L = cfg.d_model, cfg.output_dim, cfg.encoder_layers
    mel = extract_mel(WAV, n * 100, model_dir)
    mel_ref = mel.to(torch.bfloat16).to(torch.float32) if quant_mel else mel
    ref_model, _ = build_encoder(model_dir)
    all_layers = ref_model.layers
    print(f"[{tag}] n={n} mel={tuple(mel.shape)} H={H} L={L} O={O}", flush=True)
    _, wall_w = prepare_weights(model_dir)
    report = {"model": tag, "n": n, "wav": os.path.relpath(WAV, HERE), "H": H, "L": L,
              "O": O, "rows": 13 * n, "quant_mel": quant_mel, "results": []}
    for layers in layer_counts:
        if layers is not None and layers >= L:
            continue
        ref_model.layers = all_layers if layers is None else all_layers[:layers]
        t = time.time()
        with torch.no_grad():
            ref = reference(ref_model, mel_ref)
        w = wall_w if layers is None else slice_layers(wall_w, layers)
        ax.init(chip_num=1, core_num=4)
        kernel, _ = build_kernel(cfg, n, layers=layers, run_mode="PYRUN")
        out = run_forward(kernel, w, mel, n, O)
        R = 13 * n
        cos = row_cos(out[:R], ref).cpu()
        rel = ((out[:R].double() - ref.double()).norm() / ref.double().norm()).item()
        rec = {"layers": L if layers is None else layers,
               "cos_mean": cos.mean().item(), "cos_min": cos.min().item(),
               "cos_p1": torch.quantile(cos, 0.01).item(),
               "rel_l2": rel, "seconds": time.time() - t,
               "first_rows": [round(x, 8) for x in cos[: min(13, R)].tolist()]}
        report["results"].append(rec)
        print(f"[{tag}] n={n} layers={rec['layers']}: cos mean {rec['cos_mean']:.8f} "
              f"min {rec['cos_min']:.8f} p1 {rec['cos_p1']:.8f} rel {rel:.5f} "
              f"({rec['seconds']:.1f}s)", flush=True)
        if save:
            report.setdefault("arrays", {})[str(rec["layers"])] = {
                "out": out[:R].cpu().numpy(), "ref": ref.cpu().numpy()}
    ref_model.layers = all_layers
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="0.6b", help="0.6b, 1.7b or both")
    ap.add_argument("--n", type=int, action="append", default=None)
    ap.add_argument("--layers", default="1,4,all")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--quant-mel", dest="quant_mel", action="store_true", default=True,
                    help="feed the reference the same bf16 mel the kernel receives (default)")
    ap.add_argument("--fp32-mel", dest="quant_mel", action="store_false",
                    help="feed the reference the unquantized fp32 mel")
    ap.add_argument("--out", default=os.path.join(HERE, "output", "check_asr.json"))
    args = ap.parse_args()
    tags = ["0.6b", "1.7b"] if args.tag == "both" else [args.tag]
    ns = args.n or [1, 8]
    layer_counts = [None if x == "all" else int(x) for x in args.layers.split(",")]
    reports = []
    for tag in tags:
        for n in ns:
            reports.append(compare(tag, MODELS[tag], n, layer_counts, args.save, args.quant_mel))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(reports, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
