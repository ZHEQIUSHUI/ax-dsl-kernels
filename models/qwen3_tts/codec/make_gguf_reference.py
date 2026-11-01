"""PyTorch streaming reference using exactly the GGUF-loaded codec weights.

The HF checkpoint is used only as the module container.  Every value that differs
between the HF BF16 path and the GGUF is replaced: F16-stored convs, folded
codebooks, and folded SnakeBeta alpha/beta.  BF16/F32 tensors are identical in
both paths after the BF16 cast (verified by check_gguf_banks.py).
"""
from __future__ import annotations

import json
import re
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from gguf.quants import BF16

from codec_ref import load_model, streaming_ref, decoder_ref, WINDOW, SAMPLES

ROOT = Path(__file__).resolve().parent
GGUF = ROOT / "gguf" / "mmproj-qwen3-tts-0.6b-bf16.gguf"
OUT = ROOT / "artifacts" / "gguf_banks"


def read(reader, name, dtype=torch.float32):
    t = next(t for t in reader.tensors if t.name == "a.gen.wav." + name)
    a = BF16.dequantize(t.data) if t.tensor_type.name == "BF16" else np.asarray(t.data)
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True)).to(dtype)


def snake_forward(self, x):
    alpha = self._gguf_alpha.reshape(1, -1, 1)
    beta_inv = self._gguf_beta.reshape(1, -1, 1)
    return x + beta_inv * torch.pow(torch.sin(x * alpha), 2)


def main():
    reader = GGUFReader(str(GGUF))
    model, _ = load_model(bf16=True)
    sd = model.state_dict()
    patched = 0

    # F16 conv kernels: every other source is already exact after BF16 conversion.
    for t in reader.tensors:
        if not t.name.startswith("a.gen.wav.") or t.tensor_type.name != "F16":
            continue
        s = t.name.removeprefix("a.gen.wav.")
        if s == "pre_conv.weight":
            key = "decoder.pre_conv.conv.weight"
        elif s == "up.blk.0.dwconv.weight":
            key = "decoder.upsample.0.1.dwconv.conv.weight"
        elif s == "up.blk.1.dwconv.weight":
            key = "decoder.upsample.1.1.dwconv.conv.weight"
        elif s == "dac.entry.weight":
            key = "decoder.decoder.0.conv.weight"
        elif s == "dac.post_conv.weight":
            key = "decoder.decoder.6.conv.weight"
        else:
            m = re.fullmatch(r"dac\.blk\.(\d)\.res\.(\d)\.conv([12])\.weight", s)
            assert m, s
            key = (f"decoder.decoder.{int(m.group(1)) + 1}.block."
                   f"{int(m.group(2)) + 2}.conv{m.group(3)}.conv.weight")
        value = read(reader, s, torch.bfloat16)
        assert sd[key].shape == value.shape, (key, sd[key].shape, value.shape)
        sd[key].copy_(value)
        patched += 1

    # GGUF codebooks are embedding_sum / cluster_usage, so use usage=1.
    key = "decoder.quantizer.rvq_first.vq.layers.0._codebook"
    sd[key + ".cluster_usage"].fill_(1)
    sd[key + ".embedding_sum"].copy_(read(reader, "quant.first.codebook.weight", torch.bfloat16))
    for g in range(15):
        key = f"decoder.quantizer.rvq_rest.vq.layers.{g}._codebook"
        sd[key + ".cluster_usage"].fill_(1)
        sd[key + ".embedding_sum"].copy_(read(reader, "quant.rest.codebook.weight", torch.bfloat16)[g])
    patched += 16

    # SnakeBeta: use the GGUF's exp(alpha) and 1/(exp(beta)+1e-9) directly.
    for name, mod in model.named_modules():
        if not (hasattr(mod, "alpha") and hasattr(mod, "beta")):
            continue
        if name == "decoder.decoder.5":
            continue
        m = re.fullmatch(r"decoder\.decoder\.([1-4])\.block\.([2-4])\.act([12])", name)
        if m is None:
            m = re.fullmatch(r"decoder\.decoder\.([1-4])\.block\.0", name)
            assert m, name
            b, r, act = int(m.group(1)) - 1, 0, None
        else:
            b, r, act = int(m.group(1)) - 1, int(m.group(2)), m.group(3)
        if r == 0:
            stem = f"dac.blk.{b}.snake"
        else:
            stem = f"dac.blk.{b}.res.{r - 2}.act{act}"
        mod._gguf_alpha = read(reader, stem + ".alpha", torch.bfloat16)
        mod._gguf_beta = read(reader, stem + ".beta", torch.bfloat16)
        mod.forward = types.MethodType(snake_forward, mod)
        patched += 2
    key = "decoder.decoder.5"
    mod = model.get_submodule(key)
    mod._gguf_alpha = read(reader, "dac.post_snake.alpha", torch.bfloat16)
    mod._gguf_beta = read(reader, "dac.post_snake.beta", torch.bfloat16)
    mod.forward = types.MethodType(snake_forward, mod)
    patched += 2

    old = torch.load(ROOT / "artifacts" / "reference.pt", weights_only=False)
    codes = old["codes"][: 2 * WINDOW]
    with torch.no_grad():
        chunked = streaming_ref(model, codes).float()
        full = decoder_ref(model, codes).float()
    n = min(chunked.numel(), full.numel())
    cos = F.cosine_similarity(chunked[:n], full[:n], dim=0).item()
    snr = (10 * torch.log10(full[:n].square().sum()
                             / (chunked[:n] - full[:n]).square().sum().clamp_min(1e-30))).item()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"codes": codes, "chunked": chunked, "decoded": full}, OUT / "reference_gguf.pt")
    (OUT / "reference_gguf.json").write_text(json.dumps({
        "weights": "GGUF-equivalent (F16 tensors, folded codebooks/snakes)",
        "windows": 2, "window": WINDOW, "samples_per_window": SAMPLES,
        "patched_entries": patched,
        "chunked_vs_full_cosine": cos,
        "chunked_vs_full_snr_db": snr,
    }, indent=2) + "\n")
    print(f"patched {patched} entries; chunked vs full cos={cos:.9f} snr={snr:.3f}dB")


if __name__ == "__main__":
    main()
