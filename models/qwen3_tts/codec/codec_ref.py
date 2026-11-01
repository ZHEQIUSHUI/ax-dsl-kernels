"""Reference data + adapter-exact streaming decode for Qwen3-TTS-Tokenizer-12Hz.

Uses the upstream Qwen package copied into ref_src (Apache-2.0), with two compatibility
edits for the installed transformers 5.12: the removed `default` RoPE alias and the
changed `check_model_inputs` decorator are bypassed; the math is unchanged.

The kernel-side contract is one 72-frame window.  The graph below is the same contract
as llama.cpp-axera's tools/mtmd/models/qwen3tts-gen.cpp: a 71-frame transformer KV
prefix, causal-conv tails, and the transposed-conv overlap tail.  `chunked_decode`
with chunk_size=72, left_context_size=71 is mathematically the same because every
remaining receptive field is at most 72 frames / 54 samples.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref_src"))

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Config,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Model,
)

WINDOW = 72
UP = 1920
SAMPLES = WINDOW * UP
HF = ROOT / "hf"
ART = ROOT / "artifacts"
WAV = Path("/data/junda/work/qsmall/asrenc/audio/asr_en.wav")


def load_model(bf16: bool = True, device: str = "cpu"):
    cfg = Qwen3TTSTokenizerV2Config.from_pretrained(HF)
    model = Qwen3TTSTokenizerV2Model(cfg).eval()
    from safetensors.torch import load_file

    state = load_file(HF / "model.safetensors", device=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # rotary inv_freq and the quantizer's `initialized` flags are non-persistent / buffers.
    bad_missing = [k for k in missing if "rotary_emb" not in k and not k.endswith("initialized")]
    if bad_missing or unexpected:
        raise RuntimeError(f"state mismatch missing={bad_missing[:8]} unexpected={unexpected[:8]}")
    model = model.to(device)
    if bf16:
        model = model.to(torch.bfloat16)
    return model, cfg


def read_wav(path: Path = WAV, target_sr: int = 24000) -> torch.Tensor:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype == np.int32:
        x = x.astype(np.float32) / np.float32(2**31)
    elif x.dtype == np.int16:
        x = x.astype(np.float32) / np.float32(2**15)
    else:
        x = x.astype(np.float32)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return torch.from_numpy(x)


def encode_codes(model, wav: torch.Tensor, origin: int = 0, frames: int = 288) -> torch.LongTensor:
    # encoder is 12.5 Hz; give it a 4 s crop at 24 kHz.  The first frames are context.
    start = max(0, origin)
    crop = wav[start : start + int((frames + 40) * 1920)].float()
    crop = crop[: (crop.numel() // 1920) * 1920]
    if next(model.parameters()).dtype == torch.bfloat16:
        crop = crop.bfloat16()
    with torch.no_grad():
        # model.encode's final trim expression is incompatible with torch 2.13's
        # tensor-as-slice-index rule; this is the same encoder call without that trim.
        out = model.encoder.encode(input_values=crop[None, None], return_dict=True)
    codes = out.audio_codes[0][: model.encoder_valid_num_quantizers].transpose(0, 1)  # [T, 16]
    return codes[40 : 40 + frames].to(torch.long)


def decoder_ref(model, codes: torch.Tensor) -> torch.Tensor:
    """Full decoder on [T,16] codes; returns [T*1920] bf16/f32 waveform."""
    with torch.no_grad():
        return model.decoder(codes.transpose(0, 1)[None])[0, 0]


def streaming_ref(model, codes: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.decoder.chunked_decode(codes.transpose(0, 1)[None], chunk_size=WINDOW, left_context_size=WINDOW - 1)[0, 0]


def capture_stages(model, codes: torch.Tensor):
    """Run the whole decoder once and return [T,C] stage tensors.

    The caller may pass 71 frames of left context followed by one 72-frame
    window; slice each returned tensor by `context * rate` on its time axis.
    """
    dec = model.decoder
    captured = {}
    handles = []

    def hook(name):
        def fn(_module, _inputs, output):
            if name == "tfm":
                output = output.last_hidden_state
            captured[name] = output.detach()
        return fn

    modules = {
        "quant": dec.quantizer,
        "pre_conv": dec.pre_conv,
        "tfm": dec.pre_transformer,
        "up0": dec.upsample[0][1],
        "up1": dec.upsample[1][1],
        "dac0": dec.decoder[0],
        "dac1": dec.decoder[1],
        "dac2": dec.decoder[2],
        "dac3": dec.decoder[3],
        "dac4": dec.decoder[4],
        "post_snake": dec.decoder[5],
        "post_conv": dec.decoder[6],
    }
    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(hook(name)))
    try:
        wav = decoder_ref(model, codes)
    finally:
        for h in handles:
            h.remove()
    with torch.no_grad():
        q = dec.quantizer.decode(codes.transpose(0, 1)[None])[0].transpose(0, 1)
    captured["quant"] = q
    out = {}
    for name, x in captured.items():
        # Every hook output is [1,C,T] except pre_transformer [1,T,C].
        if name == "quant":
            out[name] = x.float()
        elif name == "tfm":
            out[name] = x[0].float()
        else:
            out[name] = x[0].transpose(0, 1).contiguous().float()
    out["wav"] = wav.float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=288, help="codes to encode (5+ windows)")
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    model, cfg = load_model(bf16=not args.fp32)
    wav = read_wav()
    print(f"wav {wav.numel()/24000:.3f}s peak={wav.abs().max():.4f}")
    codes = encode_codes(model, wav, frames=args.frames)
    print("codes", tuple(codes.shape), "range", int(codes.min()), int(codes.max()))
    full = decoder_ref(model, codes)
    stream = streaming_ref(model, codes)
    n = min(full.numel(), stream.numel())
    cos = F.cosine_similarity(full[:n].float(), stream[:n].float(), dim=0).item()
    snr = 10 * torch.log10(full[:n].float().square().sum() / (full[:n].float() - stream[:n].float()).square().sum().clamp_min(1e-30)).item()
    print(f"full vs chunked: n={n} cosine={cos:.9f} snr={snr:.3f}dB")
    torch.save({"codes": codes, "wav": wav, "decoded": full.float(), "chunked": stream.float()}, Path(args.out) / "reference.pt")
    (Path(args.out) / "reference.json").write_text(json.dumps({
        "window": WINDOW,
        "samples_per_window": SAMPLES,
        "codes_shape": list(codes.shape),
        "decoded_samples": int(full.numel()),
        "chunked_vs_full_cosine": cos,
        "chunked_vs_full_snr_db": snr,
        "bf16": not args.fp32,
        "source_wav": str(WAV),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
