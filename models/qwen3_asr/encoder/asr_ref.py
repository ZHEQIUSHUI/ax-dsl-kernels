"""Qwen3-ASR audio-tower reference: the official qwen-asr transformers_backend
modeling file, loaded with dequantized (bf16 checkpoint -> fp32) weights.

The pip module is imported without executing qwen_asr/__init__.py (which pulls
in inference-only dependencies) by installing the three package modules in
sys.modules first.  transformers 5.12's check_model_inputs decorator is stubbed
to the identity because the wheel targets its 4.x calling convention.
"""
from __future__ import annotations

import glob
import importlib
import importlib.machinery
import json
import os
import struct
import sys
import types

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
PIP_SRC = os.path.join(HERE, "pip_pkg", "src")
AUDIO_PREFIX = "thinker.audio_tower."


def _install_stub_packages():
    root = os.path.join(PIP_SRC, "qwen_asr")
    for name, path in (
        ("qwen_asr", root),
        ("qwen_asr.core", os.path.join(root, "core")),
        ("qwen_asr.core.transformers_backend", os.path.join(root, "core", "transformers_backend")),
    ):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__package__ = name
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        sys.modules[name] = mod


def import_modeling():
    _install_stub_packages()
    import transformers

    def _identity(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

    transformers.utils.generic.check_model_inputs = _identity
    transformers.check_model_inputs = _identity
    modeling = importlib.import_module("qwen_asr.core.transformers_backend.modeling_qwen3_asr")
    config = importlib.import_module("qwen_asr.core.transformers_backend.configuration_qwen3_asr")
    return modeling, config


def audio_config(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["thinker_config"]["audio_config"]
    _, config_mod = import_modeling()
    names = set(config_mod.Qwen3ASRAudioEncoderConfig.__init__.__code__.co_varnames)
    return config_mod.Qwen3ASRAudioEncoderConfig(**{k: v for k, v in cfg.items() if k in names})


def _safetensor_files(model_dir):
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        entries = json.load(open(index))["weight_map"].values()
        return sorted({os.path.join(model_dir, name) for name in entries})
    one = os.path.join(model_dir, "model.safetensors")
    return [one] if os.path.exists(one) else sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))


def _partial_tensors(path, prefix):
    """The prefix tensors whose bytes are already in a partially downloaded file.

    safetensors refuses a truncated file outright, but the format's header has the
    offsets.  Qwen3-ASR keeps the whole audio tower in the first ~600 MB of its
    first shard, so a download in progress can still supply the reference weights.
    """
    import ml_dtypes

    dtypes = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32,
              "F64": np.float64, "I64": np.int64, "I32": np.int32, "I16": np.int16,
              "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
    out = {}
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        data0 = 8 + hlen
        for key, info in header.items():
            if key == "__metadata__" or not key.startswith(prefix):
                continue
            b, e = info["data_offsets"]
            if data0 + e > size:
                continue
            fh.seek(data0 + b)
            arr = np.frombuffer(fh.read(e - b), dtype=dtypes[info["dtype"]]).reshape(info["shape"])
            if info["dtype"] == "BF16":
                t = torch.from_numpy(arr.astype(np.float32).copy()).to(torch.bfloat16)
            else:
                t = torch.from_numpy(arr.copy())
            out[key[len(prefix):]] = t
    return out


def audio_state_dict(model_dir):
    """stripped thinker.audio_tower.* tensors, still in their checkpoint dtype"""
    out = {}
    for path in _safetensor_files(model_dir):
        try:
            with safe_open(path, "pt") as fh:
                for key in fh.keys():
                    if key.startswith(AUDIO_PREFIX):
                        out[key[len(AUDIO_PREFIX):]] = fh.get_tensor(key)
        except Exception:
            out.update(_partial_tensors(path, AUDIO_PREFIX))
    if not out:
        raise RuntimeError(f"no {AUDIO_PREFIX}* tensors in {model_dir}")
    return out


def build_encoder(model_dir, layers=None, dtype=torch.float32):
    modeling, _ = import_modeling()
    cfg = audio_config(model_dir)
    model = modeling.Qwen3ASRAudioEncoder(cfg)
    sd = {k: v.to(dtype) for k, v in audio_state_dict(model_dir).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    hard_missing = [k for k in missing if not k.endswith("positional_embedding")]
    if hard_missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={hard_missing[:8]} unexpected={unexpected[:8]}")
    model = model.to(dtype).eval()
    if layers is not None:
        model.layers = model.layers[:layers]
    return model, cfg


@torch.no_grad()
def reference(model, mel_time_major, layers=None):
    """mel [T,128] fp32 (time-major) -> [13*nchunks, output_dim] fp32"""
    T = mel_time_major.shape[0]
    assert T % 100 == 0
    out = model(mel_time_major.T.contiguous(), feature_lens=torch.tensor([T]))
    return out.last_hidden_state.float()


def sinusoids(length, channels, max_timescale=10000.0):
    log_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv = np.exp(-log_increment * np.arange(channels // 2, dtype=np.float64))
    scaled = np.arange(length, dtype=np.float64)[:, None] * inv[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)


def load_wav_16k(path, seconds):
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, raw = wavfile.read(path)
    x = raw.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        x /= float(np.iinfo(raw.dtype).max)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    need = int(round(seconds * 16000))
    if x.shape[0] < need:
        reps = int(np.ceil(need / x.shape[0]))
        x = np.tile(x, reps)
    return np.ascontiguousarray(x[:need])


def extract_mel(wav_path, frames, model_dir):
    """real speech -> [frames,128] fp32 time-major, WhisperFeatureExtractor settings"""
    from transformers import WhisperFeatureExtractor

    wav = load_wav_16k(wav_path, frames * 160 / 16000)
    fe = WhisperFeatureExtractor.from_pretrained(model_dir)
    feats = fe(wav, sampling_rate=16000, padding="do_not_pad", return_tensors="np")["input_features"][0]
    assert feats.shape == (128, frames), feats.shape
    return torch.from_numpy(feats.T.copy()).float()


def row_cos(a, b):
    a = a.double()
    b = b.double()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--wav", default=os.path.join(HERE, "audio", "asr_en.wav"))
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()
    model, cfg = build_encoder(args.model_dir, layers=args.layers)
    mel = extract_mel(args.wav, args.frames, args.model_dir)
    with torch.no_grad():
        y = reference(model, mel)
    print("cfg", cfg.d_model, cfg.encoder_layers, cfg.encoder_attention_heads, cfg.encoder_ffn_dim,
          cfg.output_dim, cfg.n_window, cfg.n_window_infer)
    print("mel", tuple(mel.shape), "out", tuple(y.shape), "absmean", y.abs().mean().item())
