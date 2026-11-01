"""Pack the mmproj GGUF codec tensors into the kernel's three BF16 banks.

The GGUF stores every decode-path tensor in the same logical orientation as the
HF checkpoint (ggml data is contiguous in ne order, ne0 fastest).  Two things
are pre-folded by the converter and are consumed directly by decoder_kernel.py:

  codebook = embedding_sum / cluster_usage
  snake alpha = exp(alpha_raw), snake beta = 1 / (exp(beta_raw) + 1e-9)

The usage bank tensors are therefore constants (1.0), kept only so the byte
layout is identical to weight_layout.py and pack_weights.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import BF16

from weight_layout import LAYOUT

ROOT = Path(__file__).resolve().parent
GGUF = ROOT / "gguf" / "mmproj-qwen3-tts-0.6b-bf16.gguf"
OUT = ROOT / "artifacts" / "gguf_banks"
PREFIX = "a.gen.wav."


def read_tensor(reader, name):
    """Return a float32 torch tensor in logical [dimN, ..., ne0] order."""
    t = next((t for t in reader.tensors if t.name == PREFIX + name), None)
    if t is None:
        raise KeyError(PREFIX + name)
    if t.tensor_type.name == "BF16":
        a = BF16.dequantize(t.data)
    elif t.tensor_type.name in ("F16", "F32"):
        a = np.asarray(t.data)
    else:
        raise TypeError(f"{t.name}: unsupported {t.tensor_type.name}")
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True)), t


def conv(w):
    # GGUF [OC, IC, K] -> bank [OC, K*IC].
    return w.permute(0, 2, 1).contiguous().reshape(w.shape[0], w.shape[2] * w.shape[1])


def trans(w, stride):
    # GGUF [IC, OC, K] -> bank [S, OC, J*IC]; phase p uses taps p + j*S.
    k = w.shape[2]
    taps = [[p + j * stride for j in range((k - p + stride - 1) // stride)] for p in range(stride)]
    jmax = max(len(x) for x in taps)
    rows = []
    for phase in taps:
        parts = [w[:, :, t] for t in phase]
        parts += [torch.zeros_like(parts[0]) for _ in range(jmax - len(parts))]
        rows.append(torch.cat(parts, dim=0).transpose(0, 1))
    return torch.stack(rows).contiguous()


def dw_taps(w, taps):
    # GGUF [C, 1, K] -> bank [K, T, C], each tap repeated on every time row.
    w = w[:, 0, :]
    return torch.stack([w[:, k].reshape(1, -1).expand(taps, -1) for k in range(w.shape[1])])


def snake_alpha(w):
    # GGUF already stores exp(alpha_raw).
    return w


def snake_beta(w):
    # GGUF already stores 1 / (exp(beta_raw) + 1e-9).
    return w


def main():
    reader = GGUFReader(str(GGUF))
    OUT.mkdir(parents=True, exist_ok=True)
    p, info = {}, {}

    def bf(x):
        return x.to(torch.bfloat16).contiguous()

    def add_value(key, value, sources, transform):
        value = bf(value)
        want = tuple(LAYOUT[group_for(key)][key]["shape"])
        assert tuple(value.shape) == want, (key, tuple(value.shape), want)
        sources = [sources] if isinstance(sources, str) else list(sources)
        tensors = [next(t for t in reader.tensors if t.name == PREFIX + s) for s in sources]
        types = sorted({t.tensor_type.name for t in tensors})
        nes = sorted({tuple(int(x) for x in t.shape) for t in tensors})
        p[key] = value
        info[key] = {
            "source": [PREFIX + s for s in sources],
            "source_type": (types[0] if len(types) == 1 else types) if tensors else None,
            "source_ne": (([list(x) for x in nes] if len(nes) > 1 else list(nes[0]))
                          if tensors else None),
            "transform": transform,
        }

    def add(key, source, transform, fn=None):
        value, _ = read_tensor(reader, source)
        if fn is not None:
            value = fn(value)
        add_value(key, value, source, transform)

    # Quantizer: codebooks are already folded.
    add("q_sem_embed", "quant.first.codebook.weight", "codebook[o,i] = gguf[o,i]")
    add_value("q_sem_usage", torch.ones(2048), [], "constant 1.0 (unused)")
    add("q_sem_out", "quant.first.out_proj.weight", "bank[o,i] = gguf[o,i]")
    add("q_ac_embed", "quant.rest.codebook.weight", "bank[g,o,i] = gguf[g,o,i]")
    add_value("q_ac_usage", torch.ones(15, 2048), [], "constant 1.0 (unused)")
    add("q_ac_out", "quant.rest.out_proj.weight", "bank[o,i] = gguf[o,i]")

    add("pre_conv_w", "pre_conv.weight", "conv: bank[o,k*IC+i] = gguf[o,i,k]", conv)
    add("pre_conv_b", "pre_conv.bias", "bank[c] = gguf[c]")
    add("tfm_in_w", "tfm.in_proj.weight", "bank[o,i] = gguf[o,i]")
    add("tfm_in_b", "tfm.in_proj.bias", "bank[c] = gguf[c]")
    add("tfm_out_w", "tfm.out_proj.weight", "bank[o,i] = gguf[o,i]")
    add("tfm_out_b", "tfm.out_proj.bias", "bank[c] = gguf[c]")
    add("tfm_norm", "tfm.output_norm.weight", "bank[c] = gguf[c]")

    tfm = {
        "ln1": "ln1.weight", "q": "attn_q.weight", "k": "attn_k.weight",
        "v": "attn_v.weight", "o": "attn_out.weight", "attn_scale": "ls1.weight",
        "ln2": "ln2.weight", "gate": "ffn_gate.weight", "up": "ffn_up.weight",
        "down": "ffn_down.weight", "ffn_scale": "ls2.weight",
    }
    for key, suffix in tfm.items():
        src = [f"tfm.blk.{i}.{suffix}" for i in range(8)]
        vals = [read_tensor(reader, s)[0] for s in src]
        add_value("tfm_" + key, torch.stack(vals), src, f"bank[l,...] = gguf.blk[l].{suffix}")

    for u in range(2):
        pre = f"up.blk.{u}"
        add(f"up{u}_tc_w", pre + ".conv.weight", "trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S]", lambda w: trans(w, 2))
        add(f"up{u}_tc_b", pre + ".conv.bias", "bank[c] = gguf[c]")
        taps = 144 * (u + 1)
        add(f"up{u}_dw_taps", pre + ".dwconv.weight",
            "dw: bank[k,t,c] = gguf[c,0,k] (same for all t)", lambda w: dw_taps(w, taps))
        add(f"up{u}_dw_b", pre + ".dwconv.bias", "bank[c] = gguf[c]")
        add(f"up{u}_norm_w", pre + ".norm.weight", "bank[c] = gguf[c]")
        add(f"up{u}_norm_b", pre + ".norm.bias", "bank[c] = gguf[c]")
        add(f"up{u}_pw1_w", pre + ".pw1.weight", "bank[o,i] = gguf[o,i]")
        add(f"up{u}_pw1_b", pre + ".pw1.bias", "bank[c] = gguf[c]")
        add(f"up{u}_pw2_w", pre + ".pw2.weight", "bank[o,i] = gguf[o,i]")
        add(f"up{u}_pw2_b", pre + ".pw2.bias", "bank[c] = gguf[c]")
        add(f"up{u}_gamma", pre + ".gamma", "bank[c] = gguf[c]")

    add("dac_entry_w", "dac.entry.weight", "conv: bank[o,k*IC+i] = gguf[o,i,k]", conv)
    add("dac_entry_b", "dac.entry.bias", "bank[c] = gguf[c]")
    strides = (8, 5, 4, 3)
    for bid, stride in enumerate(strides):
        pre = f"dac.blk.{bid}"
        add(f"dac{bid}_snake_a", pre + ".snake.alpha", "bank[c] = gguf[c] (already exp)", snake_alpha)
        add(f"dac{bid}_snake_b", pre + ".snake.beta", "bank[c] = gguf[c] (already reciprocal)", snake_beta)
        add(f"dac{bid}_tc_w", pre + ".conv.weight",
            "trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S]", lambda w, s=stride: trans(w, s))
        add(f"dac{bid}_tc_b", pre + ".conv.bias", "bank[c] = gguf[c]")
        for r in range(3):
            rp = f"{pre}.res.{r}"
            add(f"dac{bid}_r{r}_a1", rp + ".act1.alpha", "bank[c] = gguf[c] (already exp)", snake_alpha)
            add(f"dac{bid}_r{r}_b1", rp + ".act1.beta", "bank[c] = gguf[c] (already reciprocal)", snake_beta)
            add(f"dac{bid}_r{r}_c1w", rp + ".conv1.weight", "conv: bank[o,k*IC+i] = gguf[o,i,k]", conv)
            add(f"dac{bid}_r{r}_c1b", rp + ".conv1.bias", "bank[c] = gguf[c]")
            add(f"dac{bid}_r{r}_a2", rp + ".act2.alpha", "bank[c] = gguf[c] (already exp)", snake_alpha)
            add(f"dac{bid}_r{r}_b2", rp + ".act2.beta", "bank[c] = gguf[c] (already reciprocal)", snake_beta)
            add(f"dac{bid}_r{r}_c2w", rp + ".conv2.weight", "conv: bank[o,k*IC+i] = gguf[o,i,k]", conv)
            add(f"dac{bid}_r{r}_c2b", rp + ".conv2.bias", "bank[c] = gguf[c]")
    add("post_snake_a", "dac.post_snake.alpha", "bank[c] = gguf[c] (already exp)", snake_alpha)
    add("post_snake_b", "dac.post_snake.beta", "bank[c] = gguf[c] (already reciprocal)", snake_beta)
    add("post_conv_w", "dac.post_conv.weight", "conv: bank[o,k*IC+i] = gguf[o,i,k]", conv)
    add("post_conv_b", "dac.post_conv.bias", "bank[c] = gguf[c]")

    groups = {
        "quant_w": lambda k: k.startswith(("q_", "pre_conv", "tfm_")),
        "up_w": lambda k: k.startswith("up"),
        "dac_w": lambda k: k.startswith(("dac", "post_")),
    }
    layout = {}
    md5_lines = []
    manifest = {}
    total = 0
    for group, pred in groups.items():
        names = [k for k in p if pred(k)]
        flat = torch.cat([p[k].reshape(-1) for k in names]).contiguous()
        off = 0
        for k in names:
            layout.setdefault(group, {})[k] = {"offset": off, "shape": list(p[k].shape)}
            off += p[k].numel()
        assert layout[group] == LAYOUT[group], f"{group} layout changed"
        assert flat.dtype == torch.bfloat16
        data = flat.view(torch.uint8).numpy().tobytes()
        path = OUT / f"{group}.bin"
        path.write_bytes(data)
        md5 = hashlib.md5(data).hexdigest()
        md5_lines.append(f"{md5}  {path.name}\n")
        total += len(data)
        manifest[group] = {
            "bytes": len(data), "md5": md5,
            "tensors": {k: {**info[k], "offset": layout[group][k]["offset"],
                            "shape": layout[group][k]["shape"]} for k in names},
        }
    (OUT / "md5.txt").write_text("".join(md5_lines))
    (OUT / "manifest.json").write_text(json.dumps({
        "gguf": GGUF.name,
        "layout": "weight_layout.py",
        "dtype": "bfloat16",
        "groups": manifest,
    }, indent=2) + "\n")
    print(f"{len(p)} tensors, {total/2**20:.2f} MiB -> {OUT}")


def group_for(key):
    if key.startswith(("q_", "pre_conv", "tfm_")):
        return "quant_w"
    if key.startswith("up"):
        return "up_w"
    return "dac_w"


if __name__ == "__main__":
    main()
