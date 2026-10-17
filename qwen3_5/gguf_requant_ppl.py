"""PPL cost of re-quantizing a quantized GGUF (Q4_K_M, Q8_0, ...) to the NPU weight formats.

The GGUF's 2-D projections are dequantized (gguf-py) into the Hugging Face model, then optionally re-quantized:
  gguf   : as dequantized (what llama.cpp computes with)
  mxfp8  : E4M3 + E8M0 per 32 (exponent ceil(log2(amax/448)))
  mxfp4  : E2M1 + E8M0 per 32 (exponent ceil(log2(amax/6)), round to nearest on the E2M1 grid)
  nvfp4  : E2M1 + E4M3 per 16 + fp32 per tensor
and transformers computes the PPL of the token ids (same texts as the board runs).

Usage: MODEL=<hf dir> PYTHONPATH=<llama.cpp>/gguf-py python gguf_requant_ppl.py model.gguf s_zh.ids s_en.ids
"""
import math
import os
import sys

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM

MODEL = os.environ["MODEL"]
G = 32
# HF parameter suffix (under model.language_model.layers.<i>.) -> GGUF tensor suffix (under blk.<i>.)
NAMES = {"linear_attn.in_proj_qkv.weight": "attn_qkv.weight", "linear_attn.in_proj_z.weight": "attn_gate.weight",
         "linear_attn.in_proj_b.weight": "ssm_beta.weight", "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
         "linear_attn.out_proj.weight": "ssm_out.weight", "self_attn.q_proj.weight": "attn_q.weight",
         "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
         "self_attn.o_proj.weight": "attn_output.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
         "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight"}
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp8(w):
    g = w.float().reshape(w.shape[0], -1, G)
    e = torch.ceil(torch.log2(g.abs().amax(-1, keepdim=True).clamp_min(2.0**-126) / 448.0))
    return ((g / torch.exp2(e)).to(torch.float8_e4m3fn).float() * torch.exp2(e)).reshape(w.shape)


def mxfp4(w):
    g = w.float().reshape(w.shape[0], -1, G)
    e = torch.ceil(torch.log2(g.abs().amax(-1, keepdim=True).clamp_min(2.0**-126) / 6.0))
    x = (g / torch.exp2(e)).clamp(-6, 6)
    idx = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)  # nearest grid point (ties to the lower one)
    return (torch.sign(x) * E2M1[idx] * torch.exp2(e)).reshape(w.shape)


def nvfp4(w):
    """E2M1 + an E4M3 scale per 16 + one fp32 scale per tensor (NVFP4)."""
    g = w.float().reshape(w.shape[0], -1, 16)
    amax = g.abs().amax(-1, keepdim=True)
    s2 = (amax.max() / (6.0 * 448.0)).clamp_min(2.0**-126)  # per-tensor scale: block scales fit E4M3
    s = (amax / 6.0 / s2).to(torch.float8_e4m3fn).float().clamp_min(2.0**-9) * s2
    x = (g / s).clamp(-6, 6)
    idx = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    return (torch.sign(x) * E2M1[idx] * s).reshape(w.shape)


def ppl(model, ids):
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0].float()
    return math.exp(torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(ids[1:])).item())


def main():
    reader = GGUFReader(sys.argv[1])
    gg = {t.name: t for t in reader.tensors}
    texts = [np.fromfile(p, dtype=np.int32).tolist() for p in sys.argv[2:]]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    deq = {}
    for name, p in model.named_parameters():
        if ".layers." not in name:
            continue
        i = int(name.split(".layers.")[1].split(".")[0])
        suffix = name.split(f".layers.{i}.")[1]
        if suffix in NAMES:
            t = gg[f"blk.{i}.{NAMES[suffix]}"]
            w = torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32)).reshape(p.shape)
            deq[name] = w
    # the tied embedding / output projection
    t = gg["token_embd.weight"]
    emb_name = next(n for n, _ in model.named_parameters() if n.endswith("embed_tokens.weight"))
    emb = dict(model.named_parameters())[emb_name]
    deq[emb_name] = torch.from_numpy(
        dequantize(t.data, t.tensor_type).astype(np.float32)).reshape(emb.shape)
    params = dict(model.named_parameters())
    print(f"{os.path.basename(sys.argv[1])}: {len(deq)} tensors from the GGUF")
    tags = os.environ.get("FORMATS", "gguf,mxfp8,mxfp4,nvfp4").split(",")
    fns = {"gguf": lambda w: w, "mxfp8": mxfp8, "mxfp4": mxfp4, "nvfp4": nvfp4}
    for tag, fn in ((t, fns[t]) for t in tags):
        for n, w in deq.items():
            params[n].data.copy_(fn(w).bfloat16())
        print(f"{tag:6s}", " ".join(f"{ppl(model, ids):.4f}" for ids in texts), flush=True)


if __name__ == "__main__":
    main()
