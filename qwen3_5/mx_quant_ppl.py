"""Which MXFP8 exponent rule loses least? Fake-quantize every projection of Qwen3.5-0.8B (and the lm head), then the
transformers PPL on the same 900-token texts as the board runs.

  ceil : e = ceil(log2(amax / 448))          never saturates (what the first board run used)
  ocp  : e = floor(log2(amax)) - 8           OCP MX spec, values above 448 saturate
  best : per group, the lower squared error of ceil and ceil - 1

Usage: MODEL=<hf dir> python mx_quant_ppl.py s_zh.ids s_en.ids
"""
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = os.environ["MODEL"]
G = 32


def q(w, e):
    s = torch.exp2(e)
    return ((w / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s)


def fake(w, rule):
    g = w.float().reshape(w.shape[0], -1, G)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(2.0**-126)
    if rule == "ocp":
        return q(g, torch.floor(torch.log2(amax)) - 8).reshape(w.shape)
    e = torch.ceil(torch.log2(amax / 448.0))
    a = q(g, e)
    if rule == "ceil":
        return a.reshape(w.shape)
    b = q(g, e - 1)
    pick = ((b - g) ** 2).sum(-1, keepdim=True) < ((a - g) ** 2).sum(-1, keepdim=True)
    return torch.where(pick, b, a).reshape(w.shape)


def ppl(model, ids):
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0].float()
    return math.exp(torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(ids[1:])).item())


GROUPS = {
    "lm_head/embed": lambda n: "embed_tokens" in n or "lm_head" in n,
    "mlp": lambda n: ".mlp." in n,
    "gdn proj": lambda n: ".linear_attn." in n,
    "attn proj": lambda n: ".self_attn." in n,
}


def main():
    texts = [np.fromfile(p, dtype=np.int32).tolist() for p in sys.argv[1:]]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    orig = {n: p.data.clone() for n, p in model.named_parameters() if p.dim() == 2 and p.shape[1] % G == 0}
    params = dict(model.named_parameters())
    print("bf16 ", " ".join(f"{ppl(model, t):.4f}" for t in texts), flush=True)
    if os.environ.get("ABLATE"):
        # quantize one group at a time (the rest bf16), then everything but one group
        for name, sel in GROUPS.items():
            for mode in ("only", "all but"):
                for n in orig:
                    use = sel(n) if mode == "only" else not sel(n)
                    params[n].data.copy_(fake(orig[n], "ceil").bfloat16() if use else orig[n])
                print(f"{mode} {name:14s}", " ".join(f"{ppl(model, t):.4f}" for t in texts), flush=True)
        return
    for rule in ("ceil", "ocp", "best"):
        for n in orig:
            params[n].data.copy_(fake(orig[n], rule).bfloat16())
        print(f"{rule:5s}", " ".join(f"{ppl(model, t):.4f}" for t in texts), flush=True)


if __name__ == "__main__":
    main()
