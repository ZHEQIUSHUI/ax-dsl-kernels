"""Prepare the on-board Qwen3.5-0.8B run: per-layer weights as raw kernel inputs, per-step embeddings and RoPE.

Layout (board_q35/): layers.txt (one 'gdn'/'attn' per line), L<i>/<input>.bin, x.bin [T, H] bf16,
cos.bin / sin.bin [T, ROT/2] bf16, meta.txt "T LMAX BC H". Tokens come from a q35_step.py result (teacher forced).
"""
import os, sys
import torch
import q35_step as q

res = torch.load(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
W = q.Weights(q.MODEL)


def dump(t, path):
    t = t.contiguous()
    (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tofile(path)


types = []
for i, t in enumerate(q.LAYERS):
    a = q.gdn_args(W, i) if t == "linear_attention" else q.attn_args(W, i)
    types.append("gdn" if t == "linear_attention" else "attn")
    d = f"{out}/L{i:02d}"
    os.makedirs(d, exist_ok=True)
    for k, v in a.items():
        dump(v, f"{d}/{k}.bin")
open(f"{out}/layers.txt", "w").write("\n".join(types) + "\n")
T = res["npu_logits"].shape[0]
toks = res["tokens"][:T]
emb = W("embed_tokens.weight")
dump(emb[toks], f"{out}/x.bin")
cs = [q.rope_cos_sin(p) for p in range(T)]
dump(torch.cat([c for c, _ in cs]), f"{out}/cos.bin")
dump(torch.cat([s for _, s in cs]), f"{out}/sin.bin")
open(f"{out}/meta.txt", "w").write(f"{T} {q.LMAX} {q.BC} {q.H}\n")
print("prepared", out, "T", T, "LMAX", q.LMAX, "BC", q.BC)
