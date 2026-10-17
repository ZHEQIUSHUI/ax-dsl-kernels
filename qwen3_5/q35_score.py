"""Perplexity check for the on-board generator (q35_chat with max_new_tokens = 0).

  prep <text file> <ids out> [n_tokens]      plain text -> token ids (no chat template), first n_tokens (900)
  ref  <ids> <board nll f32> [more nll ...]  transformers NLL on the same ids; compares each board run with it

Usage: python q35_score.py prep zh.txt board_mx/s_zh.ids && python q35_score.py ref s_zh.ids bf16.nll mx.nll
"""
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if sys.argv[1] == "prep":
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 900
        ids = tok(open(sys.argv[2]).read())["input_ids"][:n]
        np.array(ids, dtype=np.int32).tofile(sys.argv[3])
        print(f"{len(ids)} tokens -> {sys.argv[3]}: {tok.decode(ids[:40])!r} ...")
        return
    ids = np.fromfile(sys.argv[2], dtype=np.int32).tolist()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0].float()
    ref = torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(ids[1:]), reduction="none")
    print(f"HF bf16      ppl={math.exp(ref.mean()):.4f}  (n={len(ref)})")
    for path in sys.argv[3:]:
        b = torch.from_numpy(np.fromfile(path, dtype=np.float32))
        assert len(b) == len(ref), (path, len(b), len(ref))
        d = b - ref
        print(f"{os.path.basename(path):12s} ppl={math.exp(b.mean()):.4f}  ppl/HF={math.exp(b.mean() - ref.mean()):.4f}  "
              f"mean|dNLL|={d.abs().mean():.4f}  max|dNLL|={d.abs().max():.3f}")


if __name__ == "__main__":
    main()
