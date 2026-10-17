"""Check an on-board q35_chat run against transformers.

Teacher-forces the board's prompt + generated ids through transformers and compares the argmax at every generated
position; also runs transformers' own greedy generation from the prompt and compares the text.

Usage: python q35_chat_check.py prompt.ids out.ids
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")

prompt = np.fromfile(sys.argv[1], dtype=np.int32).tolist()
ids = np.fromfile(sys.argv[2], dtype=np.int32).tolist()
assert ids[: len(prompt)] == prompt
gen = ids[len(prompt):]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0].float()
    pred = logits[len(prompt) - 1 : len(ids) - 1].argmax(-1).tolist()
    match = sum(int(a == b) for a, b in zip(pred, gen))
    first_diff = next((i for i, (a, b) in enumerate(zip(pred, gen)) if a != b), None)
    ref = model.generate(torch.tensor([prompt]), max_new_tokens=len(gen), do_sample=False)[0, len(prompt):].tolist()
print(f"board text: {tok.decode(gen)!r}")
print(f"HF    text: {tok.decode(ref)!r}")
gap = ""
if first_diff is not None:
    top2 = logits[len(prompt) - 1 + first_diff].topk(2)
    gap = f" first_diff_at={first_diff} HF top2 gap there={float(top2.values[0] - top2.values[1]):.3f}"
print(f"RESULT gen={len(gen)} teacher_forced_top1={match}/{len(gen)} same_as_hf_greedy={gen == ref}{gap}")
