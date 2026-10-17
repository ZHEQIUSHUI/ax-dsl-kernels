"""Encode a user message with the Qwen3.5 chat template (thinking off) into prompt.ids (int32) for q35_chat.

Usage: python q35_tok.py "message" prompt.ids
"""
import os
import sys

import numpy as np
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")

tok = AutoTokenizer.from_pretrained(MODEL)
text = tok.apply_chat_template([{"role": "user", "content": sys.argv[1]}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok(text)["input_ids"]
np.array(ids, dtype=np.int32).tofile(sys.argv[2])
print(f"{len(ids)} tokens -> {sys.argv[2]}")
print(repr(text))
