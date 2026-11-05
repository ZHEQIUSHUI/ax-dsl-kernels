"""Save BF16 HF references after each four-layer block for the current real-weight check.

usage: python ref_decode_sgl35.py <hf_dir> <out_dir> [prompt]

Writes:
  x.bin        [T,2048] bf16 token embeddings
  ref.bin      [T,10,2048] f32 residual stream after layers 3,7,...,39
  final.bin    [T,2048] f32 residual stream after layer 39
  ids.json     token ids and prompt
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel as Qwen3_5TextModel

hf = Path(sys.argv[1])
out = Path(sys.argv[2])
prompt = sys.argv[3] if len(sys.argv) > 3 else "中国的首都是北京。请用一句话介绍杭州。"
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(32)

cfg = AutoConfig.from_pretrained(hf).text_config
NL = int(os.environ.get("NL", "40"))
cfg.num_hidden_layers = NL
cfg.layer_types = cfg.layer_types[:NL]
cfg._attn_implementation = "eager"
model = Qwen3_5TextModel(cfg).float().eval()
wm = json.load(open(hf / "model.safetensors.index.json"))["weight_map"]
sd = {}
for k, f in wm.items():
    if not k.startswith("model.language_model."):
        continue
    n = k[len("model.language_model."):]
    if n.startswith("layers.") and int(n.split(".")[1]) >= NL:
        continue
    with safe_open(hf / f, "pt") as fh:
        sd[n] = fh.get_tensor(k).float()
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing", len(missing), "unexpected", len(unexpected), flush=True)
del sd

ids = AutoTokenizer.from_pretrained(hf)(prompt)["input_ids"]
print("ids", ids, flush=True)
x = model.embed_tokens(torch.tensor([ids]))
x.bfloat16().view(torch.int16).numpy().tofile(out / "x.bin")
json.dump({"prompt": prompt, "ids": ids}, open(out / "ids.json", "w"), indent=1)

blocks = {}
for i in range(3, NL, 4):
    model.layers[i].register_forward_hook(
        lambda m, args, value, i=i: blocks.setdefault(i, value[0] if isinstance(value, tuple) else value)
    )
with torch.no_grad():
    model(inputs_embeds=x.float())

refs = torch.stack([blocks[i][0] for i in range(3, NL, 4)])
refs.float().contiguous().numpy().tofile(out / "ref.bin")
blocks[NL-1][0].float().contiguous().numpy().tofile(out / "final.bin")
print("T", len(ids), "block rms", [round(float(v.pow(2).mean(-1).sqrt().mean()), 5) for v in refs], flush=True)
