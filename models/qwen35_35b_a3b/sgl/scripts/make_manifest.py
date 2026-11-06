"""Build manifest.json for the qwen35_moe_sgl adapter from board iojson output."""
import json, sys
from pathlib import Path

io = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
rel_path = sys.argv[3] if len(sys.argv) > 3 else "/data/junda/work/qsmall/g35b/pkg/Qwen3.5-35B-A3B/kernels/manifest.json"
rel = json.load(open(rel_path))

def entry(file, inout):
    v = io[file]
    e = {"file": file}
    for k in ("in", "out"):
        e[k] = [{"name": n, "bytes": b, "dtype": d, "shape": list(s)} for n, b, d, s in v[k]]
    return e

kernels = {
    "decode": entry("sgl35_decode.axmodel", 0),
    "prefill.gdn": entry("sgl35_gdn.axmodel", 0),
    "prefill.attn": entry("sgl35_flash.axmodel", 0),
    "prefill.moe": entry("sgl35_moe4096.axmodel", 0),
    "prefill.padding": entry("sgl35_padding.axmodel", 0),
    "lm_head": rel["kernels"]["lm_head"],
    "vit": rel["kernels"]["vit"],
}
for role, file in (
    ("prefill.gdn128", "sgl35_gdn128.axmodel"),
    ("prefill.attn128", "sgl35_flash128.axmodel"),
    ("prefill.gdn3840", "sgl35_gdn3840.axmodel"),
    ("prefill.gdn4096", "sgl35_gdn4096.axmodel"),
    ("prefill.attn3840", "sgl35_flash3840.axmodel"),
    ("prefill.attn4096", "sgl35_flash4096.axmodel"),
    ("prefill.moev", "sgl35_moe4096v.axmodel"),
):
    if file in io:
        kernels[role] = entry(file, 0)
man = {"format": 1, "model": "qwen35_moe_sgl", "context": 16384, "cores": 4,
       "params": {}, "kernels": kernels}
(out / "manifest.json").write_text(json.dumps(man, indent=1))
for r, k in kernels.items():
    print(r, k["file"], len(k["in"]), len(k["out"]))
