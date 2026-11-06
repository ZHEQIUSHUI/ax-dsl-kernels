#!/usr/bin/env python3
"""Add the pf35 prefill roles to the gap35-rel256 release manifest."""
import json, sys
rel = json.load(open(sys.argv[1]))         # rel256 release manifest.json
iodir = sys.argv[2]                        # directory with <role>.io.json files
out = sys.argv[3]
roles = [
    ("prefill.gdn128",  "sgl35_gdn128.axmodel"),
    ("prefill.attn128", "sgl35_flash128.axmodel"),
    ("prefill.gdn3840", "sgl35_gdn3840.axmodel"),
    ("prefill.gdn4096", "sgl35_gdn4096.axmodel"),
    ("prefill.attn3840", "sgl35_flash3840.axmodel"),
    ("prefill.attn4096", "sgl35_flash4096.axmodel"),
    ("prefill.moev",    "sgl35_moe4096v.axmodel"),
]
for role, f in roles:
    d = json.load(open(f"{iodir}/{role}.io.json"))
    v = next(iter(d.values()))
    rel["kernels"][role] = {"file": f,
        "in":  [{"name": n, "bytes": b, "dtype": t, "shape": list(s)} for n, b, t, s in v["in"]],
        "out": [{"name": n, "bytes": b, "dtype": t, "shape": list(s)} for n, b, t, s in v["out"]]}
json.dump(rel, open(out, "w"), indent=1)
print("wrote", out, "roles", len(rel["kernels"]))
