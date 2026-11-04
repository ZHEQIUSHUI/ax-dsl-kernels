"""Check the real-weight 40-layer sgl decode artifact on the CRUN simulator.

usage: RUN_MODE=CRUN python check_sgl35_decode_crun.py <pack_dir> <ref_dir> [steps]

The pack directory is written by pack_sgl35_current.py.  The reference
directory holds x.bin (BF16 embeddings) and ref.bin (FP32 final hidden states),
as written by ref_decode_sgl35.py.  The decode kernel is run one token at a
time from cell zero so its state and paged KV are exercised sequentially.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import axnn.dsl as ax
from sgl_kernel_axera.models.qwen36_35b_a3b.decode import prepare

torch.set_num_threads(int(os.environ.get("THREADS", "32")))
PACK, REF = Path(sys.argv[1]), Path(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
MAN = json.load(open(PACK / "manifest.json"))

DTYPE = {
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
    "torch.int32": torch.int32,
}


def load(name: str) -> torch.Tensor:
    item = MAN[name]
    if item["state"]:
        return torch.zeros(item["shape"], dtype=DTYPE[item["dtype"]])
    raw = np.memmap(PACK / f"{name}.bin", dtype=np.uint8, mode="r")
    return torch.from_numpy(np.array(raw)).view(DTYPE[item["dtype"]]).reshape(item["shape"])


bank = {name: load(name) for name in MAN}
NB = MAN["b0_attn_input_norm_w"]["shape"][0]  # blocks (bank row count)
state_names = ("state", "q_conv_state", "k_conv_state", "v_conv_state")
state_bank_names = {f"{n}_in" for n in state_names}
moe_names = sorted(n[len("b0_moe_") :] for n in MAN if n.startswith("b0_moe_"))
attn_names = sorted(n[len("b0_attn_") :] for n in MAN if n.startswith("b0_attn_"))
gqa_names = sorted(n[len("b3_attn_") :] for n in MAN if n.startswith("b3_attn_"))


def layer_weights(il: int):
    phase, blk = il % 4, il // 4
    p = f"b{phase}_attn_"
    att = {}
    for name in (gqa_names if phase == 3 else attn_names):
        if name in state_bank_names or name == "cos_sin_cache":
            continue
        att[name] = bank[p + name][blk]
    if phase < 3:
        ord_il = (blk * 3) + phase
        for name in "qkvzabo":
            att[f"w_{name}_scale2"] = att[f"w_{name}_scale2"][:1]
        for name in state_names:
            value = bank["b0_attn_" + name + "_in"][ord_il]
            att[name + "_in"] = value
            att[name + "_out"] = value
        att["w_pre"] = None
    else:
        att["cos_sin_cache"] = bank["b3_attn_cos_sin_cache"][0]
        att["cos"] = None
        att["sin"] = None
        att["partial_rotary_factor"] = 0.25
        for name in ("q", "k", "v", "o"):
            att[f"w_scale2_{name}"] = att[f"w_scale2_{name}"][:1]
            att[f"w_zeropoint_{name}"] = att[f"w_zeropoint_{name}"][:1]
    pm = f"b{phase}_moe_"
    moe = {}
    for name in moe_names:
        moe[name] = bank[pm + name][blk]
    for name in ("gate", "up"):
        for prefix in ("w_", "w_scale_", "w_scale2_"):
            moe[prefix + name] = moe[prefix + "gate"]
    for name in ("route", "shared_expert_mlp_gate", "shared_expert_mlp_up", "shared_expert_mlp_down"):
        key = "w_scale2_" + name
        moe[key] = moe[key][:1]
    moe.update(top_k=8, num_experts=256, norm_topk_prob=True, fused_gate_up=True, add_residual=True)
    return None if phase < 3 else bank["b3_kv"][blk], att, moe


layers = [layer_weights(i) for i in range(4 * NB)]
cache = PACK / "cache"
files = glob.glob(str(cache / "*.compiled.axmodel"))
if len(files) != 1:
    raise SystemExit(f"expected one compiled artifact in {cache}, got {files}")
key = Path(files[0]).name.split(".")[1]
print("cache key", key, flush=True)

ax.init(chip_num=1, core_num=4)
kernel, weights = prepare(layers, mrope_section=(11, 11, 10), cache_path=cache, run_mode="CRUN")
del layers
del bank
gc.collect()
try:
    print("computed cache key", kernel.gen_cache_key(), flush=True)
except Exception as exc:  # noqa: BLE001
    print("computed cache key unavailable:", exc, flush=True)
kernel.gen_cache_key = lambda: key

x = torch.from_numpy(np.array(np.memmap(REF / "x.bin", dtype=np.uint8, mode="r"))).view(torch.bfloat16).reshape(STEPS, 2048)
ref_raw = torch.from_numpy(np.array(np.memmap(REF / "ref.bin", dtype=np.float32, mode="r")))
nref = ref_raw.numel() // (STEPS * 2048)
ref = ref_raw.reshape(nref, STEPS, 2048)[-1]
print("reference block boundaries", nref, flush=True)
out = torch.empty_like(x)
attention = torch.zeros_like(x)
indices = torch.arange(64, dtype=torch.int32)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


values = []
t0 = time.time()
for c in range(STEPS):
    pages = (c + 255) // 256
    md = (
        torch.tensor([[c], [c], [c]], dtype=torch.int32),
        torch.tensor([0, pages], dtype=torch.int32),
        indices,
        torch.tensor([c % 256 if pages else 0], dtype=torch.int32),
        torch.tensor([c // 256, c % 256], dtype=torch.int32),
    )
    out.zero_()
    kernel[4, 4](x[c : c + 1], attention[c : c + 1], *md, out[c : c + 1], **weights)
    values.append(cos(out[c], ref[c]))
    print(f"token {c}: cos {values[-1]:.8f}", flush=True)

print(f"steps {STEPS} wall {time.time() - t0:.1f}s", flush=True)
print(f"cos mean {sum(values) / len(values):.8f} min {min(values):.8f}", flush=True)
Path("out_sgl_check").mkdir(exist_ok=True)
out.float().numpy().tofile(Path("out_sgl_check") / "decode_out.bin")
PACK.joinpath("crun_check.json").write_text(json.dumps({
    "steps": STEPS,
    "cos": values,
    "mean": sum(values) / len(values),
    "min": min(values),
}, indent=1))
if min(values) < 0.95:
    raise SystemExit(1)
