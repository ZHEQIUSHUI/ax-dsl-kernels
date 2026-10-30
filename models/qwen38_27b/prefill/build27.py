"""Build the other team's Qwen3.8 fused forward (sgl_kernel_axera.models.qwen38_27b) at 27B shapes with MXFP4
weights (random codes, E8M0 per-32 scales) for a board timing: python build27.py <out dir>
BLOCKS (default 2): blocks of 3 GDN + 1 GQA layers (the runtime loop's time is per block); ROWS: 1,256."""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import axnn.dsl as ax

from sgl_kernel_axera.attention.gdn import runtime_weights, split_conv_state
from sgl_kernel_axera.models.qwen38_27b.forward import prepare
from test_gdn_attn import random_case

H, I, HEADS, KVH, HD = 5120, 17408, 24, 4, 256
KH, VH, KD, VD = 16, 48, 128, 128
CTX = int(os.getenv("CTX", "16384"))
rng = np.random.default_rng(1)


def mx4(n, k):
    codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    scale = rng.integers(118, 124, size=(n, k // 32), dtype=np.uint8)
    return (torch.from_numpy(codes).view(torch.float4_e2m1fn_x2), torch.from_numpy(scale).view(torch.float8_e8m0fnu),
            torch.ones(1, dtype=torch.float32))


def mlp():
    r = {"w_norm": torch.ones(H, dtype=torch.bfloat16), "input_permutation": torch.zeros(1, dtype=torch.int32)}
    for i, (n, k) in enumerate(((I, H), (I, H), (H, I)), 1):
        p, s, s2 = mx4(n, k)
        r.update({f"w{i}": p, f"w_scale{i}": s, f"w_scale2_{i}": s2, f"w_zp_scale{i}": torch.zeros(1, dtype=torch.bfloat16)})
    return r


def gdn():
    _, _, w, state, conv = random_case(1, hidden=H, key_heads=KH, value_heads=VH, key_dim=KD, value_dim=VD)
    for name in ("w_q", "w_k", "w_v", "w_z", "w_a", "w_b", "w_o"):
        n, k = w[name].shape
        w[name], w[name + "_scale"], w[name + "_scale2"] = mx4(n, k)
    w = runtime_weights(w)
    w["w_pre"] = None
    for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"),
                           (state, *split_conv_state(conv, KH * KD, VH * VD)), strict=True):
        w[name + "_in"] = w[name + "_out"] = value.contiguous()
    return None, w, mlp()


def gqa():
    w = {}
    for name, (n, k) in (("q", (HEADS * HD * 2, H)), ("k", (KVH * HD, H)), ("v", (KVH * HD, H)), ("o", (H, HEADS * HD))):
        w["w_" + name], w["w_scale_" + name], w["w_scale2_" + name] = mx4(n, k)
        w["w_zeropoint_" + name] = torch.zeros(1, dtype=torch.bfloat16)
    rot = int(HD * 0.25)
    ang = torch.arange(CTX)[:, None] / (1e7 ** (torch.arange(0, rot, 2).float() / rot))[None, :]
    w.update(w_norm_attn=torch.ones(H, dtype=torch.bfloat16), w_norm_q=torch.ones(HD) / HD**0.5,
             w_norm_k=torch.ones(HD, dtype=torch.bfloat16), cos=ang.cos(), sin=ang.sin(), partial_rotary_factor=0.25)
    cache = torch.zeros(CTX // 256, 2, 256, KVH, HD, dtype=torch.bfloat16)
    return cache, w, mlp()


def main():
    out = Path(sys.argv[1])
    blocks = int(os.getenv("BLOCKS", "2"))
    torch.set_num_threads(16)
    ax.init(chip_num=1)
    layers = [gdn() if i % 4 < 3 else gqa() for i in range(4 * blocks)]
    # one state / KV pool per kind, as production
    linear = [a for kv, a, _ in layers if kv is None]
    for name in ("state", "q_conv_state", "k_conv_state", "v_conv_state"):
        pool = torch.stack([a[name + "_in"] for a in linear])
        for i, a in enumerate(linear):
            a[name + "_in"] = a[name + "_out"] = pool[i]
    kv_pool = torch.stack([kv for kv, _, _ in layers if kv is not None])
    j = 0
    for i, (kv, a, m) in enumerate(layers):
        if kv is not None:
            layers[i] = (kv_pool[j], a, m); j += 1
    kernels, weights = prepare(layers, mrope_section=(11, 11, 10), cache_path=out, run_mode="CRUN")
    for rows in [int(r) for r in os.getenv("ROWS", "256,1").split(",")]:
        x = (torch.randn(rows, H) * 0.03).bfloat16()
        pos = torch.arange(rows, dtype=torch.int32).expand(3, -1).contiguous()
        meta = (pos, torch.tensor([0, 1], dtype=torch.int32), torch.arange(0, CTX // 256, dtype=torch.int32),
                torch.tensor([rows % 256 or 256], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32))
        vt = torch.tensor([rows], dtype=torch.int32) if rows == 256 else None
        print(f"rows {rows}: building", flush=True)
        kernels[rows][4, 4](x, *meta, x, valid_tokens=vt, **weights)
        print(f"rows {rows}: done finite={bool(torch.isfinite(x).all())}", flush=True)


if __name__ == "__main__":
    main()
