"""Build a Qwen3.5 dense whole-model prefill kernel (256 cells per call) for AX8860, four cores.

MODEL=<hf config dir> BLOCKS=<n> ROWS=256 python build.py <out dir>
WFMT4=1 builds MXFP4 weights (both released Qwen3.5 packages), otherwise MXFP8.
The source is sgl-kernel-axera's forward_qwen38_blocks at npu-codebase cad0db3d6; the block loop is
part of the kernel, and every layer weight is a runtime bank input.
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
import axnn.dsl as ax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import board_io

from sgl_kernel_axera.attention.gdn import runtime_weights, split_conv_state
from sgl_kernel_axera.models.qwen38_27b.forward import prepare
from test_gdn_attn import random_case

MODEL = os.environ["MODEL"]
C = json.load(open(f"{MODEL}/config.json"))["text_config"]
H = C["hidden_size"]
I = C["intermediate_size"]
HEADS = C["num_attention_heads"]
KVH = C["num_key_value_heads"]
HD = C["head_dim"]
KH = C["linear_num_key_heads"]
VH = C["linear_num_value_heads"]
KD = C["linear_key_head_dim"]
VD = C["linear_value_head_dim"]
CTX = int(os.environ.get("CTX", "16384"))
WFMT4 = os.environ.get("WFMT4", "0") == "1"
GROUP = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
rng = np.random.default_rng(1)


def quant_mxfp8(w):
    g = w.float().reshape(w.shape[0], -1, GROUP)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q8 = (g / torch.exp2(e)).to(torch.float8_e4m3fn).reshape(w.shape)
    scale = (e.squeeze(-1) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q8.contiguous(), scale.contiguous(), torch.ones(1, dtype=torch.float32)


def quant_mxfp4(w):
    g = w.float().reshape(-1, GROUP)
    amax = g.abs().amax(-1)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2, torch.zeros_like(amax))
    x = g / torch.exp2(e).unsqueeze(-1)
    code = ((x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1).to(torch.uint8) | ((x < 0).to(torch.uint8) << 3))
    pk = (code[..., 0::2] | (code[..., 1::2] << 4)).reshape(w.shape[0], w.shape[1] // 2)
    sc = (e + 127).clamp(0, 254).to(torch.uint8).reshape(w.shape[0], w.shape[1] // GROUP)
    return pk.view(torch.float4_e2m1fn_x2).contiguous(), sc.contiguous(), torch.ones(1, dtype=torch.float32)


quant = quant_mxfp4 if WFMT4 else quant_mxfp8


def mx(w):
    """random MX codes / E8M0 scales of w's shape: the kernel takes every weight at runtime, so the build only needs
    the shapes (quantizing the random tensors would take minutes per layer for no effect on the compiled kernel)"""
    n, k = w.shape
    if WFMT4:
        q = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        q = torch.from_numpy(q).view(torch.float4_e2m1fn_x2).contiguous()
    else:
        q = rng.integers(0, 256, (n, k), dtype=np.uint8)
        q = torch.from_numpy(q).view(torch.float8_e4m3fn).contiguous()
    s = rng.integers(118, 124, (n, k // GROUP), dtype=np.uint8)
    return q, torch.from_numpy(s).view(torch.float8_e8m0fnu).contiguous(), torch.ones(1, dtype=torch.float32)


def mlp():
    r = {"w_norm": torch.ones(H, dtype=torch.bfloat16), "input_permutation": torch.zeros(1, dtype=torch.int32)}
    for i, (n, k) in enumerate(((I, H), (I, H), (H, I)), 1):
        p, s, s2 = mx(torch.randn(n, k) * 0.02)
        r.update({f"w{i}": p, f"w_scale{i}": s, f"w_scale2_{i}": s2,
                  f"w_zp_scale{i}": torch.zeros(1, dtype=torch.bfloat16)})
    return r


def gdn():
    _, _, w, state, conv = random_case(1, hidden=H, key_heads=KH, value_heads=VH, key_dim=KD, value_dim=VD)
    for name in ("w_q", "w_k", "w_v", "w_z", "w_a", "w_b", "w_o"):
        w[name], w[name + "_scale"], w[name + "_scale2"] = mx(w[name].float())
    w = runtime_weights(w)
    w["w_pre"] = None
    for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"),
                           (state, *split_conv_state(conv, KH * KD, VH * VD)), strict=True):
        w[name + "_in"] = w[name + "_out"] = value.contiguous()
    return None, w, mlp()


def gqa():
    w = {}
    for name, (n, k) in (("q", (HEADS * HD * 2, H)), ("k", (KVH * HD, H)), ("v", (KVH * HD, H)),
                         ("o", (H, HEADS * HD))):
        w["w_" + name], w["w_scale_" + name], w["w_scale2_" + name] = mx(torch.randn(n, k) * 0.02)
        w["w_zeropoint_" + name] = torch.zeros(1, dtype=torch.bfloat16)
    rot = int(HD * C.get("partial_rotary_factor", 0.25))
    ang = torch.arange(CTX)[:, None] / (1e7 ** (torch.arange(0, rot, 2).float() / rot))[None, :]
    w.update(w_norm_attn=torch.ones(H, dtype=torch.bfloat16), w_norm_q=torch.ones(HD) / HD**0.5,
             w_norm_k=torch.ones(HD, dtype=torch.bfloat16), cos=ang.cos(), sin=ang.sin(),
             partial_rotary_factor=float(C.get("partial_rotary_factor", 0.25)))
    cache = torch.zeros(CTX // 256, 2, 256, KVH, HD, dtype=torch.bfloat16)
    return cache, w, mlp()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    blocks = int(os.environ.get("BLOCKS", C["num_hidden_layers"] // 4))
    torch.set_num_threads(16)
    ax.init(chip_num=1)
    kinds = ["linear_attention" if i % 4 < 3 else "full_attention" for i in range(4 * blocks)]
    assert kinds == C["layer_types"][: 4 * blocks]
    layers = [gdn() if i % 4 < 3 else gqa() for i in range(4 * blocks)]
    # Phase-major state pools, the same layout the adapter's decode groups write (phase p, block b -> p * blocks + b).
    # The kernel's b0 bank is one such pool per call; keeping calls' pools contiguous lets the host bind a call's
    # [3 * blocks] rows with one offset.
    order = [4 * b + p for p in range(3) for b in range(blocks)]
    for name in ("state", "q_conv_state", "k_conv_state", "v_conv_state"):
        pool = torch.stack([layers[il][1][name + "_in"] for il in order])
        for i, il in enumerate(order):
            layers[il][1][name + "_in"] = layers[il][1][name + "_out"] = pool[i]
    kv_pool = torch.stack([kv for kv, _, _ in layers if kv is not None])
    j = 0
    for i, (kv, a, m) in enumerate(layers):
        if kv is not None:
            layers[i] = (kv_pool[j], a, m)
            j += 1
    kernels, weights = prepare(layers, mrope_section=tuple(C["rope_parameters"]["mrope_section"]),
                               cache_path=out, run_mode="CRUN")
    for rows in [int(r) for r in os.environ.get("ROWS", "256,1").split(",")]:
        x = (torch.randn(rows, H) * 0.03).bfloat16()
        pos = torch.arange(rows, dtype=torch.int32).expand(3, -1).contiguous()
        meta = (pos, torch.tensor([0, 1], dtype=torch.int32), torch.arange(0, CTX // 256, dtype=torch.int32),
                torch.tensor([rows % 256 or 256], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32))
        vt = torch.tensor([rows], dtype=torch.int32) if rows == 256 else None
        print(f"rows {rows}: building", flush=True)
        kernels[rows][4, 4](x, *meta, x, valid_tokens=vt, **weights)
        print(f"rows {rows}: done finite={bool(torch.isfinite(x).all())}", flush=True)
    compiled = sorted(glob.glob(str(out / "forward_qwen38_blocks.*.compiled.axmodel")), key=os.path.getmtime)
    assert compiled, f"no compiled axmodel under {out}"
    name = os.environ.get("KNAME", "qwen35_9b_prefill256_mx4_kvc_c16384_npu4")
    dst = out / f"{name}.axmodel"
    shutil.copyfile(compiled[-1], dst)
    print("board io:", board_io.fix(str(dst)), flush=True)
    print("wrote", dst, dst.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
