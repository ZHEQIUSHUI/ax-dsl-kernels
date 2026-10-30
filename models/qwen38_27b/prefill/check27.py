"""Block 0 (layers 0..3) of Qwen3.8-27B through the other team's fused forward (PYRUN, T=256) with weights converted
from our MXFP4 GGUF, against transformers' Qwen3_5DecoderLayer with the HF weights.
NORM1=1: the kernel's norm weights are GGUF's (1 + w); NORM1=0: w (GGUF - 1)."""
import os
import sys

import numpy as np
import torch
import axnn.dsl as ax

sys.path.insert(0, "/data/junda/work/llama.cpp/gguf-py")
import gguf  # noqa: E402

from sgl_kernel_axera.attention.gdn import runtime_weights, split_conv_state  # noqa: E402
from sgl_kernel_axera.models.qwen38_27b.forward import prepare  # noqa: E402

GG = "/data/junda/gguf/qwen3.8-27b-mxfp4.gguf"
HF = "/data/junda/hf/Qwen3.8-27B"
H, I, HEADS, KVH, HD = 5120, 17408, 24, 4, 256
KH, VH, KD, VD = 16, 48, 128, 128
REP = VH // KH
T = 256
NORM1 = os.getenv("NORM1", "1") == "1"
R = gguf.GGUFReader(GG)
TS = {t.name: t for t in R.tensors}


def raw(name):
    return TS[name]


def f32(name):
    t = TS[name]
    a = gguf.quants.dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a.reshape([int(x) for x in reversed(t.shape)]))).float()


def mx4(name, row_perm=None, col_perm=None):
    """GGUF MXFP4 [N, K] -> packed [N, K/2] (channel 2i low nibble, 2i+1 high) + E8M0 scales [N, K/32]"""
    t = TS[name]
    n, k = int(t.shape[1]), int(t.shape[0])
    b = np.asarray(t.data).reshape(n, k // 32, 17)
    e, qs = b[:, :, 0], b[:, :, 1:]
    codes = np.concatenate([qs & 15, qs >> 4], axis=-1).reshape(n, k)      # element order of the block: lo 0..15, hi 16..31
    if col_perm is not None:
        assert len(col_perm) == k
        # a column permutation moves whole 32-blocks only when it permutes head_dim-sized groups (128 % 32 == 0)
        codes = codes[:, col_perm]
        e = e[:, (np.asarray(col_perm).reshape(-1, 32)[:, 0] // 32)]
    if row_perm is not None:
        codes, e = codes[row_perm], e[row_perm]
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return (torch.from_numpy(np.ascontiguousarray(packed)).view(torch.float4_e2m1fn_x2),
            torch.from_numpy(np.ascontiguousarray(e).astype(np.uint8)).view(torch.float8_e8m0fnu), torch.ones(1))


def vperm(head_dim):
    """GGUF tiled V order -> HF grouped: HF head (k, r) = k * REP + r sits at GGUF r * KH + k"""
    idx = np.arange(VH * head_dim).reshape(REP, KH, head_dim).transpose(1, 0, 2).reshape(-1)
    return idx


def nrm(name):
    w = f32(name)
    return (w if NORM1 else w - 1).bfloat16()


def mlp_w(L):
    r = {"w_norm": nrm(f"blk.{L}.post_attention_norm.weight"), "input_permutation": torch.zeros(1, dtype=torch.int32)}
    for i, n in enumerate(("ffn_gate", "ffn_up", "ffn_down"), 1):
        p, s, s2 = mx4(f"blk.{L}.{n}.weight")
        r.update({f"w{i}": p, f"w_scale{i}": s, f"w_scale2_{i}": s2, f"w_zp_scale{i}": torch.zeros(1, dtype=torch.bfloat16)})
    return r


def gdn_w(L):
    from test_gdn_attn import random_case
    _, _, w, state, conv = random_case(1, hidden=H, key_heads=KH, value_heads=VH, key_dim=KD, value_dim=VD)
    vp = vperm(VD)
    qkv_perm = np.concatenate([np.arange(2 * KH * KD), 2 * KH * KD + vp])
    p, s, s2 = mx4(f"blk.{L}.attn_qkv.weight", row_perm=qkv_perm)
    nq = KH * KD
    w["w_q"], w["w_k"], w["w_v"] = p[:nq], p[nq:2 * nq], p[2 * nq:]
    w["w_q_scale"], w["w_k_scale"], w["w_v_scale"] = s[:nq], s[nq:2 * nq], s[2 * nq:]
    for n in ("w_q", "w_k", "w_v"): w[n + "_scale2"] = s2
    w["w_z"], w["w_z_scale"], w["w_z_scale2"] = mx4(f"blk.{L}.attn_gate.weight", row_perm=vp)
    w["w_o"], w["w_o_scale"], w["w_o_scale2"] = mx4(f"blk.{L}.ssm_out.weight", col_perm=vp)
    hp = vperm(1)
    from mx4patch import quantize_mxfp4   # Q8_0 alpha / beta -> MXFP4 (all GDN projections must share the FP4 format)
    for n, g in (("w_a", "ssm_alpha"), ("w_b", "ssm_beta")):
        w[n], w[n + "_scale"], w[n + "_scale2"] = quantize_mxfp4(f32(f"blk.{L}.{g}.weight")[hp])
    w["A_log" if "A_log" in w else "a_log"] = torch.log(-f32(f"blk.{L}.ssm_a")[hp])
    w["dt_bias"] = f32(f"blk.{L}.ssm_dt.bias")[hp]
    conv_w = f32(f"blk.{L}.ssm_conv1d.weight")                              # [10240, 4]
    w["w_q_conv"], w["w_k_conv"] = conv_w[:nq].bfloat16(), conv_w[nq:2 * nq].bfloat16()
    w["w_v_conv"] = conv_w[2 * nq:][vp].bfloat16()
    w["input_norm_w"] = nrm(f"blk.{L}.attn_norm.weight")
    w["o_norm_w" if "o_norm_w" in w else "norm_w"] = f32(f"blk.{L}.ssm_norm.weight").bfloat16()
    return w, state, conv


def gqa_w(L):
    w = {}
    for n, g in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
        w["w_" + n], w["w_scale_" + n], w["w_scale2_" + n] = mx4(f"blk.{L}.{g}.weight")
        w["w_zeropoint_" + n] = torch.zeros(1, dtype=torch.bfloat16)
    rot = int(HD * 0.25)
    ang = torch.arange(16384)[:, None].double() / (1e7 ** (torch.arange(0, rot, 2).double() / rot))[None, :]
    w.update(w_norm_attn=nrm(f"blk.{L}.attn_norm.weight"), w_norm_q=f32(f"blk.{L}.attn_q_norm.weight") / HD**0.5,
             w_norm_k=f32(f"blk.{L}.attn_k_norm.weight").bfloat16(), cos=ang.cos().float(), sin=ang.sin().float(),
             partial_rotary_factor=0.25)
    return torch.zeros(64, 2, 256, KVH, HD, dtype=torch.bfloat16), w, mlp_w(L)


def gguf_into_hf(sd, L):
    """the HF layer's projection weights replaced by the GGUF's (dequantized, value heads back in HF order): the
    reference then has exactly the kernel's weights (but alpha / beta: Q8_0 here, MXFP4 in the kernel)"""
    vp, hp = vperm(VD), vperm(1)
    def put(k, v):
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
        sd[k] = v.float()
    if f"linear_attn.in_proj_qkv.weight" in sd:
        q = f32(f"blk.{L}.attn_qkv.weight"); nq = KH * KD
        put("linear_attn.in_proj_qkv.weight", torch.cat([q[:2 * nq], q[2 * nq:][vp]]))
        put("linear_attn.in_proj_z.weight", f32(f"blk.{L}.attn_gate.weight")[vp])
        put("linear_attn.in_proj_a.weight", f32(f"blk.{L}.ssm_alpha.weight")[hp])
        put("linear_attn.in_proj_b.weight", f32(f"blk.{L}.ssm_beta.weight")[hp])
        put("linear_attn.out_proj.weight", f32(f"blk.{L}.ssm_out.weight")[:, vp])
    else:
        for n, g in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            put(f"self_attn.{n}.weight", f32(f"blk.{L}.{g}.weight"))
    for n, g in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
        put(f"mlp.{n}.weight", f32(f"blk.{L}.{g}.weight"))


def hf_block(x):
    import json
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
    cfg = AutoConfig.from_pretrained(HF)
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "eager"
    idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    pos = torch.arange(T)[None]
    h = x.float().reshape(1, T, H)
    for i in range(4):
        layer = Qwen3_5DecoderLayer(cfg, i).float().eval()
        pre = [p for p in (f"model.language_model.layers.{i}.", f"model.layers.{i}.") if any(k.startswith(p) for k in idx)][0]
        sd = {}
        for k, f in idx.items():
            if k.startswith(pre):
                with safe_open(f"{HF}/{f}", "pt") as fh: sd[k[len(pre):]] = fh.get_tensor(k).float()
        if os.getenv("REF_GGUF") == "1":
            gguf_into_hf(sd, i)
        layer.load_state_dict(sd, strict=True)
        mask = torch.full((1, 1, T, T), float("-inf")).triu(1) if layer.layer_type == "full_attention" else None
        with torch.no_grad():
            pe = rot(h, pos)
            o = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
            h = o[0] if isinstance(o, tuple) else o
        print(f"hf layer {i} ({layer.layer_type}) done", flush=True)
    return h.reshape(T, H)


def main():
    torch.set_num_threads(32)
    ax.init(chip_num=1)
    torch.manual_seed(0)
    ids = torch.randint(0, 200000, (T,))
    x = f32("token_embd.weight")[ids].bfloat16()                             # [T, H]
    layers = []
    for L in range(3):
        w, state, conv = gdn_w(L)
        w = runtime_weights(w)
        w["w_pre"] = None
        for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"),
                               (torch.zeros_like(state), *split_conv_state(torch.zeros_like(conv), KH * KD, VH * VD)), strict=True):
            w[name + "_in"] = w[name + "_out"] = value.contiguous()
        layers.append((None, w, mlp_w(L)))
    layers.append(gqa_w(3))
    saved = f"/data/junda/work/q27f/y_norm{int(NORM1)}.pt"
    if os.path.exists(saved):
        y = torch.load(saved)
        return finish(x, y)
    kernels, weights = prepare(layers, mrope_section=(11, 11, 10), cache_path=os.getenv("OUT", "/data/junda/work/q27f/out_chk"),
                               run_mode=os.getenv("RUN_MODE", "PYRUN"))
    y = x.clone()
    meta = (torch.arange(T, dtype=torch.int32).expand(3, -1).contiguous(), torch.tensor([0, 0], dtype=torch.int32),
            torch.arange(64, dtype=torch.int32), torch.tensor([T], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32))
    kernels[T][4, 4](y, *meta, y, valid_tokens=torch.tensor([T], dtype=torch.int32), **weights)
    print("kernel done", flush=True)
    torch.save(y, saved)
    finish(x, y)


def finish(x, y):
    rp = f"/data/junda/work/q27f/ref_block0_g{os.getenv('REF_GGUF', '0')}.pt"
    ref = torch.load(rp) if os.path.exists(rp) else hf_block(x)
    a, b = y.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"NORM1={int(NORM1)} cos mean {cos.mean():.5f} min {cos.min():.5f}; rows 0/1/255 {cos[0]:.4f} {cos[1]:.4f} {cos[255]:.4f}; "
          f"|y| {a.abs().mean():.3f} |ref| {b.abs().mean():.3f}")
    da, db = a - x.float(), b - x.float()
    dc = torch.nn.functional.cosine_similarity(da, db, dim=-1)
    print(f"delta (block output - input): cos mean {dc.mean():.5f} min {dc.min():.5f}; |d_y| {da.abs().mean():.4f} "
          f"|d_ref| {db.abs().mean():.4f}; ratio of norms per row mean {(da.norm(dim=-1) / db.norm(dim=-1)).mean():.4f}")
    torch.save(b, f"/data/junda/work/q27f/ref_block0_g{os.getenv('REF_GGUF', '0')}.pt")


if __name__ == "__main__":
    main()
