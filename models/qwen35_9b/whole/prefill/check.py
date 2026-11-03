"""A Qwen3.5 dense model through the whole-model fused forward with weights quantized as the adapter quantizes them
(MXFP4), against transformers' Qwen3_5DecoderLayer on the same GGUF values.
MODEL=<hf dir> GG=<gguf> HF=<hf dir> FULL=1 (all layers; default: block 0) NORM1=1: the kernel's norm weights are
GGUF's (1 + w); NORM1=0: w (GGUF - 1). PRODUCE=1 copies the compiled kernel to KNAME (the production prefill file)."""
import os
from pathlib import Path
import sys

import numpy as np
import torch
import axnn.dsl as ax
import sgl_kernel_axera

sys.path.insert(0, str(Path(sgl_kernel_axera.__file__).resolve().parents[2] / "tests"))

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

from sgl_kernel_axera.attention.gdn import runtime_weights, split_conv_state  # noqa: E402
from sgl_kernel_axera.models.qwen38_27b.forward import prepare  # noqa: E402

MODEL = os.environ["MODEL"]
GG = os.environ.get("GG", "/data/junda/work/gguf/qwen3.5-9b-Q4_K_M.gguf")
HF = os.environ.get("HF", MODEL)
C = __import__("json").load(open(f"{MODEL}/config.json"))["text_config"]
H, I, HEADS, KVH, HD = C["hidden_size"], C["intermediate_size"], C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
KH, VH, KD, VD = C["linear_num_key_heads"], C["linear_num_value_heads"], C["linear_key_head_dim"], C["linear_value_head_dim"]
REP = VH // KH
T = 256
NORM1 = os.getenv("NORM1", "1") == "1"
FULL = os.getenv("FULL", "0") == "1"
NL = int(C["num_hidden_layers"]) if FULL else 4
R = gguf.GGUFReader(GG)
TS = {t.name: t for t in R.tensors}


def raw(name):
    return TS[name]


def f32(name):
    t = TS[name]
    a = gguf.quants.dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a.reshape([int(x) for x in reversed(t.shape)]))).float()


def mx4(name, row_perm=None, col_perm=None):
    """dequantized GGUF [N, K] -> MXFP4 codes/scales after the same row/column reorder (per-row quantization commutes
    with row permutations and with column permutations whose blocks are whole 32-value groups)"""
    w = f32(name)
    if col_perm is not None:
        w = w[:, torch.from_numpy(np.ascontiguousarray(col_perm)).long()]
    if row_perm is not None:
        w = w[torch.from_numpy(np.ascontiguousarray(row_perm)).long()]
    return quantize_mxfp4(w)


from mx4patch import quantize_mxfp4  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def deq_mx4(name):
    """the adapter's MXFP4 quantization of a GGUF [N, K] tensor, dequantized for the reference"""
    w = f32(name)
    p, s, _ = quantize_mxfp4(w)
    u = p.view(torch.uint8)
    code = torch.stack([u & 15, u >> 4], dim=-1).reshape(w.shape)
    v = E2M1[(code & 7).long()] * torch.where(code >= 8, -1.0, 1.0)
    return v * torch.exp2(s.view(torch.uint8).float() - 127).repeat_interleave(32, dim=-1)


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
    rot = int(HD * C.get("partial_rotary_factor", 0.25))
    ang = torch.arange(16384)[:, None].double() / (1e7 ** (torch.arange(0, rot, 2).double() / rot))[None, :]
    w.update(w_norm_attn=nrm(f"blk.{L}.attn_norm.weight"), w_norm_q=f32(f"blk.{L}.attn_q_norm.weight") / HD**0.5,
             w_norm_k=f32(f"blk.{L}.attn_k_norm.weight").bfloat16(), cos=ang.cos().float(), sin=ang.sin().float(),
             partial_rotary_factor=0.25)
    return torch.zeros(64, 2, 256, KVH, HD, dtype=torch.bfloat16), w, mlp_w(L)


def gguf_into_hf(sd, L):
    """the HF layer's projection weights replaced by the GGUF's, quantized exactly as the adapter (MXFP4, value heads
    back in HF order): the reference has the kernel's weights"""
    vp, hp = vperm(VD), vperm(1)
    def put(k, v):
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
        sd[k] = v.float()
    if f"linear_attn.in_proj_qkv.weight" in sd:
        q = deq_mx4(f"blk.{L}.attn_qkv.weight"); nq = KH * KD
        put("linear_attn.in_proj_qkv.weight", torch.cat([q[:2 * nq], q[2 * nq:][vp]]))
        put("linear_attn.in_proj_z.weight", deq_mx4(f"blk.{L}.attn_gate.weight")[vp])
        put("linear_attn.in_proj_a.weight", deq_mx4(f"blk.{L}.ssm_alpha.weight")[hp])
        put("linear_attn.in_proj_b.weight", deq_mx4(f"blk.{L}.ssm_beta.weight")[hp])
        put("linear_attn.out_proj.weight", deq_mx4(f"blk.{L}.ssm_out.weight")[:, vp])
    else:
        for n, g in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            put(f"self_attn.{n}.weight", deq_mx4(f"blk.{L}.{g}.weight"))
    for n, g in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
        put(f"mlp.{n}.weight", deq_mx4(f"blk.{L}.{g}.weight"))


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
    for i in range(NL):
        layer = Qwen3_5DecoderLayer(cfg, i).float().eval()
        pre = [p for p in (f"model.language_model.layers.{i}.", f"model.layers.{i}.") if any(k.startswith(p) for k in idx)][0]
        sd = {}
        for k, f in idx.items():
            if k.startswith(pre):
                with safe_open(f"{HF}/{f}", "pt") as fh: sd[k[len(pre):]] = fh.get_tensor(k).float()
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
    if os.getenv("REF_ONLY") == "1":
        ref = hf_block(x)
        rp = ROOT / "build" / "9b" / f"check_ref_{H}_{NL}.pt"
        torch.save(ref, rp)
        print("reference", rp, flush=True)
        return
    layers = []
    for L in range(NL):
        if L % 4 == 3:
            layers.append(gqa_w(L))
            continue
        w, state, conv = gdn_w(L)
        w = runtime_weights(w)
        w["w_pre"] = None
        for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"),
                               (torch.zeros_like(state), *split_conv_state(torch.zeros_like(conv), KH * KD, VH * VD)), strict=True):
            w[name + "_in"] = w[name + "_out"] = value.contiguous()
        layers.append((None, w, mlp_w(L)))
    saved = ROOT / "build" / "9b" / f"check_y_{H}_{NL}.pt"
    if os.path.exists(saved):
        y = torch.load(saved)
        return finish(x, y)
    out = Path(os.getenv("OUT", str(ROOT / "build" / "9b" / "check_prefill")))
    kernels, weights = prepare(layers, mrope_section=tuple(C["rope_parameters"]["mrope_section"]), cache_path=str(out),
                               run_mode=os.getenv("RUN_MODE", "PYRUN"))
    y = x.clone()
    meta = (torch.arange(T, dtype=torch.int32).expand(3, -1).contiguous(), torch.tensor([0, 0], dtype=torch.int32),
            torch.arange(64, dtype=torch.int32), torch.tensor([T], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32))
    kernels[T][4, 4](y, *meta, y, valid_tokens=torch.tensor([T], dtype=torch.int32), **weights)
    print("kernel done", flush=True)
    torch.save(y, saved)
    finish(x, y)  # asserts the acceptance gate for FULL before a kernel is produced
    if os.getenv("PRODUCE") == "1":
        import glob, shutil
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        compiled = sorted(glob.glob(str(out / "forward_qwen38_blocks.*.compiled.axmodel")), key=os.path.getmtime)
        assert compiled, f"no compiled axmodel under {out}"
        dst = out / f"{os.environ.get('KNAME', 'qwen35_9b_prefill256_mx4_kvc_c16384_npu4')}.axmodel"
        shutil.copyfile(compiled[-1], dst)
        import board_io
        print("board io:", board_io.fix(str(dst)), flush=True)
        print("produced", dst, flush=True)


def finish(x, y):
    rp = ROOT / "build" / "9b" / f"check_ref_{H}_{NL}.pt"
    ref = torch.load(rp) if os.path.exists(rp) else hf_block(x)
    a, b = y.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"NORM1={int(NORM1)} cos mean {cos.mean():.5f} min {cos.min():.5f}; rows 0/1/255 {cos[0]:.4f} {cos[1]:.4f} {cos[255]:.4f}; "
          f"|y| {a.abs().mean():.3f} |ref| {b.abs().mean():.3f}")
    print(f"ACCEPT rows cos mean {cos.mean():.6f} min {cos.min():.6f} "
          f"({'PASS' if cos.mean() >= 0.9995 and cos.min() >= 0.999 else 'FAIL'})", flush=True)
    da, db = a - x.float(), b - x.float()
    dc = torch.nn.functional.cosine_similarity(da, db, dim=-1)
    print(f"delta (block output - input): cos mean {dc.mean():.5f} min {dc.min():.5f}; |d_y| {da.abs().mean():.4f} "
          f"|d_ref| {db.abs().mean():.4f}; ratio of norms per row mean {(da.norm(dim=-1) / db.norm(dim=-1)).mean():.4f}")
    torch.save(b, rp)
    if FULL:
        assert cos.mean() >= 0.9995 and cos.min() >= 0.999, (float(cos.mean()), float(cos.min()))


if __name__ == "__main__":
    main()
