"""Reference weights and PyTorch math for the Qwen3.8-27B MTP block.

The MTP row at position p pairs token x_p with the target model's final-norm hidden h_{p-1}:

    z = fc([enorm(embed(x_p)) | hnorm(h_{p-1})])
    y = DecoderLayer(z, MTP KV at position p)
    h_next = shared_head_norm(y)              # chained into the next MTP step and lm_head

All Qwen3.5 RMSNorm tensors are stored as deltas: the multiplier is 1 + weight.
Projection weights here are quantized to the same MXFP4/E8M0 format as the kernels and
dequantized before use, so the Python reference and AX kernel see identical weights.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

H = 5120
F = 17408
HQ = 24
HKV = 4
D = 256
ROT = 64
THETA = 10_000_000.0
EPS = 1e-6
GROUP = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

MTP_LAYER_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _shape(name: str):
    if name == "mtp.fc.weight":
        return (H, 2 * H)
    if name in ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
        return (H,)
    if name.startswith("mtp.layers.0."):
        suffix = name[len("mtp.layers.0."):]
        return {
            "input_layernorm.weight": (H,),
            "post_attention_layernorm.weight": (H,),
            "self_attn.q_proj.weight": (2 * HQ * D, H),
            "self_attn.k_proj.weight": (HKV * D, H),
            "self_attn.v_proj.weight": (HKV * D, H),
            "self_attn.o_proj.weight": (H, HQ * D),
            "self_attn.q_norm.weight": (D,),
            "self_attn.k_norm.weight": (D,),
            "mlp.gate_proj.weight": (F, H),
            "mlp.up_proj.weight": (F, H),
            "mlp.down_proj.weight": (H, F),
        }[suffix]
    raise KeyError(name)


class MtpWeights:
    """HF MTP tensors, or deterministic real-shape random tensors for compilation builds."""

    def __init__(self, model_dir: str, random: bool = False):
        self.random = random
        self.where = {}
        self.files = {}
        if not random:
            idx_path = f"{model_dir}/model.safetensors.index.json"
            if os.path.exists(idx_path):
                self.where = json.load(open(idx_path))["weight_map"]
            else:
                self.where = {"model.safetensors": None}
            self.model_dir = model_dir

    def __call__(self, name: str):
        shape = _shape(name)
        if self.random:
            g = torch.Generator().manual_seed(sum(map(ord, name)) * 7919 % (1 << 31))
            if name.endswith("norm.weight"):
                return (torch.randn(shape, generator=g) * 0.02).bfloat16()
            return (torch.randn(shape, generator=g) * 0.02).bfloat16()
        filename = self.where.get(name)
        if filename is None:
            raise KeyError(name)
        if filename not in self.files:
            self.files[filename] = safe_open(f"{self.model_dir}/{filename}", "pt")
        return self.files[filename].get_tensor(name)


def one_plus(w: torch.Tensor):
    return (1.0 + w.float()).contiguous()


def quant_mxfp4(w: torch.Tensor):
    """[N, K] -> (E2M1 pairs [N, K/2] uint8 low-nibble-first, E8M0 [N, K/32] uint8). Matches q35_layer_pk."""
    n, k = w.shape
    g = w.float().reshape(n, k // GROUP, GROUP)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2, torch.zeros_like(amax))
    x = (g / torch.exp2(e)).reshape(n, k)
    code = ((x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1) | ((x < 0).long() << 3)).to(torch.uint8)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def dequant_mxfp4(pk: torch.Tensor, scale: torch.Tensor):
    u = pk.view(torch.uint8)
    code = torch.stack([u & 15, u >> 4], -1).reshape(u.shape[0], -1).long()
    v = E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    return (v.reshape(v.shape[0], -1, GROUP) * torch.exp2(scale.float() - 127).unsqueeze(-1)).reshape(v.shape[0], -1)


def quant_args(a: dict, names):
    """Quantize the named projections; return both packed args and their dequantized reference twins."""
    packed = dict(a)
    deq = dict(a)
    for name in names:
        w = a[name]
        pk, sc = quant_mxfp4(w.float())
        packed[name] = pk.view(torch.float4_e2m1fn_x2)
        packed["s_" + name[2:]] = sc.view(torch.float8_e8m0fnu)
        deq[name] = dequant_mxfp4(pk, sc).bfloat16()
    return packed, deq


MTP_WEIGHT_NAMES = (
    "w_fc", "w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down",
)
MTP_ORDER = (
    "ln_e", "ln_h", "w_fc", "s_fc",
    "ln1", "w_q", "s_q", "w_k", "s_k", "w_v", "s_v", "q_norm", "k_norm",
    "w_o", "s_o", "ln2", "w_gate", "s_gate", "w_up", "s_up", "w_down", "s_down",
    "head_norm",
)


def mtp_args(W: MtpWeights, quantized: bool = True):
    p = "mtp.layers.0."
    a = {
        "ln_e": one_plus(W("mtp.pre_fc_norm_embedding.weight")),
        "ln_h": one_plus(W("mtp.pre_fc_norm_hidden.weight")),
        "w_fc": W("mtp.fc.weight").float(),
        "ln1": one_plus(W(p + "input_layernorm.weight")),
        "w_q": W(p + "self_attn.q_proj.weight").float(),
        "w_k": W(p + "self_attn.k_proj.weight").float(),
        "w_v": W(p + "self_attn.v_proj.weight").float(),
        "q_norm": one_plus(W(p + "self_attn.q_norm.weight")),
        "k_norm": one_plus(W(p + "self_attn.k_norm.weight")),
        "w_o": W(p + "self_attn.o_proj.weight").float(),
        "ln2": one_plus(W(p + "post_attention_layernorm.weight")),
        "w_gate": W(p + "mlp.gate_proj.weight").float(),
        "w_up": W(p + "mlp.up_proj.weight").float(),
        "w_down": W(p + "mlp.down_proj.weight").float(),
        "head_norm": one_plus(W("mtp.norm.weight")),
    }
    if not quantized:
        return a, a
    return quant_args(a, MTP_WEIGHT_NAMES)


def rmsnorm(x: torch.Tensor, w: torch.Tensor):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def rope(x: torch.Tensor, pos: int, rot: int = ROT, theta: float = THETA):
    """NeoX/HF partial RoPE: adjacent halves of the first rot channels."""
    half = rot // 2
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    ang = torch.tensor(float(pos), dtype=torch.float64) * inv
    cos, sin = ang.cos().float(), ang.sin().float()
    y = x.clone()
    x1, x2 = x[..., :half], x[..., half:rot]
    y[..., :half] = x1 * cos - x2 * sin
    y[..., half:rot] = x2 * cos + x1 * sin
    return y


def kv_rows(kv: torch.Tensor, pos: int, block: int = 256):
    """[PAGES, 2, 256, HKV*D] -> the K and V row at position pos."""
    page, cell = divmod(pos, block)
    return kv[page, 0, cell].float(), kv[page, 1, cell].float()


def append_kv(kv: torch.Tensor, pos: int, k: torch.Tensor, v: torch.Tensor, block: int = 256):
    page, cell = divmod(pos, block)
    kv[page, 0, cell] = k.reshape(HKV * D).bfloat16()
    kv[page, 1, cell] = v.reshape(HKV * D).bfloat16()


def mtp_step(e: torch.Tensor, h: torch.Tensor, pos: int, kv: torch.Tensor, a: dict):
    """One MTP row. Returns (post-norm h_next, pre-norm MTP hidden)."""
    en = rmsnorm(e.reshape(H), a["ln_e"])
    hn = rmsnorm(h.reshape(H), a["ln_h"])
    z = torch.cat([en, hn]).reshape(1, 2 * H) @ a["w_fc"].float().T
    x = rmsnorm(z, a["ln1"]).reshape(H)

    qg = (x.reshape(1, H) @ a["w_q"].float().T).reshape(HQ, 2 * D)
    q = rmsnorm(qg[:, :D], a["q_norm"])
    gate = qg[:, D:]
    k = rmsnorm((x.reshape(1, H) @ a["w_k"].float().T).reshape(HKV, D), a["k_norm"])
    v = (x.reshape(1, H) @ a["w_v"].float().T).reshape(HKV, D)
    q = rope(q, pos)
    k = rope(k, pos)
    append_kv(kv, pos, k, v)

    # causal dense attention over the MTP layer's own paged KV
    past_k, past_v = [], []
    for p in range(pos + 1):
        kr, vr = kv_rows(kv, p)
        past_k.append(kr.reshape(HKV, D))
        past_v.append(vr.reshape(HKV, D))
    kb = torch.stack(past_k, 0)  # [P, HKV, D]
    vb = torch.stack(past_v, 0)
    g = HQ // HKV
    qh = q.reshape(HKV, g, D)
    scores = torch.einsum("hgd,phd->ghp", qh.float(), kb.float()) * (D ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    attn = torch.einsum("ghp,phd->hgd", probs, vb.float()).reshape(HQ * D)
    attn = attn * torch.sigmoid(gate.float().reshape(HQ * D))
    x1 = z.reshape(H) + attn.reshape(1, HQ * D) @ a["w_o"].float().T

    h2 = rmsnorm(x1, a["ln2"])
    act = torch.nn.functional.silu(h2.reshape(1, H) @ a["w_gate"].float().T) * (
        h2.reshape(1, H) @ a["w_up"].float().T)
    y = (x1 + act @ a["w_down"].float().T).reshape(H)
    return rmsnorm(y, a["head_norm"]), y

