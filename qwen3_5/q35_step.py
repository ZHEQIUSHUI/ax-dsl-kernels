"""Phase-0: one complete Qwen3.5-0.8B decode step on the NPU simulator, real weights, layer by layer.

Two single-core kernels, every weight a runtime input, so one axmodel per layer type serves all layers:
  gdn_layer   RMSNorm -> qkv/z/b/a proj -> stateful causal conv1d+SiLU -> L2norm -> gated delta rule
              (fp32 state in/out) -> gated RMSNorm -> out_proj + residual -> RMSNorm -> MLP + residual
  attn_layer  RMSNorm -> q(+gate)/k/v proj -> q/k RMSNorm -> partial RoPE (64 of 256) -> write K/V at
              runtime pos -> GQA attention (runtime nblk/mask) -> sigmoid(gate) -> o_proj + residual -> MLP
Host side: embedding lookup, final norm and lm_head (torch fp32), RoPE cos/sin for the position.
Reference: transformers Qwen3.5 forward on the same tokens; per-layer hidden states and logits compared.

Env: LMAX (1024), BC (256), WFMT (bf16), NGEN (greedy tokens to generate, 4), PROMPT
"""

import json
import math
import os
import sys
import time

import ml_dtypes
import numpy as np
import torch
from loguru import logger
from safetensors import safe_open

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_gate, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.qwen_gdn_utils import gdn_gate, gdn_recurrent_step, l2norm_lastdim
from axnn.dsl import builder_ as _builder

# 模拟器把所有输出 buffer 清零后再跑；板端宿主会把常驻 KV cache 绑到输出并保留原内容。
# 这里让模拟器的同名输出先用宿主数据预填，行为与板端一致（只在本脚本生效，不改仓库）。
RESIDENT = {}
_simulate = _builder.simulate_dynamic_assembled


def _simulate_resident(be, asm, inputs, outputs, *args, **kwargs):
    # 模拟器的内存初始化不区分输入/输出：把常驻 buffer 的内容和输入一起交给它，就预填到了输出区域
    inputs = dict(inputs)
    for name, t in RESIDENT.items():
        if name in outputs:
            o = outputs[name]
            inputs[name] = np.ascontiguousarray(t.view(torch.int16).numpy().view(o.dtype).reshape(o.shape))
    return _simulate(be, asm, inputs, outputs, *args, **kwargs)


_builder.simulate_dynamic_assembled = _simulate_resident

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")
LMAX = int(os.environ.get("LMAX", 1024))
BC = int(os.environ.get("BC", 256))
# KV cache layout of the attention kernels: "head" [HKV, LMAX, D] bf16, rows indexed by position (default);
# "cell" the llama.cpp layout [LMAX, HKV*D] f16, one row per KV cell (pos input = the cell to write)
KVL = os.environ.get("KVL", "head")
NGEN = int(os.environ.get("NGEN", 4))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
DUMP = os.path.abspath(os.environ.get("DUMP", "out_q35_step"))

cfg = json.load(open(f"{MODEL}/config.json"))["text_config"]
H = cfg["hidden_size"]  # 1024
F = cfg["intermediate_size"]  # 3584
HQ, HKV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]  # 8, 2, 256
G = HQ // HKV
ROT = int(D * cfg["rope_parameters"]["partial_rotary_factor"])  # 64
THETA = cfg["rope_parameters"]["rope_theta"]
NK, NV = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]  # 16, 16
DK, DV = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]  # 128, 128
KD, VD = NK * DK, NV * DV
# value head order with NV > NK: "tiled" (the GGUF's, llama.cpp reorders them for ggml broadcasting: value head vh
# uses key head vh % NK) or "grouped" (Hugging Face: vh // (NV / NK)); the same for NV == NK
VORDER = os.environ.get("VORDER", "tiled")
assert VORDER in ("tiled", "grouped") and NV % NK == 0
CONV = cfg["linear_conv_kernel_dim"]  # 4
EPS = cfg["rms_norm_eps"]
LAYERS = cfg["layer_types"]


def oc_tile(cols):
    """Output-channel tile so one bf16 weight tile stays around 512 KB of OCM."""
    return max(16, min(256, (512 * 1024) // (cols * 2)))


def proj(x, w, name):
    return linear(x, w, None, name, output_dtype=ax.bf16, oc_tile_size=oc_tile(w.shape[1]))


def mlp_residual(x1, ln2, w_gate, w_up, w_down, out):
    h2 = rmsnorm(x1, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
    act = silu_mul(proj(h2, w_gate, "mlp_gate"), proj(h2, w_up, "mlp_up"), activation_output_dtype=ax.fp32,
                   output_dtype=ax.bf16)
    x2 = ax.dsl.add(x1, proj(act, w_down, "mlp_down"), output_dtype=ax.bf16)
    ax.dsl.store(x2, r=out)


def gdn_layer(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
              state_in, conv_in, out, state_out, conv_out):
    xt = ax.dsl.load(x)  # [1, 1, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qkv = proj(h, w_qkv, "gdn_qkv")  # [1, 1, 2*KD+VD]
    z = proj(h, w_z, "gdn_z").reshape(1, 1, NV, DV)
    b = proj(h, w_b, "gdn_b")
    a = proj(h, w_a, "gdn_a")
    mixed = stateful_runtime_causal_depthwise_conv1d_silu(qkv, conv_w, conv_in, conv_out, CONV, "gdn_conv")
    q = mixed[:, :, 0:KD].reshape(1, 1, NK, DK)
    k = mixed[:, :, KD : 2 * KD].reshape(1, 1, NK, DK)
    v = mixed[:, :, 2 * KD :].reshape(1, 1, NV, DV)

    a_log_t = ax.dsl.load(a_log)
    dt_bias_t = ax.dsl.load(dt_bias)
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    core = ax.dsl.create_tensor((1, 1, NV, DV), ax.bf16, name="gdn_core")
    rep = NV // NK
    for kh in range(NK):
        q_tok = ax.dsl.mul(l2norm_lastdim(q[:, :, kh : kh + 1, :].reshape(1, DK)), q_scale, output_dtype=ax.bf16)
        k_tok = l2norm_lastdim(k[:, :, kh : kh + 1, :].reshape(1, DK))
        for vo in range(rep):
            vh = vo * NK + kh if VORDER == "tiled" else kh * rep + vo
            g, beta = gdn_gate(a[:, :, vh : vh + 1].reshape(1, 1), b[:, :, vh : vh + 1].reshape(1, 1),
                               a_log_t[vh : vh + 1], dt_bias_t[vh : vh + 1])
            st = ax.dsl.load(state_in[vh : vh + 1]).reshape(DK, DV)
            o_h, st = gdn_recurrent_step(q_tok, k_tok, v[:, :, vh : vh + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
            ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, :, vh : vh + 1, :])
            ax.dsl.store(x=st.reshape(1, DK, DV), r=state_out[vh : vh + 1])

    normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, 1, VD)
    x1 = ax.dsl.add(xt, proj(gated, w_out, "gdn_out"), output_dtype=ax.bf16)
    mlp_residual(x1, ln2, w_gate, w_up, w_down, out)
    return out, state_out, conv_out


def attn_layer(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
               w_down, out):
    xt = ax.dsl.load(x)  # [1, 1, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qg = proj(h, w_q, "attn_q").reshape(1, HQ, 2 * D)  # 每个头 [q | gate]
    q = rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS)
    gate = qg[:, :, D : 2 * D]
    k = rmsnorm(proj(h, w_k, "attn_k").reshape(1, HKV, D), ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
    v = proj(h, w_v, "attn_v").reshape(1, HKV, D)
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)  # [1, ROT/2]
    q = rope(q, cos_t, sin_t, rotary_dim=ROT)
    k = rope(k, cos_t, sin_t, rotary_dim=ROT)

    # 新 token 的 K/V 写入常驻 cache 的 pos 行（head 布局逐头写；cell 布局整行 f16 写入 cell 行）
    one_row = ax.dsl.from_numpy(np.array([1, 1] if KVL == "head" else [1], dtype=np.uint32))
    src0 = ax.dsl.from_numpy(np.zeros(shape=(2 if KVL == "head" else 1,), dtype=np.uint32))
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    if KVL == "cell":
        ax.dsl.dynamic_st(ax.dsl.cast(k.reshape(1, HKV * D), dtype=ax.fp16), src0, p, one_row, dim=(0,),
                          output_shape=kc.shape, r=kc)
        ax.dsl.dynamic_st(ax.dsl.cast(v.reshape(1, HKV * D), dtype=ax.fp16), src0, p, one_row, dim=(0,),
                          output_shape=vc.shape, r=vc)
    for hk in range(HKV if KVL == "head" else 0):
        dst = ax.dsl.concat(ax.dsl.from_numpy(np.array([hk], dtype=np.uint32)), p, dim=0)
        ax.dsl.dynamic_st(k[:, hk : hk + 1, :].reshape(1, 1, D), src0, dst, one_row, dim=(0, 1),
                          output_shape=kc.shape, r=kc)
        ax.dsl.dynamic_st(v[:, hk : hk + 1, :].reshape(1, 1, D), src0, dst, one_row, dim=(0, 1),
                          output_shape=vc.shape, r=vc)

    attn = ax.dsl.create_tensor((1, HQ, D), ax.bf16, ax.MemoryType.OCM, name="attn")
    kv_block_shape = ax.dsl.from_numpy(np.array([1, BC] if KVL == "head" else [BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_stop = ax.dsl.load(nblk)
    for hk in range(HKV):
        head = ax.dsl.from_numpy(np.array([hk], dtype=np.uint32))
        qh = ax.dsl.mul(q[:, hk * G : (hk + 1) * G, :].reshape(G, D), scale, output_dtype=ax.bf16)
        oi = ax.dsl.zeros(shape=(G, D), dtype=ax.bf16)
        li = ax.dsl.zeros(shape=(G,), dtype=ax.fp32)
        mi = ax.dsl.full(shape=(G, 1), dtype=ax.bf16, fill_value=-3e38)
        zero_col = ax.dsl.zeros(shape=(G, 1), dtype=ax.bf16)
        index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

        def body(index, kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape,
                 zero_col, minus_1):
            loop_mi, loop_li, loop_oi = mi, li, oi
            t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
            if KVL == "head":
                off = ax.dsl.concat(head, t0, dim=0)
                kj = ax.dsl.dynamic_ld(kc, off, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
                vj = ax.dsl.dynamic_ld(vc, off, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
            else:  # BC cells x both heads, then this head's columns
                kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D))
                vb = ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D))
                kj = ax.dsl.cast(kb[:, hk * D : (hk + 1) * D], dtype=ax.bf16)
                vj = ax.dsl.cast(vb[:, hk * D : (hk + 1) * D], dtype=ax.bf16)
            mj = ax.dsl.dynamic_ld(mask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                                   output_shape=(1, BC))
            m_s = ax.dsl.create_tensor(shape=(G, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            ax.dsl.add(sj, mj, output_dtype=ax.bf16, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(G)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
            oij = pj @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(G, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
            return (kc, vc, mask, qh, loop_mi, loop_li, loop_oi, head, block_stride, kv_block_shape,
                    mask_block_shape, zero_col, minus_1)

        (kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape, zero_col,
         minus_1) = ax.dsl.range(
            index, range_start, range_stop, range_step, body,
            kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1,
        )
        ax.dsl.div(oi, li.reshape(G, 1), output_dtype=ax.bf16, r=attn[:, hk * G : (hk + 1) * G, :].reshape(G, D))

    gated = sigmoid_mul(attn, gate, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, 1, HQ * D)
    x1 = ax.dsl.add(xt, proj(gated, w_o, "attn_o"), output_dtype=ax.bf16)
    mlp_residual(x1, ln2, w_gate, w_up, w_down, out)
    return out, kc, vc


# ---------------------------------------------------------------- host side


class Weights:
    """the checkpoint's tensors by name (one file or shards listed in model.safetensors.index.json)"""

    def __init__(self, path):
        idx = f"{path}/model.safetensors.index.json"
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
        self.f = [safe_open(f"{path}/{f}", "pt") for f in files]
        self.where = {k: f for f in self.f for k in f.keys()}

    def raw(self, name):
        return self.where[name].get_tensor(name)

    def __call__(self, name):
        return self.raw("model.language_model." + name)


def one_plus(w):
    return (1.0 + w.float()).contiguous()  # Qwen3.5 RMSNorm: x_norm * (1 + w)，fp32 运行时输入


def v_tiled(t, dim, hd):
    """value heads from the checkpoint's grouped order to the GGUF's tiled one along dim (as llama.cpp's
    _LinearAttentionVReorderBase); identity for NV == NK or VORDER=grouped"""
    if VORDER != "tiled" or NV == NK:
        return t
    sh = list(t.shape)
    t = t.reshape(sh[:dim] + [NK, NV // NK, hd] + sh[dim + 1 :]).transpose(dim, dim + 1)
    return t.contiguous().reshape(sh)


def gdn_args(W, i):
    p = f"layers.{i}."
    la = p + "linear_attn."
    qkv = W(la + "in_proj_qkv.weight")
    qkv = torch.cat([qkv[: 2 * KD], v_tiled(qkv[2 * KD :], 0, DV)])
    # checkpoint 是 PyTorch conv1d 布局（第 3 列乘当前 token）；kernel 要第 0 列乘当前 token
    conv = W(la + "conv1d.weight")[:, 0, :].flip(-1)
    conv = torch.cat([conv[: 2 * KD], v_tiled(conv[2 * KD :], 0, DV)])
    return dict(
        ln1=one_plus(W(p + "input_layernorm.weight")),
        w_qkv=qkv.contiguous(),
        w_z=v_tiled(W(la + "in_proj_z.weight"), 0, DV),
        w_b=v_tiled(W(la + "in_proj_b.weight"), 0, 1),
        w_a=v_tiled(W(la + "in_proj_a.weight"), 0, 1),
        conv_w=conv.contiguous(),
        a_log=v_tiled(W(la + "A_log").float(), 0, 1).contiguous(),
        dt_bias=v_tiled(W(la + "dt_bias").float(), 0, 1).contiguous(),
        norm_w=W(la + "norm.weight").float().contiguous(),
        w_out=v_tiled(W(la + "out_proj.weight"), 1, DV),
        **mlp_args(W, p),
    )


def attn_args(W, i):
    p = f"layers.{i}."
    sa = p + "self_attn."
    return dict(
        ln1=one_plus(W(p + "input_layernorm.weight")),
        w_q=W(sa + "q_proj.weight"),
        w_k=W(sa + "k_proj.weight"),
        w_v=W(sa + "v_proj.weight"),
        q_norm=one_plus(W(sa + "q_norm.weight")),
        k_norm=one_plus(W(sa + "k_norm.weight")),
        w_o=W(sa + "o_proj.weight"),
        **mlp_args(W, p),
    )


def mlp_args(W, p):
    return dict(
        ln2=one_plus(W(p + "post_attention_layernorm.weight")),
        w_gate=W(p + "mlp.gate_proj.weight"),
        w_up=W(p + "mlp.up_proj.weight"),
        w_down=W(p + "mlp.down_proj.weight"),
    )


def rope_cos_sin(pos):
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.int64).float() / ROT))
    f = pos * inv  # 纯文本时 mrope 三路位置相同，等价普通 RoPE
    return f.cos().bfloat16().reshape(1, -1), f.sin().bfloat16().reshape(1, -1)


GDN_ORDER = ["x", "ln1", "w_qkv", "w_z", "w_b", "w_a", "conv_w", "a_log", "dt_bias", "norm_w", "w_out", "ln2",
             "w_gate", "w_up", "w_down", "state_in", "conv_in", "out", "state_out", "conv_out"]
ATTN_ORDER = ["x", "ln1", "w_q", "w_k", "w_v", "q_norm", "k_norm", "cos", "sin", "kc", "vc", "pos", "nblk", "mask",
              "w_o", "ln2", "w_gate", "w_up", "w_down", "out"]


class NpuModel:
    def __init__(self, W):
        self.W = W
        self.k_gdn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/gdn")(gdn_layer)
        self.k_attn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/attn")(attn_layer)
        self.args = [gdn_args(W, i) if t == "linear_attention" else attn_args(W, i) for i, t in enumerate(LAYERS)]
        self.state = {}
        for i, t in enumerate(LAYERS):
            if t == "linear_attention":
                self.state[i] = dict(state=torch.zeros((NV, DK, DV), dtype=torch.float32),
                                     conv=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16))
            else:
                self.state[i] = dict(kc=torch.zeros((HKV, LMAX, D), dtype=torch.bfloat16),
                                     vc=torch.zeros((HKV, LMAX, D), dtype=torch.bfloat16))
        self.embed = W("embed_tokens.weight")
        self.norm = one_plus(W("norm.weight"))
        self.layer_ms = []

    def step(self, tok, pos):
        """One decode step; returns (per-layer outputs [L, H] fp32, logits fp32)."""
        x = self.embed[tok].reshape(1, 1, H).clone()
        cos, sin = rope_cos_sin(pos)
        m = torch.full((LMAX,), -3e38)
        m[: pos + 1] = 0
        mask = m.reshape(LMAX // BC, BC).bfloat16()
        outs = []
        for i, t in enumerate(LAYERS):
            out = torch.empty((1, 1, H), dtype=torch.bfloat16)
            a, s = self.args[i], self.state[i]
            t0 = time.time()
            if t == "linear_attention":
                st_out = torch.empty_like(s["state"])
                cv_out = torch.empty_like(s["conv"])
                case = dict(a, x=x, state_in=s["state"], conv_in=s["conv"], out=out, state_out=st_out,
                            conv_out=cv_out)
                self.k_gdn[1](*[case[n] for n in GDN_ORDER])
                s["state"], s["conv"] = st_out, cv_out
            else:
                case = dict(a, x=x, cos=cos, sin=sin, kc=s["kc"], vc=s["vc"],
                            pos=torch.tensor([pos], dtype=torch.int32),
                            nblk=torch.tensor([pos // BC + 1], dtype=torch.int32), mask=mask, out=out)
                RESIDENT.update(kc=s["kc"].clone(), vc=s["vc"].clone())  # 快照：builder 运行前会改写输出 tensor
                self.k_attn[1](*[case[n] for n in ATTN_ORDER])
                RESIDENT.clear()
            self.layer_ms.append((t, time.time() - t0))
            x = out
            outs.append(out.float().reshape(H))
        hf = x.float().reshape(1, H)
        hn = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS) * self.norm
        logits = hn @ self.embed.float().T  # tie_word_embeddings
        return torch.stack(outs), logits.reshape(-1)


def hf_reference(tokens):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        o = model(torch.tensor([tokens]), output_hidden_states=True)
    # hidden_states[0] 是 embedding，[i+1] 是第 i 层输出（最后一项已过 final norm，单独比 logits）
    hs = torch.stack([h[0].float() for h in o.hidden_states[1:]])  # [L, T, H]
    return hs, o.logits[0].float()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(DUMP)
    ax.init(chip_num=1, core_num=1)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(PROMPT)["input_ids"]
    W = Weights(MODEL)
    npu = NpuModel(W)

    tokens = list(prompt_ids)
    npu_hidden, npu_logits = [], []
    t_start = time.time()
    for pos in range(len(prompt_ids) + NGEN - 1):
        hs, lg = npu.step(tokens[pos], pos)
        npu_hidden.append(hs)
        npu_logits.append(lg)
        if pos >= len(prompt_ids) - 1:
            tokens.append(int(lg.argmax()))
        logger.info(f"step pos={pos} tok={tokens[pos]} next={int(lg.argmax())} "
                    f"({tok.decode([int(lg.argmax())])!r}) elapsed={time.time() - t_start:.0f}s")

    ref_hs, ref_logits = hf_reference(tokens[: len(npu_logits)])
    npu_hs = torch.stack(npu_hidden, dim=1)  # [L, T, H]
    L = len(LAYERS)
    worst = []
    for i in range(L - 1):
        c = min(cos(npu_hs[i, t], ref_hs[i, t]) for t in range(npu_hs.shape[1]))
        worst.append(c)
        logger.info(f"layer {i:2d} {LAYERS[i]:17s} min_cos_over_steps={c:.6f}")
    lg = torch.stack(npu_logits)
    lc = min(cos(lg[t], ref_logits[t]) for t in range(lg.shape[0]))
    top_match = sum(int(lg[t].argmax() == ref_logits[t].argmax()) for t in range(lg.shape[0]))
    ref_greedy = [int(ref_logits[t].argmax()) for t in range(len(prompt_ids) - 1, lg.shape[0])]
    gdn_s = [s for t, s in npu.layer_ms if t == "linear_attention"]
    att_s = [s for t, s in npu.layer_ms if t != "linear_attention"]
    logger.info(f"NPU text: {PROMPT!r} + {tok.decode(tokens[len(prompt_ids):])!r}")
    logger.info(f"HF  greedy next tokens: {tok.decode(ref_greedy)!r}")
    logger.info(f"RESULT steps={lg.shape[0]} layer_min_cos={min(worst):.6f} logits_min_cos={lc:.6f} "
                f"top1_match={top_match}/{lg.shape[0]} sim_s_per_layer gdn={np.mean(gdn_s):.1f} "
                f"attn={np.mean(att_s):.1f}")
    torch.save(dict(tokens=tokens, npu_hs=npu_hs, ref_hs=ref_hs, npu_logits=lg, ref_logits=ref_logits),
               f"{DUMP}/result.pt")
    if lc < 0.99:
        sys.exit(3)


if __name__ == "__main__":
    main()
