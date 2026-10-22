"""Qwen3.5 prefill layers on CORES NPU cores without synchronization between them: each call splits its work over the
cores (projections by output channel or by head, the delta rule by head), and the projections whose input is split
(out / o / down) leave one fp32 partial result per core. The residual stream passes between calls as a base plus the
CORES partials of the previous call: x = base + sum_c part[c], computed by every core, stored by core 0.

  mlp_mc<P>   x = base + sum(part); x2 = x + down(silu(gate(h)) * up(h)), h = rmsnorm(x): gate / up rows and the
              matching down columns per core -> base_out = x, part_out[c] = the core's down partial
  gdn_mc<P>   x = base + sum(part); the delta-net half of the layer for NV / CORES value heads per core (their q / k / v
              / z rows, conv channels, state rows; b / a for all heads; with NV > NK the core's NK / CORES key heads
              and their value heads) -> base_out = x, part_out[c] = the core's
              out-projection partial (its heads' columns); then mlp_mc. P = 1 (decode): the recurrent step per head
              instead of the chunked delta rule
  attn_head_mc<P>   x = base + sum(part) -> base_out = x; per core HQ / CORES query heads (q pre-scaled, gate) and,
              on cores below HKV, KV head c (k / v f16 rows) -> as q35_flash attn_fd_head
  mul_mat_bf16_n<C>_k<K>   the GEMV of ggml/gguf_gemv_kernels.py (x [1, K] bf16, w [C, K] -> out [1, C]
              f32), C / CORES weight rows per core: the output projection in decode
  attn_tail_mc<P>   the core's query heads of the attention output, sigmoid gate, o-projection partial -> part_out[c];
              then mlp_mc with base = the head's base_out
  moe_pre_mc<P>   MoE models, in place of mlp_mc: x = base + sum(part), P / CORES tokens per core (each core reads all
              of the dense part's weights, a few MB): h2 = rmsnorm(x), the shared expert with its sigmoid gate and the
              router logits -> base_out = x + shared expert, h2_out, logits_out [1, P, NE] f32 (q35_moe.moe_pre);
              the host routes the tokens to the experts (the slots kernel) and adds their results to base_out

WFMT=mx8 also builds gdn_mc with MXFP8 q / k / v / z / b / a / out weights (the core's rows of qkv and z and
columns of out sliced with their scales).

Qwen3 (a config without delta-net layers: KP = qwen3): no gdn_mc, no output gate (no gate_out / gate), q_proj rows
of q only; with more KV heads than cores each core takes HKV / CORES of them, and a core's query heads may span
several KV heads (the tail merges each KV head's rows). WFMT=mx8: head / tail / mlp with MXFP8 weights (w_<x> fp8
and s_<x> scales, sliced per core along with their weights), written <kernel><P>_mx8; the output projection as
mul_mat_mx8_n<C>_k<K> (x, w, s -> out f32), C / CORES rows per core.

Written as <KP>_<kernel><P>[_mx8]_npu<CORES>.axmodel (mul_mat_<bf16|mx8>_n<C>_k<K>_npu<CORES>.axmodel).
Check on the simulator against torch (fp32 reference; MXFP8: on the dequantized weights). Usage: MODEL=<hf dir> P=64
CORES=4 [WFMT=mx8] [TEST=mlp,head,tail,lm] python q35_mc.py [kernel_dir]
"""
import glob
import inspect
import os
import shutil
import sys

os.environ.setdefault("KVL", "cell")

import torch
from loguru import logger

import math

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_gate, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.qwen_gdn_utils import (chunk_prefix_sum_fp32, gdn_gate, gdn_recurrent_step, gdn_stable_chunk,
                                             l2norm_lastdim)
import board_io
import q35_step as q

MOE = "num_experts" in q.cfg  # 35B-A3B: the layers end in the MoE's dense part (moe_pre_mc) instead of the MLP
if MOE:
    import q35_moe as mo  # q.mlp_args: the shared expert, router and shared gate

WFMT = os.environ.get("WFMT", "bf16")
MX = WFMT == "mx8"
if MX:
    import q35_mx_build as mb  # q.proj / q.slice_w: the MX linear, weight slices carry their scales

P = int(os.environ.get("P", 64))
CORES = int(os.environ.get("CORES", 4))
H, F, EPS = q.H, q.F, q.EPS
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_mc"))
FC = F // CORES  # MLP rows per core
NK, NV, DK, DV, KD, VD, CONV = q.NK, q.NV, q.DK, q.DV, q.KD, q.VD, q.CONV
HQ, HKV, D, G, ROT = q.HQ, q.HKV, q.D, q.G, q.ROT
KHC = NK // CORES  # delta-net key heads per core
REP = NV // NK  # value heads per key head
QH = HQ // CORES  # query heads per core
QW = (2 if q.GATE else 1) * D  # q_proj rows per query head ([q | gate] with the output gate)
KH = max(1, HKV // CORES)  # KV heads per core (the cores below HKV when there are fewer)
assert HQ % CORES == 0 and (HKV % CORES == 0 or CORES % HKV == 0)
CHUNK = min(P, int(os.environ.get("CHUNK", 32)))
NP = int(os.environ.get("NP", 8))  # partial attention slots the tail merges (segments x segment-kernel cores)


def residual_in(base, part):
    """x = base + sum_c part[c] (fp32 sum), bf16 [1, P, H]; core 0 stores it to base_out by the caller"""
    acc = ax.dsl.cast(ax.dsl.load(base), dtype=ax.fp32)
    for c in range(CORES):
        acc = ax.dsl.add(acc, ax.dsl.load(part[c : c + 1]), output_dtype=ax.fp32)
    return ax.dsl.cast(acc, dtype=ax.bf16)


def mlp_body(base, part, ln2, w_gate, w_up, w_down, base_out, part_out):
    c = ax.program_id(0)
    x = residual_in(base, part)  # [1, P, H]
    if c == 0:
        ax.dsl.store(x, r=base_out)
    h = rmsnorm(x, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
    rows = slice(c * FC, (c + 1) * FC)
    act = silu_mul(q.proj(h, q.slice_w(w_gate, rows=rows), "mlp_gate"), q.proj(h, q.slice_w(w_up, rows=rows), "mlp_up"),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)  # [1, P, FC]
    down = q.proj(act, q.slice_w(w_down, cols=rows), "mlp_down", output_dtype=ax.fp32)
    ax.dsl.store(down, r=part_out[c : c + 1])


def moe_pre_body(base, part, ln2, w_gate, w_up, w_down, w_router, w_shg, base_out, h2_out, logits_out):
    c = ax.program_id(0)
    TP = P // CORES
    rows = slice(c * TP, (c + 1) * TP)
    acc = ax.dsl.cast(ax.dsl.load(base[:, rows]), dtype=ax.fp32)
    for k in range(CORES):
        acc = ax.dsl.add(acc, ax.dsl.load(part[k : k + 1, rows]), output_dtype=ax.fp32)
    x = ax.dsl.cast(acc, dtype=ax.bf16)  # [1, TP, H]
    h2 = rmsnorm(x, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
    act = silu_mul(q.proj(h2, w_gate, "sh_gate"), q.proj(h2, w_up, "sh_up"), activation_output_dtype=ax.fp32,
                   output_dtype=ax.bf16)
    sh = q.proj(act, w_down, "sh_down", output_dtype=ax.fp32)
    g = ax.dsl.sum(ax.dsl.mul(h2, ax.dsl.load(w_shg).reshape(1, 1, H), output_dtype=ax.fp32), dim=-1, keepdim=True,
                   output_dtype=ax.fp32)
    ax.dsl.store(ax.dsl.add(x, ax.dsl.mul(sh, ax.dsl.sigmoid(g, output_dtype=ax.fp32), output_dtype=ax.fp32),
                            output_dtype=ax.bf16), r=base_out[:, rows])
    ax.dsl.store(h2, r=h2_out[:, rows])
    logits = linear(h2, w_router, None, "router", output_dtype=ax.fp32, oc_tile_size=mo.NE)
    ax.dsl.store(logits, r=logits_out[:, rows])


def moe_pre_mc(base, part, ln2, w_gate, w_up, w_down, w_router, w_shg, base_out, h2_out, logits_out):
    moe_pre_body(base, part, ln2, w_gate, w_up, w_down, w_router, w_shg, base_out, h2_out, logits_out)
    return base_out, h2_out, logits_out


def moe_pre_mc_mx(base, part, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, base_out, h2_out,
                  logits_out):
    scales(w_gate, s_gate, w_up, s_up, w_down, s_down)
    moe_pre_body(base, part, ln2, w_gate, w_up, w_down, w_router, w_shg, base_out, h2_out, logits_out)
    return base_out, h2_out, logits_out


def scales(*ws):
    """MXFP8: the scales of the kernel's weights (w, s pairs), looked up by q.proj / q.slice_w"""
    mb.SCALES.clear()
    for w, s in zip(ws[::2], ws[1::2]):
        mb.SCALES[id(w)] = s


# the builder takes a kernel's inputs from its parameters and its outputs from its return statement: one function per
# IO set (bf16 / MXFP8, with and without the output gate)
def mlp_mc(base, part, ln2, w_gate, w_up, w_down, base_out, part_out):
    mlp_body(base, part, ln2, w_gate, w_up, w_down, base_out, part_out)
    return base_out, part_out


def mlp_mc_mx(base, part, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, base_out, part_out):
    scales(w_gate, s_gate, w_up, s_up, w_down, s_down)
    mlp_body(base, part, ln2, w_gate, w_up, w_down, base_out, part_out)
    return base_out, part_out


def gdn_mc(base, part, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, state_in, conv_in,
           base_out, part_out, state_out, conv_out):
    gdn_body(base, part, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, state_in, conv_in,
             base_out, part_out, state_out, conv_out)
    return base_out, part_out, state_out, conv_out


def gdn_mc_mx(base, part, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out, s_out,
              state_in, conv_in, base_out, part_out, state_out, conv_out):
    scales(w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, w_out, s_out)
    gdn_body(base, part, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, state_in, conv_in,
             base_out, part_out, state_out, conv_out)
    return base_out, part_out, state_out, conv_out


def gdn_body(base, part, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, state_in, conv_in,
             base_out, part_out, state_out, conv_out):
    """The core's NK / CORES key heads and the value heads that use them: in the GGUF's tiled order (value head vh
    uses key head vh % NK) REP = NV / NK strips of KHC consecutive value heads, vo * NK + c * KHC .. + KHC. Every
    q / k / v channel, state row and out column belongs to one core (conv and state are updated in place)."""
    assert NK % CORES == 0 and (REP == 1 or q.VORDER == "tiled") and (P == 1 or P % CHUNK == 0)
    c = ax.program_id(0)
    x = residual_in(base, part)
    if c == 0:
        ax.dsl.store(x, r=base_out)
    h = rmsnorm(x, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)

    def conv(off, n, i):  # the projection and causal conv of channels off .. off + n
        ch = slice(off, off + n)
        pr = q.proj(h, q.slice_w(w_qkv, rows=ch), f"gdn_qkv{i}")
        return stateful_runtime_causal_depthwise_conv1d_silu(pr, conv_w[ch], conv_in[:, :, ch], conv_out[:, :, ch],
                                                             CONV, f"gdn_conv{i}")

    qm = conv(c * KHC * DK, KHC * DK, 0).reshape(1, P, KHC, DK)
    km = conv(KD + c * KHC * DK, KHC * DK, 1).reshape(1, P, KHC, DK)
    b_all = q.proj(h, w_b, "gdn_b")  # all value heads (NV rows), then the strip's
    a_all = q.proj(h, w_a, "gdn_a")
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    if P == 1:  # decode: the recurrent step per head, as q35_step.gdn_layer
        qt = [ax.dsl.mul(l2norm_lastdim(qm[:, :, j : j + 1, :].reshape(1, DK)), q_scale, output_dtype=ax.bf16)
              for j in range(KHC)]
        kt = [l2norm_lastdim(km[:, :, j : j + 1, :].reshape(1, DK)) for j in range(KHC)]
    else:
        qn = ax.dsl.mul(l2norm_lastdim(qm), q_scale, output_dtype=ax.bf16)
        kn = l2norm_lastdim(km)
    o_sum = None
    for vo in range(REP):
        v0 = vo * NK + c * KHC  # the strip's first value head
        vs = slice(v0, v0 + KHC)
        t = "" if REP == 1 else str(vo)
        vm = conv(2 * KD + v0 * DV, KHC * DV, 2 + vo).reshape(1, P, KHC, DV)
        z = q.proj(h, q.slice_w(w_z, rows=slice(v0 * DV, (v0 + KHC) * DV)), "gdn_z" + t).reshape(1, P, KHC, DV)
        b, a = b_all[:, :, vs], a_all[:, :, vs]
        a_log_t, dt_bias_t = ax.dsl.load(a_log[vs]), ax.dsl.load(dt_bias[vs])
        if P == 1:
            core = ax.dsl.create_tensor((1, 1, KHC, DV), ax.bf16, name="gdn_core" + t)
            for j in range(KHC):
                g, beta = gdn_gate(a[:, :, j : j + 1].reshape(1, 1), b[:, :, j : j + 1].reshape(1, 1),
                                   a_log_t[j : j + 1], dt_bias_t[j : j + 1])
                vh = v0 + j
                st = ax.dsl.load(state_in[vh : vh + 1]).reshape(DK, DV)
                o_h, st = gdn_recurrent_step(qt[j], kt[j], vm[:, :, j : j + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
                ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, :, j : j + 1, :])
                ax.dsl.store(x=st.reshape(1, DK, DV), r=state_out[vh : vh + 1])
        else:
            g_all, beta_all = gdn_gate(a, b, a_log_t, dt_bias_t)  # [1, P, KHC]
            prefix_all = chunk_prefix_sum_fp32(g_all, CHUNK, "gdn_prefix_g" + t)
            exp_prefix_all = ax.dsl.exp(prefix_all, output_dtype=ax.fp32)
            core = ax.dsl.create_tensor((1, P, KHC, DV), ax.bf16, name="gdn_core" + t)
            st = ax.dsl.load(state_in[vs]).reshape(KHC, DK, DV)
            for c0 in range(0, P, CHUNK):
                cs = slice(c0, c0 + CHUNK)
                q_tile = ax.dsl.transpose(qn[:, cs], perm=(0, 2, 1, 3)).reshape(KHC, CHUNK, DK)
                k_tile = ax.dsl.transpose(kn[:, cs], perm=(0, 2, 1, 3)).reshape(KHC, CHUNK, DK)
                v_tile = ax.dsl.transpose(vm[:, cs], perm=(0, 2, 1, 3)).reshape(KHC, CHUNK, DV)
                prefix_g = ax.dsl.transpose(prefix_all[:, cs], perm=(0, 2, 1)).reshape(KHC, CHUNK, 1)
                exp_prefix_g = ax.dsl.transpose(exp_prefix_all[:, cs], perm=(0, 2, 1)).reshape(KHC, CHUNK, 1)
                beta = ax.dsl.transpose(beta_all[:, cs], perm=(0, 2, 1)).reshape(KHC, CHUNK, 1)
                out_tile, st = gdn_stable_chunk(q_tile, k_tile, v_tile, beta, st, prefix_g, exp_prefix_g,
                                                f"gdn{t}_c{c0}", solver="triton")
                ax.dsl.move(x=ax.dsl.transpose(out_tile.reshape(1, KHC, CHUNK, DV), perm=(0, 2, 1, 3)), r=core[:, cs])
            ax.dsl.store(x=st.reshape(KHC, DK, DV), r=state_out[vs])
        # gated RMSNorm of the strip's heads, their out-projection partial
        normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
        gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, P, KHC * DV)
        o = q.proj(gated, q.slice_w(w_out, cols=slice(v0 * DV, (v0 + KHC) * DV)), "gdn_out" + t, output_dtype=ax.fp32)
        o_sum = o if o_sum is None else ax.dsl.add(o_sum, o, output_dtype=ax.fp32)
    ax.dsl.store(o_sum, r=part_out[c : c + 1])


def head_body(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, gate_out, k_out, v_out):
    """gate_out None: no output gate (Qwen3)"""
    c = ax.program_id(0)
    x = residual_in(base, part)
    if c == 0:
        ax.dsl.store(x, r=base_out)
    h = rmsnorm(x, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)  # [P, ROT/2]
    qg = q.proj(h, q.slice_w(w_q, rows=slice(c * QH * QW, (c + 1) * QH * QW)), "attn_q").reshape(P, QH, QW)  # [q | gate]
    qh = rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS)
    qh = rope(qh, cos_t, sin_t, rotary_dim=ROT)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    ax.dsl.store(ax.dsl.mul(qh, scale, output_dtype=ax.bf16), r=q_out[:, c * QH : (c + 1) * QH])
    if gate_out is not None:
        ax.dsl.store(qg[:, :, D : 2 * D], r=gate_out[:, c * QH : (c + 1) * QH])
    if c * KH < HKV:  # the core's KV heads
        ks = slice(c * KH * D, (c + 1) * KH * D)
        k = rmsnorm(q.proj(h, q.slice_w(w_k, rows=ks), "attn_k").reshape(P, KH, D),
                    ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
        k = rope(k, cos_t, sin_t, rotary_dim=ROT)
        v = q.proj(h, q.slice_w(w_v, rows=ks), "attn_v").reshape(P, KH * D)
        ax.dsl.store(ax.dsl.cast(k.reshape(P, KH * D), dtype=ax.fp16), r=k_out[:, ks])
        ax.dsl.store(ax.dsl.cast(v, dtype=ax.fp16), r=v_out[:, ks])


def attn_head_mc(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, gate_out, k_out, v_out):
    head_body(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, gate_out, k_out, v_out)
    return base_out, q_out, gate_out, k_out, v_out


def attn_head_mc_ng(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, k_out, v_out):
    head_body(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, None, k_out, v_out)
    return base_out, q_out, k_out, v_out


def attn_head_mc_mx(base, part, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, base_out, q_out, gate_out,
                    k_out, v_out):
    scales(w_q, s_q, w_k, s_k, w_v, s_v)
    head_body(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, gate_out, k_out, v_out)
    return base_out, q_out, gate_out, k_out, v_out


def attn_head_mc_mx_ng(base, part, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, base_out, q_out, k_out,
                       v_out):
    scales(w_q, s_q, w_k, s_k, w_v, s_v)
    head_body(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, None, k_out, v_out)
    return base_out, q_out, k_out, v_out


def tail_body(m_all, l_all, o_all, gate, w_o, part_out):
    """The NP partial attention results (segment kernel outputs [NP][HKV][G * P] rows g * P + t; a slot without
    data has m = -3e38) merged for the core's query heads, per KV head they belong to: m = max m_i, att = sum
    e^(m_i - m) o_i / sum e^(m_i - m) l_i; then the gate (if any) and the o-projection partial. A row that sees no
    cell of a slot has m_i near -3e38 there, so its weight is 0."""
    c = ax.program_id(0)
    h0 = c * QH
    att = ax.dsl.create_tensor((P, QH, D), ax.bf16, name="att")
    for hk in range(h0 // G, (h0 + QH - 1) // G + 1):  # the KV heads of the core's query heads
        g0, g1 = max(h0, hk * G) - hk * G, min(h0 + QH, (hk + 1) * G) - hk * G
        n, rows = g1 - g0, slice(g0 * P, g1 * P)
        mi = ax.dsl.cast(ax.dsl.load(m_all[:, hk : hk + 1, rows]), dtype=ax.fp32).reshape(NP, n * P, 1)
        li = ax.dsl.load(l_all[:, hk : hk + 1, rows]).reshape(NP, n * P, 1)
        oi = ax.dsl.load(o_all[:, hk : hk + 1, rows]).reshape(NP, n * P, D)
        mx = ax.dsl.max(mi, dim=0, keepdim=True)  # [1, n * P, 1]
        w = ax.dsl.exp(ax.dsl.sub(mi, mx, output_dtype=ax.fp32), output_dtype=ax.fp32)  # [NP, n * P, 1]
        lsum = ax.dsl.sum(ax.dsl.mul(w, li, output_dtype=ax.fp32), dim=0, output_dtype=ax.fp32)  # [n * P, 1]
        osum = ax.dsl.sum(ax.dsl.mul(w, oi, output_dtype=ax.fp32), dim=0, output_dtype=ax.fp32)  # [n * P, D]
        a = ax.dsl.div(osum, lsum, output_dtype=ax.bf16).reshape(n, P, D)
        j0 = hk * G + g0 - h0
        ax.dsl.move(x=ax.dsl.transpose(a, perm=(1, 0, 2)), r=att[:, j0 : j0 + n])  # [P, n, D]
    hs = slice(h0, h0 + QH)
    gated = (att if gate is None else sigmoid_mul(att, ax.dsl.load(gate[:, hs]), activation_output_dtype=ax.fp32,
                                                    output_dtype=ax.bf16)).reshape(1, P, QH * D)
    o = q.proj(gated, q.slice_w(w_o, cols=slice(h0 * D, (h0 + QH) * D)), "attn_o", output_dtype=ax.fp32)
    ax.dsl.store(o, r=part_out[c : c + 1])


def attn_tail_mc(m_all, l_all, o_all, gate, w_o, part_out):
    tail_body(m_all, l_all, o_all, gate, w_o, part_out)
    return part_out


def attn_tail_mc_ng(m_all, l_all, o_all, w_o, part_out):
    tail_body(m_all, l_all, o_all, None, w_o, part_out)
    return part_out


def attn_tail_mc_mx(m_all, l_all, o_all, gate, w_o, s_o, part_out):
    scales(w_o, s_o)
    tail_body(m_all, l_all, o_all, gate, w_o, part_out)
    return part_out


def attn_tail_mc_mx_ng(m_all, l_all, o_all, w_o, s_o, part_out):
    scales(w_o, s_o)
    tail_body(m_all, l_all, o_all, None, w_o, part_out)
    return part_out


LM_C, LM_K = int(os.environ.get("LM_C", 31040)), H  # output projection rows per call (the vocabulary in 8 calls)


def mul_mat_bf16(x, w, out):
    c = ax.program_id(0)
    rows = slice(c * (LM_C // CORES), (c + 1) * (LM_C // CORES))
    o = q.linear(ax.dsl.load(x), w[rows], None, "mm", output_dtype=ax.fp32, oc_tile_size=q.oc_tile(w.shape[1]))
    ax.dsl.store(o, r=out[:, rows])
    return out


def mul_mat_mx8(x, w, s, out):
    c = ax.program_id(0)
    rows = slice(c * (LM_C // CORES), (c + 1) * (LM_C // CORES))
    o = q.linear(ax.dsl.load(x), w[rows], s[rows], "mm", output_dtype=ax.fp32, oc_tile_size=256)
    ax.dsl.store(o, r=out[:, rows])
    return out


MLP_ORDER = ["base", "part", "ln2", "w_gate", "w_up", "w_down", "base_out", "part_out"]
GDN_ORDER = ["base", "part", "ln1", "w_qkv", "w_z", "w_b", "w_a", "conv_w", "a_log", "dt_bias", "norm_w", "w_out",
             "state_in", "conv_in", "base_out", "part_out", "state_out", "conv_out"]
HEAD_ORDER = ["base", "part", "ln1", "w_q", "w_k", "w_v", "q_norm", "k_norm", "cos", "sin", "base_out", "q_out",
              "gate_out", "k_out", "v_out"]
TAIL_ORDER = ["m_all", "l_all", "o_all", "gate", "w_o", "part_out"]
# the kernels of this config and weight format: (function, parameter order)
_ng = "" if q.GATE else "_ng"
MLP_FN = mlp_mc_mx if MX else mlp_mc
GDN_FN = gdn_mc_mx if MX else gdn_mc
MOE_FN = moe_pre_mc_mx if MX else moe_pre_mc
HEAD_FN = globals()["attn_head_mc" + ("_mx" if MX else "") + _ng]
TAIL_FN = globals()["attn_tail_mc" + ("_mx" if MX else "") + _ng]


def order(fn):
    return list(inspect.signature(fn).parameters)


def mx_args(a, names):
    """MXFP8: the kernel's arguments (fp8 w_<x>, s_<x>) and the dequantized ones for the reference; bf16: a twice"""
    if not MX:
        return a, a
    return mb.mx_args(a, [n for n in names if n in a])


def ref_mlp(a):
    x = (a["base"].float() + a["part"].float().sum(0, keepdim=True)).bfloat16().float()
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * a["ln2"].float()
    g, u = h @ a["w_gate"].float().T, h @ a["w_up"].float().T
    return x, x + (torch.nn.functional.silu(g) * u) @ a["w_down"].float().T


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.environ["PCHUNK"] = os.environ["R"] = str(P)
    import q35_flash as fl  # noqa: E402  (R = P from the environment)
    import q35_prefill as pf  # noqa: E402

    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    emb = W("embed_tokens.weight")
    tests = os.environ.get("TEST", "mlp,gdn,head,tail").split(",")
    base = emb[torch.randint(0, min(200000, emb.shape[0]), (P,))].reshape(1, P, H).contiguous()
    part = (torch.randn((CORES, P, H)) * 0.02).float()
    x_in = (base.float() + part.sum(0, keepdim=True)).bfloat16()  # the residual both versions see
    zeros = lambda *sh, dt=torch.bfloat16: torch.zeros(sh, dtype=dt)  # noqa: E731
    built = []

    sfx = "_mx8" if MX else ""

    def run(fn, sub, args, order, cores, name=None):
        ax.init(chip_num=1, core_num=cores)
        k = kern(fn, sub)
        if MX:
            mb.SCALES.clear()
        k[cores](*[args[n] for n in order])
        if cores > 1:
            built.append((sub, fn.__name__, f"{q.KP}_{name}{P}{sfx}"))

    if "lm" in tests:
        a = dict(x=torch.randn((1, LM_K), dtype=torch.bfloat16), w=(torch.randn((LM_C, LM_K)) * LM_K**-0.5).bfloat16(),
                 out=zeros(1, LM_C, dt=torch.float32))
        ax.init(chip_num=1, core_num=CORES)
        if MX:
            w8, s8 = mb.quant_mxfp8(a["w"])
            k = kern(mul_mat_mx8, f"lm_mx8_npu{CORES}")
            k[CORES](a["x"], w8, s8, a["out"])
            ref = a["x"].float() @ mb.dequant(w8, s8).T
            built.append((f"lm_mx8_npu{CORES}", "mul_mat_mx8", f"mul_mat_mx8_n{LM_C}_k{LM_K}"))
        else:
            k = kern(mul_mat_bf16, f"lm_npu{CORES}")
            k[CORES](a["x"], a["w"], a["out"])
            ref = a["x"].float() @ a["w"].float().T
            built.append((f"lm_npu{CORES}", "mul_mat_bf16", f"mul_mat_bf16_n{LM_C}_k{LM_K}"))
        logger.info(f"mul_mat_{WFMT} n{LM_C} k{LM_K} npu{CORES}: cos {cos(a['out'], ref):.6f}")

    if "mlp" in tests:
        ma, md = mx_args(q.mlp_args(W, "layers.0."), ["w_gate", "w_up", "w_down"])
        a = dict(ma, base=base, part=part, base_out=zeros(1, P, H), part_out=zeros(CORES, P, H, dt=torch.float32))
        run(MLP_FN, f"mlp{P}{sfx}_npu{CORES}", a, order(MLP_FN), CORES, "mlp_mc")
        x_ref, x2_ref = ref_mlp(dict(md, base=base, part=part))
        x2 = a["base_out"].float() + a["part_out"].sum(0, keepdim=True)
        logger.info(f"mlp_mc P={P} npu{CORES}: base cos {cos(a['base_out'], x_ref):.6f}, out cos {cos(x2, x2_ref):.6f}, "
                    f"delta cos {cos(x2 - a['base_out'].float(), x2_ref - x_ref):.6f}")

    if "gdn" in tests:  # gdn_mc + mlp_mc vs the single-core prefill layer
        st0 = torch.randn((NV, DK, DV)) * 0.02
        cv0 = (torch.randn((1, CONV - 1, 2 * KD + VD)) * 0.1).bfloat16()
        g, _ = mx_args(q.gdn_args(W, 0), ["w_qkv", "w_z", "w_b", "w_a", "w_out", "w_gate", "w_up", "w_down"])
        mw = ("ln2", "w_gate", "w_up", "w_down", "s_gate", "s_up", "s_down")
        ga = {k: v for k, v in g.items() if k not in mw}
        a = dict(ga, base=base, part=part, state_in=st0.clone(), conv_in=cv0.clone(), base_out=zeros(1, P, H),
                 part_out=zeros(CORES, P, H, dt=torch.float32), state_out=torch.zeros_like(st0),
                 conv_out=torch.zeros_like(cv0))
        run(GDN_FN, f"gdn{P}{sfx}_npu{CORES}", a, order(GDN_FN), CORES, "gdn_mc")
        if MOE:  # the reference layer up to its residual (no MoE part): out = x + the delta net
            q.mlp_residual = lambda x1, ln2, w_gate, w_up, w_down, out: ax.dsl.store(x1, r=out)
        m = dict({k: v for k, v in g.items() if k in mw}, base=a["base_out"], part=a["part_out"], base_out=zeros(1, P, H),
                 part_out=zeros(CORES, P, H, dt=torch.float32))
        if not MOE:
            run(MLP_FN, f"mlp{P}{sfx}_npu{CORES}", m, order(MLP_FN), CORES, "mlp_mc")
            out = m["base_out"].float() + m["part_out"].sum(0, keepdim=True)
        else:
            out = a["base_out"].float() + a["part_out"].sum(0, keepdim=True)
        r = dict(g, x=x_in, state_in=st0.clone(), conv_in=cv0.clone(), out=zeros(1, P, H),
                 state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
        # the single-core layer in the same weight format (MXFP8: the MX decode / prefill kernels)
        ref_fn = (mb.gdn_layer_mx if MX else q.gdn_layer) if P == 1 else (pf.gdn_layer_prefill_mx if MX else pf.gdn_layer_prefill)
        run(ref_fn, f"gdn_ref{P}{sfx}", r, order(ref_fn), 1)
        logger.info(f"gdn_mc P={P} npu{CORES} vs the single-core layer: out cos {cos(out, r['out']):.6f}, state cos "
                    f"{cos(a['state_out'], r['state_out']):.6f}, conv equal {torch.equal(a['conv_out'], r['conv_out'])} (differing "
                    f"{(a['conv_out'] != r['conv_out']).sum().item()} of {a['conv_out'].numel()}, max diff "
                    f"{(a['conv_out'].float() - r['conv_out'].float()).abs().max().item():.3g})")

    if "moe" in tests:  # moe_pre_mc vs torch (MXFP8: on the dequantized weights)
        ma, md = mx_args(q.mlp_args(W, "layers.0."), ["w_gate", "w_up", "w_down"])
        a = dict(ma, base=base, part=part, base_out=zeros(1, P, H), h2_out=zeros(1, P, H),
                 logits_out=zeros(1, P, mo.NE, dt=torch.float32))
        run(MOE_FN, f"moe{P}{sfx}_npu{CORES}", a, order(MOE_FN), CORES, "moe_pre_mc")
        x = x_in.float()
        h2 = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * md["ln2"].float()).bfloat16().float()
        sh = (torch.nn.functional.silu(h2 @ md["w_gate"].float().T) * (h2 @ md["w_up"].float().T)) @ md["w_down"].float().T
        bo = x + sh * torch.sigmoid(h2 @ md["w_shg"].float().reshape(H, 1))
        lg = h2 @ md["w_router"].float().T
        logger.info(f"moe_pre_mc P={P} {WFMT} npu{CORES}: base cos {cos(a['base_out'], bo):.6f}, h2 cos "
                    f"{cos(a['h2_out'], h2):.6f}, logits cos {cos(a['logits_out'], lg):.6f}, top-8 agree "
                    f"{(a['logits_out'].topk(8, -1).indices.sort(-1).values == lg.topk(8, -1).indices.sort(-1).values).float().mean().item():.4f}")

    if "head" in tests or "tail" in tests:  # vs q35_flash head (R = P) and torch
        att, att_deq = mx_args(dict(q.attn_args(W, 3)), ["w_q", "w_k", "w_v", "w_o"])
        posn = torch.arange(1000, 1000 + P)
        cs = [q.rope_cos_sin(int(p)) for p in posn]
        cos_t, sin_t = torch.cat([c for c, _ in cs]), torch.cat([s_ for _, s_ in cs])
        outs = lambda: dict(q_out=zeros(P, HQ, D), gate_out=zeros(P, HQ, D), k_out=zeros(P, HKV * D, dt=torch.half),  # noqa: E731
                            v_out=zeros(P, HKV * D, dt=torch.half))
        names = ("q_out", "gate_out", "k_out", "v_out") if q.GATE else ("q_out", "k_out", "v_out")
        if "head" in tests:  # the single-core split-attention head on the same (quantized) weights
            a = dict(att, base=base, part=part, cos=cos_t, sin=sin_t, base_out=zeros(1, P, H), **outs())
            run(HEAD_FN, f"head{P}{sfx}_npu{CORES}", a, order(HEAD_FN), CORES, "attn_head_mc")
            r = dict(att, x=x_in, cos=cos_t, sin=sin_t, **outs())
            ref_fn = getattr(fl, "attn_fd_head" + ("_mx" if MX else "") + ("" if q.GATE else "_ng"))
            run(ref_fn, f"head_ref{P}{sfx}", r, order(ref_fn), 1)
            logger.info(f"attn_head_mc P={P} {WFMT} npu{CORES}: " + ", ".join(
                f"{n} cos {cos(a[n], r[n]):.6f}" for n in names) + f", base equal {torch.equal(a['base_out'], x_in)}")
        if "tail" in tests:
            # scores over NC cells split into NP - 1 slots (the last one empty), rows hk, g * P + t
            NC = 128 * NP
            sc = torch.randn((HKV, G * P, NC)) * 2
            vv = torch.randn((HKV, NC, D)) * 0.5
            sc[:, :, NC // 2 : NC // 2 + 64] = -3e38  # one slot sees nothing
            ma, la, oa = zeros(NP, HKV, G * P, 1), zeros(NP, HKV, G * P, dt=torch.float32), zeros(NP, HKV, G * P, D)
            ma[:] = -3e38
            per = NC // (NP - 1) // 64 * 64
            for i in range(NP - 1):
                cs_ = slice(i * per, NC if i == NP - 2 else (i + 1) * per)
                mm = sc[:, :, cs_].max(-1, keepdim=True).values
                pp = torch.exp(sc[:, :, cs_] - mm)
                ma[i], la[i], oa[i] = mm.bfloat16(), pp.sum(-1), (pp @ vv[:, cs_]).bfloat16()
            ref_att = torch.softmax(sc, -1) @ vv  # [HKV, G * P, D]
            at = ref_att.reshape(HKV, G, P, D).permute(2, 0, 1, 3).reshape(P, HQ, D)
            gt = (torch.randn((P, HQ, D))).bfloat16()
            a = dict(att, m_all=ma, l_all=la, o_all=oa, gate=gt, part_out=zeros(CORES, P, H, dt=torch.float32))
            run(TAIL_FN, f"tail{P}{sfx}_npu{CORES}", a, order(TAIL_FN), CORES, "attn_tail_mc")
            g = torch.sigmoid(gt.float()) if q.GATE else 1.0
            ref = (g * at.float()).reshape(1, P, HQ * D) @ att_deq["w_o"].float().T
            logger.info(f"attn_tail_mc P={P} {WFMT} npu{CORES}: o cos {cos(a['part_out'].sum(0, keepdim=True), ref):.6f}")

    if out_dir:
        for sub, fn, name in built:
            (path,) = glob.glob(f"{BUILD}/{sub}/cache/{fn}.*.compiled.axmodel")
            shutil.copy(path, board_io.kernel_file(out_dir, name, CORES))
            logger.info(f"wrote {board_io.kernel_file(out_dir, name, CORES)}")


if __name__ == "__main__":
    main()
