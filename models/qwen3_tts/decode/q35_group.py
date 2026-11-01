"""Several consecutive decode layers of a Qwen3.5 model in one four-core kernel (MXFP8, packed + column-blocked
parameters as q35_layer_pk): the layers run one after another, each layer's output (every core has all of it after the
MLP's reduction) feeds the next layer on chip, each reduction through shared OCM slots of its own (no slot reused within
a call). One call per group of G layers instead of G calls.

Inputs: x [1, 1, H] (the group's input), the packs of all G layers (pw fp8, ps e8m0, pf f32, pb bf16: layer j's
members as "L<j>.<name>"), the attention's cs [2, ROT/2] (cos, sin), pn [2] int32 (pos, nblk), mask; outputs out (the
group's output), per gated delta net layer j its recurrent state st<j> and conv state cv<j> (read and written in place),
per attention layer j its kc<j> / vc<j>. Written as <KP>_group<G>_<kinds>_mx8_kvc_c<LMAX>_npu<C> with a .layout file
("member pack offset bytes [blk rows cols bw]" per line, the packs as "pack - 0 bytes"); kinds: g / a per layer.

Usage: MODEL=<hf dir> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=mx8 G=4 FIRST=0 python q35_group.py <dir>
"""
import glob
import math
import os
import shutil
import sys

import ml_dtypes

os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("DEC_SPLIT", "1")

import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import silu_gate, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.qwen_gdn_utils import gdn_gate, gdn_recurrent_step, l2norm_lastdim
import board_io
import q35_layer_mc as L
import q35_layer_pk as K
import q35_mx_build as mb
import q35_step as q


def rmsnorm(x, weight, eps=L.EPS, r=None):
    """RMSNorm with FP32 squares / reduction and a BF16 round before the weight multiply, exactly like
    transformers' Qwen3RMSNorm and qwen3_kernel.forward_qwen3_blocks' final norm.  (The shared helper
    keeps the normalized value in FP32 through the weight multiply; that one rounding point is enough to
    shift the 28-layer cosine by ~3e-4.)"""
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=x.shape[-1])
    square_mean = ax.dsl.div(square_sum, hidden)
    epsilon = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=eps)
    denominator = ax.dsl.sqrt(ax.dsl.add(square_mean, epsilon))
    normalized = ax.dsl.div(x, denominator, output_dtype=ax.bf16)
    if r is None:
        return ax.dsl.mul(normalized, weight, output_dtype=ax.bf16)
    return ax.dsl.mul(normalized, weight, r=r)


assert L.MX and not L.MOE and L.P == 1 and K.BLOCK
G = int(os.environ.get("G", 4))
# KVX=1: the KV heads split over the cores and exchanged through shared OCM (needs HKV % CORES == 0)
KVX = os.environ.get("KVX", "1") == "1"
# LMHEAD=1 (the group covers every layer): also the final norm and the output projection (ASR lm_head or TTS
# codec_head, in the layers' weight format, or LMFMT=mx8 / mx4), logits [1, V] f32 as one more output; V / CORES rows
# per core
LMHEAD = os.environ.get("LMHEAD", "0") == "1"
LMFMT = os.environ.get("LMFMT", "mx4" if K.MX4 and (not K.MIX or K.NVF) else "mx8")
FIRST = int(os.environ.get("FIRST", 0))
KINDS = ["gdn" if q.LAYERS[FIRST + j] == "linear_attention" else "attn" for j in range(G)]
CORES, H, EPS = L.CORES, L.H, L.EPS
SPEC = {}  # "L<j>.<name>" -> (pack, element offset, shape)
# a kernel IO's size is kept modulo 2 GiB (a larger pack is bound short and the task fails): the layers' packs in S sets
# of consecutive layers (S the fewest dividing G with every pack under PACKMAX bytes), set s > 0 named "<pack>_<s>"
PACKMAX = int(os.environ.get("PACKMAX", 7 << 28))
PDT = {}  # pack name -> dtype, in kernel-input order


def base(p):
    return p.split("_")[0]
FULL = {}  # (j, w_<x>) -> columns before blocking
# the single-core layer kernels the group is checked against (NVFP4: their twins, t_<x> after each s_<x>)
_only = K.MLP_W if K.MIX else None
GL, AL = ((mb.nv_twin(mb.gdn_layer_mx, _only), mb.nv_twin(mb.attn_layer_mx, _only)) if K.NVF else
          (mb.gdn_layer_mx, mb.attn_layer_mx))




def reduce(c, x, partial, tag):
    """x + the cores' fp32 partials, through a shared OCM slot of its own"""
    part = L.shm_tensor((CORES, 1, H), ax.fp32, name="part_" + tag)
    ax.dsl.move(r=part[c : c + 1], x=partial.reshape(1, 1, H))
    axd.pub(part)
    axd.sub(part)
    acc = part[0:1]
    for k in range(1, CORES):
        acc = ax.dsl.add(acc, part[k : k + 1], output_dtype=ax.fp32)
    return ax.dsl.add(x, ax.dsl.cast(acc, dtype=ax.bf16), output_dtype=ax.bf16)


def mlp(c, x1, m, t):
    fs = slice(c * L.FC, (c + 1) * L.FC)
    h2 = rmsnorm(x1, ax.dsl.load(m["ln2"]).reshape(1, 1, H), eps=EPS)
    act = silu_mul(q.proj(h2, q.slice_w(m["w_gate"], rows=fs), t + "mlp_gate"),
                   q.proj(h2, q.slice_w(m["w_up"], rows=fs), t + "mlp_up"), activation_output_dtype=ax.fp32,
                   output_dtype=ax.bf16)
    d = q.proj(act, q.slice_w(m["w_down"], cols=fs), t + "mlp_down", output_dtype=ax.fp32)
    return reduce(c, x1, d, t + "m")


def gdn_layer(c, xt, m, state, conv, t):
    NK, DK, DV, KD, CONV, KHC, REP = L.NK, L.DK, L.DV, L.KD, L.CONV, L.KHC, L.REP
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, H), eps=EPS)

    def cv(off, n, i):
        ch = slice(off, off + n)
        pr = q.proj(h, q.slice_w(m["w_qkv"], rows=ch), f"{t}gdn_qkv{i}")
        return stateful_runtime_causal_depthwise_conv1d_silu(pr, m["conv_w"][ch], conv[:, :, ch], conv[:, :, ch], CONV,
                                                             f"{t}gdn_conv{i}")

    qm = cv(c * KHC * DK, KHC * DK, 0).reshape(1, 1, KHC, DK)
    km = cv(KD + c * KHC * DK, KHC * DK, 1).reshape(1, 1, KHC, DK)
    b_all = q.proj(h, m["w_b"], t + "gdn_b")
    a_all = q.proj(h, m["w_a"], t + "gdn_a")
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    qt = [ax.dsl.mul(l2norm_lastdim(qm[:, :, j : j + 1, :].reshape(1, DK)), q_scale, output_dtype=ax.bf16)
          for j in range(KHC)]
    kt = [l2norm_lastdim(km[:, :, j : j + 1, :].reshape(1, DK)) for j in range(KHC)]
    o_sum = None
    for vo in range(REP):
        v0 = vo * NK + c * KHC
        vs = slice(v0, v0 + KHC)
        u = "" if REP == 1 else str(vo)
        vm = cv(2 * KD + v0 * DV, KHC * DV, 2 + vo).reshape(1, 1, KHC, DV)
        z = q.proj(h, q.slice_w(m["w_z"], rows=slice(v0 * DV, (v0 + KHC) * DV)), t + "gdn_z" + u).reshape(1, 1, KHC, DV)
        b, a = b_all[:, :, vs], a_all[:, :, vs]
        a_log_t, dt_bias_t = ax.dsl.load(m["a_log"][vs]), ax.dsl.load(m["dt_bias"][vs])
        core = ax.dsl.create_tensor((1, 1, KHC, DV), ax.bf16, name=t + "gdn_core" + u)
        for j in range(KHC):
            g, beta = gdn_gate(a[:, :, j : j + 1].reshape(1, 1), b[:, :, j : j + 1].reshape(1, 1), a_log_t[j : j + 1],
                               dt_bias_t[j : j + 1])
            vh = v0 + j
            st = ax.dsl.load(state[vh : vh + 1]).reshape(DK, DV)
            o_h, st = gdn_recurrent_step(qt[j], kt[j], vm[:, :, j : j + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
            ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, :, j : j + 1, :])
            ax.dsl.store(x=st.reshape(1, DK, DV), r=state[vh : vh + 1])
        normed = rmsnorm(core, ax.dsl.load(m["norm_w"]).reshape(1, 1, 1, DV), eps=EPS)
        gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, 1, KHC * DV)
        o = q.proj(gated, q.slice_w(m["w_out"], cols=slice(v0 * DV, (v0 + KHC) * DV)), t + "gdn_out" + u,
                   output_dtype=ax.fp32)
        o_sum = o if o_sum is None else ax.dsl.add(o_sum, o, output_dtype=ax.fp32)
    return mlp(c, reduce(c, xt, o_sum, t + "a"), m, t)


def kv_write_paged(kv, k, v, krow):
    """the new K / V row (all heads) into the paged cache: K at flat row krow (page p, row r: p * 2 * BC + r), V at
    krow + BC.  Returns zero after a local read-back: that read is only a store fence, never the attention value (the
    board may return another core's stale DDR row here)."""
    HKV, D, BC, NBLK = L.HKV, L.D, L.BC, L.NBLK
    flat = kv.reshape(NBLK * 2 * BC, HKV * D)
    vrow = ax.dsl.add(krow, ax.dsl.from_numpy(np.array([BC], dtype=np.int32)), output_dtype=ax.int32)
    ax.dsl.select_scatter(k.reshape(1, HKV * D), krow, dim=0, r=flat)
    ax.dsl.select_scatter(v.reshape(1, HKV * D), vrow, dim=0, r=flat)
    kb = ax.dsl.index_select(flat, krow, dim=0)
    vb = ax.dsl.index_select(flat, vrow, dim=0)
    back = ax.dsl.add(ax.dsl.cast(kb[:, 0:1], dtype=ax.fp32), ax.dsl.cast(vb[:, 0:1], dtype=ax.fp32), output_dtype=ax.fp32)
    return ax.dsl.mul(back.reshape(1), ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=0.0), output_dtype=ax.fp32)


def attn_layer(c, xt, m, cs, pn, mask, kv, t):
    assert KVX and L.DEC_SPLIT
    QH, QW, D, HKV, ROT = L.QH, L.QW, L.D, L.HKV, L.ROT
    cos_t, sin_t = ax.dsl.load(cs[0:1]), ax.dsl.load(cs[1:2])
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, q.slice_w(m["w_q"], rows=slice(c * QH * QW, (c + 1) * QH * QW)), t + "attn_q").reshape(1, QH, QW)
    qh = rmsnorm(qg[:, :, 0:D], ax.dsl.load(m["q_norm"]).reshape(1, 1, D), eps=EPS)
    qh = rope(qh, cos_t, sin_t, rotary_dim=ROT)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, ax.MemoryType.OCM, name=t + "attn")
    if KVX:
        # Each core projects HKV / CORES KV heads; the K / V rows (and, splitting the blocks, the scaled queries) go
        # through shared OCM in one exchange.  Attention takes the current cell from that copy; the DDR row is written
        # only for later calls.
        HC, QD = HKV // CORES, QH * D
        ks = slice(c * HC * D, (c + 1) * HC * D)
        kc_ = rmsnorm(q.proj(h, q.slice_w(m["w_k"], rows=ks), t + "attn_k").reshape(1, HC, D),
                      ax.dsl.load(m["k_norm"]).reshape(1, 1, D), eps=EPS)
        kc_ = rope(kc_, cos_t, sin_t, rotary_dim=ROT)
        vc_ = q.proj(h, q.slice_w(m["w_v"], rows=ks), t + "attn_v").reshape(1, HC, D)
        qs = ax.dsl.mul(qh.reshape(QH, D), scale, output_dtype=ax.bf16)
        W_ = (QD if L.DEC_SPLIT else 0) + 2 * HC * D
        ex = L.shm_tensor((CORES, W_), ax.bf16, name=t + "qkv_all")
        o0 = QD if L.DEC_SPLIT else 0
        if L.DEC_SPLIT:
            ax.dsl.move(r=ex[c : c + 1, 0:QD], x=qs.reshape(1, QD))
        ax.dsl.move(r=ex[c : c + 1, o0 : o0 + HC * D], x=kc_.reshape(1, HC * D))
        ax.dsl.move(r=ex[c : c + 1, o0 + HC * D : o0 + 2 * HC * D], x=vc_.reshape(1, HC * D))
        axd.pub(ex)
        axd.sub(ex)
        k = ax.dsl.create_tensor((CORES, HC * D), ax.bf16, ax.MemoryType.OCM, name=t + "k_row")
        v = ax.dsl.create_tensor((CORES, HC * D), ax.bf16, ax.MemoryType.OCM, name=t + "v_row")
        ax.dsl.move(r=k, x=ex[:, o0 : o0 + HC * D])
        ax.dsl.move(r=v, x=ex[:, o0 + HC * D : o0 + 2 * HC * D])
        zero = kv_write_paged(kv, k.reshape(1, HKV, D), v.reshape(1, HKV, D), ax.dsl.load(pn[2:3])).reshape(1, 1)
        if L.DEC_SPLIT:
            qa = ax.dsl.create_tensor((CORES, QD), ax.bf16, ax.MemoryType.OCM, name=t + "q_heads")
            ax.dsl.move(r=qa, x=ex[:, 0:QD])
            qa = qa.reshape(L.HQ, D)
            attention_split_qa(c, qa, kv, pn[1:2], mask, attn, k.reshape(HKV, D), v.reshape(HKV, D), t)
    gated = (sigmoid_mul(attn, qg[:, :, D : 2 * D], activation_output_dtype=ax.fp32, output_dtype=ax.bf16) if q.GATE
             else attn).reshape(1, 1, QH * D)
    # Keep the DDR row write in the dependency graph of this call.  The value is exactly zero, so this is a fence only;
    # attention above used the shared K / V copy and never this read-back.
    gated = ax.dsl.add(gated, ax.dsl.cast(zero.reshape(1, 1, 1), dtype=ax.bf16), output_dtype=ax.bf16)
    o = q.proj(gated, q.slice_w(m["w_o"], cols=slice(c * QH * D, (c + 1) * QH * D)), t + "attn_o", output_dtype=ax.fp32)
    return mlp(c, reduce(c, xt, o, t + "a"), m, t)


def attention_split_qa(c, qa, kv, nblk, mask, attn, kcur, vcur, tag=""):
    """q35_layer_mc.attention_split on all scaled queries qa [HQ, D].  The paged DDR history excludes the current cell
    (the caller replaced that mask entry with -3e38); the current cell's score / value are computed from the shared-OCM
    K / V copy kcur / vcur [HKV, D] and merged into the online softmax after the block scan."""
    HQ, QH, D, HKV, BC, NBLK, G_ = L.HQ, L.QH, L.D, L.HKV, L.BC, L.NBLK, L.G
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([c], dtype=np.int32), prefix=f"{tag}split_start")
    range_step = ax.dsl.from_numpy(np.array([CORES], dtype=np.int32), prefix=f"{tag}split_step")
    range_stop = ax.dsl.load(nblk)
    oi = ax.dsl.zeros(shape=(HQ, D), dtype=ax.bf16)
    li = ax.dsl.zeros(shape=(HQ,), dtype=ax.fp32)
    mi = ax.dsl.full(shape=(HQ, 1), dtype=ax.bf16, fill_value=-3e38)
    zero_col = ax.dsl.zeros(shape=(HQ, 1), dtype=ax.bf16)
    index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

    def body(index, kv, mask, qa, mi, li, oi, zero_col, minus_1):
        loop_mi, loop_li, loop_oi = mi, li, oi
        page = ax.dsl.index_select(kv.reshape(NBLK, 2 * BC * HKV * D), index, dim=0).reshape(2 * BC, HKV * D)
        kb, vb = page[0:BC], page[BC : 2 * BC]
        mj = L.mask_block(mask, index, 1)
        m_s = ax.dsl.create_tensor(shape=(HQ, 1 + BC + 1), dtype=ax.bf16)
        ax.dsl.move(r=m_s[:, :1], x=mi)
        ax.dsl.move(r=m_s[:, -1:], x=zero_col)
        for hk in range(HKV):
            sj = ax.dsl.matmul(qa[hk * G_ : (hk + 1) * G_, :], kb[:, hk * D : (hk + 1) * D], transposed=True,
                               output_dtype=ax.bf16)
            ax.dsl.add(sj, mj, output_dtype=ax.bf16, r=m_s[hk * G_ : (hk + 1) * G_, 1 : 1 + BC])
        # Exclude the trailing zero sentinel from the block maximum, as the
        # reference paged kernels do.  An all-negative block would otherwise get
        # m_next == 0, underflow every lane, and turn the final reduction into 0/0.
        m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
        m_s = ax.dsl.sub(m_s, m_next, r=m_s)
        pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
        p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
        lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
        ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(HQ)
        ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
        ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
        oij = ax.dsl.create_tensor(shape=(HQ, D), dtype=ax.fp32)
        for hk in range(HKV):
            ax.dsl.matmul(pj[hk * G_ : (hk + 1) * G_, :], vb[:, hk * D : (hk + 1) * D], output_dtype=ax.fp32,
                          r=oij[hk * G_ : (hk + 1) * G_, :])
        ratio_oi = ax.dsl.mul(ratio.reshape(HQ, 1), oi, output_dtype=ax.fp32)
        ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
        ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
        return (kv, mask, qa, loop_mi, loop_li, loop_oi, zero_col, minus_1)

    (kv, mask, qa, mi, li, oi, zero_col, minus_1) = ax.dsl.range(
        index, range_start, range_stop, range_step, body, kv, mask, qa, mi, li, oi, zero_col, minus_1)
    part = L.shm_tensor((CORES, HQ, 2 + D), ax.fp32, name=tag + "attn_part")
    ax.dsl.move(r=part[c : c + 1, :, 0:1], x=ax.dsl.cast(mi, dtype=ax.fp32).reshape(1, HQ, 1))
    ax.dsl.move(r=part[c : c + 1, :, 1:2], x=li.reshape(1, HQ, 1))
    ax.dsl.move(r=part[c : c + 1, :, 2:], x=ax.dsl.cast(oi, dtype=ax.fp32).reshape(1, HQ, D))
    axd.pub(part)
    axd.sub(part)
    hs = slice(c * QH, (c + 1) * QH)
    m_all = part[0:1, hs, 0:1].reshape(QH, 1)
    for k_ in range(1, CORES):
        m_all = ax.dsl.maximum(m_all, part[k_ : k_ + 1, hs, 0:1].reshape(QH, 1))
    l_sum, o_sum = None, None
    for k_ in range(CORES):
        w = ax.dsl.exp(ax.dsl.sub(part[k_ : k_ + 1, hs, 0:1].reshape(QH, 1), m_all, output_dtype=ax.fp32),
                       output_dtype=ax.fp32)
        lw = ax.dsl.mul(part[k_ : k_ + 1, hs, 1:2].reshape(QH, 1), w, output_dtype=ax.fp32)
        ow = ax.dsl.mul(part[k_ : k_ + 1, hs, 2:].reshape(QH, D), w, output_dtype=ax.fp32)
        l_sum = lw if l_sum is None else ax.dsl.add(l_sum, lw, output_dtype=ax.fp32)
        o_sum = ow if o_sum is None else ax.dsl.add(o_sum, ow, output_dtype=ax.fp32)
    # current token from the shared exchange, not from the DDR row this call wrote
    hk0, hk1_ = hs.start // G_, (hs.stop - 1) // G_
    cur_s = ax.dsl.create_tensor((QH, 1), ax.bf16, name=tag + "cur_s")
    for hk in range(hk0, hk1_ + 1):
        lo = hk * G_ - hs.start
        ax.dsl.matmul(qa[hk * G_ : (hk + 1) * G_, :], kcur[hk : hk + 1, :], transposed=True,
                      output_dtype=ax.bf16, r=cur_s[lo : lo + G_, :])
    m_new = ax.dsl.maximum(m_all, ax.dsl.cast(cur_s, dtype=ax.fp32), output_dtype=ax.fp32)
    w_hist = ax.dsl.exp(ax.dsl.sub(m_all, m_new, output_dtype=ax.fp32), output_dtype=ax.fp32)
    cur_p = ax.dsl.exp(ax.dsl.sub(ax.dsl.cast(cur_s, dtype=ax.fp32), m_new, output_dtype=ax.fp32),
                       output_dtype=ax.fp32)
    cur_v = ax.dsl.create_tensor((QH, D), ax.fp32, name=tag + "cur_v")
    cur_p_bf = ax.dsl.cast(cur_p, dtype=ax.bf16)
    for hk in range(hk0, hk1_ + 1):
        lo = hk * G_ - hs.start
        ax.dsl.matmul(cur_p_bf[lo : lo + G_, :], vcur[hk : hk + 1, :], output_dtype=ax.fp32,
                      r=cur_v[lo : lo + G_, :])
    l_total = ax.dsl.add(ax.dsl.mul(l_sum, w_hist, output_dtype=ax.fp32), cur_p, output_dtype=ax.fp32)
    o_total = ax.dsl.add(ax.dsl.mul(o_sum, w_hist, output_dtype=ax.fp32), cur_v, output_dtype=ax.fp32)
    ax.dsl.div(o_total, l_total, output_dtype=ax.bf16, r=attn.reshape(QH, D))


def members(src, j):
    """layer j's parameters: static slices of the packs (scales and column strips registered as q35_layer_pk.unpack)"""
    mb.SCALES.clear()  # the previous layer's: their tensors are gone, their ids may come back (an MX weight of this
    mb.SCALES2.clear()  # layer would find an NVFP4 scale2 under its id)
    m = {}
    for key, (p, off, shape) in SPEC.items():
        lj, n = key.split(".", 1)
        if lj != f"L{j}":  # j == "M": the output head's ("LM.<name>")
            continue
        off, shape = K.dsl_dims(base(p), off, shape)
        m[n] = src[p][off : off + int(np.prod(shape))].reshape(*shape)
    for n in list(m):
        if n.startswith("w_") and "s_" + n[2:] in m:
            mb.SCALES[id(m[n])] = m["s_" + n[2:]]
            if "t_" + n[2:] in m:
                mb.SCALES2[id(m[n])] = m["t_" + n[2:]]
        if n in K.BW:
            K.BLK[id(m[n])] = (m[n].shape[0] // (FULL[(j, n)] // K.BW[n]), K.BW[n])
    return m


def group_body(x, packs, cs, pn, mask, kv, fnorm, hbuf, out, logits=None):
    """One decode token through every layer.  ``kv`` is the whole paged BF16 cache
    [layers, NBLK, 2, BC, HKV, D]; layer j reads / writes kv[j].  ``pn`` is
    [cell, nblk, krow]."""
    c = ax.program_id(0)
    mb.SCALES.clear()
    mb.SCALES2.clear()
    K.BLK.clear()
    L.SHM_LIVE.clear()
    m_lm = members(packs, "M") if LMHEAD else None
    xt = ax.dsl.load(x)
    # The host supplies the causal mask with the current cell visible.  The current K / V cannot be read back from DDR
    # on the board (another core may still see the old row), so exclude it here and add it from shared OCM instead.
    mask_hist = ax.dsl.load(mask)
    minus_inf = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-3e38))
    cell = ax.dsl.cast(ax.dsl.load(pn[0:1]), dtype=ax.int32)
    ax.dsl.select_scatter(minus_inf, cell, dim=0, r=mask_hist.reshape(L.NBLK * L.BC))
    for j, kind in enumerate(KINDS):
        m = members(packs, j)
        if kind == "gdn":
            raise RuntimeError("the Qwen3 dense group has attention layers only")
        xt = attn_layer(c, xt, m, cs, pn, mask_hist, kv[j : j + 1].reshape(*kv.shape[1:]), f"L{j}_")
    # Materialize the last hidden row in DDR (a runtime scratch input): derived OCM values from the loop's last row are
    # recycled by later OCM allocations in PYRUN.  Only core 0 touches hbuf, and that is its own store/load; all
    # cross-core use of the final row goes through shared OCM below.
    hrow = L.shm_tensor((CORES, 1, H), ax.bf16, name="final_row")
    if c == 0:
        ax.dsl.store(xt, r=hbuf.reshape(1, 1, H))
        xr = ax.dsl.load(hbuf).reshape(1, 1, H)
        # TTS's real final norm is packed with the head: the E2 host still supplies a unit fnorm, so reading the
        # runtime input would leave out and logits unweighted.
        if m_lm is not None and "fnorm" in m_lm:
            fw = ax.dsl.cast(ax.dsl.load(m_lm["fnorm"]).reshape(1, 1, H), dtype=ax.bf16)
        else:
            fw = ax.dsl.load(fnorm).reshape(1, 1, H)
        hn = rmsnorm(xr, fw, eps=EPS)
        ax.dsl.move(r=hrow[0:1], x=hn.reshape(1, 1, H))
    axd.pub(hrow)
    axd.sub(hrow)
    hrow = hrow[0:1]
    if c == 0:
        ax.dsl.store(hrow, r=out)
    if LMHEAD:
        m = m_lm
        V = SPEC["LM.w_lm"][2][0]
        rows = slice(c * (V // CORES), (c + 1) * (V // CORES))
        w = m["w_lm"]
        ch = int(os.environ.get("LM_CHUNK", "512"))
        for k0 in range(rows.start, rows.stop, ch):
            k1 = min(k0 + ch, rows.stop)
            wt = ax.dsl.load(w[k0:k1])
            ws = ax.dsl.load(m["s_lm"][k0:k1])
            xh = hrow
            lg = ax.dsl.matmul(xh, wt, transposed=True, output_dtype=ax.fp32,
                               w_scale=ws.reshape(1, k1 - k0, H // 32))
            ax.dsl.store(lg.reshape(1, k1 - k0), r=logits[:, k0:k1].reshape(1, k1 - k0))


def kernel_source():
    """the kernel as a module source (the builder reads it).  The KV cache is one
    runtime input; the layer index is a static slice of it."""
    packs = list(PDT)
    lo = ["logits"] if LMHEAD else []
    params = ", ".join(["x"] + packs + ["cs", "pn", "mask", "kv", "fnorm", "hbuf", "out"] + lo)
    pd = ", ".join(f"{p}={p}" for p in packs)
    lo_arg = ", logits" if LMHEAD else ""
    return (f"import q35_group as Q\n\n\ndef group_kernel({params}):\n"
            f"    Q.group_body(x, dict({pd}), cs, pn, mask, kv, fnorm, hbuf, out{lo_arg})\n"
            f"    return {', '.join(['out'] + lo)}\n")


def set_kinds(n):
    global G
    G = n
    KINDS[:] = ["attn"] * n


def build_layout(W, layers, *, lmhead):
    """quantize each layer's weights to the group's MXFP8 packs and lay the members out."""
    args, raw = [], []
    for j in range(layers):
        a = mb.mx_args(q.attn_args(W, j), mb.ATTN_W)[0]
        raw.append(a)
        for n in K.BW:
            if n in a:
                FULL[(j, n)] = a[n].shape[1] * (2 if a[n].dtype == torch.float4_e2m1fn_x2 else 1)
        K.FULL_COLS["attn"] = {n: FULL[(j, n)] for n in K.BW if n in a}
        args.append(K.block_args(a))
    lm = None
    if lmhead:
        if q.MODEL_KIND == "tts":
            name = "codec_head.weight"  # talker.codec_head: GGUF output.weight
        else:
            name = "lm_head.weight" if hasattr(W, "has") and W.has("lm_head.weight") else "embed_tokens.weight"
        emb = W(name)
        V = emb.shape[0]
        assert V % CORES == 0
        wq, sq = K.quant_mxfp4(emb.float()) if LMFMT == "mx4" else mb.quant_mxfp8(emb)
        wq = wq.view(torch.float4_e2m1fn_x2) if LMFMT == "mx4" else wq
        lm = dict(w_lm=wq, s_lm=sq.view(torch.float8_e8m0fnu))
        if q.MODEL_KIND == "tts":
            lm["fnorm"] = W("norm.weight").float()
        if "p4" not in K.PACKS and LMFMT == "mx4":
            K.PACKS = {"p4": torch.float4_e2m1fn_x2, **K.PACKS}
        if K.NVF and not K.MIX:
            K.PACKS["pe"] = torch.float8_e8m0fnu
    order = {"attn": L.order(AL)}
    items = [(f"L{j}.{n}", j, args[j][n]) for j in range(layers) for n in order["attn"]
             if torch.is_tensor(args[j].get(n)) and n != "x"]
    items += ([(f"LM.{n}", layers - 1, lm[n]) for n in ("w_lm", "s_lm", "fnorm") if n in lm]
              if lmhead else [])
    for S in [s for s in range(1, layers + 1) if layers % s == 0]:
        SPEC.clear()
        used = {p: 0 for p in K.PACKS}
        for key, j, tns in items:
            p = K.pack_of(tns)
            es_ = tns.element_size()
            off = (used[p] * es_ + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es_
            SPEC[key] = (p, off, tuple(tns.shape))
            used[p] = off + tns.numel()
        if max(used[p] * torch.tensor([], dtype=K.PACKS[base(p)]).element_size() for p in used) < PACKMAX:
            break
    PDT.update({p: K.PACKS[base(p)] for p in used if used[p]})
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in PDT.items()}
    sizes = {p: max(1, (used[p] * es[p] + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es[p]) for p in PDT}
    pk = {p: torch.zeros(sizes[p], dtype=torch.uint8 if es[p] == 1 else dt) for p, dt in PDT.items()}
    for key, (p, off, shape) in SPEC.items():
        n = key.split(".", 1)[1]
        tns = (lm[n] if key.startswith("LM.") else args[int(key.split(".")[0][1:])][n]).reshape(-1)
        tns = tns.view(torch.uint8) if es[p] == 1 else tns
        pk[p][off : off + tns.numel()] = tns
    pk = {p: (pk[p].view(dt) if es[p] == 1 else pk[p]) for p, dt in PDT.items()}
    logger.info(f"{S} pack set(s): " + ", ".join(f"{p} {used[p]}" for p in PDT))
    return pk, lm, sizes, es, args


def write_layout(dst, sizes, es):
    with open(dst, "w") as f:
        for p in PDT:
            f.write(f"{p} - 0 {sizes[p] * es[p]}\n")
        for key, (p, off, shape) in SPEC.items():
            j = -1 if key.startswith("LM.") else int(key.split(".")[0][1:])
            n = key.split(".", 1)[1]
            numel = int(np.prod(shape))
            w = n if n.startswith("w_") else "w_" + n[2:]
            blk = ""
            if w in K.BW and (j, w) in FULL and not n.startswith("t_"):
                g = 1 if n == w else K.sg(n)
                cols, bw = FULL[(j, w)] // g, K.BW[w] // g
                if base(p) == "p4":
                    cols, bw = cols // 2, bw // 2
                blk = f" blk {numel // cols} {cols} {bw}"
            f.write(f"{key} {p} {off * es[p]} {numel * es[p]}{blk}\n")


def prepare_group(W, out_dir, *, layers, lmhead, run_mode, tag="", cache_path=None):
    """Build the one-call group kernel.  Returns (kernel, packs, lm, meta)."""
    global LMHEAD
    LMHEAD = bool(lmhead)
    set_kinds(layers)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pk, lm, sizes, es, args = build_layout(W, layers, lmhead=lmhead)
    src = kernel_source()
    build = os.path.abspath(cache_path or os.path.join(out_dir, "grp"))
    os.makedirs(build, exist_ok=True)
    with open(os.path.join(build, "group_kernel.py"), "w") as f:
        f.write(src)
    if build not in sys.path:
        sys.path.insert(0, build)
    import importlib
    importlib.invalidate_caches()
    fn = importlib.import_module("group_kernel").group_kernel
    kernel = ax.builder(run_mode=ax.RunMode(run_mode), enable_cache=True, unique_axmodel=True,
                        cache_path=os.path.join(build, "cache"), dump_path=build)(fn)
    meta = dict(SPEC=dict(SPEC), packs=list(PDT), sizes=sizes, elem=es, args=args, lm=lm, build=build,
                layers=layers, lmhead=lmhead)
    return kernel, pk, lm, meta


def main():
    sys.modules["q35_group"] = sys.modules[__name__]
    out_dir = os.path.abspath(sys.argv[1])
    W = q.Weights(q.MODEL)
    G = int(os.environ.get("G", len(q.LAYERS)))
    kernel, pk, lm, meta = prepare_group(W, out_dir, layers=G, lmhead=LMHEAD, run_mode="CRUN")
    ax.init(chip_num=1, core_num=CORES)
    name = f"{q.KP}_group{G}_attn{K.FMT}{'_lm' + LMFMT[-1] if LMHEAD else ''}_paged_c{q.LMAX}"
    dst = board_io.kernel_file(out_dir, name, CORES)
    path = max(glob.glob(os.path.join(meta["build"], "cache", "group_kernel.*.compiled.axmodel")),
               key=os.path.getmtime)
    shutil.copy(path, dst)
    write_layout(dst[: -len(".axmodel")] + ".layout", meta["sizes"], meta["elem"])
    logger.info(f"wrote {dst} ({os.path.getsize(dst)} bytes) and its layout")


if __name__ == "__main__":
    main()
