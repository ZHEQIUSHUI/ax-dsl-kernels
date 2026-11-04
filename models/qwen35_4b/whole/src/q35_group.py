"""Several consecutive decode layers of a Qwen3.5 model in one four-core kernel (MXFP8, packed + column-blocked
parameters as q35_layer_pk): the layers run one after another, each layer's output (every core has all of it after the
MLP's reduction) feeds the next layer on chip, each reduction through shared OCM slots of its own (no slot reused within
a call). One call per group of G layers instead of G calls.

Inputs: x [1, 1, H] (the group's input), the packs of all G layers (pw fp8, ps e8m0, pf f32, pb bf16: layer j's
members as "L<j>.<name>"), the attention's cs [2, ROT/2] (cos, sin), pn [2] int32 (pos, nblk), mask; outputs out (the
group's output), per gated delta net layer j its recurrent state st<j> and conv state cv<j> (read and written in place),
per attention layer j its kc<j> / vc<j>. Written as <KP>_group<G>_<kinds>_mx8_kvc_c<LMAX>_npu<C> with a .layout file
("member pack offset bytes [blk rows cols bw]" per line, the packs as "pack - 0 bytes"); kinds: g / a per layer.

Usage: MODEL=<hf dir> KVL=cell LMAX=16384 BC=256 CORES=4 WFMT=mx8 G=16 FIRST=0 BANKS=1 KSTRIDE=1 \
       python q35_group.py <dir>
"""
import glob
import math
import os
import shutil
import sys

os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("BLOCK", "1")

import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import rmsnorm, silu_gate, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.qwen_gdn_utils import gdn_gate, gdn_recurrent_step, l2norm_lastdim
import board_io
import q35_layer_mc as L
import q35_layer_pk as K
import q35_mx_build as mb
import q35_step as q

assert L.MX and not L.MOE and L.P == 1 and K.BLOCK
G = int(os.environ.get("G", 4))
# KVX=1: the KV heads split over the cores and exchanged through shared OCM (needs HKV % CORES == 0)
KVX = os.environ.get("KVX", "0") == "1"
# LMHEAD=1 (the group covers every layer): also the final norm and the output projection (tied embedding rows, in the
# layers' weight format, or LMFMT=mx8 / mx4), logits [1, V] f32 as one more output; V / CORES rows per core
LMHEAD = os.environ.get("LMHEAD", "0") == "1"
LMFMT = os.environ.get("LMFMT", "mx4" if K.MX4 and not K.MIX else "mx8")
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


# BANKS=1: the row-split projection weights read from the prefill kernel's banks (b<phase>_<group>_<name> [blocks, ..],
# layer j of the group in block j // 4 of phase j % 4: inputs k_<bank name> holding the group's 4 blocks), not packed;
# KSTRIDE=1 moves the column-split ones (w_down, w_out, w_o) to the banks too; otherwise they stay in the packs
BANKS = os.environ.get("BANKS", "0") == "1"
BANK_OF = {  # kind -> member name -> bank name (without b<phase>_)
    "gdn": {"w_gq": "attn_w_q", "s_gq": "attn_w_q_scale", "w_gk": "attn_w_k", "s_gk": "attn_w_k_scale",
            "w_gv": "attn_w_v", "s_gv": "attn_w_v_scale", "w_z": "attn_w_z", "s_z": "attn_w_z_scale",
            "w_a": "attn_w_a", "w_b": "attn_w_b", "w_gate": "mlp_w1", "s_gate": "mlp_w_scale1",
            "w_up": "mlp_w2", "s_up": "mlp_w_scale2"},
    "attn": {"w_q": "attn_w_q", "s_q": "attn_w_scale_q", "w_k": "attn_w_k", "s_k": "attn_w_scale_k",
             "w_v": "attn_w_v", "s_v": "attn_w_scale_v", "w_gate": "mlp_w1", "s_gate": "mlp_w_scale1",
             "w_up": "mlp_w2", "s_up": "mlp_w_scale2"},
}
# KSTRIDE=1 (with BANKS): every projection reads the prefill banks. Each core's column strip of w_down / w_out / w_o
# comes straight from the row-major bank (a strided read), with the same K split and reduction as a packed strip.
KSTRIDE = BANKS and os.environ.get("KSTRIDE", "0") == "1"
# KS_DOWN_ONLY=1: only w_down is read from the banks (w_out / w_o stay packed, column-blocked).
KS_DOWN_ONLY = KSTRIDE and os.environ.get("KS_DOWN_ONLY", "0") == "1"
if KSTRIDE:
    BANK_OF["gdn"].update(w_down="mlp_w3", s_down="mlp_w_scale3")
    BANK_OF["attn"].update(w_down="mlp_w3", s_down="mlp_w_scale3")
    K.BW.pop("w_down")
    if not KS_DOWN_ONLY:
        BANK_OF["gdn"].update(w_out="attn_w_o", s_out="attn_w_o_scale")
        BANK_OF["attn"].update(w_o="attn_w_o", s_o="attn_w_scale_o")
        K.BW.clear()  # row-major bank rows, not column-blocked pack strips
BANK_NAMES = []  # the bank inputs k_<name>, in kernel-input order


def strip(w, c0, c1):
    """KSTRIDE: input columns [c0, c1) of a row-major MX weight (FP4 tensors in logical columns), with its scales."""
    ws = w[:, c0:c1]
    mb.SCALES[id(ws)] = mb.SCALES[id(w)][:, c0 // mb.GROUP : c1 // mb.GROUP]
    return ws


def bank_split(a, kind):
    """a layer's args with w_qkv / s_qkv split into the GDN's q / k / v parts (the banks' tensors)"""
    if kind != "gdn" or "w_qkv" not in a:
        return a
    a = dict(a)
    KD = L.KD
    for n in ("w_qkv", "s_qkv"):
        t = a.pop(n)
        a[n[:2] + "gq"], a[n[:2] + "gk"], a[n[:2] + "gv"] = t[:KD], t[KD : 2 * KD], t[2 * KD :]
    return a
FULL = {}  # (j, w_<x>) -> columns before blocking
# the single-core layer kernels the group is checked against (NVFP4: their twins, t_<x> after each s_<x>)
GL, AL = (mb.nv_twin(mb.gdn_layer_mx), mb.nv_twin(mb.attn_layer_mx)) if K.NVF else (mb.gdn_layer_mx, mb.attn_layer_mx)




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
    wd = strip(m["w_down"], fs.start, fs.stop) if KSTRIDE else q.slice_w(m["w_down"], cols=fs)
    d = q.proj(act, wd, t + "mlp_down", output_dtype=ax.fp32)
    return reduce(c, x1, d, t + "m")


def gdn_layer(c, xt, m, state, cq, ck, cv_, t):
    """one GDN layer; value heads in HF order (core c: key heads c * KHC .., value heads c * KHC * REP .., contiguous),
    conv state as q / k / v tensors [1, CONV - 1, width] (oldest row first)"""
    NK, DK, DV, KD, CONV, KHC, REP = L.NK, L.DK, L.DV, L.KD, L.CONV, L.KHC, L.REP
    VC = KHC * REP  # value heads per core
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, H), eps=EPS)

    def cv(row0, col0, n, st, i):
        if "w_qkv" in m:
            w = q.slice_w(m["w_qkv"], rows=slice(row0, row0 + n))
        else:  # (BANKS: the q / k / v parts)
            part, r0 = ("w_gq", 0) if row0 < KD else ("w_gk", KD) if row0 < 2 * KD else ("w_gv", 2 * KD)
            w = q.slice_w(m[part], rows=slice(row0 - r0, row0 - r0 + n))
        pr = q.proj(h, w, f"{t}gdn_qkv{i}")
        return stateful_runtime_causal_depthwise_conv1d_silu(pr, m["conv_w"][row0 : row0 + n], st[:, :, col0 : col0 + n],
                                                             st[:, :, col0 : col0 + n], CONV, f"{t}gdn_conv{i}")

    qm = cv(c * KHC * DK, c * KHC * DK, KHC * DK, cq, 0).reshape(1, 1, KHC, DK)
    km = cv(KD + c * KHC * DK, c * KHC * DK, KHC * DK, ck, 1).reshape(1, 1, KHC, DK)
    vm = cv(2 * KD + c * VC * DV, c * VC * DV, VC * DV, cv_, 2).reshape(1, 1, VC, DV)
    vs = slice(c * VC, (c + 1) * VC)
    z = q.proj(h, q.slice_w(m["w_z"], rows=slice(c * VC * DV, (c + 1) * VC * DV)), t + "gdn_z").reshape(1, 1, VC, DV)
    b = q.proj(h, m["w_b"], t + "gdn_b")[:, :, vs]
    a = q.proj(h, m["w_a"], t + "gdn_a")[:, :, vs]
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    qt = [ax.dsl.mul(l2norm_lastdim(qm[:, :, j : j + 1, :].reshape(1, DK)), q_scale, output_dtype=ax.bf16)
          for j in range(KHC)]
    kt = [l2norm_lastdim(km[:, :, j : j + 1, :].reshape(1, DK)) for j in range(KHC)]
    a_log_t, dt_bias_t = ax.dsl.load(m["a_log"][vs]), ax.dsl.load(m["dt_bias"][vs])
    core = ax.dsl.create_tensor((1, 1, VC, DV), ax.bf16, name=t + "gdn_core")
    for hl in range(VC):
        j = hl // REP  # the key head of HF value head c * VC + hl
        g, beta = gdn_gate(a[:, :, hl : hl + 1].reshape(1, 1), b[:, :, hl : hl + 1].reshape(1, 1), a_log_t[hl : hl + 1],
                           dt_bias_t[hl : hl + 1])
        vh = c * VC + hl
        st = ax.dsl.load(state[vh : vh + 1]).reshape(DK, DV)
        o_h, st = gdn_recurrent_step(qt[j], kt[j], vm[:, :, hl : hl + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
        ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, :, hl : hl + 1, :])
        ax.dsl.store(x=st.reshape(1, DK, DV), r=state[vh : vh + 1])
    normed = rmsnorm(core, ax.dsl.load(m["norm_w"]).reshape(1, 1, 1, DV), eps=EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, 1, VC * DV)
    wo = (strip(m["w_out"], c * VC * DV, (c + 1) * VC * DV) if KSTRIDE and not KS_DOWN_ONLY
          else q.slice_w(m["w_out"], cols=slice(c * VC * DV, (c + 1) * VC * DV)))
    o = q.proj(gated, wo, t + "gdn_out", output_dtype=ax.fp32)
    return mlp(c, reduce(c, xt, o, t + "a"), m, t)


def kv_write_paged(kv, k, v, krow, kvs=None, rzero=None):
    """the new K / V row (all heads) into the paged cache: K at flat row krow (page p, row r: p * 2 * BC + r), V at
    krow + BC; then a zero read back from them (whatever depends on it runs after this core's store landed)"""
    HKV, D, BC, NBLK = L.HKV, L.D, L.BC, L.NBLK
    flat = kv.reshape(NBLK * 2 * BC, HKV * D)
    vrow = ax.dsl.add(krow, ax.dsl.from_numpy(np.array([BC], dtype=np.int32)), output_dtype=ax.int32)
    ax.dsl.select_scatter(k.reshape(1, HKV * D), krow, dim=0, r=flat)
    ax.dsl.select_scatter(v.reshape(1, HKV * D), vrow, dim=0, r=flat)
    kb = ax.dsl.index_select(flat, krow, dim=0)
    vb = ax.dsl.index_select(flat, vrow, dim=0)
    back = ax.dsl.add(ax.dsl.cast(kb[:, 0:1], dtype=ax.fp32), ax.dsl.cast(vb[:, 0:1], dtype=ax.fp32), output_dtype=ax.fp32)
    if rzero is None:
        z = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=0.0)
    else:  # a runtime-false scalar: the read-back cannot be folded, so the stores really land
        z = ax.dsl.cast(rzero, dtype=ax.fp32).reshape(1)
    return ax.dsl.mul(back.reshape(1), z, output_dtype=ax.fp32)


def attn_layer(c, xt, m, cs, pn, mask, kv, t):
    assert KVX and L.DEC_SPLIT
    QH, QW, D, G_, HKV, ROT, LMAX = L.QH, L.QW, L.D, L.G, L.HKV, L.ROT, L.LMAX
    cos_t, sin_t = ax.dsl.load(cs[0:1]), ax.dsl.load(cs[1:2])
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, q.slice_w(m["w_q"], rows=slice(c * QH * QW, (c + 1) * QH * QW)), t + "attn_q").reshape(1, QH, QW)
    qh = rmsnorm(qg[:, :, 0:D], ax.dsl.load(m["q_norm"]).reshape(1, 1, D), eps=EPS)
    qh = rope(qh, cos_t, sin_t, rotary_dim=ROT)
    hk0, hk1 = (c * QH) // G_, ((c + 1) * QH - 1) // G_
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, ax.MemoryType.OCM, name=t + "attn")
    p = ax.dsl.cast(ax.dsl.load(pn[0:1]), dtype=ax.uint32)
    if KVX:
        # each core projects HKV / CORES KV heads; the K / V rows (and, splitting the blocks, the scaled queries) go
        # through shared OCM in one exchange, then every core writes the whole K / V row itself and reads only after
        # its own write: the KV weights read once per call, not once per core
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
            qa = ax.dsl.add(qa, ax.dsl.cast(zero, dtype=ax.bf16), output_dtype=ax.bf16)  # exact: + 0
            attention_split_qa(c, qa, kv, pn[1:2], mask, attn, t)
    gated = (sigmoid_mul(attn, qg[:, :, D : 2 * D], activation_output_dtype=ax.fp32, output_dtype=ax.bf16) if q.GATE
             else attn).reshape(1, 1, QH * D)
    wo = (strip(m["w_o"], c * QH * D, (c + 1) * QH * D) if KSTRIDE and not KS_DOWN_ONLY
          else q.slice_w(m["w_o"], cols=slice(c * QH * D, (c + 1) * QH * D)))
    o = q.proj(gated, wo, t + "attn_o", output_dtype=ax.fp32)
    return mlp(c, reduce(c, xt, o, t + "a"), m, t)


def attention_split_qa(c, qa, kv, nblk, mask, attn, tag=""):
    """q35_layer_mc.attention_split on the queries of all heads qa [HQ, D] (scaled, + 0 after the own KV write) the
    caller exchanged already: the cache blocks over the cores, partials merged per core's own heads into attn"""
    import ml_dtypes
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
    ax.dsl.div(o_sum, l_sum, output_dtype=ax.bf16, r=attn.reshape(QH, D))


def attention_split_qa_rows(c, qa, kv, nblk, masks, attn, tag=""):
    """attention_split_qa for T query rows at once (the verify kernel): qa [HKV, T, G, D] the rows' scaled queries (all
    heads, exchanged already, + 0 after the rows' KV writes), masks the T rows' masks ([NBLK, BC] each: a row sees the
    cells up to its own), nblk the last row's block count; one pass over the cache blocks for all rows (the scores of
    the T * G rows of a KV head in one matmul); attn [HC, T, G, D] = this core's heads. bf16 KV cache (the Qwen3.5
    whole-model kernels' format)."""
    import ml_dtypes
    HQ, QH, D, HKV, BC, NBLK, G_ = L.HQ, L.QH, L.D, L.HKV, L.BC, L.NBLK, L.G
    T = len(masks)
    RT = T * G_  # rows per KV head: [t][g]
    R = HKV * RT  # all rows: [hk][t][g]
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([c], dtype=np.int32), prefix=f"{tag}split_start")
    range_step = ax.dsl.from_numpy(np.array([CORES], dtype=np.int32), prefix=f"{tag}split_step")
    range_stop = ax.dsl.load(nblk)
    qa2 = qa.reshape(R, D)
    oi = ax.dsl.zeros(shape=(R, D), dtype=ax.bf16)
    li = ax.dsl.zeros(shape=(R,), dtype=ax.fp32)
    mi = ax.dsl.full(shape=(R, 1), dtype=ax.bf16, fill_value=-3e38)
    zero_col = ax.dsl.zeros(shape=(R, 1), dtype=ax.bf16)
    index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

    def body(index, kv, qa2, mi, li, oi, zero_col, minus_1, *rest):
        loop_mi, loop_li, loop_oi = mi, li, oi
        masks = rest
        page = ax.dsl.index_select(kv.reshape(NBLK, 2 * BC * HKV * D), index, dim=0).reshape(2 * BC, HKV * D)
        kb, vb = page[0:BC], page[BC : 2 * BC]
        mj = [L.mask_block(mk, index, 1) for mk in masks]
        m_s = ax.dsl.create_tensor(shape=(R, 1 + BC + 1), dtype=ax.bf16)
        ax.dsl.move(r=m_s[:, :1], x=mi)
        ax.dsl.move(r=m_s[:, -1:], x=zero_col)
        for hk in range(HKV):
            sj = ax.dsl.matmul(qa2[hk * RT : (hk + 1) * RT, :], kb[:, hk * D : (hk + 1) * D], transposed=True,
                               output_dtype=ax.bf16)
            for t in range(T):
                r0 = hk * RT + t * G_
                ax.dsl.add(sj[t * G_ : (t + 1) * G_, :], mj[t], output_dtype=ax.bf16, r=m_s[r0 : r0 + G_, 1 : 1 + BC])
        m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
        m_s = ax.dsl.sub(m_s, m_next, r=m_s)
        pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
        p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
        lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
        ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(R)
        ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
        ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
        oij = ax.dsl.create_tensor(shape=(R, D), dtype=ax.fp32)
        for hk in range(HKV):
            ax.dsl.matmul(pj[hk * RT : (hk + 1) * RT, :], vb[:, hk * D : (hk + 1) * D], output_dtype=ax.fp32,
                          r=oij[hk * RT : (hk + 1) * RT, :])
        ratio_oi = ax.dsl.mul(ratio.reshape(R, 1), oi, output_dtype=ax.fp32)
        ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
        ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
        return (kv, qa2, loop_mi, loop_li, loop_oi, zero_col, minus_1, *rest)

    (kv, qa2, mi, li, oi, zero_col, minus_1, *rest) = ax.dsl.range(
        index, range_start, range_stop, range_step, body, kv, qa2, mi, li, oi, zero_col, minus_1, *masks)
    part = L.shm_tensor((CORES, R, 2 + D), ax.fp32, name=tag + "attn_part")
    ax.dsl.move(r=part[c : c + 1, :, 0:1], x=ax.dsl.cast(mi, dtype=ax.fp32).reshape(1, R, 1))
    ax.dsl.move(r=part[c : c + 1, :, 1:2], x=li.reshape(1, R, 1))
    ax.dsl.move(r=part[c : c + 1, :, 2:], x=ax.dsl.cast(oi, dtype=ax.fp32).reshape(1, R, D))
    axd.pub(part)
    axd.sub(part)
    RQ = QH * T  # this core's rows: its KV heads' [t][g] blocks are contiguous in [hk][t][g]
    hs = slice(c * RQ, (c + 1) * RQ)
    m_all = part[0:1, hs, 0:1].reshape(RQ, 1)
    for k_ in range(1, CORES):
        m_all = ax.dsl.maximum(m_all, part[k_ : k_ + 1, hs, 0:1].reshape(RQ, 1))
    l_sum, o_sum = None, None
    for k_ in range(CORES):
        w = ax.dsl.exp(ax.dsl.sub(part[k_ : k_ + 1, hs, 0:1].reshape(RQ, 1), m_all, output_dtype=ax.fp32),
                       output_dtype=ax.fp32)
        lw = ax.dsl.mul(part[k_ : k_ + 1, hs, 1:2].reshape(RQ, 1), w, output_dtype=ax.fp32)
        ow = ax.dsl.mul(part[k_ : k_ + 1, hs, 2:].reshape(RQ, D), w, output_dtype=ax.fp32)
        l_sum = lw if l_sum is None else ax.dsl.add(l_sum, lw, output_dtype=ax.fp32)
        o_sum = ow if o_sum is None else ax.dsl.add(o_sum, ow, output_dtype=ax.fp32)
    ax.dsl.div(o_sum, l_sum, output_dtype=ax.bf16, r=attn.reshape(RQ, D))


def members(src, j, banks=None):
    """layer j's parameters: static slices of the packs (scales and column strips registered as q35_layer_pk.unpack),
    with BANKS the row-split weights block j // 4 of the bank inputs"""
    m = {}
    if banks is not None and j != "M":
        kind = KINDS[j]
        for n, bn in BANK_OF[kind].items():
            t = banks["k_b%d_%s" % (j % 4, bn)]
            m[n] = t[j // 4 : j // 4 + 1].reshape(*t.shape[1:])
    for key, (p, off, shape) in SPEC.items():
        lj, n = key.split(".", 1)
        if lj != f"L{j}":  # j == "M": the output head's ("LM.<name>")
            continue
        off, shape = K.dsl_dims(base(p), off, shape)
        m[n] = src[p][off : off + int(np.prod(shape))].reshape(*shape)
    for n in list(m):
        if n.startswith("w_") and "s_" + n[2:] in m:  # (w_gq: s_gq, ...)
            mb.SCALES[id(m[n])] = m["s_" + n[2:]]
            if "t_" + n[2:] in m:
                mb.SCALES2[id(m[n])] = m["t_" + n[2:]]
        if n in K.BW:
            K.BLK[id(m[n])] = (m[n].shape[0] // (FULL[(j, n)] // K.BW[n]), K.BW[n])
    return m


def group_body(x, packs, cs, pn, mask, out, io, banks=None):
    """packs: name -> pack; io: per layer j its state / conv or kc / vc tensors"""
    c = ax.program_id(0)
    mb.SCALES.clear()
    mb.SCALES2.clear()
    K.BLK.clear()
    L.SHM_LIVE.clear()
    xt = ax.dsl.load(x)
    for j, kind in enumerate(KINDS):
        m = members(packs, j, banks)
        if kind == "gdn":
            xt = gdn_layer(c, xt, m, *io[j], f"L{j}_")
        else:
            xt = attn_layer(c, xt, m, cs, pn, mask, io[j][0], f"L{j}_")
    if c == 0:
        ax.dsl.store(L.keep_shm(xt), r=out)  # every shared OCM tensor of the call read once more (q35_layer_mc)
    if LMHEAD:
        m = members(packs, "M")
        hn = rmsnorm(xt, ax.dsl.load(m["norm"]).reshape(1, 1, H), eps=EPS)
        V = SPEC["LM.w_lm"][2][0]
        rows = slice(c * (V // CORES), (c + 1) * (V // CORES))
        w = m["w_lm"]
        tile = max(16, min(1024, (256 * 1024) // (H // 2))) if LMFMT == "mx4" else 256
        lg = linear(hn, w[rows], m["s_lm"][rows], "lm_head", output_dtype=ax.fp32, oc_tile_size=tile)
        ax.dsl.store(lg.reshape(1, V // CORES), r=io[-1][:, rows])


def kernel_source():
    """the kernel as module source (the builder reads it): its parameters name each layer's state / KV"""
    outs, ios = [], []
    for j, kind in enumerate(KINDS):
        names = [f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"] if kind == "gdn" else [f"kv{j}"]
        outs += names
        ios.append("(" + ", ".join(names) + ",)")
    packs = list(PDT)
    lo = ["logits"] if LMHEAD else []
    params = ", ".join(["x"] + packs + BANK_NAMES + ["cs", "pn", "mask", "out"] + outs + lo)
    pd = ", ".join(f"{p}={p}" for p in packs)
    bd = ", ".join(f"{p}={p}" for p in BANK_NAMES)
    io = ", ".join(ios + (["logits"] if LMHEAD else []))
    return (f"import q35_group as Q\n\n\ndef group_kernel({params}):\n"
            f"    Q.group_body(x, dict({pd}), cs, pn, mask, out, [{io}], {'dict(' + bd + ')' if BANKS else 'None'})\n"
            f"    return {', '.join(['out'] + outs + lo)}\n"), outs


def main():
    sys.modules["q35_group"] = sys.modules[__name__]  # the kernel module's "import q35_group" sees this layout
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(q.MODEL)
    attn_li = q.LAYERS.index("full_attention")
    args, raw = [], []
    for j, kind in enumerate(KINDS):  # random weights of each layer (QW_RANDOM: per layer index)
        conv = (lambda a, names: K.mx4_args(a, names)) if K.MX4 else (lambda a, names: mb.mx_args(a, names)[0])
        mk = lambda: (conv(q.gdn_args(W, FIRST + j), mb.GDN_W) if kind == "gdn"  # noqa: E731
                      else conv(q.attn_args(W, FIRST + j), mb.ATTN_W))
        q.VORDER = "tiled"
        raw.append(mk())
        q.VORDER = "grouped"
        a = mk()
        q.VORDER = "tiled"
        for n in K.BW:
            if n in a:
                FULL[(j, n)] = a[n].shape[1] * (2 if a[n].dtype == torch.float4_e2m1fn_x2 else 1)
        K.FULL_COLS[kind] = {n: FULL[(j, n)] for n in K.BW if n in a}
        args.append(bank_split(K.block_args(a), kind) if BANKS else K.block_args(a))
    # layout: every layer's members in kernel-input order, by dtype, aligned; the layers in S pack sets
    lm = None
    if LMHEAD:
        assert FIRST == 0 and G == len(q.LAYERS), "LMHEAD needs the group of every layer"
        emb = W("embed_tokens.weight")
        wq, sq = K.quant_mxfp4(emb.float()) if LMFMT == "mx4" else mb.quant_mxfp8(emb)
        wq = wq.view(torch.float4_e2m1fn_x2) if LMFMT == "mx4" else wq
        lm = dict(norm=(torch.randn(H) * 0.1 + 1).float(), w_lm=wq, s_lm=sq.view(torch.float8_e8m0fnu))
        if "p4" not in K.PACKS and LMFMT == "mx4":
            K.PACKS = {"p4": torch.float4_e2m1fn_x2, **K.PACKS}
        if K.NVF:  # the head's E8M0 scales: ps holds the layers' E4M3 ones
            K.PACKS["pe"] = torch.float8_e8m0fnu
    order = {"gdn": L.order(GL), "attn": L.order(AL)}
    if BANKS:  # order: the layer kernels' names with w_qkv / s_qkv as their parts
        order = {k: [x for n in o for x in ((n[:2] + "gq", n[:2] + "gk", n[:2] + "gv") if n in ("w_qkv", "s_qkv") else (n,))]
                 for k, o in order.items()}
    items = [(f"L{j}.{n}", j, args[j][n]) for j, kind in enumerate(KINDS) for n in order[kind]
             if torch.is_tensor(args[j].get(n)) and n != "x" and not (BANKS and n in BANK_OF[kind])]
    bank_t = {}
    if BANKS:
        assert G % 4 == 0 and KINDS == ["gdn", "gdn", "gdn", "attn"] * (G // 4)
        for ph in range(4):
            kind = KINDS[ph]
            for n, bn in BANK_OF[kind].items():
                bank_t["k_b%d_%s" % (ph, bn)] = torch.stack([args[j][n] for j in range(ph, G, 4)])
        BANK_NAMES[:] = list(bank_t)
    items += [(f"LM.{n}", G - 1, lm[n]) for n in ("norm", "w_lm", "s_lm")] if LMHEAD else []
    for S in [s for s in range(1, G + 1) if G % s == 0]:
        SPEC.clear()
        used = {(f"{p}_{s}" if s else p): 0 for s in range(S) for p in K.PACKS}
        for key, j, tns in items:
            s = j // (G // S)
            p = K.pack_of(tns) + (f"_{s}" if s else "")
            es_ = tns.element_size()
            off = (used[p] * es_ + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es_
            SPEC[key] = (p, off, tuple(tns.shape))
            used[p] = off + tns.numel()
        if max(used[p] * torch.tensor([], dtype=K.PACKS[base(p)]).element_size() for p in used) < PACKMAX:
            break
    PDT.update({p: K.PACKS[base(p)] for p in used if p in K.PACKS or used[p]})  # set 0 every pack, later sets used
    logger.info(f"{S} pack set(s): " + ", ".join(f"{p} {used[p]}" for p in PDT))
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in PDT.items()}
    sizes = {p: max(1, (used[p] * es[p] + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es[p]) for p in PDT}
    pk = {p: torch.zeros(sizes[p], dtype=torch.uint8 if es[p] == 1 else dt) for p, dt in PDT.items()}
    for key, (p, off, shape) in SPEC.items():
        n = key.split(".", 1)[1]
        tns = (lm[n] if key.startswith("LM.") else args[int(key.split(".")[0][1:])][n]).reshape(-1)
        tns = tns.view(torch.uint8) if es[p] == 1 else tns
        pk[p][off : off + tns.numel()] = tns
    pk = {p: (pk[p].view(dt) if es[p] == 1 else pk[p]) for p, dt in PDT.items()}
    # the kernel module
    src, outs = kernel_source()
    build = os.path.abspath(os.environ.get("DUMP", "out_q35_group"))
    os.makedirs(f"{build}/grp", exist_ok=True)
    tag = "".join("g" if k == "gdn" else "a" for k in KINDS)
    modname = f"group_k_{tag}_{FIRST}"
    with open(f"{build}/grp/{modname}.py", "w") as f:
        f.write(src)
    sys.path.insert(0, f"{build}/grp")
    fn = __import__(modname).group_kernel
    single = {"gdn": L.kern(GL, "g1"), "attn": L.kern(AL, "a1")}  # MXFP8, MXFP4 or NVFP4 args
    os.chdir(f"{build}/grp")
    PERF = os.environ.get("PERF") == "1"  # simulator trace of the group kernel (DUMP_DSL_PERF artifacts in <DUMP>/grp)
    kg = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=not PERF, dump_dsl_perf=PERF, dump_path=f"{build}/grp")(fn)
    LMAX, BC, NV, DK, DV, CONV, KD, VD, HKV, D = L.LMAX, L.BC, L.NV, L.DK, L.DV, L.CONV, L.KD, L.VD, L.HKV, L.D
    state = {"1": [torch.zeros((NV, DK, DV)) if k == "gdn" else torch.zeros((LMAX, HKV * D), dtype=torch.half)
                   for k in KINDS]}
    conv = {"1": [torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16) if k == "gdn" else
                  torch.zeros((LMAX, HKV * D), dtype=torch.half) for k in KINDS]}
    gio = [[torch.zeros((NV, DK, DV)), torch.zeros((1, CONV - 1, KD), dtype=torch.bfloat16),
            torch.zeros((1, CONV - 1, KD), dtype=torch.bfloat16), torch.zeros((1, CONV - 1, VD), dtype=torch.bfloat16)]
           if k == "gdn" else [torch.zeros((LMAX // BC, 2, BC, HKV * D), dtype=torch.bfloat16)] for k in KINDS]
    REP_, NK_ = NV // L.NK, L.NK
    to_tiled = torch.tensor([(t % NK_) * REP_ + t // NK_ for t in range(NV)])  # tiled head t = r * NK + k <- grouped k * REP + r
    kpfx = os.environ.get("KPFX", "qwen35")
    name = f"{kpfx}_group{G}_{tag}{K.FMT}{os.environ.get('NAMESFX', '')}{'_lm' + LMFMT[-1] if LMHEAD else ''}_kvc_c{LMAX}"
    dst = board_io.kernel_file(out_dir, name, CORES)
    with open(dst[: -len(".axmodel")] + ".layout", "w") as f:
        for p in PDT:
            f.write(f"{p} - 0 {sizes[p] * es[p]}\n")
        for key, (p, off, shape) in SPEC.items():
            j = -1 if key.startswith("LM.") else int(key.split(".")[0][1:])
            n = key.split(".", 1)[1]
            numel = int(np.prod(shape))
            w = n if n.startswith("w_") else "w_" + n[2:]
            blk = ""
            if w in K.BW and (j, w) in FULL and not n.startswith("t_"):
                g = 1 if n == w else K.SG
                cols, bw = FULL[(j, w)] // g, K.BW[w] // g
                if base(p) == "p4":  # bytes: two fp4 values each
                    cols, bw = cols // 2, bw // 2
                blk = f" blk {numel // cols} {cols} {bw}"
            f.write(f"{key} {p} {off * es[p]} {numel * es[p]}{blk}\n")
    logger.info(f"wrote {dst[: -len('.axmodel')]}.layout (before the simulation)")
    # INIT_PT / SAVE_PT: chain two groups (the full model): start from a previous group's kernel / reference x and
    # states, then save both at the end. Default: start from the token embedding and zero states, save nothing.
    init = torch.load(os.path.abspath(os.environ["INIT_PT"])) if os.environ.get("INIT_PT") else None
    if init and os.environ.get("INIT_STATE", "carry") != "zero":
        state["1"] = [t.clone() for t in init["ref_state"]]
        conv["1"] = [t.clone() for t in init["ref_conv"]]
        for j in range(len(KINDS)):
            for d, t_ in zip(gio[j], init["k_gio"][j]):
                d.copy_(t_)
    elif init:
        logger.info("chain: group states start at zero (step 0)")
    worst = 1.0
    for step in range(L.STEPS):
        x0 = init["x"].clone() if init and step == 0 else L.tokens(W, step)
        x0_ref = init["ref_x"].clone() if init and step == 0 else x0
        cs, sn = L.rope_rows(step)
        att = dict(cos=cs, sin=sn, pos=torch.tensor([step], dtype=torch.int32),
                   nblk=torch.tensor([step // BC + 1], dtype=torch.int32), mask=L.mask_rows(step))
        x = x0_ref
        for j, kind in enumerate(KINDS):  # the single-core layers one after another
            o = torch.zeros((1, 1, H), dtype=torch.bfloat16)
            case = dict(raw[j], x=x, out=o)
            if kind == "gdn":
                so, co = torch.zeros_like(state["1"][j]), torch.zeros_like(conv["1"][j])
                case.update(state_in=state["1"][j], conv_in=conv["1"][j], state_out=so, conv_out=co)
            else:
                case.update(att, kc=state["1"][j], vc=conv["1"][j])
                q.RESIDENT.update(kc=state["1"][j].clone(), vc=conv["1"][j].clone())
            os.chdir(f"{L.BUILD}/{'g1' if kind == 'gdn' else 'a1'}")
            L.run(single[kind], 1, case, L.order(GL if kind == "gdn" else AL))
            q.RESIDENT.clear()
            if kind == "gdn":
                state["1"][j], conv["1"][j] = so, co
            x = o
        ref = x
        o = torch.zeros((1, 1, H), dtype=torch.bfloat16)
        krow = (step // BC) * 2 * BC + step % BC
        case = dict(pk, **bank_t, x=x0, out=o, cs=torch.cat([cs, sn]), pn=torch.tensor([step, step // BC + 1, krow], dtype=torch.int32),
                    mask=att["mask"])
        if LMHEAD:
            case["logits"] = torch.zeros((1, lm["w_lm"].shape[0]))
        names = iter(outs)
        for j in range(len(KINDS)):  # in place: the simulator zeroes outputs, the board keeps them
            for t_ in gio[j]:
                n = next(names)
                case[n] = t_
                q.RESIDENT[n] = t_.clone()
        os.chdir(f"{build}/grp")
        ax.init(chip_num=1, core_num=CORES)
        mb.SCALES.clear()
        kg[CORES](*[case[n] for n in L.order(fn)])
        q.RESIDENT.clear()
        if LMHEAD:  # the output head on the reference's last hidden state (torch, dequantized weights)
            hf = ref.float().reshape(1, H)
            hn = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS) * lm["norm"]).bfloat16().float()
            if LMFMT == "mx4":
                pk_ = lm["w_lm"].view(torch.uint8)
                code = torch.stack([pk_ & 15, pk_ >> 4], -1).reshape(pk_.shape[0], -1).long()
                dq = K.E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0)
            else:
                dq = lm["w_lm"].float()
            dq = dq * torch.exp2(lm["s_lm"].view(torch.uint8).float() - 127).repeat_interleave(mb.GROUP, -1)
            logger.info(f"group {tag} step {step}: logits cos {L.cos(case['logits'], hn @ dq.T):.6f}")
        cval = L.cos(o, ref)
        same = bool(torch.equal(o.view(torch.int16), ref.view(torch.int16)))
        sc = min([L.cos(gio[j][0][to_tiled], state["1"][j]) for j, k in enumerate(KINDS) if k == "gdn"] or [1.0])
        kvc = min([L.cos(gio[j][0][: step // BC + 1, 0].reshape(-1, HKV * D)[: step + 1].float(),
                         state["1"][j][: step + 1].float()) for j, k in enumerate(KINDS) if k == "attn"] or [1.0])
        logger.info(f"group step {step}: worst K cache cos {kvc:.6f}")
        logger.info(f"group {tag} step {step}: out cos {cval:.6f} (bitwise {same}), worst gdn state cos {sc:.6f}")
        worst = min(worst, cval)
    if os.environ.get("SAVE_PT"):
        torch.save({"x": o, "ref_x": ref,
                    "k_gio": [[t.detach().cpu() for t in gio[j]] for j in range(len(KINDS))],
                    "ref_state": [t.detach().cpu() for t in state["1"]],
                    "ref_conv": [t.detach().cpu() for t in conv["1"]]}, os.path.abspath(os.environ["SAVE_PT"]))
        logger.info(f"saved chain state {os.environ['SAVE_PT']}")
    path = max(glob.glob(f"{build}/grp/cache/group_kernel.*.compiled.axmodel"), key=os.path.getmtime)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst} ({os.path.getsize(dst)} bytes) and its layout (worst cos {worst:.6f})")
    logger.info(f"ACCEPT group out cos {worst:.6f} ({'PASS' if worst >= 0.9995 else 'FAIL'})")
    assert worst >= 0.9995, worst


if __name__ == "__main__":
    main()
