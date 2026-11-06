"""AX kernels for the Qwen3.8-27B MTP block.

Two shape variants share one packed MXFP4 weight set:
  * mtp_t1       one draft row, T=1
  * mtp_prompt   up to 256 prompt rows, a runtime row loop (the MTP KV is filled causally)

The MTP layer is the same full-gated-attention + dense-MLP body as q27m's decode
attention layer, so the attention code is imported from q35_group unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import math
import ml_dtypes

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.linear import _weight_format

import q35_group as Q
import q35_layer_mc as L
import q35_layer_pk as K
import q35_step as q
import q35_mx_build as MB
from mtp_ref import H, MTP_ORDER, ROT, mtp_args

EPS = L.EPS
P = int(__import__("os").environ.get("P", "256"))
CH = int(__import__("os").environ.get("CH", "256"))
KVCH = int(__import__("os").environ.get("KVCH", "128"))
HF32 = int(__import__("os").environ.get("HF32", "0"))
EF32 = int(__import__("os").environ.get("EF32", "0"))
XC = 32
MT = int(__import__("os").environ.get("MT", "64"))
ROWSPLIT = int(__import__("os").environ.get("ROWSPLIT", "0"))  # partition prompt rows over the NPU cores
KV_ORDER = ("ln_e", "ln_h", "w_fc", "s_fc", "ln1", "w_k", "s_k", "w_v", "s_v", "k_norm")

_linear = MB.linear


def _linear_mt(x, w, ws, name, **kw):
    if x.ndim >= 2 and x.shape[-2] > MT:
        kw.setdefault("m_tile_size", MT)
    return _linear(x, w, ws, name, **kw)


MB.linear = _linear_mt


def _ddr_linear(x, w, s, name, *, r, output_dtype=ax.bf16, oc_tile=64):
    weight_format, group_size, packed_factor = _weight_format(w, s)
    assert weight_format == "mx"
    oc, packed_ic = w.shape
    ic, rows = packed_ic * packed_factor, x.shape[-2]
    assert x.shape[-1] == ic and rows % MT == 0 and oc % oc_tile == 0
    for m0 in range(0, rows, MT):
        xb = ax.dsl.load(x[..., m0 : m0 + MT, :])
        for o0 in range(0, oc, oc_tile):
            os = slice(o0, o0 + oc_tile)
            wb = ax.dsl.load(w[os, :])
            sb = ax.dsl.load(s[os, :])
            ob = ax.dsl.create_tensor((*x.shape[:-2], MT, oc_tile), output_dtype,
                                      ax.MemoryType.OCM, name=f"{name}_o{m0}_{o0}")
            ax.dsl.matmul(xb, wb, transposed=True, output_dtype=output_dtype,
                          w_scale=sb.reshape(1, oc_tile, ic // group_size), r=ob)
            dst = r[..., m0 : m0 + MT, os]
            if r.memory_type == ax.MemoryType.DDR:
                ax.dsl.store(ob, r=dst)
            else:
                ax.dsl.move(x=ob, r=dst)
    return r


def mproj(x, w, name, output_dtype=ax.bf16, r=None):
    if r is None:
        return q.proj(x, w, name, output_dtype=output_dtype)
    assert not MB.PRJ_ROWS and not MB.PRJ_KSPLIT, "an explicit mproj output needs the plain projection path"
    s, s2 = MB.SCALES.get(id(w)), MB.SCALES2.get(id(w))
    if (x.memory_type == ax.MemoryType.DDR or r.memory_type == ax.MemoryType.DDR):
        assert s is not None and s2 is None
        return _ddr_linear(x, w, s, name, r=r, output_dtype=output_dtype)
    nv = dict(weight_scale2=s2) if s2 is not None else {}
    tile = MB.oc_tile_fp4(w.shape[1]) if s2 is not None else \
        MB.oc_tile_fp8(w.shape[1]) if s is not None else q.oc_tile(w.shape[1])
    return _linear_mt(x, w, s, name, output_dtype=output_dtype, oc_tile_size=tile, r=r, **nv)


class MtpBundle:
    def __init__(self, packs, sizes, pdt, lines):
        self.packs = packs
        self.sizes = sizes
        self.pdt = pdt
        self.lines = lines


def make_bundle(W):
    args, deq = mtp_args(W, quantized=True)
    K.FULL_COLS["mtp"] = {n: args[n].shape[1] * (2 if args[n].dtype == torch.float4_e2m1fn_x2 else 1)
                          for n in K.BW if n in args}
    ab = K.block_args(args)
    sizes = K.layout("mtp", ab, MTP_ORDER)
    packs = K.packed(ab, "mtp", sizes)
    lines = K.layout_lines("mtp", sizes)
    return MtpBundle(packs, sizes, dict(K.PACKS), lines), args, deq


def make_kv_bundle(W):
    """The compact K/V-only prompt pack: concat norm, eh_proj, ln1 and K/V."""
    args, deq = mtp_args(W, quantized=True)
    sizes = K.layout("mtpkv", args, KV_ORDER)
    packs = K.packed(args, "mtpkv", sizes)
    lines = K.layout_lines("mtpkv", sizes)
    return MtpBundle(packs, sizes, dict(K.PACKS), lines), args, deq


def _ld(t):
    return t if t.memory_type == ax.MemoryType.OCM else ax.dsl.load(t)


def attention_split_qa_pre(c, qa, kv, nblk, mask, attn, tag=""):
    HQ, QH, D, HKV, BC, NBLK, G_ = L.HQ, L.QH, L.D, L.HKV, L.BC, L.NBLK, L.G
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([c], dtype=np.int32), prefix=f"{tag}split_start")
    range_step = ax.dsl.from_numpy(np.array([L.CORES], dtype=np.int32), prefix=f"{tag}split_step")
    range_stop = _ld(nblk)
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
    part = L.shm_tensor((L.CORES, HQ, 2 + D), ax.fp32, name=tag + "attn_part")
    ax.dsl.move(r=part[c : c + 1, :, 0:1], x=ax.dsl.cast(mi, dtype=ax.fp32).reshape(1, HQ, 1))
    ax.dsl.move(r=part[c : c + 1, :, 1:2], x=li.reshape(1, HQ, 1))
    ax.dsl.move(r=part[c : c + 1, :, 2:], x=ax.dsl.cast(oi, dtype=ax.fp32).reshape(1, HQ, D))
    L.axd.pub(part)
    L.axd.sub(part)
    hs = slice(c * QH, (c + 1) * QH)
    m_all = part[0:1, hs, 0:1].reshape(QH, 1)
    for k_ in range(1, L.CORES):
        m_all = ax.dsl.maximum(m_all, part[k_ : k_ + 1, hs, 0:1].reshape(QH, 1))
    l_sum, o_sum = None, None
    for k_ in range(L.CORES):
        w = ax.dsl.exp(ax.dsl.sub(part[k_ : k_ + 1, hs, 0:1].reshape(QH, 1), m_all, output_dtype=ax.fp32),
                       output_dtype=ax.fp32)
        lw = ax.dsl.mul(part[k_ : k_ + 1, hs, 1:2].reshape(QH, 1), w, output_dtype=ax.fp32)
        ow = ax.dsl.mul(part[k_ : k_ + 1, hs, 2:].reshape(QH, D), w, output_dtype=ax.fp32)
        l_sum = lw if l_sum is None else ax.dsl.add(l_sum, lw, output_dtype=ax.fp32)
        o_sum = ow if o_sum is None else ax.dsl.add(o_sum, ow, output_dtype=ax.fp32)
    ax.dsl.div(o_sum, l_sum, output_dtype=ax.bf16, r=attn.reshape(QH, D))



def mlp_pack(c, x1, m, tag):
    """The MTP block's own MLP pack (never the prefill bank): q35_group.mlp's math, KSTRIDE-independent."""
    fs = slice(c * L.FC, (c + 1) * L.FC)
    h2 = rmsnorm(x1, _ld(m["ln2"]).reshape(1, 1, H), eps=EPS)
    act = silu_mul(q.proj(h2, q.slice_w(m["w_gate"], rows=fs), tag + "mlp_gate"),
                   q.proj(h2, q.slice_w(m["w_up"], rows=fs), tag + "mlp_up"),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
    d = q.proj(act, q.slice_w(m["w_down"], cols=fs), tag + "mlp_down", output_dtype=ax.fp32)
    return Q.reduce(c, x1, d, tag + "mlp")

def attn_layer_preloaded(c, xt, m, cs, pn, mask, kv, t):
    """q35_group.attn_layer with cos/sin and pn allowed to arrive as OCM row views."""
    QH, QW, D, G_, HKV, ROT, LMAX = L.QH, L.QW, L.D, L.G, L.HKV, L.ROT, L.LMAX
    cos_t, sin_t = _ld(cs[0:1]), _ld(cs[1:2])
    h = rmsnorm(xt, _ld(m["ln1"]).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, q.slice_w(m["w_q"], rows=slice(c * QH * QW, (c + 1) * QH * QW)), t + "attn_q").reshape(1, QH, QW)
    qh = rmsnorm(qg[:, :, 0:D], _ld(m["q_norm"]).reshape(1, 1, D), eps=EPS)
    qh = q.rope(qh, cos_t, sin_t, rotary_dim=ROT)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, ax.MemoryType.OCM, name=t + "attn")
    p = ax.dsl.cast(_ld(pn[0:1]), dtype=ax.uint32)
    HC, QD = HKV // L.CORES, QH * D
    ks = slice(c * HC * D, (c + 1) * HC * D)
    kc_ = rmsnorm(q.proj(h, q.slice_w(m["w_k"], rows=ks), t + "attn_k").reshape(1, HC, D),
                  _ld(m["k_norm"]).reshape(1, 1, D), eps=EPS)
    kc_ = q.rope(kc_, cos_t, sin_t, rotary_dim=ROT)
    vc_ = q.proj(h, q.slice_w(m["w_v"], rows=ks), t + "attn_v").reshape(1, HC, D)
    qs = ax.dsl.mul(qh.reshape(QH, D), scale, output_dtype=ax.bf16)
    W_ = QD + 2 * HC * D
    ex = L.shm_tensor((L.CORES, W_), ax.bf16, name=t + "qkv_all")
    o0 = QD
    ax.dsl.move(r=ex[c : c + 1, 0:QD], x=qs.reshape(1, QD))
    ax.dsl.move(r=ex[c : c + 1, o0 : o0 + HC * D], x=kc_.reshape(1, HC * D))
    ax.dsl.move(r=ex[c : c + 1, o0 + HC * D : o0 + 2 * HC * D], x=vc_.reshape(1, HC * D))
    L.axd.pub(ex)
    L.axd.sub(ex)
    k = ax.dsl.create_tensor((L.CORES, HC * D), ax.bf16, ax.MemoryType.OCM, name=t + "k_row")
    v = ax.dsl.create_tensor((L.CORES, HC * D), ax.bf16, ax.MemoryType.OCM, name=t + "v_row")
    ax.dsl.move(r=k, x=ex[:, o0 : o0 + HC * D])
    ax.dsl.move(r=v, x=ex[:, o0 + HC * D : o0 + 2 * HC * D])
    zero = Q.kv_write_paged(kv, k.reshape(1, HKV, D), v.reshape(1, HKV, D), _ld(pn[2:3])).reshape(1, 1)
    qa = ax.dsl.create_tensor((L.CORES, QD), ax.bf16, ax.MemoryType.OCM, name=t + "q_heads")
    ax.dsl.move(r=qa, x=ex[:, 0:QD])
    qa = qa.reshape(L.HQ, D)
    qa = ax.dsl.add(qa, ax.dsl.cast(zero, dtype=ax.bf16), output_dtype=ax.bf16)
    attention_split_qa_pre(c, qa, kv, pn[1:2], mask, attn, t)
    gated = (q.sigmoid_mul(attn, qg[:, :, D : 2 * D], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
             if q.GATE else attn).reshape(1, 1, QH * D)
    o = q.proj(gated, q.slice_w(m["w_o"], cols=slice(c * QH * D, (c + 1) * QH * D)), t + "attn_o",
               output_dtype=ax.fp32)
    return mlp_pack(c, Q.reduce(c, xt, o, t + "attn_a"), m, t)


def mtp_compute(c, e, h, m, cs, pn, mask, kv, tag):
    en = rmsnorm(e, ax.dsl.load(m["ln_e"]).reshape(1, 1, H), eps=EPS)
    hn = rmsnorm(h, ax.dsl.load(m["ln_h"]).reshape(1, 1, H), eps=EPS)
    z = q.proj(ax.dsl.concat(en, hn, dim=-1), m["w_fc"], tag + "fc")
    xt = attn_layer_preloaded(c, z, m, cs, pn, mask, kv, tag)
    hnext = rmsnorm(xt, ax.dsl.load(m["head_norm"]).reshape(1, 1, H), eps=EPS)
    return hnext, xt, kv


def mtp_t1(e, h, p4, pw, ps, pf, pb, cs, pn, mask, kv, hnext, hraw):
    c = ax.program_id(0)
    m = K.unpack("mtp", pw, ps, pf, pb, p4=p4)
    hn, xr, kv = mtp_compute(c, ax.dsl.load(e), ax.dsl.load(h), m, cs, pn, mask, kv, "mtp_")
    if c == 0:
        ax.dsl.store(hn, r=hnext)
        ax.dsl.store(xr, r=hraw)
    return hnext, hraw, kv


def norm_sliced(x, w, rows, tag):
    out = ax.dsl.create_tensor((x.shape[0], rows, H), x.dtype, ax.MemoryType.DDR, name=tag + "_norm")
    for r0 in range(0, rows, XC):
        tile = ax.dsl.create_tensor((x.shape[0], XC, H), x.dtype,
                                    name=f"{tag}_norm_tile{r0}")
        rmsnorm(_ld(x[:, r0 : r0 + XC]), _ld(w).reshape(1, 1, H), eps=EPS, r=tile)
        ax.dsl.store(tile, r=out[:, r0 : r0 + XC])
    return out


def eh_proj_rows(e, h, m, c0, rows, tag):
    """z = eh_proj(concat(norm(e), norm(h))) for `rows` rows in one linear call."""
    cat = ax.dsl.create_tensor((1, rows, 2 * H), ax.bf16, ax.MemoryType.DDR, name=tag + "_cat")
    for r0 in range(0, rows, XC):
        en = ax.dsl.create_tensor((1, XC, H), ax.bf16, name=f"{tag}_en{r0}")
        es = _ld(e[:, c0 + r0 : c0 + r0 + XC])
        if es.dtype == ax.fp32:
            es = ax.dsl.cast(es, dtype=ax.bf16)
        rmsnorm(es, _ld(m["ln_e"]).reshape(1, 1, H), eps=EPS, r=en)
        hs = _ld(h[:, c0 + r0 : c0 + r0 + XC])
        if hs.dtype == ax.fp32:
            hs = ax.dsl.cast(hs, dtype=ax.bf16)
        hn = ax.dsl.create_tensor((1, XC, H), ax.bf16, name=f"{tag}_hn{r0}")
        rmsnorm(hs, _ld(m["ln_h"]).reshape(1, 1, H), eps=EPS, r=hn)
        ct = ax.dsl.create_tensor((1, XC, 2 * H), ax.bf16, name=f"{tag}_cat{r0}")
        ax.dsl.concat(en, hn, dim=-1, r=ct)
        ax.dsl.store(ct, r=cat[:, r0 : r0 + XC])
    z = ax.dsl.create_tensor((1, rows, H), ax.bf16, ax.MemoryType.DDR, name=tag + "_z")
    return mproj(cat, m["w_fc"], tag + "fc", r=z)


def mtp_prompt_kv(e, h, p4, pw, ps, pf, pb, cs, kout, vout):
    """K/V of the prompt rows; the host scatters them into the paged MTP KV."""
    c = ax.program_id(0)
    m = K.unpack("mtpkv", pw, ps, pf, pb, p4=p4)
    QH, QW, D, G_, HKV, ROT_, BC = L.QH, L.QW, L.D, L.G, L.HKV, L.ROT, L.BC
    if ROWSPLIT:
        assert P % L.CORES == 0
        parts = [(c * (P // L.CORES), P // L.CORES, f"kvcore{c}_")]
    else:
        parts = [(c0, min(KVCH, P - c0), f"kv{c0}_") for c0 in range(0, P, KVCH)]
    for c0, rows, tag in parts:
        z = eh_proj_rows(e, h, m, c0, rows, tag)
        hh = norm_sliced(z, m["ln1"], rows, tag + "ln1")
        cos_t = _ld(cs[c0 : c0 + rows, 0:1, :]).reshape(rows, ROT // 2)
        sin_t = _ld(cs[c0 : c0 + rows, 1:2, :]).reshape(rows, ROT // 2)
        kraw = ax.dsl.create_tensor((1, rows, HKV * D), ax.bf16, name=tag + "kraw")
        mproj(hh, m["w_k"], tag + "k", r=kraw)
        knorm = ax.dsl.create_tensor((rows, HKV, D), ax.bf16, name=tag + "knorm")
        rmsnorm(kraw.reshape(rows, HKV, D), _ld(m["k_norm"]).reshape(1, 1, D), eps=EPS, r=knorm)
        krope = ax.dsl.create_tensor((rows, HKV, D), ax.bf16, name=tag + "krope")
        q.rope(knorm, cos_t, sin_t, rotary_dim=ROT_, r=krope)
        ax.dsl.store(krope.reshape(rows, HKV * D), r=kout[c0 : c0 + rows])
        mproj(hh, m["w_v"], tag + "v", r=vout.reshape(1, P, HKV * D)[:, c0 : c0 + rows])
    return kout, vout


def reduce_rows(c, x, partial, tag):
    part = L.shm_tensor((L.CORES, CH, H), ax.fp32, name=tag)
    ax.dsl.move(r=part[c : c + 1], x=partial.reshape(1, CH, H))
    L.axd.pub(part)
    L.axd.sub(part)
    acc = part[0:1]
    for k in range(1, L.CORES):
        acc = ax.dsl.add(acc, part[k : k + 1], output_dtype=ax.fp32)
    return ax.dsl.add(x, ax.dsl.cast(acc, dtype=ax.bf16), output_dtype=ax.bf16)


def keep_shm_safe(y):
    live = list(L.SHM_LIVE)
    L.SHM_LIVE.clear()
    one = (1,) * len(y.shape)
    off = ax.dsl.full(shape=one, dtype=ax.uint8, fill_value=0)
    for i, t in enumerate(live):
        try:
            e = t[tuple(slice(0, 1) for _ in t.shape)].reshape(*one)
        except Exception as exc:
            print(f"keep_shm: skip live[{i}] shape={t.shape} name={getattr(t, 'name', None)}: {exc}")
            continue
        y = ax.dsl.where(off, ax.dsl.cast(e, dtype=y.dtype), y, output_dtype=y.dtype)
    return y


def mtp_prompt(e, h, p4, pw, ps, pf, pb, cs, pn, mask, nblk, kv, hnext, hraw):
    """256 prompt rows in four static 64-row attention/MLP chunks."""
    c = ax.program_id(0)
    m = K.unpack("mtp", pw, ps, pf, pb, p4=p4)
    # Chunk the input norms / eh_proj: a 256-row fp32 RMSNorm output does not
    # fit one CRUN OCM tile (5.2 MB output + 2.6 MB input > 6.9 MB).
    z_all = ax.dsl.create_tensor((1, P, H), ax.bf16, name="mtp_z_all")
    for c0 in range(0, P, CH):
        L.SHM_LIVE.clear()
        en = rmsnorm(ax.dsl.load(e[:, c0 : c0 + CH]), _ld(m["ln_e"]).reshape(1, 1, H), eps=EPS)
        hn_in = rmsnorm(ax.dsl.load(h[:, c0 : c0 + CH]), _ld(m["ln_h"]).reshape(1, 1, H), eps=EPS)
        zc = q.proj(ax.dsl.concat(en, hn_in, dim=-1), m["w_fc"], f"mtp_fc{c0}")
        ax.dsl.move(r=z_all[:, c0 : c0 + CH], x=zc)
    L.SHM_LIVE.clear()
    QH, QW, D, G_, HKV, ROT_, BC = L.QH, L.QW, L.D, L.G, L.HKV, L.ROT, L.BC
    FC, HC, NBLK = L.FC, L.HC, L.NBLK
    kvflat = kv.reshape(NBLK * 2 * BC, HKV * D)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="mtp_split_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="mtp_split_step")
    range_stop = _ld(nblk)

    for c0 in range(0, P, CH):
        tag = f"mtp{c0}_"
        x = z_all[:, c0 : c0 + CH]
        xt = _ld(x)
        hh = rmsnorm(xt, _ld(m["ln1"]).reshape(1, 1, H), eps=EPS)
        cos_t = _ld(cs[c0 : c0 + CH, 0:1, :]).reshape(CH, ROT // 2)
        sin_t = _ld(cs[c0 : c0 + CH, 1:2, :]).reshape(CH, ROT // 2)
        qg = q.proj(hh, q.slice_w(m["w_q"], rows=slice(c * QH * QW, (c + 1) * QH * QW)), tag + "q").reshape(
            CH, QH, QW)
        qh = q.rope(rmsnorm(qg[:, :, 0:D], _ld(m["q_norm"]).reshape(1, 1, D), eps=EPS), cos_t, sin_t,
                    rotary_dim=ROT_)
        k = rmsnorm(q.proj(hh, m["w_k"], tag + "k").reshape(CH, HKV, D), _ld(m["k_norm"]).reshape(1, 1, D),
                    eps=EPS)
        k = q.rope(k, cos_t, sin_t, rotary_dim=ROT_)
        v = q.proj(hh, m["w_v"], tag + "v").reshape(CH, HKV, D)
        krow = _ld(pn[c0 : c0 + CH, 2:3]).reshape(CH)
        vrow = ax.dsl.add(krow, ax.dsl.from_numpy(np.array([BC], dtype=np.int32)), output_dtype=ax.int32)
        ax.dsl.select_scatter(k.reshape(CH, HKV * D), krow, dim=0, r=kvflat)
        ax.dsl.select_scatter(v.reshape(CH, HKV * D), vrow, dim=0, r=kvflat)
        kb0 = ax.dsl.index_select(kvflat, krow[0:1], dim=0)
        vb0 = ax.dsl.index_select(kvflat, vrow[0:1], dim=0)
        zero = ax.dsl.mul(ax.dsl.add(ax.dsl.cast(kb0[:, 0:1], dtype=ax.fp32),
                                     ax.dsl.cast(vb0[:, 0:1], dtype=ax.fp32), output_dtype=ax.fp32),
                          ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=0.0), output_dtype=ax.fp32)
        scl = ax.dsl.add(scale, zero, output_dtype=ax.fp32)
        hk0, hk1 = (c * QH) // G_, ((c + 1) * QH - 1) // G_
        attn = ax.dsl.create_tensor((CH, QH, D), ax.bf16, ax.MemoryType.OCM, name=tag + "attn")
        j = 0
        for hk in range(hk0, hk1 + 1):
            j1 = min(QH, (hk + 1) * G_ - c * QH)
            n, R = j1 - j, (j1 - j) * CH
            qs = ax.dsl.create_tensor((R, D), ax.bf16, name=tag + f"qs{hk}")
            for g in range(n):
                ax.dsl.mul(qh[:, j + g : j + g + 1, :].reshape(CH, D), scl, output_dtype=ax.bf16,
                           r=qs[g * CH : (g + 1) * CH, :])
            oi = ax.dsl.zeros(shape=(R, D), dtype=ax.bf16)
            li = ax.dsl.zeros(shape=(R,), dtype=ax.fp32)
            mi = ax.dsl.full(shape=(R, 1), dtype=ax.bf16, fill_value=-3e38)
            zero_col = ax.dsl.zeros(shape=(R, 1), dtype=ax.bf16)
            index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

            def abody(index, kv, mask, qs, mi, li, oi, zero_col, minus_1):
                loop_mi, loop_li, loop_oi = mi, li, oi
                page = ax.dsl.index_select(kv.reshape(NBLK, 2 * BC, HKV * D), index, dim=0).reshape(2 * BC, HKV * D)
                mj = L.mask_block(mask, index, CH)
                m_s = ax.dsl.create_tensor(shape=(R, 1 + BC + 1), dtype=ax.bf16)
                ax.dsl.move(r=m_s[:, :1], x=mi)
                ax.dsl.move(r=m_s[:, -1:], x=zero_col)
                sj = ax.dsl.matmul(qs, page[0:BC, hk * D : (hk + 1) * D], transposed=True, output_dtype=ax.bf16)
                for g in range(n):
                    ax.dsl.add(sj[g * CH : (g + 1) * CH, :], mj, output_dtype=ax.bf16,
                               r=m_s[g * CH : (g + 1) * CH, 1 : 1 + BC])
                m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
                m_s = ax.dsl.sub(m_s, m_next, r=m_s)
                pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
                p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
                lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
                ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(R)
                ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
                ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
                oij = ax.dsl.matmul(pj, page[BC : 2 * BC, hk * D : (hk + 1) * D], output_dtype=ax.fp32)
                ratio_oi = ax.dsl.mul(ratio.reshape(R, 1), oi, output_dtype=ax.fp32)
                ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
                ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
                return kv, mask, qs, loop_mi, loop_li, loop_oi, zero_col, minus_1

            mask_c0 = mask[:, c0 : c0 + CH, :]
            kv, mask_c, qs, mi, li, oi, zero_col, minus_1 = ax.dsl.range(
                index, range_start, range_stop, range_step, abody, kv, mask_c0, qs, mi, li, oi, zero_col, minus_1)
            o = ax.dsl.div(oi, li.reshape(R, 1), output_dtype=ax.bf16)
            for g in range(n):
                ax.dsl.move(x=o[g * CH : (g + 1) * CH, :].reshape(CH, 1, D), r=attn[:, j + g : j + g + 1, :])
            j = j1
        gated = (sigmoid_mul(attn, qg[:, :, D : 2 * D], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
                 if q.GATE else attn).reshape(1, CH, QH * D)
        hs = slice(c * HC, (c + 1) * HC)
        o = q.proj(gated.reshape(1, CH, QH * D),
                   q.slice_w(m["w_o"], cols=slice(c * QH * D, (c + 1) * QH * D)), tag + "o", output_dtype=ax.fp32)
        x1 = reduce_rows(c, xt, o, tag + "o_part")
        h2 = rmsnorm(x1, _ld(m["ln2"]).reshape(1, 1, H), eps=EPS)
        fs = slice(c * FC, (c + 1) * FC)
        act = silu_mul(q.proj(h2, q.slice_w(m["w_gate"], rows=fs), tag + "gate"),
                       q.proj(h2, q.slice_w(m["w_up"], rows=fs), tag + "up"),
                       activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = q.proj(act, q.slice_w(m["w_down"], cols=fs), tag + "down", output_dtype=ax.fp32)
        x2 = reduce_rows(c, x1, d, tag + "d_part")
        hn = rmsnorm(x2, _ld(m["head_norm"]).reshape(1, 1, H), eps=EPS)
        hn = keep_shm_safe(hn)
        if c == 0:
            ax.dsl.store(hn, r=hnext[:, c0 : c0 + CH])
            ax.dsl.store(x2, r=hraw[:, c0 : c0 + CH])
    return hnext, hraw, kv


T1_INPUTS = ["e", "h", "p4", "pw", "ps", "pf", "pb", "cs", "pn", "mask"]
T1_OUTPUTS = ["hnext", "hraw", "kv"]
PROMPT_INPUTS = ["e", "h", "p4", "pw", "ps", "pf", "pb", "cs", "pn", "mask", "nblk"]
PROMPT_OUTPUTS = ["hnext", "hraw", "kv"]
PROMPTKV_INPUTS = ["e", "h", "p4", "pw", "ps", "pf", "pb", "cs"]
PROMPTKV_OUTPUTS = ["kout", "vout"]

