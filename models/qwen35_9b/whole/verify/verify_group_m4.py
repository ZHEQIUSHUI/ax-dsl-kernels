"""Qwen3.5-9B T=4 verification group with one pass over every weight tile.

All four rows share each projection (M=4): the packed MXFP4 weights are read
once per layer/call.  Only the row recurrences stay sequential: the GDN conv
shift and delta-rule state update, plus each row's KV write and attention.

The kernel ABI is identical to ``verify_group``: ``ntok`` selects the valid
prefix and every per-row state/conv snapshot is still written.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, sigmoid_mul, silu_gate, silu_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.qwen_gdn_utils import gdn_gate, gdn_recurrent_step, l2norm_lastdim
from axnn.dsl.kernels.utils.rope import rope

import q35_group as G
import q35_layer_mc as L
import q35_layer_pk as K
import q35_step as q

T = 4


def layer_io_names(kind, j):
    return [f"st{j}", f"cq{j}", f"ck{j}", f"cv{j}"] if kind == "gdn" else [f"kv{j}"]


def snapshot_shapes():
    nl = sum(k == "gdn" for k in G.KINDS)
    conv = L.CONV - 1
    return [(T, nl, L.NV, L.DK, L.DV), (T, nl, 1, conv, L.KD), (T, nl, 1, conv, L.KD),
            (T, nl, 1, conv, L.VD)]


def reduce_rows(c, x, partial, tag):
    """x + the four cores' fp32 projection partials, for all T rows at once."""
    part = L.shm_tensor((L.CORES, T, L.H), ax.fp32, name="part_" + tag)
    ax.dsl.move(r=part[c : c + 1], x=partial.reshape(1, T, L.H))
    L.axd.pub(part)
    L.axd.sub(part)
    acc = part[0:1]
    for k in range(1, L.CORES):
        acc = ax.dsl.add(acc, part[k : k + 1], output_dtype=ax.fp32)
    return ax.dsl.add(x, ax.dsl.cast(acc, dtype=ax.bf16), output_dtype=ax.bf16)


def mlp_rows(c, x1, m, tag):
    fs = slice(c * L.FC, (c + 1) * L.FC)
    h2 = rmsnorm(x1, ax.dsl.load(m["ln2"]).reshape(1, 1, L.H), eps=L.EPS)
    act = silu_mul(q.proj(h2, q.slice_w(m["w_gate"], rows=fs), tag + "mlp_gate"),
                   q.proj(h2, q.slice_w(m["w_up"], rows=fs), tag + "mlp_up"),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
    wd = q.slice_w(m["w_down"], cols=fs)
    d = q.proj(act, wd, tag + "mlp_down", output_dtype=ax.fp32)
    return reduce_rows(c, x1, d, tag + "m")


def conv_rows(projected, weight, state_in, state_out, name):
    """Stateful causal conv over all T new rows; returns output and [old state | new rows]."""
    conv_size = L.CONV
    channels = projected.shape[2]
    state_len = conv_size - 1
    conv_input = ax.dsl.concat(ax.dsl.load(state_in), projected, dim=1)
    ax.dsl.store(conv_input[:, T : T + state_len, :], r=state_out)
    conv_output = ax.dsl.create_tensor((1, T, channels), projected.dtype, name=f"{name}_conv_output")
    for window_offset in range(conv_size):
        source = conv_input[:, window_offset : window_offset + T, :]
        checkpoint_offset = conv_size - 1 - window_offset
        tap = ax.dsl.load(weight[:, checkpoint_offset : checkpoint_offset + 1]).reshape(1, 1, channels)
        product = ax.dsl.mul(source, tap, output_dtype=projected.dtype)
        ax.dsl.move(product, r=conv_output, op=None if window_offset == 0 else "add")
    return ax.dsl.silu(conv_output, output_dtype=projected.dtype, r=conv_output), conv_input


def stateful_conv(projected, weight, state, name):
    return stateful_runtime_causal_depthwise_conv1d_silu(
        projected, weight, state, state, L.CONV, name)


RUNTIME_ZERO = None  # the verify's runtime-false scalar (set by verify_body): keeps the guards unfoldable


def snapshot_guard(parts):
    """A runtime zero that depends on every snapshot read-back in ``parts``."""
    def scalar(part):
        part = ax.dsl.cast(part, dtype=ax.fp32)
        if len(part.shape) > 1:
            part = ax.dsl.sum(part, dim=list(range(1, len(part.shape))), keepdim=True, output_dtype=ax.fp32)
        return part.reshape(1)

    acc = scalar(parts[0])
    for part in parts[1:]:
        acc = ax.dsl.add(acc, scalar(part), output_dtype=ax.fp32)
    z = (ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=0.0) if RUNTIME_ZERO is None
         else ax.dsl.cast(RUNTIME_ZERO, dtype=ax.fp32).reshape(1))
    return ax.dsl.mul(acc, z, output_dtype=ax.fp32)


def runtime_zero(ntok):
    """A [1, 1, 1] uint8 zero that depends on a runtime kernel input.

    keep_shm below needs real WAR dependencies on every shared-OCM buffer.
    A constant-false condition can be folded away by the CRUN compiler,
    letting a later layer reuse an address while an asynchronous reader from
    the previous layer is still in flight.
    """
    one = ax.dsl.full(shape=(1,), dtype=ax.int32, fill_value=1)
    huge = ax.dsl.full(shape=(1,), dtype=ax.int32, fill_value=1 << 29)
    zero = ax.dsl.zeros(shape=(1,), dtype=ax.int32)
    low = ax.dsl.add(ax.dsl.sub(ax.dsl.load(ntok), huge, output_dtype=ax.int32), one, output_dtype=ax.int32)
    return ax.dsl.minimum(ax.dsl.maximum(low, zero, output_dtype=ax.int32), one, output_dtype=ax.uint8
                          ).reshape(1, 1, 1)


def keep_shm_runtime(y, off):
    """L.keep_shm with a runtime-false condition, so CRUN cannot fold the reads away."""
    live = list(L.SHM_LIVE)
    L.SHM_LIVE.clear()
    one = (1,) * len(y.shape)
    for t in live:
        e = t[tuple(slice(0, 1) for _ in t.shape)].reshape(*one)
        y = ax.dsl.where(off, ax.dsl.cast(e, dtype=y.dtype), y, output_dtype=y.dtype)
    return y


def store_conv_state(c, gi, t, cq, ck, cv, cqsnap, cksnap, cvsnap):
    CVD = L.CONV - 1
    qch = slice(c * L.KHC * L.DK, (c + 1) * L.KHC * L.DK)
    vch = slice(c * L.KHC * L.REP * L.DV, (c + 1) * L.KHC * L.REP * L.DV)
    cqv = ax.dsl.load(cq[:, :, qch]).reshape(1, 1, 1, CVD, L.KHC * L.DK)
    ckv = ax.dsl.load(ck[:, :, qch]).reshape(1, 1, 1, CVD, L.KHC * L.DK)
    cvv = ax.dsl.load(cv[:, :, vch]).reshape(1, 1, 1, CVD, L.KHC * L.REP * L.DV)
    ax.dsl.store(cqv, r=cqsnap[t : t + 1, gi : gi + 1, :, :, qch])
    ax.dsl.store(ckv, r=cksnap[t : t + 1, gi : gi + 1, :, :, qch])
    ax.dsl.store(cvv, r=cvsnap[t : t + 1, gi : gi + 1, :, :, vch])
    # The board compiler may otherwise move the next row's conv update ahead
    # of these asynchronous DDR stores, leaving a snapshot from row t+1.
    q0, v0 = qch.start, vch.start
    return snapshot_guard([
        ax.dsl.load(cqsnap[t : t + 1, gi : gi + 1, :, 0:1, q0 : q0 + 1]),
        ax.dsl.load(cksnap[t : t + 1, gi : gi + 1, :, 0:1, q0 : q0 + 1]),
        ax.dsl.load(cvsnap[t : t + 1, gi : gi + 1, :, 0:1, v0 : v0 + 1]),
    ])


def store_state_snapshots(c, gi, t, snapbuf, ssnap):
    v0 = c * L.KHC * L.REP
    v1 = v0 + L.KHC * L.REP
    ax.dsl.store(snapbuf, r=ssnap[t : t + 1, gi : gi + 1, v0:v1])
    return snapshot_guard([ax.dsl.load(ssnap[t : t + 1, gi : gi + 1, v0:v1, 0:1, 0:1])])


def gdn_rows(c, xt, m, state, cq, ck, cv_, gi, ssnap, cqsnap, cksnap, cvsnap, tag, debug=None):
    KHC, DK, DV, KD, CONV, REP = L.KHC, L.DK, L.DV, L.KD, L.CONV, L.REP
    VC = KHC * REP
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, L.H), eps=L.EPS)
    if debug is not None:
        debug["h"] = h

    def wqkv(r0, n, part, base):  # w_qkv rows, or (the banks) its part's
        if "w_qkv" in m:
            return q.slice_w(m["w_qkv"], rows=slice(r0, r0 + n))
        return q.slice_w(m[part], rows=slice(r0 - base, r0 - base + n))

    qproj = q.proj(h, wqkv(c * KHC * DK, KHC * DK, "w_gq", 0), tag + "gdn_qkv_q")
    kproj = q.proj(h, wqkv(KD + c * KHC * DK, KHC * DK, "w_gk", KD),
                   tag + "gdn_qkv_k")
    vproj = q.proj(h, wqkv(2 * KD + c * VC * DV, VC * DV, "w_gv", 2 * KD),
                   tag + "gdn_qkv_v")
    if debug is not None:
        debug.update(h=h, qproj=qproj, kproj=kproj, vproj=vproj)
    # the causal conv of all T rows at once over [old state | the rows]; the conv state after row t is that
    # sequence's rows t + 1 .. t + CONV - 1 (stored straight from it: no per-row in-place update to order)
    qs = slice(c * KHC * DK, (c + 1) * KHC * DK)
    ks = slice(KD + c * KHC * DK, KD + (c + 1) * KHC * DK)
    vs_conv = slice(2 * KD + c * VC * DV, 2 * KD + (c + 1) * VC * DV)
    vch = slice(c * VC * DV, (c + 1) * VC * DV)
    CVD = CONV - 1
    qmt, qin = conv_rows(qproj, m["conv_w"][qs], cq[:, :, qs], cq[:, :, qs], tag + "gdn_conv_q")
    kmt, kin = conv_rows(kproj, m["conv_w"][ks], ck[:, :, qs], ck[:, :, qs], tag + "gdn_conv_k")
    vmt, vin = conv_rows(vproj, m["conv_w"][vs_conv], cv_[:, :, vch], cv_[:, :, vch], tag + "gdn_conv_v")
    for t in range(T):
        ax.dsl.store(qin[:, t + 1 : t + 1 + CVD, :].reshape(1, 1, 1, CVD, KHC * DK), r=cqsnap[t : t + 1, gi : gi + 1, :, :, qs])
        ax.dsl.store(kin[:, t + 1 : t + 1 + CVD, :].reshape(1, 1, 1, CVD, KHC * DK), r=cksnap[t : t + 1, gi : gi + 1, :, :, qs])
        ax.dsl.store(vin[:, t + 1 : t + 1 + CVD, :].reshape(1, 1, 1, CVD, VC * DV), r=cvsnap[t : t + 1, gi : gi + 1, :, :, vch])
    # (the snapshot stores are asynchronous: read one value of each back before the buffers they come from are reused)
    cguard = snapshot_guard([ax.dsl.load(sn[t : t + 1, gi : gi + 1, :, 0:1, sl.start : sl.start + 1])
                             for t in range(T) for sn, sl in ((cqsnap, qs), (cksnap, qs), (cvsnap, vch))] +
                            [ax.dsl.load(sn[t : t + 1, gi : gi + 1, :, CVD - 1 : CVD, sl.stop - 1 : sl.stop])
                             for t in range(T) for sn, sl in ((cqsnap, qs), (cksnap, qs), (cvsnap, vch))])
    qm = qmt.reshape(1, T, KHC, DK)
    km = kmt.reshape(1, T, KHC, DK)
    vm = ax.dsl.add(vmt.reshape(1, T, VC, DV), ax.dsl.cast(cguard, dtype=ax.bf16).reshape(1, 1, 1, 1), output_dtype=ax.bf16)
    if debug is not None:
        debug.update(qm=qm.reshape(1, T, KHC * DK), km=km.reshape(1, T, KHC * DK),
                     vm=vm.reshape(1, T, VC * DV))
    z = q.proj(h, q.slice_w(m["w_z"], rows=slice(c * VC * DV, (c + 1) * VC * DV)), tag + "gdn_z")
    z = z.reshape(1, T, VC, DV)
    vs = slice(c * VC, (c + 1) * VC)
    b = q.proj(h, m["w_b"], tag + "gdn_b")[:, :, vs]
    a = q.proj(h, m["w_a"], tag + "gdn_a")[:, :, vs]

    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    qt = ax.dsl.mul(l2norm_lastdim(qm), q_scale, output_dtype=ax.bf16)
    kt = l2norm_lastdim(km)
    if debug is not None:
        debug.update(qt=qt.reshape(1, T, KHC * DK), kt=kt.reshape(1, T, KHC * DK),
                     z=z.reshape(1, T, VC * DV), a=a, b=b)
    a_log_t, dt_bias_t = ax.dsl.load(m["a_log"][vs]), ax.dsl.load(m["dt_bias"][vs])
    # head by head: its state loaded once, the T rows' delta-rule steps on it, row t's state into that row's snapshot
    # buffer (one per row: none reused within the layer); then the snapshots and the in-place state (the last row's)
    # stored whole, and read back before this layer's output (the next layer reuses the buffers)
    core = ax.dsl.create_tensor((1, T, VC, DV), ax.bf16, name=tag + "gdn_core")
    snapbufs = [ax.dsl.create_tensor((1, 1, VC, DK, DV), ax.fp32, name=tag + f"gdn_state_snap{t}") for t in range(T)]
    for hl in range(VC):
        j = hl // REP
        vh = c * VC + hl
        st = ax.dsl.load(state[vh : vh + 1]).reshape(DK, DV)
        for t in range(T):
            g, beta = gdn_gate(a[:, t : t + 1, hl : hl + 1].reshape(1, 1),
                               b[:, t : t + 1, hl : hl + 1].reshape(1, 1),
                               a_log_t[hl : hl + 1], dt_bias_t[hl : hl + 1])
            o_h, st = gdn_recurrent_step(qt[:, t : t + 1, j : j + 1, :].reshape(1, DK),
                                         kt[:, t : t + 1, j : j + 1, :].reshape(1, DK),
                                         vm[:, t : t + 1, hl : hl + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
            ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, t : t + 1, hl : hl + 1, :])
            ax.dsl.move(x=st.reshape(1, 1, 1, DK, DV), r=snapbufs[t][:, :, hl : hl + 1])
    v0, v1 = c * VC, (c + 1) * VC
    for t in range(T):
        ax.dsl.store(snapbufs[t], r=ssnap[t : t + 1, gi : gi + 1, v0:v1])
    ax.dsl.store(snapbufs[T - 1].reshape(VC, DK, DV), r=state[v0:v1])
    sguard = snapshot_guard([ax.dsl.load(ssnap[t : t + 1, gi : gi + 1, v0:v1, 0:1, 0:1]) for t in range(T)] +
                            [ax.dsl.load(ssnap[t : t + 1, gi : gi + 1, v1 - 1 : v1, DK - 1 : DK, DV - 1 : DV]) for t in range(T)] +
                            [ax.dsl.load(state[v0 : v0 + 1, 0:1, 0:1]), ax.dsl.load(state[v1 - 1 : v1, DK - 1 : DK, DV - 1 : DV])])
    core = ax.dsl.add(core, ax.dsl.cast(sguard, dtype=ax.bf16).reshape(1, 1, 1, 1), output_dtype=ax.bf16)

    if debug is not None:
        debug["core"] = core.reshape(1, T, VC * DV)

    normed = rmsnorm(core, ax.dsl.load(m["norm_w"]).reshape(1, 1, 1, DV), eps=L.EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, T, VC * DV)
    wo = q.slice_w(m["w_out"], cols=slice(c * VC * DV, (c + 1) * VC * DV))
    o = q.proj(gated, wo, tag + "gdn_out", output_dtype=ax.fp32)
    x1 = reduce_rows(c, xt, o, tag + "a")
    if debug is not None:
        debug.update(gated=gated, o=o, x1=x1, normed=normed)
    y = mlp_rows(c, x1, m, tag)
    if debug is not None:
        debug["y"] = y
    return y


def attn_rows(c, xt, m, cs, pn, mask, kv, tag):
    QH, QW, D, HKV, ROT = L.QH, L.QW, L.D, L.HKV, L.ROT
    HC, QD = HKV // L.CORES, QH * D
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, L.H), eps=L.EPS)
    qg = q.proj(h, q.slice_w(m["w_q"], rows=slice(c * QH * QW, (c + 1) * QH * QW)), tag + "attn_q")
    qg = qg.reshape(1, T, QH, QW)

    cs_all = ax.dsl.load(cs)
    cos_t = cs_all[:, 0:1].reshape(T, ROT // 2)
    sin_t = cs_all[:, 1:2].reshape(T, ROT // 2)
    qh = rmsnorm(qg[:, :, :, 0:D], ax.dsl.load(m["q_norm"]).reshape(1, 1, 1, D), eps=L.EPS)
    qh = rope(qh, cos_t, sin_t, rotary_dim=ROT)

    ks = slice(c * HC * D, (c + 1) * HC * D)
    kc_ = q.proj(h, q.slice_w(m["w_k"], rows=ks), tag + "attn_k").reshape(1, T, HC, D)
    kc_ = rmsnorm(kc_, ax.dsl.load(m["k_norm"]).reshape(1, 1, 1, D), eps=L.EPS)
    kc_ = rope(kc_, cos_t, sin_t, rotary_dim=ROT)
    vc_ = q.proj(h, q.slice_w(m["w_v"], rows=ks), tag + "attn_v").reshape(1, T, HC, D)

    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / np.sqrt(D))
    qs = ax.dsl.mul(qh.reshape(1, T, QD), scale, output_dtype=ax.bf16)
    W_ = QD + 2 * HC * D
    ex = L.shm_tensor((L.CORES, T, W_), ax.bf16, name=tag + "qkv_all")
    ax.dsl.move(r=ex[c : c + 1, :, 0:QD], x=qs)
    ax.dsl.move(r=ex[c : c + 1, :, QD : QD + HC * D], x=kc_.reshape(1, T, HC * D))
    ax.dsl.move(r=ex[c : c + 1, :, QD + HC * D : W_], x=vc_.reshape(1, T, HC * D))
    L.axd.pub(ex)
    L.axd.sub(ex)

    gated = ax.dsl.create_tensor((1, T, QH * D), ax.bf16, name=tag + "attn_gated")
    # the T rows' K / V written, then their attention in one pass over the cache (a row's mask hides the cells past it)
    G_ = L.G
    qa = ax.dsl.create_tensor((HKV, T, G_, D), ax.bf16, name=tag + "q_heads")
    masks = []
    zero = None
    for t in range(T):
        pn_t = pn[t : t + 1].reshape(3)
        masks.append(mask[t : t + 1].reshape(L.NBLK, L.BC))
        k = ax.dsl.create_tensor((1, HKV, D), ax.bf16, name=tag + f"k_row_{t}")
        v = ax.dsl.create_tensor((1, HKV, D), ax.bf16, name=tag + f"v_row_{t}")
        for kc in range(L.CORES):
            src = ex[kc : kc + 1, t : t + 1, QD : QD + HC * D].reshape(1, HC, D)
            ax.dsl.move(r=k[:, kc : kc + 1, :], x=src)
            src = ex[kc : kc + 1, t : t + 1, QD + HC * D : W_].reshape(1, HC, D)
            ax.dsl.move(r=v[:, kc : kc + 1, :], x=src)
            src = ex[kc : kc + 1, t : t + 1, 0:QD].reshape(HC, 1, G_, D)
            ax.dsl.move(r=qa[kc * HC : (kc + 1) * HC, t : t + 1, :, :], x=src)
        z = G.kv_write_paged(kv, k, v, ax.dsl.load(pn_t[2:3]), rzero=RUNTIME_ZERO).reshape(1, 1)
        zero = z if zero is None else ax.dsl.add(zero, z, output_dtype=zero.dtype)
    qa = ax.dsl.add(qa, ax.dsl.cast(zero, dtype=ax.bf16).reshape(1, 1, 1, 1), output_dtype=ax.bf16)
    attn4 = ax.dsl.create_tensor((HC, T, G_, D), ax.bf16, name=tag + "attn_rows")
    G.attention_split_qa_rows(c, qa, kv, pn[T - 1 : T].reshape(3)[1:2], masks, attn4, tag + "rows_")
    for t in range(T):
        attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, name=tag + f"attn_{t}")
        for hl in range(HC):
            ax.dsl.move(r=attn[:, hl * G_ : (hl + 1) * G_, :], x=attn4[hl : hl + 1, t : t + 1, :, :].reshape(1, G_, D))
        gt = sigmoid_mul(attn, qg[:, t : t + 1, :, D : 2 * D], activation_output_dtype=ax.fp32,
                         output_dtype=ax.bf16).reshape(1, 1, QH * D)
        ax.dsl.move(r=gated[:, t : t + 1, :], x=gt)

    wo = q.slice_w(m["w_o"], cols=slice(c * QH * D, (c + 1) * QH * D))
    o = q.proj(gated, wo, tag + "attn_o", output_dtype=ax.fp32)
    return mlp_rows(c, reduce_rows(c, xt, o, tag + "a"), m, tag)


def verify_body(x, packs, cs, pn, mask, out, ios, ssnap, cqsnap, cksnap, cvsnap, ntok, banks=None, limit=None):
    global RUNTIME_ZERO
    c = ax.program_id(0)
    __import__("q35_mx_build").SCALES.clear()
    __import__("q35_mx_build").SCALES2.clear()
    K.BLK.clear()
    L.SHM_LIVE.clear()
    off = runtime_zero(ntok)
    RUNTIME_ZERO = off
    xt = ax.dsl.load(x)
    gi = 0
    kinds = G.KINDS if limit is None else G.KINDS[:limit]
    for j, kind in enumerate(kinds):
        m = G.members(packs, j, banks)
        if kind == "gdn":
            xt = gdn_rows(c, xt, m, ios[j][0], ios[j][1], ios[j][2], ios[j][3], gi,
                          ssnap, cqsnap, cksnap, cvsnap, f"L{j}_")
            gi += 1
        else:
            xt = attn_rows(c, xt, m, cs, pn, mask, ios[j][0], f"L{j}_")
        # Retire this layer's shared-OCM buffers before the next layer.  The
        # M=4 reductions are 4x a T=1 group's, so holding all 16 live would
        # make the allocator reuse an address still being read.
        xt = keep_shm_runtime(xt, off)
    xt = keep_shm_runtime(xt, off)
    if c == 0:
        ax.dsl.store(xt, r=out)


def make_source(pdt, limit=None):
    outs, ios = [], []
    for j, kind in enumerate(G.KINDS):
        names = layer_io_names(kind, j)
        outs += names
        ios.append("(" + ", ".join(names) + ",)")
    snaps = ["ssnap", "cqsnap", "cksnap", "cvsnap"]
    packs = list(pdt)
    params = ", ".join(["x"] + packs + G.BANK_NAMES + ["cs", "pn", "mask", "out"] + outs + snaps + ["ntok"])
    pd = ", ".join(f"{p}={p}" for p in packs)
    bd = ", ".join(f"{p}={p}" for p in G.BANK_NAMES)
    ret = ", ".join(["out"] + outs + snaps)
    tail = "" if limit is None else f", limit={limit}"
    return (f"import verify_group_m4 as V\n\n\ndef verify_kernel({params}):\n"
            f"    V.verify_body(x, dict({pd}), cs, pn, mask, out, [{', '.join(ios)}], "
            f"ssnap, cqsnap, cksnap, cvsnap, ntok, {'dict(' + bd + ')' if G.BANKS else 'None'}{tail})\n"
            f"    return {ret}\n"), outs
