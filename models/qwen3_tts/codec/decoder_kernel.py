"""Qwen3-TTS-12Hz codec decoder as one AX8860 kernel.

One call consumes 72 RVQ code frames and the persisted state from the previous
call, and writes 72*1920 = 138240 samples plus the next state.  The prefix
(quantizer, causal pre-conv, eight-layer sliding-window transformer and the two
2x ConvNeXt upsamplers) is replicated on all four cores.  The DAC decoder is
channel-sharded across the four cores, with one shared full-channel tile per
time block and explicit pub/sub handoffs.

Weight banks are flat BF16 runtime inputs, sliced by the generated
`weight_layout.py`.  See NOTES.md for the exact GGUF source of every bank.
"""
from __future__ import annotations

import math

import numpy as np

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl import util
from axnn.dsl.dtype import typing_dtype

from weight_layout import LAYOUT

WINDOW = 72
UP = 1920
SAMPLES = WINDOW * UP
PREFIX = WINDOW - 1
N_LAYER = 8
HEAD_DIM = 64
N_HEAD = 16
HIDDEN = 512
LATENT = 1024
DEC = 1536


def _sin(x, *, output_dtype=None, r=None):
    return util.op_convert("AxSin", {"output_dtype": output_dtype}, {"x": x}, {"r": r})


def _cos(x, *, output_dtype=None, r=None):
    return util.op_convert("AxCos", {"output_dtype": output_dtype}, {"x": x}, {"r": r})


def _nshape(name):
    return math.prod(LAYOUT_NAME_SHAPES[name])


LAYOUT_NAME_SHAPES = {}
for _group in LAYOUT.values():
    for _name, _spec in _group.items():
        LAYOUT_NAME_SHAPES[_name] = tuple(_spec["shape"])


def _view(flat, group, name):
    spec = LAYOUT[group][name]
    n = math.prod(spec["shape"])
    return flat[spec["offset"] : spec["offset"] + n].reshape(*spec["shape"])


def _load(flat, group, name):
    return ax.dsl.load(_view(flat, group, name)).reshape(*LAYOUT[group][name]["shape"])


def _tile(flat, group, name, row0, rows, cols):
    spec = LAYOUT[group][name]
    off = spec["offset"] + row0 * cols
    return ax.dsl.load(flat[off : off + rows * cols]).reshape(rows, cols)


def _load_idx(flat, group, name, i):
    """Load one index of a stacked bank [L, ...] instead of the whole stack.

    Loading the whole [8,1024,512] transformer bank is an 8 MB OCM constant,
    above the 7 MB AX8860 OCM; one layer is 0.5 MB.
    """
    spec = LAYOUT[group][name]
    inner = tuple(spec["shape"][1:])
    n = math.prod(inner)
    off = spec["offset"] + i * n
    return ax.dsl.load(flat[off : off + n]).reshape(*inner)


def _f32(v):
    return ax.dsl.full((1,), ax.fp32, v)


def _bf(v):
    return ax.dsl.full((1,), ax.bf16, v)


def _int(v):
    return ax.dsl.full((1,), ax.int32, v)


def _take(x, i):
    """Static integer indexing via a length-1 slice (DSL forbids scalar indices)."""
    return x[i : i + 1].reshape(*x.shape[1:])


def _cast(x, dtype):
    return ax.dsl.cast(x, dtype=dtype)


def _linear(x, w, b=None):
    """BF16 linear with FP32 accumulation, bias, one BF16 rounding at the end."""
    acc = ax.dsl.matmul(x, w, transposed=True, output_dtype=ax.fp32)
    if b is not None:
        acc = ax.dsl.add(acc, b.reshape(1, b.shape[0]), output_dtype=ax.fp32)
    return _cast(acc, ax.bf16)


def _linear_bank(x, flat, group, name, bias_name, layer=None):
    w = _load(flat, group, name)
    if layer is not None:
        w = _take(w, layer)
    b = _load(flat, group, bias_name) if bias_name else None
    return _linear(x, w, b)


def _rms_norm(x, weight, eps=1e-5):
    """HF RMSNorm boundary: FP32 variance, BF16 normalized value, BF16 weight."""
    x32 = _cast(x, ax.fp32)
    sq = ax.dsl.mul(x32, x32, output_dtype=ax.fp32)
    mean = ax.dsl.div(
        ax.dsl.sum(sq, dim=-1, keepdim=True, output_dtype=ax.fp32),
        _f32(x.shape[-1]),
        output_dtype=ax.fp32,
        high_precision=True,
    )
    denom = ax.dsl.sqrt(ax.dsl.add(mean, _f32(eps), output_dtype=ax.fp32), output_dtype=ax.fp32)
    normed = _cast(ax.dsl.div(x32, denom, output_dtype=ax.fp32, high_precision=True), ax.bf16)
    return ax.dsl.mul(normed, weight.reshape(1, weight.shape[0]), output_dtype=ax.bf16)


def _layer_norm(x, weight, bias, eps=1e-6):
    x32 = _cast(x, ax.fp32)
    mean = ax.dsl.div(
        ax.dsl.sum(x32, dim=-1, keepdim=True, output_dtype=ax.fp32),
        _f32(x.shape[-1]),
        output_dtype=ax.fp32,
        high_precision=True,
    )
    d = ax.dsl.sub(x32, mean, output_dtype=ax.fp32)
    var = ax.dsl.div(
        ax.dsl.sum(ax.dsl.mul(d, d, output_dtype=ax.fp32), dim=-1, keepdim=True, output_dtype=ax.fp32),
        _f32(x.shape[-1]),
        output_dtype=ax.fp32,
        high_precision=True,
    )
    inv = ax.dsl.div(
        _f32(1.0),
        ax.dsl.sqrt(ax.dsl.add(var, _f32(eps), output_dtype=ax.fp32), output_dtype=ax.fp32),
        output_dtype=ax.fp32,
        high_precision=True,
    )
    y = ax.dsl.mul(d, inv, output_dtype=ax.fp32)
    y = ax.dsl.mul(y, weight.reshape(1, weight.shape[0]), output_dtype=ax.fp32)
    y = ax.dsl.add(y, bias.reshape(1, bias.shape[0]), output_dtype=ax.fp32)
    return _cast(y, ax.bf16)


def _snake(x, alpha, inv_beta):
    # Both parameters are already folded by the GGUF converter:
    # alpha = exp(alpha_raw), inv_beta = 1/(exp(beta_raw)+1e-9).
    s = _sin(ax.dsl.mul(x, alpha.reshape(1, alpha.shape[0]), output_dtype=ax.bf16), output_dtype=ax.bf16)
    s2 = ax.dsl.mul(s, s, output_dtype=ax.bf16)
    return ax.dsl.add(
        x,
        ax.dsl.mul(s2, inv_beta.reshape(1, inv_beta.shape[0]), output_dtype=ax.bf16),
        output_dtype=ax.bf16,
    )


def _rope(x, base, head_dim=HEAD_DIM):
    """HF rotate_half RoPE, NEOX layout, BF16 table and BF16 arithmetic."""
    half = head_dim // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freq = ax.dsl.from_numpy(inv)
    pos = ax.dsl.add(_cast(base.reshape(1), ax.fp32), ax.dsl.from_numpy(np.arange(x.shape[0], dtype=np.float32)))
    angle = ax.dsl.mul(pos.reshape(x.shape[0], 1), freq.reshape(1, half), output_dtype=ax.fp32)
    c = _cast(_cos(angle, output_dtype=ax.fp32), ax.bf16)
    s = _cast(_sin(angle, output_dtype=ax.fp32), ax.bf16)
    lo = x[:, :half]
    hi = x[:, half:]
    top = ax.dsl.sub(
        ax.dsl.mul(lo, c, output_dtype=ax.bf16),
        ax.dsl.mul(hi, s, output_dtype=ax.bf16),
        output_dtype=ax.bf16,
    )
    bot = ax.dsl.add(
        ax.dsl.mul(hi, c, output_dtype=ax.bf16),
        ax.dsl.mul(lo, s, output_dtype=ax.bf16),
        output_dtype=ax.bf16,
    )
    return ax.dsl.concat(top, bot, dim=1)


def _attention_mask(base):
    i = np.arange(WINDOW, dtype=np.int64)[:, None]
    j = np.arange(PREFIX + WINDOW, dtype=np.int64)[None, :]
    band = (j >= i) & (j <= PREFIX + i)
    cold = band & (j >= PREFIX)
    def values(k):
        return np.where(k, np.float32(0.0), np.float32(-3.0e38)).astype(np.float32)
    cold_v = values(cold)
    delta_v = np.ascontiguousarray(values(band) - cold_v)
    basef = _cast(base.reshape(1), ax.fp32)
    warm = ax.dsl.ceil(ax.dsl.clip(basef, min=0.0, max=1.0))
    return ax.dsl.add(
        ax.dsl.from_numpy(cold_v),
        ax.dsl.mul(ax.dsl.from_numpy(delta_v), warm.reshape(1, 1), output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )


def _quant_decode(codes, qw):
    c = ax.dsl.load(codes)
    # CRUN mis-lowers index_select when its indices are a strided column view
    # (only one index per 8 frames survived).  Materialize transposed rows so
    # every gather reads a contiguous [WINDOW] index vector.
    ct = ax.dsl.transpose(c, perm=(1, 0))
    sem_ids = ct[0:1, :].reshape(WINDOW)
    # The GGUF codebook is already embedding_sum/cluster_usage; the usage
    # tensors remain in the layout as constants for byte-identical banks.
    sem = ax.dsl.index_select(_view(qw, "quant_w", "q_sem_embed"), sem_ids, dim=0)
    sem = _linear(sem, _load(qw, "quant_w", "q_sem_out"))

    acc = ax.dsl.zeros((WINDOW, 256), ax.bf16)
    for g in range(15):
        ids = ct[g + 1 : g + 2, :].reshape(WINDOW)
        e = ax.dsl.index_select(_take(_view(qw, "quant_w", "q_ac_embed"), g), ids, dim=0)
        acc = ax.dsl.add(acc, e, output_dtype=ax.bf16)
    ac = _linear(acc, _load(qw, "quant_w", "q_ac_out"))
    return ax.dsl.add(sem, ac, output_dtype=ax.bf16)


def _causal_conv(x, state, w, b, dilation=1):
    """Full-sequence stride-1 causal conv: x [T,IC], state [pad,IC]."""
    k = w.shape[1] // x.shape[1]
    pad = (k - 1) * dilation
    if state.memory_type == ax.MemoryType.DDR:
        state = ax.dsl.load(state)
    xp = ax.dsl.concat(state, x, dim=0)
    parts = [xp[kk * dilation : kk * dilation + x.shape[0]] for kk in range(k)]
    xk = ax.dsl.concat(*parts, dim=1)
    return _linear(xk, w, b)


def _convnext(x, state, state_out, uw, u):
    # Depthwise causal k=7, FP32 accumulation and one BF16 rounding.  Each tap
    # is materialized as [T, C] so both fma operands have the same shape, and it
    # is loaded one tap at a time (the full [7,T,C] would be 4 MB for up1).
    b = _load(uw, "up_w", f"up{u}_dw_b")
    t, c = x.shape
    if state.memory_type == ax.MemoryType.DDR:
        state = ax.dsl.load(state)
    xp = ax.dsl.concat(state, x, dim=0)
    acc = ax.dsl.zeros((t, c), ax.fp32)
    for k in range(7):
        wk = _load_idx(uw, "up_w", f"up{u}_dw_taps", k)
        assert wk.shape == (t, c), (u, k, wk.shape, t, c)
        acc = ax.dsl.add(
            acc,
            ax.dsl.mul(xp[k : k + t], wk, output_dtype=ax.fp32),
            output_dtype=ax.fp32,
        )
    acc = ax.dsl.add(acc, b.reshape(1, c), output_dtype=ax.fp32)
    h = _cast(acc, ax.bf16)
    h = _layer_norm(h, _load(uw, "up_w", f"up{u}_norm_w"), _load(uw, "up_w", f"up{u}_norm_b"))

    # pwconv1 is 1024 -> 4096; tile its output rows to fit private OCM.
    gs = []
    for ob in range(0, 4096, 512):
        w1 = _tile(uw, "up_w", f"up{u}_pw1_w", ob, 512, 1024)
        b1 = _load(uw, "up_w", f"up{u}_pw1_b")[ob : ob + 512]
        g = _linear(h, w1, b1)
        gs.append(ax.dsl.gelu(g, output_dtype=ax.bf16))
    g = ax.dsl.concat(*gs, dim=1)

    # The next call's depthwise-conv left context is the last six input rows.
    c = ax.program_id(0)
    cw = c * (x.shape[1] // 4)
    ax.dsl.store(x[t - 6 : t, cw : cw + x.shape[1] // 4], r=state_out[:, cw : cw + x.shape[1] // 4])

    # pwconv2 is 4096 -> 1024; accumulate four FP32 IC tiles, one rounding.
    outs = []
    for ob in range(0, 1024, 256):
        w2 = _tile(uw, "up_w", f"up{u}_pw2_w", ob, 256, 4096)
        acc = ax.dsl.zeros((t, 256), ax.fp32)
        for ib in range(0, 4096, 1024):
            acc = ax.dsl.add(
                ax.dsl.matmul(g[:, ib : ib + 1024], w2[:, ib : ib + 1024],
                              transposed=True, output_dtype=ax.fp32),
                acc,
                output_dtype=ax.fp32,
            )
        b2 = _load(uw, "up_w", f"up{u}_pw2_b")[ob : ob + 256]
        acc = ax.dsl.add(acc, b2.reshape(1, 256), output_dtype=ax.fp32)
        y = _cast(acc, ax.bf16)
        gamma = _load(uw, "up_w", f"up{u}_gamma")[ob : ob + 256]
        outs.append(ax.dsl.mul(y, gamma.reshape(1, 256), output_dtype=ax.bf16))
    y = ax.dsl.concat(*outs, dim=1)
    return ax.dsl.add(x, y, output_dtype=ax.bf16)


def _upsample_once(x, state, state_out, uw, u):
    t, c = x.shape
    b = _load(uw, "up_w", f"up{u}_tc_b")
    # One phase weight at a time: the [1024,1024] view is 2 MB, the [2,...]
    # stack would be 4 MB.
    w0 = _load_idx(uw, "up_w", f"up{u}_tc_w", 0)
    y0 = ax.dsl.add(ax.dsl.matmul(x, w0, transposed=True, output_dtype=ax.fp32),
                   b.reshape(1, c), output_dtype=ax.fp32)
    y0 = _cast(y0, ax.bf16).reshape(t, 1, c)
    w1 = _load_idx(uw, "up_w", f"up{u}_tc_w", 1)
    y1 = ax.dsl.add(ax.dsl.matmul(x, w1, transposed=True, output_dtype=ax.fp32),
                   b.reshape(1, c), output_dtype=ax.fp32)
    y1 = _cast(y1, ax.bf16).reshape(t, 1, c)
    y = ax.dsl.concat(y0, y1, dim=1).reshape(2 * t, c)
    return _convnext(y, state, state_out, uw, u)


def _tfm_layer(cur, qw, layer, base, mask, kv_in):
    ln1 = _load_idx(qw, "quant_w", "tfm_ln1", layer)
    ln2 = _load_idx(qw, "quant_w", "tfm_ln2", layer)
    h = _rms_norm(cur, ln1)
    q = _linear(h, _load_idx(qw, "quant_w", "tfm_q", layer))
    k = _linear(h, _load_idx(qw, "quant_w", "tfm_k", layer))
    v = _linear(h, _load_idx(qw, "quant_w", "tfm_v", layer))

    heads, kheads = [], []
    for head in range(N_HEAD):
        sl = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        qh = _rope(q[:, sl], base)
        kh = _rope(k[:, sl], base)
        kheads.append(kh)
        old_k = ax.dsl.load(kv_in[layer : layer + 1, 0:1, sl, :].reshape(HEAD_DIM, PREFIX))
        old_v = ax.dsl.load(kv_in[layer : layer + 1, 1:2, sl, :].reshape(HEAD_DIM, PREFIX))
        kfull = ax.dsl.concat(old_k, ax.dsl.transpose(kh, perm=(1, 0)), dim=1)  # [64, 143]
        vfull = ax.dsl.concat(old_v, ax.dsl.transpose(v[:, sl], perm=(1, 0)), dim=1)  # [64, 143]
        score = ax.dsl.mul(
            ax.dsl.matmul(qh, kfull, output_dtype=ax.bf16), _bf(HEAD_DIM**-0.5), output_dtype=ax.bf16
        )
        score = ax.dsl.add(score, _cast(mask, ax.bf16), output_dtype=ax.bf16)
        prob = _cast(ax.dsl.softmax(score, dim=-1, output_dtype=ax.fp32), ax.bf16)
        out = ax.dsl.matmul(prob, ax.dsl.transpose(vfull, perm=(1, 0)), output_dtype=ax.bf16)
        heads.append(out)
    attn = ax.dsl.concat(*heads, dim=1)
    attn = _linear(attn, _load_idx(qw, "quant_w", "tfm_o", layer))
    scale = _load_idx(qw, "quant_w", "tfm_attn_scale", layer)
    cur = ax.dsl.add(cur, ax.dsl.mul(attn, scale.reshape(1, HIDDEN), output_dtype=ax.bf16), output_dtype=ax.bf16)

    residual = cur
    h = _rms_norm(cur, ln2)
    gate = _linear(h, _load_idx(qw, "quant_w", "tfm_gate", layer))
    up = _linear(h, _load_idx(qw, "quant_w", "tfm_up", layer))
    gu = ax.dsl.mul(ax.dsl.silu(gate, output_dtype=ax.bf16), up, output_dtype=ax.bf16)
    down = _linear(gu, _load_idx(qw, "quant_w", "tfm_down", layer))
    scale = _load_idx(qw, "quant_w", "tfm_ffn_scale", layer)
    cur = ax.dsl.add(residual, ax.dsl.mul(down, scale.reshape(1, HIDDEN), output_dtype=ax.bf16), output_dtype=ax.bf16)
    return cur, ax.dsl.concat(*kheads, dim=1), v


def prefix(codes, qw, uw, base, kv_in, kv_out, pre_state, pre_state_out, up0_state, up0_state_out, up1_state, up1_state_out):
    """Full prefix duplicated on every core.  Returns [288,1024] and state tails."""
    base = ax.dsl.load(base)
    hidden = _quant_decode(codes, qw)
    x = _causal_conv(hidden, pre_state, _load(qw, "quant_w", "pre_conv_w"), _load(qw, "quant_w", "pre_conv_b"))
    c = ax.program_id(0)
    cw = c * (hidden.shape[1] // 4)
    ax.dsl.store(hidden[WINDOW - 2 : WINDOW, cw : cw + hidden.shape[1] // 4],
                 r=pre_state_out[:, cw : cw + hidden.shape[1] // 4])

    h = _linear(x, _load(qw, "quant_w", "tfm_in_w"), _load(qw, "quant_w", "tfm_in_b"))
    mask = _attention_mask(base)
    for layer in range(N_LAYER):
        h, k_new, v_new = _tfm_layer(h, qw, layer, base, mask, kv_in)
        # State is channel-major [1024,71].  Each core writes its four heads.
        lo = c * (1024 // 4)
        hi = lo + 1024 // 4
        ax.dsl.store(ax.dsl.transpose(k_new, perm=(1, 0))[lo:hi, 1:WINDOW],
                     r=kv_out[layer : layer + 1, 0:1, lo:hi, :].reshape(hi - lo, PREFIX))
        ax.dsl.store(ax.dsl.transpose(v_new, perm=(1, 0))[lo:hi, 1:WINDOW],
                     r=kv_out[layer : layer + 1, 1:2, lo:hi, :].reshape(hi - lo, PREFIX))
    h = _rms_norm(h, _load(qw, "quant_w", "tfm_norm"))
    y = _linear(h, _load(qw, "quant_w", "tfm_out_w"), _load(qw, "quant_w", "tfm_out_b"))
    y = _upsample_once(y, up0_state, up0_state_out, uw, 0)
    y = _upsample_once(y, up1_state, up1_state_out, uw, 1)
    return y


# ---------------------------------------------------------------------------
# DAC decoder.  Core c computes the causal prefix [0, 72*(c+1)] of the up1
# latent (the prefix is replicated, so every core has all of it) and writes the
# c-th quarter of the final waveform.  Only the last core reaches the end of
# the window, so it persists all DAC state.
# ---------------------------------------------------------------------------
Q = SAMPLES // 4              # 34560 output samples per core
UP_PER_FRAME = 4              # two 2x upsample blocks
DAC_PER_FRAME = 480           # 8*5*4*3

DAC_TAILS = ((0, 8, 768), (1, 5, 384), (2, 4, 192), (3, 3, 96))
DAC_RES = ((0, 768), (1, 384), (2, 192), (3, 96))


def _buf(ws, base, max_elem, slot, shape):
    lo = base + slot * max_elem
    n = math.prod(shape)
    return ws[lo : lo + n].reshape(*shape)


def _causal_conv_ddr(flat, group, x, T, IC, y, OC, wname, bname, K, dil, *,
                     state_in=None, state_out=None, residual=None, write_state=False,
                     tile_t=256, tile_oc=256, out_dtype=None):
    """out[o,t] = sum_{k,i} w[o,k*IC+i] * x[t-pad+k*dil, i] + b[o] (+ residual)."""
    pad = (K - 1) * dil
    for t0 in range(0, T, tile_t):
        tsz = min(tile_t, T - t0)
        if pad:
            if t0 == 0:
                xc = ax.dsl.concat(ax.dsl.load(state_in), ax.dsl.load(x[0:tsz]), dim=0)
            else:
                xc = ax.dsl.load(x[t0 - pad : t0 + tsz])
        else:
            xc = ax.dsl.load(x[t0 : t0 + tsz])
        if K == 1:
            xk = xc
        else:
            xk = ax.dsl.concat(*[xc[k * dil : k * dil + tsz] for k in range(K)], dim=1)
        for oc0 in range(0, OC, tile_oc):
            ob = min(tile_oc, OC - oc0)
            wt = _tile(flat, group, wname, oc0, ob, K * IC)
            acc = ax.dsl.matmul(xk, wt, transposed=True, output_dtype=ax.fp32)
            if bname:
                acc = ax.dsl.add(acc, _load(flat, group, bname)[oc0 : oc0 + ob].reshape(1, ob),
                                 output_dtype=ax.fp32)
            if residual is not None:
                acc = ax.dsl.add(acc, ax.dsl.load(residual[t0 : t0 + tsz, oc0 : oc0 + ob]),
                                 output_dtype=ax.fp32)
            ax.dsl.store(_cast(acc, out_dtype or ax.bf16), r=y[t0 : t0 + tsz, oc0 : oc0 + ob])
    if write_state and pad:
        ax.dsl.store(ax.dsl.load(x[T - pad : T]), r=state_out)


def _snake_ddr(flat, group, x, y, T, C, aname, bname, tile_t=1920):
    a = _load(flat, group, aname)
    b = _load(flat, group, bname)
    for t0 in range(0, T, tile_t):
        tsz = min(tile_t, T - t0)
        ax.dsl.store(_snake(ax.dsl.load(x[t0 : t0 + tsz]), a, b), r=y[t0 : t0 + tsz])


def _trans_conv_ddr(flat, group, x, T, IC, y, OC, wname, bname, S, J, tail_in, *,
                    tile_t, tile_oc, tail_out=None, write_tail=False):
    """Causal ConvTranspose1d as per-phase GEMMs; raw col2im tail is the state."""
    trim = S if J > 1 else 0
    for t0 in range(0, T, tile_t):
        tsz = min(tile_t, T - t0)
        cur = ax.dsl.load(x[t0 : t0 + tsz])
        if J > 1:
            if t0 == 0:
                prev = ax.dsl.concat(ax.dsl.zeros((1, IC), ax.bf16), ax.dsl.load(x[0 : tsz - 1]), dim=0)
            else:
                prev = ax.dsl.load(x[t0 - 1 : t0 + tsz - 1])
            xcat = ax.dsl.concat(cur, prev, dim=1)
        else:
            xcat = cur
        for oc0 in range(0, OC, tile_oc):
            ob = min(tile_oc, OC - oc0)
            phases = []
            for p in range(S):
                wp = _tile(flat, group, wname, p * OC + oc0, ob, J * IC)
                acc = ax.dsl.matmul(xcat, wp, transposed=True, output_dtype=ax.fp32)
                phases.append(_cast(acc, ax.bf16))
            yb = ax.dsl.concat(*[ph.reshape(tsz, 1, ob) for ph in phases], dim=1).reshape(tsz * S, ob)
            if t0 == 0 and trim:
                yh = ax.dsl.add(yb[0:trim], ax.dsl.load(tail_in[:, oc0 : oc0 + ob]), output_dtype=ax.bf16)
                yb = ax.dsl.concat(yh, yb[trim : tsz * S], dim=0)
            yb = ax.dsl.add(yb, _load(flat, group, bname)[oc0 : oc0 + ob].reshape(1, ob),
                            output_dtype=ax.bf16)
            ax.dsl.store(yb, r=y[t0 * S : (t0 + tsz) * S, oc0 : oc0 + ob])
    if write_tail:
        xlast = ax.dsl.load(x[T - 1 : T])
        for oc0 in range(0, OC, tile_oc):
            ob = min(tile_oc, OC - oc0)
            phs = []
            for p in range(S):
                wp2 = _tile(flat, group, wname, p * OC + oc0, ob, J * IC)[:, IC : 2 * IC]
                acc = ax.dsl.matmul(xlast, wp2, transposed=True, output_dtype=ax.fp32)
                phs.append(_cast(acc, ax.bf16))
            ax.dsl.store(ax.dsl.concat(*phs, dim=0), r=tail_out[:, oc0 : oc0 + ob])


def _res_unit(flat, x, T, C, p, b1, b2, a1, b1n, c1, c1b, a2, b2n, c2, c2b, dil,
              st_in, st_out, *, write_state, tile_t, tile_k1, tile_oc):
    s = b1((T, C))
    _snake_ddr(flat, "dac_w", x, s, T, C, a1, b1n, tile_t=tile_t)
    h = b2((T, C))
    _causal_conv_ddr(flat, "dac_w", s, T, C, h, C, c1, c1b, 7, dil,
                     state_in=st_in, state_out=st_out, write_state=write_state,
                     tile_t=tile_t, tile_oc=tile_oc)
    s2 = b1((T, C))
    _snake_ddr(flat, "dac_w", h, s2, T, C, a2, b2n, tile_t=tile_t)
    _causal_conv_ddr(flat, "dac_w", s2, T, C, x, C, c2, c2b, 1, 1,
                     residual=x, tile_t=tile_k1, tile_oc=C)


def _final_conv(flat, x, T, wname, bname, state_in, state_out, wav, q0, q1, *,
                write_state):
    pad = 6
    for t0 in range(q0, q1, 1920):
        tsz = min(1920, q1 - t0)
        if t0 == 0:
            xc = ax.dsl.concat(ax.dsl.load(state_in), ax.dsl.load(x[0:tsz]), dim=0)
        else:
            xc = ax.dsl.load(x[t0 - pad : t0 + tsz])
        xk = ax.dsl.concat(*[xc[k : k + tsz] for k in range(7)], dim=1)
        wt = _tile(flat, "dac_w", wname, 0, 1, 7 * 96)
        acc = ax.dsl.matmul(xk, wt, transposed=True, output_dtype=ax.fp32)
        acc = ax.dsl.add(acc, _load(flat, "dac_w", bname)[0:1].reshape(1, 1), output_dtype=ax.fp32)
        y = _cast(acc, ax.bf16)
        y = ax.dsl.clip(y, min=-1.0, max=1.0)
        ax.dsl.store(_cast(y, ax.fp32).reshape(tsz), r=wav[t0 : t0 + tsz])
    if write_state:
        ax.dsl.store(ax.dsl.load(x[T - pad : T]), r=state_out)


def decode_window(codes, qw, uw, dw, pos,
                  kv_in, kv_out, pre_in, pre_out, up0_in, up0_out, up1_in, up1_out,
                  dent_in, dent_out,
                  t0_in, t0_out, t1_in, t1_out, t2_in, t2_out, t3_in, t3_out,
                  r00_in, r00_out, r01_in, r01_out, r02_in, r02_out,
                  r10_in, r10_out, r11_in, r11_out, r12_in, r12_out,
                  r20_in, r20_out, r21_in, r21_out, r22_in, r22_out,
                  r30_in, r30_out, r31_in, r31_out, r32_in, r32_out,
                  post_in, post_out, wav):
    c = ax.program_id(0)
    R = 72 * (c + 1)                      # up1 rows computed by this core
    max_elem = 34560 * (c + 1) * 96       # max intermediate element count
    ws = ax.dsl.create_tensor((3 * max_elem,), ax.bf16, ax.MemoryType.DDR, name=f"dac_ws_{c}")

    def buf(slot, shape):
        return _buf(ws, 0, max_elem, slot, shape)

    y = prefix(codes, qw, uw, pos, kv_in, kv_out, pre_in, pre_out,
               up0_in, up0_out, up1_in, up1_out)
    ax.dsl.store(y[0:R], r=buf(0, (R, 1024)))

    last = (c == ax.num_programs(0) - 1)

    _causal_conv_ddr(dw, "dac_w", buf(0, (R, 1024)), R, 1024, buf(1, (R, 1536)), 1536,
                     "dac_entry_w", "dac_entry_b", 7, 1, state_in=dent_in, state_out=dent_out,
                     write_state=last, tile_t=72, tile_oc=256)

    # block 0: 1536 -> 768, stride 8
    T = R * 8
    _snake_ddr(dw, "dac_w", buf(1, (R, 1536)), buf(2, (R, 1536)), R, 1536,
               "dac0_snake_a", "dac0_snake_b", tile_t=72)
    _trans_conv_ddr(dw, "dac_w", buf(2, (R, 1536)), R, 1536, buf(0, (T, 768)), 768,
                    "dac0_tc_w", "dac0_tc_b", 8, 2, t0_in, tile_t=R, tile_oc=384,
                    tail_out=t0_out, write_tail=last)
    for r in range(3):
        st = (r00_in, r01_in, r02_in)[r]
        so = (r00_out, r01_out, r02_out)[r]
        _res_unit(dw, buf(0, (T, 768)), T, 768, "dac0", 
                  lambda sh, _T=T: buf(1, sh), lambda sh: buf(2, sh),
                  f"dac0_r{r}_a1", f"dac0_r{r}_b1", f"dac0_r{r}_c1w", f"dac0_r{r}_c1b",
                  f"dac0_r{r}_a2", f"dac0_r{r}_b2", f"dac0_r{r}_c2w", f"dac0_r{r}_c2b",
                  (1, 3, 9)[r], st, so, write_state=last, tile_t=192, tile_k1=192, tile_oc=192)

    # block 1: 768 -> 384, stride 5
    T1 = R * 40
    _snake_ddr(dw, "dac_w", buf(0, (T, 768)), buf(1, (T, 768)), T, 768,
               "dac1_snake_a", "dac1_snake_b", tile_t=192)
    _trans_conv_ddr(dw, "dac_w", buf(1, (T, 768)), T, 768, buf(2, (T1, 384)), 384,
                    "dac1_tc_w", "dac1_tc_b", 5, 2, t1_in, tile_t=576, tile_oc=384,
                    tail_out=t1_out, write_tail=last)
    for r in range(3):
        st = (r10_in, r11_in, r12_in)[r]
        so = (r10_out, r11_out, r12_out)[r]
        _res_unit(dw, buf(2, (T1, 384)), T1, 384, "dac1",
                  lambda sh: buf(0, sh), lambda sh: buf(1, sh),
                  f"dac1_r{r}_a1", f"dac1_r{r}_b1", f"dac1_r{r}_c1w", f"dac1_r{r}_c1b",
                  f"dac1_r{r}_a2", f"dac1_r{r}_b2", f"dac1_r{r}_c2w", f"dac1_r{r}_c2b",
                  (1, 3, 9)[r], st, so, write_state=last, tile_t=480, tile_k1=960, tile_oc=384)

    # block 2: 384 -> 192, stride 4
    T2 = R * 160
    _snake_ddr(dw, "dac_w", buf(2, (T1, 384)), buf(0, (T1, 384)), T1, 384,
               "dac2_snake_a", "dac2_snake_b", tile_t=480)
    _trans_conv_ddr(dw, "dac_w", buf(0, (T1, 384)), T1, 384, buf(1, (T2, 192)), 192,
                    "dac2_tc_w", "dac2_tc_b", 4, 2, t2_in, tile_t=960, tile_oc=192,
                    tail_out=t2_out, write_tail=last)
    for r in range(3):
        st = (r20_in, r21_in, r22_in)[r]
        so = (r20_out, r21_out, r22_out)[r]
        _res_unit(dw, buf(1, (T2, 192)), T2, 192, "dac2",
                  lambda sh: buf(2, sh), lambda sh: buf(0, sh),
                  f"dac2_r{r}_a1", f"dac2_r{r}_b1", f"dac2_r{r}_c1w", f"dac2_r{r}_c1b",
                  f"dac2_r{r}_a2", f"dac2_r{r}_b2", f"dac2_r{r}_c2w", f"dac2_r{r}_c2b",
                  (1, 3, 9)[r], st, so, write_state=last, tile_t=1440, tile_k1=2880, tile_oc=192)

    # block 3: 192 -> 96, stride 3
    T3 = R * 480
    _snake_ddr(dw, "dac_w", buf(1, (T2, 192)), buf(2, (T2, 192)), T2, 192,
               "dac3_snake_a", "dac3_snake_b", tile_t=1440)
    _trans_conv_ddr(dw, "dac_w", buf(2, (T2, 192)), T2, 192, buf(0, (T3, 96)), 96,
                    "dac3_tc_w", "dac3_tc_b", 3, 2, t3_in, tile_t=2304, tile_oc=96,
                    tail_out=t3_out, write_tail=last)
    for r in range(3):
        st = (r30_in, r31_in, r32_in)[r]
        so = (r30_out, r31_out, r32_out)[r]
        _res_unit(dw, buf(0, (T3, 96)), T3, 96, "dac3",
                  lambda sh: buf(1, sh), lambda sh: buf(2, sh),
                  f"dac3_r{r}_a1", f"dac3_r{r}_b1", f"dac3_r{r}_c1w", f"dac3_r{r}_c1b",
                  f"dac3_r{r}_a2", f"dac3_r{r}_b2", f"dac3_r{r}_c2w", f"dac3_r{r}_c2b",
                  (1, 3, 9)[r], st, so, write_state=last, tile_t=2880, tile_k1=5760, tile_oc=96)

    _snake_ddr(dw, "dac_w", buf(0, (T3, 96)), buf(1, (T3, 96)), T3, 96,
               "post_snake_a", "post_snake_b", tile_t=2880)
    q0, q1 = c * Q, (c + 1) * Q
    _final_conv(dw, buf(1, (T3, 96)), T3, "post_conv_w", "post_conv_b",
                post_in, post_out, wav, q0, q1, write_state=last)
    return wav


def decoder(codes, quant_w, up_w, dac_w, pos,
            kv_state, kv_next, pre_state, pre_next,
            up0_state, up0_next, up1_state, up1_next,
            dac_entry_state, dac_entry_next,
            dac0_tail, dac0_tail_next, dac1_tail, dac1_tail_next,
            dac2_tail, dac2_tail_next, dac3_tail, dac3_tail_next,
            dac0_res0_state, dac0_res0_next, dac0_res1_state, dac0_res1_next,
            dac0_res2_state, dac0_res2_next,
            dac1_res0_state, dac1_res0_next, dac1_res1_state, dac1_res1_next,
            dac1_res2_state, dac1_res2_next,
            dac2_res0_state, dac2_res0_next, dac2_res1_state, dac2_res1_next,
            dac2_res2_state, dac2_res2_next,
            dac3_res0_state, dac3_res0_next, dac3_res1_state, dac3_res1_next,
            dac3_res2_state, dac3_res2_next,
            post_state, post_next, wav):
    """One 72-frame window: codes + persisted state -> 138240 wav samples + next state."""
    decode_window(
        codes, quant_w, up_w, dac_w, pos,
        kv_state, kv_next, pre_state, pre_next, up0_state, up0_next,
        up1_state, up1_next, dac_entry_state, dac_entry_next,
        dac0_tail, dac0_tail_next, dac1_tail, dac1_tail_next,
        dac2_tail, dac2_tail_next, dac3_tail, dac3_tail_next,
        dac0_res0_state, dac0_res0_next, dac0_res1_state, dac0_res1_next,
        dac0_res2_state, dac0_res2_next,
        dac1_res0_state, dac1_res0_next, dac1_res1_state, dac1_res1_next,
        dac1_res2_state, dac1_res2_next,
        dac2_res0_state, dac2_res0_next, dac2_res1_state, dac2_res1_next,
        dac2_res2_state, dac2_res2_next,
        dac3_res0_state, dac3_res0_next, dac3_res1_state, dac3_res1_next,
        dac3_res2_state, dac3_res2_next,
        post_state, post_next, wav,
    )
    return wav
