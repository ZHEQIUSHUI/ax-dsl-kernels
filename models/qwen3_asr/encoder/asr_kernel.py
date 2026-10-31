"""Qwen3-ASR audio tower as ONE AX8860 kernel per model (n = 8 chunks).

Input : log-mel [n*100, 128] BF16 (time-major), kmask [P] BF16
Output: projector rows [P, output_dim] FP32; the first 13*nch rows are valid
        for a call with nch real chunks (nch <= n = 8, P = 128).

Inside: on-chip im2col for all three stride-2 3x3 convolutions, GELU, conv_out,
sinusoidal position add, a runtime range over the encoder layers with every
weight bank [L, ...], the runtime key mask, final LayerNorm and the projector.
Four cores: each owns a 1/4 band of conv3's 16 frequency rows for the front-end
(no cross-core front-end traffic; the tiny conv3 map is exchanged through
SHARED_OCM) and 1/4 of the padded rows for the layer stack.

The host zero-pads the mel to n=8 chunks and passes kmask = 0 for rows
[0, 13*nch) and -3e38 for rows [13*nch, P).  Only keys are masked; queries of
padding rows are don't-care.  This matches the official tower, whose attention
never sees rows beyond the real chunks of the call.
"""
from __future__ import annotations

import json
import os

import ml_dtypes
import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils.linear import linear as _linear
from sgl_kernel_axera.models.bank_access import load_tile

from asr_ref import audio_config, audio_state_dict, sinusoids

EPS = 1e-5
CORES = 4
HD = 64  # both 0.6B and 1.7B: d_model / encoder_attention_heads
FQ = 4  # conv3 frequency rows per core (16 / 4)


def padded_rows(n):
    return 64 if n <= 4 else 128


def _np_bf16(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32), dtype=ml_dtypes.bfloat16)


def _bank_rows(bank, idx, rows, cols, r0=0, c0=0):
    view = bank[0:1, r0 : r0 + rows, c0 : c0 + cols].reshape(rows, cols)
    return load_tile(view, banks={view.name: (bank, (idx, bank.shape[1], 0))})


def _bank_vec(bank, idx):
    rows = bank.shape[1]
    view = bank[0:1, 0:rows].reshape(1, rows)
    return load_tile(view, banks={view.name: (bank, (idx, 1, 0))}).reshape(rows)


def _layernorm(x, w, b, out_dtype=ax.bf16):
    n = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=x.shape[-1])
    mean = ax.dsl.div(ax.dsl.sum(x, dim=-1, keepdim=True, output_dtype=ax.fp32), n)
    xc = ax.dsl.sub(x, mean, output_dtype=ax.fp32)
    var = ax.dsl.div(ax.dsl.sum(ax.dsl.mul(xc, xc, output_dtype=ax.fp32), dim=-1, keepdim=True,
                                output_dtype=ax.fp32), n)
    den = ax.dsl.sqrt(ax.dsl.add(var, ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=EPS)))
    y = ax.dsl.mul(ax.dsl.div(xc, den, output_dtype=ax.fp32), w.reshape(1, 1, -1), output_dtype=ax.fp32)
    return ax.dsl.add(y, b.reshape(1, 1, -1), output_dtype=out_dtype)


def _bank_linear(x, bank, bbank, layer, row0, cols, out_dim, ob, name):
    """[.., cols] @ bank[layer, row0:row0+out_dim, :cols].T + bias, fp32 out"""
    y = ax.dsl.create_tensor((*x.shape[:-1], out_dim), ax.fp32, ax.MemoryType.OCM, name=name)
    bias = _bank_vec(bbank, layer)
    for o0 in range(0, out_dim, ob):
        ow = min(ob, out_dim - o0)
        wt = _bank_rows(bank, layer, ow, cols, r0=row0 + o0)
        yt = ax.dsl.matmul(x, wt, transposed=True, output_dtype=ax.fp32)
        yt = ax.dsl.add(yt, bias[row0 + o0 : row0 + o0 + ow].reshape(*((1,) * (yt.ndim - 1)), ow),
                        output_dtype=ax.fp32)
        ax.dsl.move(yt, r=y[..., o0 : o0 + ow])
    return y


def _front_indices_core(n, core):
    total = n * 100 * 128
    f0 = core * FQ
    h2_lo = 2 * f0 - 1
    h1_lo = 2 * h2_lo - 1
    idx1 = np.zeros((n, 19 * 50, 16), np.int32)
    idx2 = np.zeros((n, 9 * 25, 9), np.int32)
    idx3 = np.zeros((n, FQ * 13, 9), np.int32)
    for c in range(n):
        for i in range(19):
            h1 = h1_lo + i
            for w1 in range(50):
                p = i * 50 + w1
                for t in range(16):
                    if t >= 9:
                        idx1[c, p, t] = total
                        continue
                    kh, kw = divmod(t, 3)
                    h = 2 * h1 + kh - 1
                    f = 2 * w1 + kw - 1
                    if 0 <= h < 128 and 0 <= f < 100:
                        idx1[c, p, t] = (c * 100 + f) * 128 + h
                    else:
                        idx1[c, p, t] = total
        for j in range(9):
            h2 = h2_lo + j
            for w2 in range(25):
                p = j * 25 + w2
                for t in range(9):
                    kh, kw = divmod(t, 3)
                    h1 = 2 * h2 + kh - 1
                    w1 = 2 * w2 + kw - 1
                    if 0 <= h1 < 64 and 0 <= w1 < 50:
                        idx2[c, p, t] = (h1 - h1_lo) * 50 + w1
                    else:
                        idx2[c, p, t] = 19 * 50
        for jf in range(FQ):
            f = f0 + jf
            for tt in range(13):
                p = jf * 13 + tt
                for t in range(9):
                    kh, kw = divmod(t, 3)
                    h2 = 2 * f + kh - 1
                    w2 = 2 * tt + kw - 1
                    if 0 <= h2 < 32 and 0 <= w2 < 25:
                        idx3[c, p, t] = (h2 - h2_lo) * 25 + w2
                    else:
                        idx3[c, p, t] = 9 * 25
    return idx1, idx2, idx3


def _conv_out_index(n, P, NQ, core):
    R = 13 * n
    D = n * 208
    idx = np.full((NQ, 16), D, np.int32)
    for j in range(NQ):
        r = core * NQ + j
        if r < R:
            c, t = divmod(r, 13)
            for f in range(16):
                idx[j, f] = c * 208 + f * 13 + t
    return idx


def _pe(P, H):
    pe = sinusoids(1500, H)[np.arange(P) % 13].astype(np.float32)
    return _np_bf16(pe)


def key_mask(P, valid_rows):
    """Host-side kmask for a call with valid_rows = 13 * real chunks."""
    mask = np.zeros((P,), np.float32)
    mask[valid_rows:] = -3e38
    return torch.from_numpy(mask).to(torch.bfloat16).contiguous()


def make_fused(H, F, O, NH, L, n, layers=None):
    """Return the fused DSL kernel for one shape variant (layers overrides L for diagnostics)."""
    Lc = L if layers is None else layers
    P = padded_rows(n)
    NQ = P // CORES
    assert H % NH == 0 and H // NH == HD and F % 256 == 0 and H % 128 == 0
    assert P % CORES == 0 and NQ % 16 == 0
    R = 13 * n
    D = n * 208
    idx1 = [_front_indices_core(n, c)[0] for c in range(CORES)]
    idx2 = [_front_indices_core(n, c)[1] for c in range(CORES)]
    idx3 = [_front_indices_core(n, c)[2] for c in range(CORES)]
    idxo = [_conv_out_index(n, P, NQ, c) for c in range(CORES)]
    pe_all = _pe(P, H)

    def fused(
        mel, kmask,
        w1, b1, w2, b2, w3, b3, conv_out_w,
        ln1w, ln1b, qkvw, qkvb, ow, ob, ln2w, ln2b, upw, upb, downw, downb,
        postw, postb, p1w, p1b, p2w, p2b,
        out,
    ):
        core = ax.program_id(0)
        rows = slice(core * NQ, (core + 1) * NQ)
        f0 = core * FQ
        total = n * 100 * 128

        # ---- front-end: mel to the layers' input rows --------------------
        melp = ax.dsl.create_tensor((total + 1,), ax.bf16, ax.MemoryType.OCM, name="mel_flat")
        ax.dsl.move(r=melp[0:total], x=ax.dsl.load(mel).reshape(total))
        ax.dsl.move(r=melp[total : total + 1], x=ax.dsl.zeros((1,), ax.bf16))

        c3sh = ax.dsl.create_tensor((D + 1, 480), ax.bf16, ax.MemoryType.SHARED_OCM, name="conv3_all")
        b1o = ax.dsl.load(b1).reshape(1, 480)
        b2o = ax.dsl.load(b2).reshape(1, 480)
        b3o = ax.dsl.load(b3).reshape(1, 480)

        for ch in range(n):
            # conv1: local h1 band of 19 rows, all 50 columns, 480 channels
            i1 = ax.dsl.from_numpy(np.ascontiguousarray(idx1[core][ch].reshape(-1)),
                                   prefix=f"idx1_{core}_{ch}")
            x1 = ax.dsl.index_select(melp, i1, dim=0).reshape(19 * 50, 16)
            c1 = ax.dsl.create_tensor((19 * 50 + 1, 480), ax.bf16, ax.MemoryType.OCM, name=f"c1_{ch}")
            ax.dsl.move(r=c1[19 * 50 : 19 * 50 + 1], x=ax.dsl.zeros((1, 480), ax.bf16))
            for r0 in range(0, 19 * 50, 190):
                y = _linear(x1[r0 : r0 + 190], w1, None, "c1", output_dtype=ax.fp32, oc_tile_size=480)
                y = ax.dsl.add(y, b1o, output_dtype=ax.fp32)
                ax.dsl.move(ax.dsl.gelu(y, output_dtype=ax.bf16), r=c1[r0 : r0 + 190])

            # conv2: local h2 band of 9 rows x 25 columns, 480 channels
            i2 = ax.dsl.from_numpy(np.ascontiguousarray(idx2[core][ch].reshape(-1)),
                                   prefix=f"idx2_{core}_{ch}")
            x2 = ax.dsl.index_select(c1, i2, dim=0).reshape(9 * 25, 9 * 480)
            y2 = _linear(x2, w2, None, "c2", output_dtype=ax.fp32, oc_tile_size=32)
            y2 = ax.dsl.add(y2, b2o, output_dtype=ax.fp32)
            y2 = ax.dsl.gelu(y2, output_dtype=ax.bf16)
            c2 = ax.dsl.create_tensor((9 * 25 + 1, 480), ax.bf16, ax.MemoryType.OCM, name=f"c2_{ch}")
            ax.dsl.move(r=c2[9 * 25 : 9 * 25 + 1], x=ax.dsl.zeros((1, 480), ax.bf16))
            ax.dsl.move(r=c2[0 : 9 * 25], x=y2)

            # conv3: local 4 frequency rows x 13 time rows, 480 channels
            i3 = ax.dsl.from_numpy(np.ascontiguousarray(idx3[core][ch].reshape(-1)),
                                   prefix=f"idx3_{core}_{ch}")
            x3 = ax.dsl.index_select(c2, i3, dim=0).reshape(FQ * 13, 9 * 480)
            y3 = _linear(x3, w3, None, "c3", output_dtype=ax.fp32, oc_tile_size=32)
            y3 = ax.dsl.add(y3, b3o, output_dtype=ax.fp32)
            y3 = ax.dsl.gelu(y3, output_dtype=ax.bf16)
            d0 = ch * 208 + f0 * 13
            ax.dsl.move(r=c3sh[d0 : d0 + FQ * 13], x=y3)

        ax.dsl.move(r=c3sh[D : D + 1], x=ax.dsl.zeros((1, 480), ax.bf16))
        axd.pub(c3sh)
        axd.sub(c3sh)
        c3 = ax.dsl.create_tensor((D + 1, 480), ax.bf16, ax.MemoryType.OCM, name="conv3_local")
        ax.dsl.move(r=c3, x=c3sh)

        # conv_out: [rows, c*16+f] @ W^T + sinusoidal position
        io = ax.dsl.from_numpy(np.ascontiguousarray(idxo[core].reshape(-1)), prefix=f"idxo_{core}")
        xo = ax.dsl.index_select(c3, io, dim=0).reshape(NQ, 16, 480)
        xo = ax.dsl.transpose(xo, perm=(0, 2, 1)).reshape(NQ, 16 * 480)
        x = _linear(xo, conv_out_w, None, "conv_out", output_dtype=ax.fp32, oc_tile_size=32)
        pe_idx = ax.dsl.add(ax.dsl.from_numpy(np.arange(NQ, dtype=np.int32)),
                            ax.dsl.full((1,), ax.int32, core * NQ))
        pe = ax.dsl.index_select(ax.dsl.from_numpy(pe_all), pe_idx, dim=0)
        x = ax.dsl.add(x, ax.dsl.cast(pe, dtype=ax.fp32), output_dtype=ax.fp32)
        # Keep the layer stack in fp32: the reference tower is fp32 and the only
        # quantization it shares with this kernel is the checkpoint's bf16 weights.
        x0 = ax.dsl.create_tensor((NQ, H), ax.fp32, ax.MemoryType.OCM, name="x0")
        ax.dsl.move(r=x0, x=x)

        # ---- encoder layers in a runtime loop ---------------------------
        zero = ax.dsl.zeros((1,), ax.int32)
        one = ax.dsl.full((1,), ax.int32, 1)
        stop = ax.dsl.full((1,), ax.int32, Lc)
        mask = ax.dsl.cast(ax.dsl.load(kmask).reshape(1, P), dtype=ax.fp32)
        scale = ax.dsl.full((1,), ax.fp32, HD**-0.5)

        def body(layer, hidden):
            h = _layernorm(hidden.reshape(1, NQ, H), _bank_vec(ln1w, layer), _bank_vec(ln1b, layer),
                           out_dtype=ax.fp32).reshape(NQ, H)
            hb = ax.dsl.cast(h, dtype=ax.bf16)
            q = _bank_linear(hb, qkvw, qkvb, layer, 0, H, H, 256, f"q")
            k = _bank_linear(hb, qkvw, qkvb, layer, H, H, H, 256, f"k")
            v = _bank_linear(hb, qkvw, qkvb, layer, 2 * H, H, H, 256, f"v")
            q = ax.dsl.cast(q, dtype=ax.bf16)
            k = ax.dsl.cast(k, dtype=ax.bf16)
            v = ax.dsl.cast(v, dtype=ax.bf16)

            ksh = ax.dsl.create_tensor((P, H), ax.bf16, ax.MemoryType.SHARED_OCM, name="k_all")
            vsh = ax.dsl.create_tensor((P, H), ax.bf16, ax.MemoryType.SHARED_OCM, name="v_all")
            ax.dsl.move(r=ksh[rows], x=k)
            ax.dsl.move(r=vsh[rows], x=v)
            axd.pub(ksh)
            axd.pub(vsh)
            axd.sub(ksh)
            axd.sub(vsh)
            kl = ax.dsl.create_tensor((P, H), ax.bf16, ax.MemoryType.OCM, name="k_local")
            vl = ax.dsl.create_tensor((P, H), ax.bf16, ax.MemoryType.OCM, name="v_local")
            ax.dsl.move(r=kl, x=ksh)
            ax.dsl.move(r=vl, x=vsh)
            # second barrier: no core may rewrite the shared buffers before every
            # core has copied them out of SHARED_OCM.
            axd.pub(ksh)
            axd.pub(vsh)
            axd.sub(ksh)
            axd.sub(vsh)

            o = ax.dsl.create_tensor((NQ, H), ax.fp32, ax.MemoryType.OCM, name="attn")
            for hh in range(NH):
                cs = slice(hh * HD, (hh + 1) * HD)
                s = ax.dsl.matmul(q[:, cs], kl[:, cs], transposed=True, output_dtype=ax.fp32)
                s = ax.dsl.mul(s, scale, output_dtype=ax.fp32)
                s = ax.dsl.add(s, mask, output_dtype=ax.fp32)
                p = ax.dsl.softmax(s, dim=-1, output_dtype=ax.bf16)
                ax.dsl.move(r=o[:, cs], x=ax.dsl.matmul(p, vl[:, cs], output_dtype=ax.fp32))

            a = _bank_linear(ax.dsl.cast(o, dtype=ax.bf16), ow, ob, layer, 0, H, H, 256, "o")
            x1 = ax.dsl.add(hidden, a, output_dtype=ax.fp32)
            h2 = _layernorm(x1.reshape(1, NQ, H), _bank_vec(ln2w, layer), _bank_vec(ln2b, layer),
                            out_dtype=ax.fp32).reshape(NQ, H)
            h2 = ax.dsl.cast(h2, dtype=ax.bf16)
            upbias = _bank_vec(upb, layer)
            u = ax.dsl.create_tensor((NQ, F), ax.bf16, ax.MemoryType.OCM, name="up")
            for o0 in range(0, F, 256):
                ow_ = min(256, F - o0)
                wt = _bank_rows(upw, layer, ow_, H, r0=o0)
                yt = ax.dsl.matmul(h2, wt, transposed=True, output_dtype=ax.fp32)
                yt = ax.dsl.add(yt, upbias[o0 : o0 + ow_].reshape(1, ow_), output_dtype=ax.fp32)
                ax.dsl.move(ax.dsl.gelu(yt, output_dtype=ax.bf16), r=u[:, o0 : o0 + ow_])
            d = _bank_linear(u, downw, downb, layer, 0, F, H, 128, "down")
            xn = ax.dsl.create_tensor((NQ, H), ax.fp32, ax.MemoryType.OCM, name="x_new")
            ax.dsl.move(r=xn, x=ax.dsl.add(x1, d, output_dtype=ax.fp32))
            return (xn,)

        (hout,) = ax.dsl.range(zero, zero, stop, one, body, x0)

        # ---- final norm + projector ------------------------------------
        h = _layernorm(hout.reshape(1, NQ, H), ax.dsl.load(postw), ax.dsl.load(postb),
                       out_dtype=ax.fp32).reshape(NQ, H)
        u = _linear(ax.dsl.cast(h, dtype=ax.bf16), p1w, None, "proj1", output_dtype=ax.fp32,
                    oc_tile_size=256)
        u = ax.dsl.add(u, ax.dsl.load(p1b).reshape(1, H), output_dtype=ax.fp32)
        u = ax.dsl.gelu(u, output_dtype=ax.bf16)
        y = _linear(u, p2w, None, "proj2", output_dtype=ax.fp32, oc_tile_size=256)
        y = ax.dsl.add(y, ax.dsl.load(p2b).reshape(1, O), output_dtype=ax.fp32)
        ax.dsl.store(y, r=out[rows])
        return out

    return fused


ABI = [
    "mel", "kmask",
    "w1", "b1", "w2", "b2", "w3", "b3", "conv_out_w",
    "ln1w", "ln1b", "qkvw", "qkvb", "ow", "ob", "ln2w", "ln2b", "upw", "upb", "downw", "downb",
    "postw", "postb", "p1w", "p1b", "p2w", "p2b",
    "out",
]


def _stack(xs):
    return torch.stack([t.to(torch.bfloat16).contiguous() for t in xs]).contiguous()


def prepare_weights(model_dir, layers=None):
    cfg = audio_config(model_dir)
    sd = audio_state_dict(model_dir)
    H, F, O = cfg.d_model, cfg.encoder_ffn_dim, cfg.output_dim
    L = cfg.encoder_layers if layers is None else layers
    p = "layers."
    w = {}
    w1 = sd["conv2d1.weight"].reshape(480, 9).to(torch.float32)
    w1 = torch.nn.functional.pad(w1, (0, 7)).to(torch.bfloat16).contiguous()
    w["w1"], w["b1"] = w1, sd["conv2d1.bias"].to(torch.bfloat16).contiguous()
    w["w2"] = sd["conv2d2.weight"].permute(0, 2, 3, 1).reshape(480, 4320).to(torch.bfloat16).contiguous()
    w["b2"] = sd["conv2d2.bias"].to(torch.bfloat16).contiguous()
    w["w3"] = sd["conv2d3.weight"].permute(0, 2, 3, 1).reshape(480, 4320).to(torch.bfloat16).contiguous()
    w["b3"] = sd["conv2d3.bias"].to(torch.bfloat16).contiguous()
    w["conv_out_w"] = sd["conv_out.weight"].to(torch.bfloat16).contiguous()

    def g(l, name):
        return sd[f"{p}{l}.{name}"]

    ln1w, ln1b, qkvw, qkvb, ow, ob = [], [], [], [], [], []
    ln2w, ln2b, upw, upb, downw, downb = [], [], [], [], [], []
    for l in range(L):
        ln1w.append(g(l, "self_attn_layer_norm.weight"))
        ln1b.append(g(l, "self_attn_layer_norm.bias"))
        qkvw.append(torch.cat([g(l, f"self_attn.{x}_proj.weight") for x in "qkv"]))
        qkvb.append(torch.cat([g(l, f"self_attn.{x}_proj.bias") for x in "qkv"]))
        ow.append(g(l, "self_attn.out_proj.weight"))
        ob.append(g(l, "self_attn.out_proj.bias"))
        ln2w.append(g(l, "final_layer_norm.weight"))
        ln2b.append(g(l, "final_layer_norm.bias"))
        upw.append(g(l, "fc1.weight"))
        upb.append(g(l, "fc1.bias"))
        downw.append(g(l, "fc2.weight"))
        downb.append(g(l, "fc2.bias"))
    for name, xs in (("ln1w", ln1w), ("ln1b", ln1b), ("qkvw", qkvw), ("qkvb", qkvb), ("ow", ow), ("ob", ob),
                     ("ln2w", ln2w), ("ln2b", ln2b), ("upw", upw), ("upb", upb), ("downw", downw), ("downb", downb)):
        w[name] = _stack(xs)
    w["postw"] = sd["ln_post.weight"].to(torch.bfloat16).contiguous()
    w["postb"] = sd["ln_post.bias"].to(torch.bfloat16).contiguous()
    w["p1w"] = sd["proj1.weight"].to(torch.bfloat16).contiguous()
    w["p1b"] = sd["proj1.bias"].to(torch.bfloat16).contiguous()
    w["p2w"] = sd["proj2.weight"].to(torch.bfloat16).contiguous()
    w["p2b"] = sd["proj2.bias"].to(torch.bfloat16).contiguous()
    return cfg, w


def random_weights(cfg, layers=None, seed=0):
    """real shapes, random BF16 data (the builder's compile smoke test)"""
    torch.manual_seed(seed)
    H, F, O = cfg.d_model, cfg.encoder_ffn_dim, cfg.output_dim
    L = cfg.encoder_layers if layers is None else layers
    L = max(1, L)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape) * scale).to(torch.bfloat16).contiguous()

    w = {
        "w1": rnd(480, 16, scale=0.1), "b1": rnd(480, scale=0.01),
        "w2": rnd(480, 4320, scale=0.01), "b2": rnd(480, scale=0.01),
        "w3": rnd(480, 4320, scale=0.01), "b3": rnd(480, scale=0.01),
        "conv_out_w": rnd(H, 7680, scale=0.01),
        "ln1w": rnd(L, H, scale=1.0) + 1, "ln1b": rnd(L, H, scale=0.01),
        "qkvw": rnd(L, 3 * H, H, scale=H**-0.5), "qkvb": rnd(L, 3 * H, scale=0.01),
        "ow": rnd(L, H, H, scale=H**-0.5), "ob": rnd(L, H, scale=0.01),
        "ln2w": rnd(L, H, scale=1.0) + 1, "ln2b": rnd(L, H, scale=0.01),
        "upw": rnd(L, F, H, scale=H**-0.5), "upb": rnd(L, F, scale=0.01),
        "downw": rnd(L, H, F, scale=F**-0.5), "downb": rnd(L, H, scale=0.01),
        "postw": rnd(H) + 1, "postb": rnd(H, scale=0.01),
        "p1w": rnd(H, H, scale=H**-0.5), "p1b": rnd(H, scale=0.01),
        "p2w": rnd(O, H, scale=H**-0.5), "p2b": rnd(O, scale=0.01),
    }
    return w


def run_forward(kernel, w, mel, n, out_dim, cores=CORES, dtype=torch.float32, kmask=None):
    P = padded_rows(n)
    if kmask is None:
        kmask = key_mask(P, 13 * n)
    out = torch.zeros((P, out_dim), dtype=dtype)
    args = {"mel": mel.to(torch.bfloat16).contiguous(),
            "kmask": kmask.to(torch.bfloat16).contiguous(), "out": out, **w}
    kernel[cores](*[args[name] for name in ABI])
    return out


def build_kernel(cfg, n, layers=None, run_mode="CRUN", cache_path=None, dump_path=None):
    H, F, O, NH, L = cfg.d_model, cfg.encoder_ffn_dim, cfg.output_dim, cfg.encoder_attention_heads, cfg.encoder_layers
    fn = make_fused(H, F, O, NH, L, n, layers=layers)
    kwargs = dict(run_mode=ax.RunMode(run_mode), enable_cache=True)
    if cache_path is not None:
        kwargs["cache_path"] = cache_path
    if dump_path is not None:
        kwargs["dump_path"] = dump_path
    return ax.builder(**kwargs)(fn), fn
