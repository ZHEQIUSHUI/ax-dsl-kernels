"""Qwen3.5 prefill layer kernels: P tokens per call, so every weight is read once per P tokens instead of per token.

  gdn_layer_prefill   projections and MLP as P-row matrix products; causal conv over the chunk; the gated delta rule
                      runs token by token inside the kernel with each head's state resident on chip (same arithmetic
                      as the decode kernel), state updated in place like the decode kernel
  attn_layer_prefill  P new K/V rows written at the runtime position; the G query heads of a KV head times P tokens are
                      one G*P-row block attention against the cache (runtime block count), with a host-built causal mask
                      mask [LMAX/BC, P, BC] (row t: query token t, shared by the G heads)

Inputs as the decode kernels in q35_step.py, with x/out [1, P, H] and cos/sin [P, ROT/2].

Check on the simulator: P tokens through the prefill kernels once vs the decode kernels P times (layer outputs,
state, conv state, KV rows). Usage: MODEL=<hf dir> LMAX=4096 BC=64 PCHUNK=16 [WFMT=mx8] python q35_prefill.py
[kernel_dir] (with kernel_dir: also copy qwen35_{gdn,attn}_prefill<P>[_mx8].axmodel there). WFMT=mx8 builds the
MXFP8 kernels (a s_* scale input per projection, as q35_mx_build.py) and checks them against the MXFP8 decode kernels.
"""
import glob
import math
import os
import shutil
import sys

os.environ.setdefault("LMAX", "4096")
os.environ.setdefault("BC", "64")

import ml_dtypes
import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_gate, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.qwen_gdn_utils import chunk_prefix_sum_fp32, gdn_gate, gdn_recurrent_step, gdn_stable_chunk
from axnn.dsl.kernels.qwen_gdn_utils import l2norm_lastdim
import q35_step as q

P = int(os.environ.get("PCHUNK", 16))
# GDN=chunk: delta rule in the chunked (WY / triangular-solve) form, official gdn_stable_chunk with the FP32 "triton"
# solver, CHUNK tokens per solve and HG value heads batched per solve; GDN=seq: token by token as the decode kernel
GDN = os.environ.get("GDN", "chunk")
CHUNK = min(P, int(os.environ.get("CHUNK", 32)))
HG = int(os.environ.get("HG", 8))
H, D, HQ, HKV, G = q.H, q.D, q.HQ, q.HKV, q.G
NK, NV, DK, DV, KD, VD = q.NK, q.NV, q.DK, q.DV, q.KD, q.VD
EPS, ROT, LMAX, BC = q.EPS, q.ROT, q.LMAX, q.BC
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_prefill"))
# attention: P rows of independent tokens (batched decode, q35_batch.py): K/V row t goes to cell pos[t] (pos [P])
SCATTER = False


def gdn_layer_prefill(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                      state_in, conv_in, out, state_out, conv_out):
    xt = ax.dsl.load(x)  # [1, P, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qkv = q.proj(h, w_qkv, "gdn_qkv")  # [1, P, 2*KD+VD]
    z = q.proj(h, w_z, "gdn_z").reshape(1, P, NV, DV)
    b = q.proj(h, w_b, "gdn_b")
    a = q.proj(h, w_a, "gdn_a")
    mixed = stateful_runtime_causal_depthwise_conv1d_silu(qkv, conv_w, conv_in, conv_out, q.CONV, "gdn_conv")
    qm = mixed[:, :, 0:KD].reshape(1, P, NK, DK)
    km = mixed[:, :, KD : 2 * KD].reshape(1, P, NK, DK)
    vm = mixed[:, :, 2 * KD :].reshape(1, P, NV, DV)

    a_log_t = ax.dsl.load(a_log)
    dt_bias_t = ax.dsl.load(dt_bias)
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    core = ax.dsl.create_tensor((1, P, NV, DV), ax.bf16, name="gdn_core")
    rep = NV // NK
    if GDN == "chunk":
        gdn_chunked(qm, km, vm, a, b, a_log_t, dt_bias_t, q_scale, state_in, state_out, core)
    for kh in range(NK if GDN == "seq" else 0):
        for vo in range(rep):
            vh = vo * NK + kh if q.VORDER == "tiled" else kh * rep + vo
            st = ax.dsl.load(state_in[vh : vh + 1]).reshape(DK, DV)
            for t in range(P):
                q_tok = ax.dsl.mul(l2norm_lastdim(qm[:, t : t + 1, kh : kh + 1, :].reshape(1, DK)), q_scale,
                                   output_dtype=ax.bf16)
                k_tok = l2norm_lastdim(km[:, t : t + 1, kh : kh + 1, :].reshape(1, DK))
                g, beta = gdn_gate(a[:, t : t + 1, vh : vh + 1].reshape(1, 1), b[:, t : t + 1, vh : vh + 1].reshape(1, 1),
                                   a_log_t[vh : vh + 1], dt_bias_t[vh : vh + 1])
                o_h, st = gdn_recurrent_step(q_tok, k_tok, vm[:, t : t + 1, vh : vh + 1, :].reshape(1, DV), g, beta, st,
                                             ax.bf16)
                ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, t : t + 1, vh : vh + 1, :])
            ax.dsl.store(x=st.reshape(1, DK, DV), r=state_out[vh : vh + 1])

    normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, P, VD)
    x1 = ax.dsl.add(xt, q.proj(gated, w_out, "gdn_out"), output_dtype=ax.bf16)
    q.mlp_residual(x1, ln2, w_gate, w_up, w_down, out)
    return out, state_out, conv_out


def gdn_chunked(qm, km, vm, a, b, a_log_t, dt_bias_t, q_scale, state_in, state_out, core):
    """Delta rule for the P chunk tokens in the chunked form: HG heads per solve, CHUNK tokens per solve."""
    # the HG value heads h0 .. h0 + HG - 1 use key heads h0 % NK .. (+ HG): consecutive in the tiled order
    assert (NV == NK or q.VORDER == "tiled") and NK % HG == 0 and P % CHUNK == 0
    qn = ax.dsl.mul(l2norm_lastdim(qm), q_scale, output_dtype=ax.bf16)  # [1, P, NK, DK]
    kn = l2norm_lastdim(km)
    g_all, beta_all = gdn_gate(a.reshape(1, P, NV), b.reshape(1, P, NV), a_log_t, dt_bias_t)  # [1, P, NV]
    prefix_all = chunk_prefix_sum_fp32(g_all, CHUNK, "gdn_prefix_g")  # restarts every CHUNK tokens
    exp_prefix_all = ax.dsl.exp(prefix_all, output_dtype=ax.fp32)
    for h0 in range(0, NV, HG):
        hs = slice(h0, h0 + HG)
        ks = slice(h0 % NK, h0 % NK + HG)
        st = ax.dsl.load(state_in[hs]).reshape(HG, DK, DV)
        for c0 in range(0, P, CHUNK):
            cs = slice(c0, c0 + CHUNK)
            q_tile = ax.dsl.transpose(qn[:, cs, ks, :], perm=(0, 2, 1, 3)).reshape(HG, CHUNK, DK)
            k_tile = ax.dsl.transpose(kn[:, cs, ks, :], perm=(0, 2, 1, 3)).reshape(HG, CHUNK, DK)
            v_tile = ax.dsl.transpose(vm[:, cs, hs, :], perm=(0, 2, 1, 3)).reshape(HG, CHUNK, DV)
            prefix_g = ax.dsl.transpose(prefix_all[:, cs, hs], perm=(0, 2, 1)).reshape(HG, CHUNK, 1)
            exp_prefix_g = ax.dsl.transpose(exp_prefix_all[:, cs, hs], perm=(0, 2, 1)).reshape(HG, CHUNK, 1)
            beta = ax.dsl.transpose(beta_all[:, cs, hs], perm=(0, 2, 1)).reshape(HG, CHUNK, 1)
            out_tile, st = gdn_stable_chunk(q_tile, k_tile, v_tile, beta, st, prefix_g, exp_prefix_g,
                                            f"gdn_h{h0}_c{c0}", solver="triton")
            ax.dsl.move(x=ax.dsl.transpose(out_tile.reshape(1, HG, CHUNK, DV), perm=(0, 2, 1, 3)),
                        r=core[:, cs, hs, :])
        ax.dsl.store(x=st.reshape(HG, DK, DV), r=state_out[hs])


def attn_layer_prefill(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                       w_down, out):
    xt = ax.dsl.load(x)  # [1, P, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, w_q, "attn_q").reshape(P, HQ, 2 * D)  # per head [q | gate]
    qh_all = rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS)
    gate = qg[:, :, D : 2 * D]
    k = rmsnorm(q.proj(h, w_k, "attn_k").reshape(P, HKV, D), ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
    v = q.proj(h, w_v, "attn_v").reshape(P, HKV, D)
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)  # [P, ROT/2]
    qh_all = rope(qh_all, cos_t, sin_t, rotary_dim=ROT)
    k = rope(k, cos_t, sin_t, rotary_dim=ROT)

    # P new K/V rows at the runtime position (head layout: per KV head rows pos .. pos+P-1; cell layout: cells
    # pos .. pos+P-1, both heads per row, f16)
    rows = ax.dsl.from_numpy(np.array([1, P] if q.KVL == "head" else [P], dtype=np.uint32))
    src0 = ax.dsl.from_numpy(np.zeros(shape=(2 if q.KVL == "head" else 1,), dtype=np.uint32))
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    if q.KVL == "cell" and SCATTER:
        one = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
        k16 = ax.dsl.cast(k.reshape(P, HKV * D), dtype=ax.fp16)
        v16 = ax.dsl.cast(v.reshape(P, HKV * D), dtype=ax.fp16)
        for t in range(P):
            ax.dsl.dynamic_st(k16[t : t + 1, :], src0, p[t : t + 1], one, dim=(0,), output_shape=kc.shape, r=kc)
            ax.dsl.dynamic_st(v16[t : t + 1, :], src0, p[t : t + 1], one, dim=(0,), output_shape=vc.shape, r=vc)
    elif q.KVL == "cell":
        ax.dsl.dynamic_st(ax.dsl.cast(k.reshape(P, HKV * D), dtype=ax.fp16), src0, p, rows, dim=(0,),
                          output_shape=kc.shape, r=kc)
        ax.dsl.dynamic_st(ax.dsl.cast(v.reshape(P, HKV * D), dtype=ax.fp16), src0, p, rows, dim=(0,),
                          output_shape=vc.shape, r=vc)
    for hk in range(HKV if q.KVL == "head" else 0):
        dst = ax.dsl.concat(ax.dsl.from_numpy(np.array([hk], dtype=np.uint32)), p, dim=0)
        kt = ax.dsl.create_tensor((P, 1, D), ax.bf16, name=f"k_rows{hk}")
        vt = ax.dsl.create_tensor((P, 1, D), ax.bf16, name=f"v_rows{hk}")
        ax.dsl.move(x=k[:, hk : hk + 1, :], r=kt)
        ax.dsl.move(x=v[:, hk : hk + 1, :], r=vt)
        ax.dsl.dynamic_st(kt.reshape(1, P, D), src0, dst, rows, dim=(0, 1), output_shape=kc.shape, r=kc)
        ax.dsl.dynamic_st(vt.reshape(1, P, D), src0, dst, rows, dim=(0, 1), output_shape=vc.shape, r=vc)

    R = G * P  # query rows per KV head: g * P + t
    attn = ax.dsl.create_tensor((P, HQ, D), ax.bf16, ax.MemoryType.OCM, name="attn")
    kv_block_shape = ax.dsl.from_numpy(np.array([1, BC] if q.KVL == "head" else [BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_stop = ax.dsl.load(nblk)
    for hk in range(HKV):
        head = ax.dsl.from_numpy(np.array([hk], dtype=np.uint32))
        qh = ax.dsl.create_tensor((R, D), ax.bf16, name=f"qh{hk}")
        for g in range(G):
            ax.dsl.mul(qh_all[:, hk * G + g : hk * G + g + 1, :].reshape(P, D), scale, output_dtype=ax.bf16,
                       r=qh[g * P : (g + 1) * P, :])
        oi = ax.dsl.zeros(shape=(R, D), dtype=ax.bf16)
        li = ax.dsl.zeros(shape=(R,), dtype=ax.fp32)
        mi = ax.dsl.full(shape=(R, 1), dtype=ax.bf16, fill_value=-3e38)
        zero_col = ax.dsl.zeros(shape=(R, 1), dtype=ax.bf16)
        index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

        def body(index, kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape,
                 zero_col, minus_1):
            loop_mi, loop_li, loop_oi = mi, li, oi
            t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
            if q.KVL == "head":
                off = ax.dsl.concat(head, t0, dim=0)
                kj = ax.dsl.dynamic_ld(kc, off, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
                vj = ax.dsl.dynamic_ld(vc, off, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
            else:  # BC cells x both heads, then this head's columns
                kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D))
                vb = ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D))
                kj = ax.dsl.cast(kb[:, hk * D : (hk + 1) * D], dtype=ax.bf16)
                vj = ax.dsl.cast(vb[:, hk * D : (hk + 1) * D], dtype=ax.bf16)
            mj = ax.dsl.dynamic_ld(mask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                                   output_shape=(1, P, BC)).reshape(P, BC)
            m_s = ax.dsl.create_tensor(shape=(R, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            for g in range(G):  # the P mask rows apply to each of the G heads' P rows
                ax.dsl.add(sj[g * P : (g + 1) * P, :], mj, output_dtype=ax.bf16, r=m_s[g * P : (g + 1) * P, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(R)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
            oij = pj @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(R, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
            return (kc, vc, mask, qh, loop_mi, loop_li, loop_oi, head, block_stride, kv_block_shape,
                    mask_block_shape, zero_col, minus_1)

        (kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape, zero_col,
         minus_1) = ax.dsl.range(
            index, range_start, range_stop, range_step, body,
            kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1,
        )
        o = ax.dsl.div(oi, li.reshape(R, 1), output_dtype=ax.bf16)
        for g in range(G):
            ax.dsl.move(x=o[g * P : (g + 1) * P, :].reshape(P, 1, D), r=attn[:, hk * G + g : hk * G + g + 1, :])

    gated = sigmoid_mul(attn, gate, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, P, HQ * D)
    x1 = ax.dsl.add(xt, q.proj(gated, w_o, "attn_o"), output_dtype=ax.bf16)
    q.mlp_residual(x1, ln2, w_gate, w_up, w_down, out)
    return out, kc, vc


GDN_ORDER = q.GDN_ORDER
ATTN_ORDER = q.ATTN_ORDER
WFMT = os.environ.get("WFMT", "bf16")
if WFMT == "mx8":
    import q35_mx_build as mb  # patches q.proj to the MX linear (scales looked up per weight)

    def gdn_layer_prefill_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out,
                             s_out, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, state_in, conv_in, out, state_out,
                             conv_out):
        mb.SCALES.clear()
        for w, s in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate),
                     (w_up, s_up), (w_down, s_down)]:
            mb.SCALES[id(w)] = s
        gdn_layer_prefill(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up,
                          w_down, state_in, conv_in, out, state_out, conv_out)
        return out, state_out, conv_out

    def attn_layer_prefill_mx(x, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask,
                              w_o, s_o, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, out):
        mb.SCALES.clear()
        for w, s in [(w_q, s_q), (w_k, s_k), (w_v, s_v), (w_o, s_o), (w_gate, s_gate), (w_up, s_up), (w_down, s_down)]:
            mb.SCALES[id(w)] = s
        attn_layer_prefill(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate,
                           w_up, w_down, out)
        return out, kc, vc


def prefill_mask(pos0):
    """[LMAX/BC, P, BC] bf16: row t sees positions <= pos0 + t."""
    col = torch.arange(LMAX).reshape(LMAX // BC, 1, BC)
    lim = (pos0 + torch.arange(P)).reshape(1, P, 1)
    return torch.where(col <= lim, 0.0, -3e38).bfloat16()


def decode_mask(pos):
    m = torch.full((LMAX,), -3e38)
    m[: pos + 1] = 0
    return m.reshape(LMAX // BC, BC).bfloat16()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None  # before the chdir below
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    emb = W("embed_tokens.weight")
    torch.manual_seed(0)
    pos0 = int(os.environ.get("POS0", 70))  # chunk start: earlier KV rows / state are random "history"
    xs = emb[torch.randint(0, 200000, (P,))].reshape(1, P, H).contiguous()

    def kern(fn, sub):
        os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
        os.chdir(f"{BUILD}/{sub}")
        return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)

    mx = WFMT == "mx8"
    gdn_dec, gdn_pre = (mb.gdn_layer_mx, gdn_layer_prefill_mx) if mx else (q.gdn_layer, gdn_layer_prefill)
    attn_dec, attn_pre = (mb.attn_layer_mx, attn_layer_prefill_mx) if mx else (q.attn_layer, attn_layer_prefill)
    gdn_order = mb.order(mb.gdn_layer_mx) if mx else GDN_ORDER
    attn_order = mb.order(mb.attn_layer_mx) if mx else ATTN_ORDER
    sfx = "_mx8" if mx else ""
    only = os.environ.get("ONLY")  # "attn": skip the GDN layer (e.g. a KV-layout change)
    kv_sfx = f"_kvc_c{LMAX}" if q.KVL == "cell" else ""  # llama.cpp KV layout, named by context

    # --- GDN layer 0
    a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0] if mx else q.gdn_args(W, 0)
    gdn_check(a, xs, gdn_dec, gdn_pre, gdn_order, sfx, kern) if only != "attn" else None

    # --- attention layer 3
    a = mb.mx_args(q.attn_args(W, 3), mb.ATTN_W)[0] if mx else q.attn_args(W, 3)
    if q.KVL == "head":
        kc0 = (torch.randn((HKV, LMAX, D)) * 0.5).bfloat16()
        vc0 = (torch.randn((HKV, LMAX, D)) * 0.5).bfloat16()
        sel = lambda c, r: c[:, r]  # noqa: E731
    else:  # llama.cpp cells: [LMAX, HKV * D] f16
        kc0 = (torch.randn((LMAX, HKV * D)) * 0.5).half()
        vc0 = (torch.randn((LMAX, HKV * D)) * 0.5).half()
        sel = lambda c, r: c[r]  # noqa: E731
    attn_check(a, xs, attn_dec, attn_pre, attn_order, sfx + kv_sfx, kern, pos0, kc0, vc0, sel)

    if out_dir:
        for layer, fn in (("gdn", gdn_pre), ("attn", attn_pre)):
            if layer == "gdn" and only == "attn":
                continue
            tag = sfx + (kv_sfx if layer == "attn" else "")
            (path,) = glob.glob(f"{BUILD}/{layer}_pre{P}{tag}/cache/{fn.__name__}.*.compiled.axmodel")
            shutil.copy(path, f"{out_dir}/qwen35_{layer}_prefill{P}{tag}.axmodel")
            logger.info(f"wrote {out_dir}/qwen35_{layer}_prefill{P}{tag}.axmodel")


NODEC = os.environ.get("NODEC") == "1"  # only the prefill kernels (build, cycles), no token-by-token decode reference


def gdn_check(a, xs, gdn_dec, gdn_pre, gdn_order, sfx, kern):
    st0 = torch.randn((NV, DK, DV)) * 0.02
    cv0 = (torch.randn((1, q.CONV - 1, 2 * KD + VD)) * 0.1).bfloat16()
    if NODEC:
        case = dict(a, x=xs, state_in=st0.clone(), conv_in=cv0.clone(), out=torch.zeros((1, P, H), dtype=torch.bfloat16),
                    state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
        kern(gdn_pre, f"gdn_pre{P}{sfx}")[1](*[case[n] for n in gdn_order])
        logger.info(f"gdn prefill P={P}: built, out finite {bool(case['out'].float().isfinite().all())}")
        return
    k_dec, k_pre = kern(gdn_dec, f"gdn_dec{sfx}"), kern(gdn_pre, f"gdn_pre{P}{sfx}")
    st, cv, dec_out = st0.clone(), cv0.clone(), []
    for t in range(P):
        case = dict(a, x=xs[:, t : t + 1].contiguous(), state_in=st, conv_in=cv,
                    out=torch.zeros((1, 1, H), dtype=torch.bfloat16), state_out=torch.zeros_like(st),
                    conv_out=torch.zeros_like(cv))
        k_dec[1](*[case[n] for n in gdn_order])
        dec_out.append(case["out"].float())
        st, cv = case["state_out"], case["conv_out"]
    case = dict(a, x=xs, state_in=st0.clone(), conv_in=cv0.clone(), out=torch.zeros((1, P, H), dtype=torch.bfloat16),
                state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
    k_pre[1](*[case[n] for n in gdn_order])
    po = case["out"].float()
    worst = min(cos(po[:, t], dec_out[t]) for t in range(P))
    logger.info(f"gdn prefill P={P}: out min cos {worst:.6f} bitwise={all(torch.equal(po[:, t:t+1], dec_out[t]) for t in range(P))}"
                f" state cos {cos(case['state_out'], st):.6f} conv equal {torch.equal(case['conv_out'], cv)}")

def attn_check(a, xs, attn_dec, attn_pre, attn_order, sfx, kern, pos0, kc0, vc0, sel):
    k_dec, k_pre = (None if NODEC else kern(attn_dec, f"attn_dec{sfx}")), kern(attn_pre, f"attn_pre{P}{sfx}")
    kc, vc, dec_out = kc0.clone(), vc0.clone(), []
    for t in range(0 if NODEC else P):
        p = pos0 + t
        c, s = q.rope_cos_sin(p)
        case = dict(a, x=xs[:, t : t + 1].contiguous(), cos=c, sin=s, kc=kc, vc=vc,
                    pos=torch.tensor([p], dtype=torch.int32), nblk=torch.tensor([p // BC + 1], dtype=torch.int32),
                    mask=decode_mask(p), out=torch.zeros((1, 1, H), dtype=torch.bfloat16))
        q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
        k_dec[1](*[case[n] for n in attn_order])
        q.RESIDENT.clear()
        dec_out.append(case["out"].float())
        kc, vc = case["kc"], case["vc"]
    cs = [q.rope_cos_sin(pos0 + t) for t in range(P)]
    case = dict(a, x=xs, cos=torch.cat([c for c, _ in cs]), sin=torch.cat([s for _, s in cs]), kc=kc0.clone(),
                vc=vc0.clone(), pos=torch.tensor([pos0], dtype=torch.int32),
                nblk=torch.tensor([(pos0 + P - 1) // BC + 1], dtype=torch.int32), mask=prefill_mask(pos0),
                out=torch.zeros((1, P, H), dtype=torch.bfloat16))
    q.RESIDENT.update(kc=kc0.clone(), vc=vc0.clone())
    k_pre[1](*[case[n] for n in attn_order])
    q.RESIDENT.clear()
    po = case["out"].float()
    if NODEC:
        logger.info(f"attn prefill P={P}: built, out finite {bool(po.isfinite().all())}")
        return
    worst = min(cos(po[:, t], dec_out[t]) for t in range(P))
    rows = slice(pos0, pos0 + P)
    logger.info(f"attn prefill P={P} pos0={pos0}: out min cos {worst:.6f} "
                f"kv rows equal {torch.equal(sel(case['kc'], rows), sel(kc, rows)) and torch.equal(sel(case['vc'], rows), sel(vc, rows))}"
                f" other rows untouched {torch.equal(sel(case['kc'], slice(0, pos0)), sel(kc0, slice(0, pos0)))}")



if __name__ == "__main__":
    main()
