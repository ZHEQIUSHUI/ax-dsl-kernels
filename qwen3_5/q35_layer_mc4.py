"""The decode layer kernels of the ggml-axera template (ggml_template_build.py: q35_step.gdn_layer / attn_layer, P = 1)
on CORES NPU cores in one call per layer, the cores synchronized through shared OCM (tensor pub / sub) between the
halves of the layer, as q3tts_cp_mc.py:

  gdn_layer   every core: rmsnorm of x; its NK / CORES key heads and the value heads that use them (as q35_mc.gdn_body:
              their q / k / v channels of the projection and causal conv, conv and state rows in place, the recurrent
              step, gated rmsnorm), the out-projection partial of their columns -> shared part_a[c]; pub / sub;
              x1 = x + sum_c part_a[c]; its F / CORES MLP channels -> shared part_m[c]; pub / sub; x2 = x1 + sum
  attn_layer  every core: its HQ / CORES query heads ([q | gate] rows), the KV heads they use (k / v rows, written to
              the cache row pos: a KV head shared by several cores is written by each, the same bytes), their attention
              over the cache, the output gate, the o-projection partial -> part_a; then the MLP as above

Core 0 stores out. Same IO and names as the single-core kernels (bf16, or WFMT=mx8 the MXFP8 ones), written as
<name>_npu<CORES>: qwen35_gdn[_mx8], <KP>_attn[_mx8]_kvc_c<LMAX> (KVL=cell). Checked on the simulator against the
single-core kernel for positions 0 .. STEPS-1 (out, state / conv, K / V).

Usage: MODEL=<hf dir> KVL=cell LMAX=4096 BC=64 CORES=4 [WFMT=mx8] [ONLY=gdn|attn] [STEPS=2] python q35_layer_mc.py <dir>
"""
import glob
import math
import os
import shutil
import sys

os.environ.setdefault("KVL", "cell")
os.environ.setdefault("LMAX", "4096")
os.environ.setdefault("BC", "64")

import ml_dtypes
import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import rmsnorm, silu_gate, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.qwen_gdn_utils import (chunk_prefix_sum_fp32, gdn_gate, gdn_recurrent_step, gdn_stable_chunk,
                                             l2norm_lastdim)
import board_io
import q35_step as q

MOE = "num_experts" in q.cfg  # 35B-A3B: the MLP replaced by the MoE's dense part, the experts a kernel of their own
if MOE:
    import q35_moe as mo  # q.mlp_args: the shared expert, router and its gate
MX = os.environ.get("WFMT", "bf16") in ("mx8", "nv4")  # nv4: the projections in NVFP4 (q35_mx_build.NV4)
NV4 = os.environ.get("WFMT") == "nv4"
if MX:
    import q35_mx_build as mb  # q.proj / q.slice_w: the MX linear, weight slices carry their scales
CORES = int(os.environ.get("CORES", 4))
STEPS = int(os.environ.get("STEPS", 2))
TOPK_OUT = os.environ.get("TOPK_OUT") == "1"  # MoE prefill: also the router's top-K per token (topk_v / topk_i, _tk)
TK = {}  # the top-K outputs of the kernel being traced (topk wrappers -> finish_pre)
SAVE = os.environ.get("SAVE")  # attention check: torch.save each step's outputs to <SAVE>_<step>.pt
NOREF = os.environ.get("NOREF") == "1"  # attention: build the four-core kernel without the single-core reference run
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_layer_mc"))
H, F, EPS, LMAX, BC = q.H, q.F, q.EPS, q.LMAX, q.BC
HQ, HKV, D, G, ROT = q.HQ, q.HKV, q.D, q.G, q.ROT
NK, NV, DK, DV, KD, VD, CONV = q.NK, q.NV, q.DK, q.DV, q.KD, q.VD, q.CONV
QH, FC = HQ // CORES, F // CORES  # query heads, MLP channels per core
QW = (2 if q.GATE else 1) * D  # q_proj rows per query head ([q | gate])
KHC, REP = max(1, NK // CORES), NV // NK  # delta-net key heads per core, value heads per key head
assert HQ % CORES == 0 and F % CORES == 0 and q.KVL == "cell"
SFX = "_nv4" if NV4 else "_mx8" if MX else ""
# WFMT4=1 (with WFMT=mx8): the layer weights in MXFP4 (E2M1 pairs, low nibble first, an E8M0 scale per 32, as the
# 27B GGUF's): the same bodies, kernels _mx4 with the single-core _mx4 kernels' IO (q35_layer_pk.py); NOREF=1 needed
MX4 = MX and os.environ.get("WFMT4", "0") == "1"
if MX4:
    SFX = "_mx4"


def mx4_args(a, names):
    """the weights of names (bf16 / f32) as MXFP4 w_<x> (fp4 pairs) and s_<x> (e8m0), as q35_layer_pk.mx4_args"""
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    a = dict(a)
    for n in names:
        if n not in a or not torch.is_tensor(a[n]):
            continue
        w = a[n].float()
        r, k = w.shape
        g = w.reshape(r, k // mb.GROUP, mb.GROUP)
        amax = g.abs().amax(-1, keepdim=True)
        e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2, torch.zeros_like(amax))
        x = (g / torch.exp2(e)).reshape(r, k)
        code = ((x.abs().unsqueeze(-1) - e2m1).abs().argmin(-1) | ((x < 0).long() << 3)).to(torch.uint8)
        a[n] = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
        a["s_" + n[2:]] = (e.squeeze(-1) + 127).to(torch.uint8).contiguous().view(torch.float8_e8m0fnu)
    return a


PFX = [""]  # a prefix of every named tensor of a layer (several layers in one kernel: q35_moe_group.py)


def T(name):
    return PFX[0] + name


class Cols:
    """a weight split by columns over the cores, stored as one contiguous block per (core, strip) (packed kernels): the
    block of a column range, its scales registered; a plain weight is sliced"""

    def __init__(self, blocks):
        self.blocks = blocks  # (start, stop) -> weight block [rows, stop - start]


def wcols(w, sl):
    return w.blocks[(sl.start, sl.stop)] if isinstance(w, Cols) else q.slice_w(w, cols=sl)



# KEEP_SHM (default on): every shared OCM tensor of a call is read once more right before the call's output is stored,
# so its space is not handed to a later tensor before then. An extra safeguard only: onepass allocates shared OCM once
# for all cores, and where it reuses an address update_job_deps already orders the later writes after the earlier reads
# (the NVFP4 layer groups' run-to-run differences have another cause). The reads are selected away by a false condition
# (no arithmetic: a -inf / NaN in a slot cannot reach the output)
KEEP_SHM = os.environ.get("KEEP_SHM", "1") == "1"
SHM_LIVE = []  # the shared OCM tensors since the last keep_shm


def shm_tensor(shape, dtype, name):
    t = ax.dsl.create_tensor(shape, dtype, ax.MemoryType.SHARED_OCM, name=name)
    SHM_LIVE.append(t)
    return t


def keep_shm(y):
    """y itself, read after every shared OCM tensor recorded since the last call (then forgotten)"""
    live = list(SHM_LIVE)
    SHM_LIVE.clear()
    if not KEEP_SHM or not live:
        return y
    one = (1,) * len(y.shape)
    off = ax.dsl.full(shape=one, dtype=ax.uint8, fill_value=0)
    for t in live:
        e = t[tuple(slice(0, 1) for _ in t.shape)].reshape(*one)
        y = ax.dsl.where(off, ax.dsl.cast(e, dtype=y.dtype), y, output_dtype=y.dtype)
    return y

def reduce(c, x, part, partial):
    """x + the cores' partials (fp32 [1, 1, H]): the core's into shared part[c], published, all of them summed; the sum
    rounded to bf16 and added to x, as the single-core layer's bf16 projection output"""
    ax.dsl.move(r=part[c : c + 1], x=partial.reshape(1, 1, H))
    axd.pub(part)
    axd.sub(part)
    acc = part[0:1]
    for j in range(1, CORES):
        acc = ax.dsl.add(acc, part[j : j + 1], output_dtype=ax.fp32)
    return ax.dsl.add(x, ax.dsl.cast(acc, dtype=ax.bf16), output_dtype=ax.bf16)


def mlp(c, x1, ln2, w_gate, w_up, w_down):
    fs = slice(c * FC, (c + 1) * FC)
    h2 = rmsnorm(x1, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
    act = silu_mul(q.proj(h2, q.slice_w(w_gate, rows=fs), T("mlp_gate")), q.proj(h2, q.slice_w(w_up, rows=fs), T("mlp_up")),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
    d = q.proj(act, wcols(w_down, fs), T("mlp_down"), output_dtype=ax.fp32)
    part_m = shm_tensor((CORES, 1, H), ax.fp32, name=T("part_m"))
    return reduce(c, x1, part_m, d)


def moe_part(c, x1, ln2, w_gate, w_up, w_down, out, w_router, w_shg, h2_out, logits_out):
    """the MoE's dense part (q35_moe.moe_pre) on the cores: the shared expert's F / CORES channels (fp32 partials
    summed through shared OCM), its sigmoid gate, the router logits of NE / CORES experts per core"""
    fs, ne = slice(c * FC, (c + 1) * FC), mo.NE // CORES
    h2 = rmsnorm(x1, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
    act = silu_mul(q.proj(h2, q.slice_w(w_gate, rows=fs), "sh_gate"), q.proj(h2, q.slice_w(w_up, rows=fs), "sh_up"),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
    d = q.proj(act, wcols(w_down, fs), "sh_down", output_dtype=ax.fp32)
    part = shm_tensor((CORES, 1, H), ax.fp32, name="part_m")
    ax.dsl.move(r=part[c : c + 1], x=d.reshape(1, 1, H))
    axd.pub(part)
    axd.sub(part)
    sh = part[0:1]
    for j in range(1, CORES):
        sh = ax.dsl.add(sh, part[j : j + 1], output_dtype=ax.fp32)
    rs = slice(c * ne, (c + 1) * ne)
    logits = linear(h2, w_router[rs], None, "router", output_dtype=ax.fp32, oc_tile_size=ne)
    ax.dsl.store(logits, r=logits_out[:, :, rs])
    if c == 0:
        g = ax.dsl.sum(ax.dsl.mul(h2, ax.dsl.load(w_shg).reshape(1, 1, H), output_dtype=ax.fp32), dim=-1, keepdim=True,
                       output_dtype=ax.fp32)
        base = ax.dsl.add(x1, ax.dsl.mul(sh, ax.dsl.sigmoid(g, output_dtype=ax.fp32), output_dtype=ax.fp32),
                          output_dtype=ax.bf16)
        ax.dsl.store(base, r=out)
        ax.dsl.store(h2, r=h2_out)


def finish(c, x1, ln2, w_gate, w_up, w_down, out, moe):
    """the layer after its attention / delta-net half: the MLP (core 0 stores out), or the MoE's dense part"""
    if moe is not None and "e_gw" in moe:
        return moe_full(c, x1, ln2, w_gate, w_up, w_down, out, **moe)
    if moe is not None:
        moe_part(c, x1, ln2, w_gate, w_up, w_down, out, **moe)
        return
    x2 = keep_shm(mlp(c, x1, ln2, w_gate, w_up, w_down))
    if c == 0:
        ax.dsl.store(x2, r=out)


def gdn_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
             state_in, conv_in, out, state_out, conv_out, moe=None, xt=None):
    assert NK % CORES == 0 and (REP == 1 or q.VORDER == "tiled")
    c = ax.program_id(0)
    SHM_LIVE.clear()
    xt = ax.dsl.load(x) if xt is None else xt  # xt: the layer input already on chip (after the experts)
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)

    def conv(off, n, i):  # the projection and causal conv of channels off .. off + n
        ch = slice(off, off + n)
        pr = q.proj(h, q.slice_w(w_qkv, rows=ch), T(f"gdn_qkv{i}"))
        return stateful_runtime_causal_depthwise_conv1d_silu(pr, conv_w[ch], conv_in[:, :, ch], conv_out[:, :, ch],
                                                             CONV, T(f"gdn_conv{i}"))

    qm = conv(c * KHC * DK, KHC * DK, 0).reshape(1, 1, KHC, DK)
    km = conv(KD + c * KHC * DK, KHC * DK, 1).reshape(1, 1, KHC, DK)
    b_all = q.proj(h, w_b, T("gdn_b"))
    a_all = q.proj(h, w_a, T("gdn_a"))
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    qt = [ax.dsl.mul(l2norm_lastdim(qm[:, :, j : j + 1, :].reshape(1, DK)), q_scale, output_dtype=ax.bf16)
          for j in range(KHC)]
    kt = [l2norm_lastdim(km[:, :, j : j + 1, :].reshape(1, DK)) for j in range(KHC)]
    o_sum = None
    for vo in range(REP):
        v0 = vo * NK + c * KHC  # the strip's first value head (tiled order: value head vh uses key head vh % NK)
        vs = slice(v0, v0 + KHC)
        t = "" if REP == 1 else str(vo)
        vm = conv(2 * KD + v0 * DV, KHC * DV, 2 + vo).reshape(1, 1, KHC, DV)
        z = q.proj(h, q.slice_w(w_z, rows=slice(v0 * DV, (v0 + KHC) * DV)), T("gdn_z" + t)).reshape(1, 1, KHC, DV)
        b, a = b_all[:, :, vs], a_all[:, :, vs]
        a_log_t, dt_bias_t = ax.dsl.load(a_log[vs]), ax.dsl.load(dt_bias[vs])
        core = ax.dsl.create_tensor((1, 1, KHC, DV), ax.bf16, name=T("gdn_core" + t))
        for j in range(KHC):
            g, beta = gdn_gate(a[:, :, j : j + 1].reshape(1, 1), b[:, :, j : j + 1].reshape(1, 1), a_log_t[j : j + 1],
                               dt_bias_t[j : j + 1])
            vh = v0 + j
            st = ax.dsl.load(state_in[vh : vh + 1]).reshape(DK, DV)
            o_h, st = gdn_recurrent_step(qt[j], kt[j], vm[:, :, j : j + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
            ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, :, j : j + 1, :])
            ax.dsl.store(x=st.reshape(1, DK, DV), r=state_out[vh : vh + 1])
        normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
        gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, 1, KHC * DV)
        o = q.proj(gated, wcols(w_out, slice(v0 * DV, (v0 + KHC) * DV)), T("gdn_out" + t), output_dtype=ax.fp32)
        o_sum = o if o_sum is None else ax.dsl.add(o_sum, o, output_dtype=ax.fp32)
    part_a = shm_tensor((CORES, 1, H), ax.fp32, name=T("part_a"))
    return finish(c, reduce(c, xt, part_a, o_sum), ln2, w_gate, w_up, w_down, out, moe)


# KV cache access for any context: dynamic_ld / dynamic_st check the whole tensor against the SDMA's 26-bit range (a
# cache over 64 MB fails to build), index_select / select_scatter do not
NBLK = LMAX // BC


def kv_block(cache, index, hk):
    """block `index` (int32 [1]) of the cache [LMAX, HKV * D] f16, KV head hk's columns: [BC, D] bf16"""
    blk = ax.dsl.index_select(cache.reshape(NBLK, BC, HKV, D)[:, :, hk : hk + 1, :], index, dim=0)
    return ax.dsl.cast(blk.reshape(BC, D), dtype=ax.bf16)


def mask_block(mask, index, rows):
    """block `index` of the mask: [rows, BC]"""
    return ax.dsl.index_select(mask.reshape(NBLK, rows, BC), index, dim=0).reshape(rows, BC)


def kv_write(kc, vc, k, v, p, rows):
    """rows new K / V rows (all heads) at cells p .. p + rows - 1, then a zero read back from them: whatever depends on
    it runs after this core's store has landed (the dynamic indices hide that dependency from the scheduler)"""
    idx = p if rows == 1 else ax.dsl.add(p, ax.dsl.from_numpy(np.arange(rows, dtype=np.uint32)), output_dtype=ax.uint32)
    idx = ax.dsl.cast(idx, dtype=ax.int32)
    for cache, t in ((kc, k), (vc, v)):
        ax.dsl.select_scatter(ax.dsl.cast(t.reshape(rows, HKV * D), dtype=ax.fp16), idx, dim=0,
                              r=cache.reshape(LMAX, HKV * D))
    first = idx[0:1]
    kb = ax.dsl.index_select(kc.reshape(LMAX, HKV * D), first, dim=0)
    vb = ax.dsl.index_select(vc.reshape(LMAX, HKV * D), first, dim=0)
    back = ax.dsl.add(ax.dsl.cast(kb[:, 0:1], dtype=ax.fp32), ax.dsl.cast(vb[:, 0:1], dtype=ax.fp32),
                      output_dtype=ax.fp32)
    return ax.dsl.mul(back.reshape(1), ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=0.0), output_dtype=ax.fp32)


def attention(hk, qh, kc, vc, nblk, mask, attn, j0):
    """the G' query heads qh [n, D] of KV head hk over the cache, into attn rows j0 .. j0 + n"""
    n = qh.shape[0]
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix=T(f"range_start{j0}"))
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix=T(f"range_step{j0}"))
    range_stop = ax.dsl.load(nblk)
    oi = ax.dsl.zeros(shape=(n, D), dtype=ax.bf16)
    li = ax.dsl.zeros(shape=(n,), dtype=ax.fp32)
    mi = ax.dsl.full(shape=(n, 1), dtype=ax.bf16, fill_value=-3e38)
    zero_col = ax.dsl.zeros(shape=(n, 1), dtype=ax.bf16)
    index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

    def body(index, kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1):
        loop_mi, loop_li, loop_oi = mi, li, oi
        kj, vj = kv_block(kc, index, hk), kv_block(vc, index, hk)
        mj = mask_block(mask, index, 1)
        m_s = ax.dsl.create_tensor(shape=(n, 1 + BC + 1), dtype=ax.bf16)
        ax.dsl.move(r=m_s[:, :1], x=mi)
        ax.dsl.move(r=m_s[:, -1:], x=zero_col)
        sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
        ax.dsl.add(sj, mj, output_dtype=ax.bf16, r=m_s[:, 1 : 1 + BC])
        m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
        m_s = ax.dsl.sub(m_s, m_next, r=m_s)
        pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
        p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
        lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
        ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(n)
        ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
        ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
        oij = pj @ vj
        ratio_oi = ax.dsl.mul(ratio.reshape(n, 1), oi, output_dtype=ax.fp32)
        ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
        ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
        return (kc, vc, mask, qh, loop_mi, loop_li, loop_oi, block_stride, kv_block_shape, mask_block_shape, zero_col,
                minus_1)

    (kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1) = ax.dsl.range(
        index, range_start, range_stop, range_step, body,
        kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1,
    )
    ax.dsl.div(oi, li.reshape(n, 1), output_dtype=ax.bf16, r=attn[:, j0 : j0 + n, :].reshape(n, D))


DEC_SPLIT = os.environ.get("DEC_SPLIT", "0") == "1"


def attention_split(c, qs, kc, vc, nblk, mask, attn, tag=""):
    """decode attention of the core's scaled queries qs [QH, D] (+ 0 after its own KV write) into attn [1, QH, D]: the
    cache blocks split over the cores (block c, c + CORES, ...), all query heads per block, partials merged"""
    q_all = shm_tensor((CORES, QH, D), ax.bf16, name=tag + "q_all")
    ax.dsl.move(r=q_all[c : c + 1], x=qs.reshape(1, QH, D))
    axd.pub(q_all)
    axd.sub(q_all)
    qa = ax.dsl.create_tensor((HQ, D), ax.bf16, ax.MemoryType.OCM, name=tag + "q_heads")  # global head order
    ax.dsl.move(r=qa, x=q_all.reshape(HQ, D))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([c], dtype=np.int32), prefix=f"{tag}split_start")
    range_step = ax.dsl.from_numpy(np.array([CORES], dtype=np.int32), prefix=f"{tag}split_step")
    range_stop = ax.dsl.load(nblk)
    oi = ax.dsl.zeros(shape=(HQ, D), dtype=ax.bf16)
    li = ax.dsl.zeros(shape=(HQ,), dtype=ax.fp32)
    mi = ax.dsl.full(shape=(HQ, 1), dtype=ax.bf16, fill_value=-3e38)
    zero_col = ax.dsl.zeros(shape=(HQ, 1), dtype=ax.bf16)
    index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

    def body(index, kc, vc, mask, qa, mi, li, oi, zero_col, minus_1):
        loop_mi, loop_li, loop_oi = mi, li, oi
        kb = ax.dsl.cast(ax.dsl.index_select(kc.reshape(NBLK, BC, HKV * D), index, dim=0).reshape(BC, HKV * D),
                         dtype=ax.bf16)
        vb = ax.dsl.cast(ax.dsl.index_select(vc.reshape(NBLK, BC, HKV * D), index, dim=0).reshape(BC, HKV * D),
                         dtype=ax.bf16)
        mj = mask_block(mask, index, 1)
        m_s = ax.dsl.create_tensor(shape=(HQ, 1 + BC + 1), dtype=ax.bf16)
        ax.dsl.move(r=m_s[:, :1], x=mi)
        ax.dsl.move(r=m_s[:, -1:], x=zero_col)
        for hk in range(HKV):
            sj = ax.dsl.matmul(qa[hk * G : (hk + 1) * G, :], kb[:, hk * D : (hk + 1) * D], transposed=True,
                               output_dtype=ax.bf16)
            ax.dsl.add(sj, mj, output_dtype=ax.bf16, r=m_s[hk * G : (hk + 1) * G, 1 : 1 + BC])
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
            ax.dsl.matmul(pj[hk * G : (hk + 1) * G, :], vb[:, hk * D : (hk + 1) * D], output_dtype=ax.fp32,
                          r=oij[hk * G : (hk + 1) * G, :])
        ratio_oi = ax.dsl.mul(ratio.reshape(HQ, 1), oi, output_dtype=ax.fp32)
        ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
        ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
        return (kc, vc, mask, qa, loop_mi, loop_li, loop_oi, zero_col, minus_1)

    (kc, vc, mask, qa, mi, li, oi, zero_col, minus_1) = ax.dsl.range(
        index, range_start, range_stop, range_step, body, kc, vc, mask, qa, mi, li, oi, zero_col, minus_1)
    # the cores' partials [m | l | o] (fp32) through shared OCM; each core merges those of its own query heads
    part = shm_tensor((CORES, HQ, 2 + D), ax.fp32, name=tag + "attn_part")
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


def attn_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
              w_down, out, moe=None, xt=None):
    c = ax.program_id(0)
    SHM_LIVE.clear()
    xt = ax.dsl.load(x) if xt is None else xt
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, q.slice_w(w_q, rows=slice(c * QH * QW, (c + 1) * QH * QW)), T("attn_q")).reshape(1, QH, QW)
    qh = rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS)
    qh = rope(qh, cos_t, sin_t, rotary_dim=ROT)
    hk0, hk1 = (c * QH) // G, ((c + 1) * QH - 1) // G  # the KV heads of the core's query heads
    # the new token's K / V: every core projects all KV heads and writes the whole row (the same code on every core:
    # the same bytes), then reads only after its own write (kv_write's zero scales the queries). A core reading K / V
    # rows another core wrote in the same call may read them before they reach DDR (pub / sub does not order the DMA
    # writes): no core does
    k = rmsnorm(q.proj(h, w_k, T("attn_k")).reshape(1, HKV, D), ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
    k = rope(k, cos_t, sin_t, rotary_dim=ROT)
    v = q.proj(h, w_v, T("attn_v")).reshape(1, HKV, D)
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    zero = kv_write(kc, vc, k, v, p, 1)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, ax.MemoryType.OCM, name=T("attn"))
    if DEC_SPLIT:  # the blocks over the cores, all query heads (every core wrote the whole row: its block reads see it)
        qs = ax.dsl.mul(qh.reshape(QH, D), scale, output_dtype=ax.bf16)
        qs = ax.dsl.add(qs, ax.dsl.cast(zero.reshape(1, 1), dtype=ax.bf16), output_dtype=ax.bf16)  # + 0, after the store
        attention_split(c, qs, kc, vc, nblk, mask, attn, T(""))
    j = 0
    for hk in range(hk0, (hk1 + 1) if not DEC_SPLIT else hk0):  # the core's query heads of KV head hk
        j1 = min(QH, (hk + 1) * G - c * QH)
        qs = ax.dsl.mul(qh[:, j:j1, :].reshape(j1 - j, D), scale, output_dtype=ax.bf16)
        qs = ax.dsl.add(qs, ax.dsl.cast(zero.reshape(1, 1), dtype=ax.bf16), output_dtype=ax.bf16)  # + 0, after the store
        attention(hk, qs, kc, vc, nblk, mask, attn, j)
        j = j1
    gated = (sigmoid_mul(attn, qg[:, :, D : 2 * D], activation_output_dtype=ax.fp32, output_dtype=ax.bf16) if q.GATE
             else attn).reshape(1, 1, QH * D)
    o = q.proj(gated, wcols(w_o, slice(c * QH * D, (c + 1) * QH * D)), T("attn_o"), output_dtype=ax.fp32)
    part_a = shm_tensor((CORES, 1, H), ax.fp32, name=T("part_a"))
    return finish(c, reduce(c, xt, part_a, o), ln2, w_gate, w_up, w_down, out, moe)


# ---------------------------------------------------------------- prefill (PCHUNK = P > 1 tokens per call)
# Every core needs the whole input of a projection whose output it splits by columns: the attention / delta-net
# output, x1 and the MLP activations pass through shared OCM, each core writing its slice; then each core computes
# its H / CORES output columns over the whole input (the single-core kernel's arithmetic per element) and stores them.
P = int(os.environ.get("PCHUNK", 1))
HC = H // CORES  # output columns per core
CHUNK = min(P, int(os.environ.get("CHUNK", 32)))
# PBLK=<rows>: a chunk of more rows runs in halves down to PBLK rows inside the one call (the layer body per block of
# rows, the second half continuing from the first's state, conv rows and K / V, as q35_prefill's REF_HALF): a larger
# chunk per call (fewer calls per layer) whose blocks each fit the OCM as a PBLK-row chunk does; each block's
# arithmetic is the PBLK-row kernel's
PBLK = int(os.environ.get("PBLK", 0))
POS_OFF = [0]  # the block's first position after pos


def gather(c, t, cols, width, name):
    """[1, P, width] bf16 in shared OCM: every core's t into its columns, published, then all of them"""
    sh = shm_tensor((1, P, width), ax.bf16, name=name)
    ax.dsl.move(r=sh[:, :, cols], x=t.reshape(1, P, cols.stop - cols.start))
    axd.pub(sh)
    axd.sub(sh)
    return sh


def finish_pre(c, x1_c, ln2, w_gate, w_up, w_down, out, moe):
    """x1's columns of this core -> x1 (shared), then the MLP (or the MoE's dense part) by output columns"""
    hs = slice(c * HC, (c + 1) * HC)
    x1 = gather(c, x1_c, hs, H, "x1_all")
    # the earlier shared tensors (gated_all / attn_all, x1_all) read once more after x1_all's pub / sub, when every core
    # has read them: their space may then go to act_all, not before (all kept to the end do not fit: 27B 7.2 of 8 MB)
    h2 = keep_shm(rmsnorm(x1, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS))
    fs = slice(c * FC, (c + 1) * FC)
    if moe is None:
        act = silu_mul(q.proj(h2, q.slice_w(w_gate, rows=fs), "mlp_gate"),
                       q.proj(h2, q.slice_w(w_up, rows=fs), "mlp_up"), activation_output_dtype=ax.fp32,
                       output_dtype=ax.bf16)
        act_all = gather(c, act, fs, F, "act_all")
        d = q.proj(act_all, q.slice_w(w_down, rows=hs), "mlp_down")  # [1, P, HC] bf16
        ax.dsl.store(keep_shm(ax.dsl.add(x1[:, :, hs], d, output_dtype=ax.bf16)), r=out[:, :, hs])
        return
    act = silu_mul(q.proj(h2, q.slice_w(w_gate, rows=fs), "sh_gate"), q.proj(h2, q.slice_w(w_up, rows=fs), "sh_up"),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
    act_all = gather(c, act, fs, F, "act_all")
    sh = q.proj(act_all, q.slice_w(w_down, rows=hs), "sh_down", output_dtype=ax.fp32)  # [1, P, HC]
    g = ax.dsl.sum(ax.dsl.mul(h2, ax.dsl.load(moe["w_shg"]).reshape(1, 1, H), output_dtype=ax.fp32), dim=-1,
                   keepdim=True, output_dtype=ax.fp32)
    base = ax.dsl.add(x1[:, :, hs], ax.dsl.mul(sh, ax.dsl.sigmoid(g, output_dtype=ax.fp32), output_dtype=ax.fp32),
                      output_dtype=ax.bf16)
    ax.dsl.store(base, r=out[:, :, hs])
    ax.dsl.store(h2[:, :, hs], r=moe["h2_out"][:, :, hs])
    ne = mo.NE // CORES
    rs = slice(c * ne, (c + 1) * ne)
    logits = linear(h2, moe["w_router"][rs], None, "router", output_dtype=ax.fp32, oc_tile_size=ne)
    ax.dsl.store(logits, r=moe["logits_out"][:, :, rs])
    if TK:  # the top-K experts of each token: all logits through shared OCM (no core reads another's DDR writes),
        # each core its P / CORES tokens' top-K: the raw logits (descending) and their experts
        K, tp = mo.TOPK, P // CORES
        lg = shm_tensor((P, mo.NE), ax.fp32, name="logits_all")
        ax.dsl.move(r=lg[:, rs], x=logits.reshape(P, ne))
        axd.pub(lg)
        axd.sub(lg)
        tr = slice(c * tp, (c + 1) * tp)
        vals = ax.dsl.create_tensor((tp, K), ax.fp32, name="topk_vals")
        ids = ax.dsl.create_tensor((tp, K), ax.int32, name="topk_ids")
        ax.dsl.topk(lg[tr], k=K, dim=-1, r_values=vals, r_indices=ids)
        ax.dsl.store(vals, r=TK["v"][tr])
        ax.dsl.store(ids, r=TK["i"][tr])


def gdn_pre_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                 state_in, conv_in, out, state_out, conv_out, moe=None):
    """the core's key heads and their value heads (as gdn_body) over the P tokens, the delta rule in the chunked form
    (q35_prefill.gdn_chunked, the core's KHC heads per solve)"""
    global P
    if PBLK and P > PBLK:
        full, P = P, P // 2
        try:
            w = (ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down)
            gdn_pre_body(x[:, :P], *w, state_in, conv_in, out[:, :P], state_out, conv_out, moe)
            gdn_pre_body(x[:, P:], *w, state_out, conv_out, out[:, P:], state_out, conv_out, moe)
        finally:
            P = full
        return
    assert NK % CORES == 0 and (REP == 1 or q.VORDER == "tiled") and P % CHUNK == 0
    c = ax.program_id(0)
    SHM_LIVE.clear()
    xt = ax.dsl.load(x)  # [1, P, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)

    def conv(off, n, i):
        ch = slice(off, off + n)
        pr = q.proj(h, q.slice_w(w_qkv, rows=ch), f"gdn_qkv{i}")
        return q.causal_conv(pr, conv_w[ch], conv_in[:, :, ch], conv_out[:, :, ch], CONV, f"gdn_conv{i}")

    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    qn = ax.dsl.mul(l2norm_lastdim(conv(c * KHC * DK, KHC * DK, 0).reshape(1, P, KHC, DK)), q_scale,
                    output_dtype=ax.bf16)
    kn = l2norm_lastdim(conv(KD + c * KHC * DK, KHC * DK, 1).reshape(1, P, KHC, DK))
    b_all = q.proj(h, w_b, "gdn_b")
    a_all = q.proj(h, w_a, "gdn_a")
    g_sh = shm_tensor((1, P, VD), ax.bf16, name="gated_all")
    for vo in range(REP):
        v0 = vo * NK + c * KHC
        vs = slice(v0, v0 + KHC)
        t = "" if REP == 1 else str(vo)
        vm = conv(2 * KD + v0 * DV, KHC * DV, 2 + vo).reshape(1, P, KHC, DV)
        z = q.proj(h, q.slice_w(w_z, rows=slice(v0 * DV, (v0 + KHC) * DV)), "gdn_z" + t).reshape(1, P, KHC, DV)
        g_all, beta_all = q.pad_gate(*gdn_gate(a_all[:, :, vs], b_all[:, :, vs], ax.dsl.load(a_log[vs]),
                                               ax.dsl.load(dt_bias[vs])))
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
        normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
        gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, P, KHC * DV)
        ax.dsl.move(r=g_sh[:, :, v0 * DV : (v0 + KHC) * DV], x=gated)
    axd.pub(g_sh)
    axd.sub(g_sh)
    hs = slice(c * HC, (c + 1) * HC)
    x1_c = ax.dsl.add(xt[:, :, hs], q.proj(g_sh, q.slice_w(w_out, rows=hs), "gdn_out"), output_dtype=ax.bf16)
    finish_pre(c, x1_c, ln2, w_gate, w_up, w_down, out, moe)


def attn_pre_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                  w_down, out, moe=None):
    global P
    if PBLK and P > PBLK:  # halves: the second's rows at pos + P / 2, its mask and rope rows the second half's
        full, P = P, P // 2
        off = POS_OFF[0]
        try:
            w = (w_o, ln2, w_gate, w_up, w_down)
            attn_pre_body(x[:, :P], ln1, w_q, w_k, w_v, q_norm, k_norm, cos[:P], sin[:P], kc, vc, pos, nblk,
                          mask[:, :P], *w, out[:, :P], moe)
            POS_OFF[0] = off + P
            attn_pre_body(x[:, P:], ln1, w_q, w_k, w_v, q_norm, k_norm, cos[P:], sin[P:], kc, vc, pos, nblk,
                          mask[:, P:], *w, out[:, P:], moe)
        finally:
            P, POS_OFF[0] = full, off
        return
    c = ax.program_id(0)
    SHM_LIVE.clear()
    xt = ax.dsl.load(x)  # [1, P, H]
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)  # [P, ROT/2]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, q.slice_w(w_q, rows=slice(c * QH * QW, (c + 1) * QH * QW)), "attn_q").reshape(P, QH, QW)
    qh = rope(rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS), cos_t, sin_t, rotary_dim=ROT)
    hk0, hk1 = (c * QH) // G, ((c + 1) * QH - 1) // G
    # all KV heads on every core (a block of some heads of P rows is not one contiguous run to write)
    k = rmsnorm(q.proj(h, w_k, "attn_k").reshape(P, HKV, D), ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
    k = rope(k, cos_t, sin_t, rotary_dim=ROT)
    v = q.proj(h, w_v, "attn_v").reshape(P, HKV, D)
    # the chunk's K / V rows: every core writes all of them (each computes all KV heads alike: the same bits) and
    # reads only what it wrote itself (rows another core wrote in this call may not have reached DDR yet)
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    if POS_OFF[0]:
        p = ax.dsl.add(p, ax.dsl.from_numpy(np.array([POS_OFF[0]], dtype=np.uint32)), output_dtype=ax.uint32)
    zero = kv_write(kc, vc, k, v, p, P)
    # + 0 read back from this core's rows: the attention after its own store
    scale = ax.dsl.add(ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D)), zero, output_dtype=ax.fp32)
    attn = ax.dsl.create_tensor((P, QH, D), ax.bf16, ax.MemoryType.OCM, name="attn")
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_stop = ax.dsl.load(nblk)
    j = 0
    for hk in range(hk0, hk1 + 1):  # the core's query heads of KV head hk: R = n * P rows, row g * P + t
        j1 = min(QH, (hk + 1) * G - c * QH)
        n, R = j1 - j, (j1 - j) * P
        qs = ax.dsl.create_tensor((R, D), ax.bf16, name=f"qs{hk}")
        for g in range(n):
            ax.dsl.mul(qh[:, j + g : j + g + 1, :].reshape(P, D), scale, output_dtype=ax.bf16,
                       r=qs[g * P : (g + 1) * P, :])
        oi = ax.dsl.zeros(shape=(R, D), dtype=ax.bf16)
        li = ax.dsl.zeros(shape=(R,), dtype=ax.fp32)
        mi = ax.dsl.full(shape=(R, 1), dtype=ax.bf16, fill_value=-3e38)
        zero_col = ax.dsl.zeros(shape=(R, 1), dtype=ax.bf16)
        index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

        def body(index, kc, vc, mask, qs, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col,
                 minus_1):
            loop_mi, loop_li, loop_oi = mi, li, oi
            kj, vj = kv_block(kc, index, hk), kv_block(vc, index, hk)
            mj = mask_block(mask, index, P)
            m_s = ax.dsl.create_tensor(shape=(R, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qs, kj, transposed=True, output_dtype=ax.bf16)
            for g in range(n):
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
            return (kc, vc, mask, qs, loop_mi, loop_li, loop_oi, block_stride, kv_block_shape, mask_block_shape,
                    zero_col, minus_1)

        (kc, vc, mask, qs, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1) = \
            ax.dsl.range(index, range_start, range_stop, range_step, body,
                         kc, vc, mask, qs, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1)
        o = ax.dsl.div(oi, li.reshape(R, 1), output_dtype=ax.bf16)
        for g in range(n):
            ax.dsl.move(x=o[g * P : (g + 1) * P, :].reshape(P, 1, D), r=attn[:, j + g : j + g + 1, :])
        j = j1
    gated = (sigmoid_mul(attn, qg[:, :, D : 2 * D], activation_output_dtype=ax.fp32, output_dtype=ax.bf16) if q.GATE
             else attn).reshape(1, P, QH * D)
    a_all = gather(c, gated, slice(c * QH * D, (c + 1) * QH * D), HQ * D, "attn_all")
    hs = slice(c * HC, (c + 1) * HC)
    x1_c = ax.dsl.add(xt[:, :, hs], q.proj(a_all, q.slice_w(w_o, rows=hs), "attn_o"), output_dtype=ax.bf16)
    finish_pre(c, x1_c, ln2, w_gate, w_up, w_down, out, moe)


if P > 1:  # the kernels' bodies: the prefill ones (the decode wrappers below call gdn_body / attn_body by name)
    gdn_body, attn_body = gdn_pre_body, attn_pre_body


# one function per IO set (the builder takes a kernel's inputs from its parameters, its outputs from its return)
def gdn_layer(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
              state_in, conv_in, out, state_out, conv_out):
    gdn_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down, state_in,
             conv_in, out, state_out, conv_out)
    return out, state_out, conv_out


def gdn_layer_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out, s_out,
                 ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, state_in, conv_in, out, state_out, conv_out):
    mb.SCALES.clear()
    for w, s in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate), (w_up, s_up),
                 (w_down, s_down)]:
        mb.SCALES[id(w)] = s
    gdn_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down, state_in,
             conv_in, out, state_out, conv_out)
    return out, state_out, conv_out


def attn_layer(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
               w_down, out):
    attn_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up, w_down,
              out)
    return out, kc, vc


def attn_layer_mx(x, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, s_o,
                  ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, out):
    mb.SCALES.clear()
    for w, s in [(w_q, s_q), (w_k, s_k), (w_v, s_v), (w_o, s_o), (w_gate, s_gate), (w_up, s_up), (w_down, s_down)]:
        mb.SCALES[id(w)] = s
    attn_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up, w_down,
              out)
    return out, kc, vc


if MOE:
    def gdn_layer_moe(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                      w_router, w_shg, state_in, conv_in, out, h2_out, logits_out, state_out, conv_out):
        gdn_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                 state_in, conv_in, out, state_out, conv_out,
                 dict(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out))
        return out, h2_out, logits_out, state_out, conv_out

    def attn_layer_moe(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                       w_down, w_router, w_shg, out, h2_out, logits_out):
        attn_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                  w_down, out, dict(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out))
        return out, kc, vc, h2_out, logits_out

    def gdn_layer_moe_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out,
                         s_out, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, state_in, conv_in,
                         out, h2_out, logits_out, state_out, conv_out):
        mb.SCALES.clear()
        for w, sc in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate),
                      (w_up, s_up), (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        gdn_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                 state_in, conv_in, out, state_out, conv_out,
                 dict(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out))
        return out, h2_out, logits_out, state_out, conv_out

    def attn_layer_moe_mx(x, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o,
                          s_o, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, out, h2_out,
                          logits_out):
        mb.SCALES.clear()
        for w, sc in [(w_q, s_q), (w_k, s_k), (w_v, s_v), (w_o, s_o), (w_gate, s_gate), (w_up, s_up),
                      (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        attn_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                  w_down, out, dict(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out))
        return out, kc, vc, h2_out, logits_out

    if NV4:  # the NVFP4 inputs: t_<n> (scale2) beside each s_<n>
        gdn_layer_moe_mx, attn_layer_moe_mx = mb.nv_twin(gdn_layer_moe_mx), mb.nv_twin(attn_layer_moe_mx)

    def _experts_src(k):
        """q35_moe.moe_experts_gu (each expert's gate and up rows in one weight) on the cores: core c takes experts
        c * k / CORES .., their weighted fp32 sum into shared part[c]; pub / sub; core 0 adds base and stores out"""
        ins = ", ".join(f"w_c{i}, s_c{i}, w_d{i}, s_d{i}" for i in range(k))
        return f"""
def moe_experts_gu(h2, base, rw, {ins}, out):
    c = ax.program_id(0)
    h = ax.dsl.load(h2)  # [1, 1, H] bf16
    r = ax.dsl.load(rw)  # [1, K] f32
    acc = None
    ws = [{", ".join(f"(w_c{i}, s_c{i}, w_d{i}, s_d{i})" for i in range(k))}]
    for i, (wc, sc, wd, sd) in enumerate(ws):
        if i // ({k} // CORES) != c:
            continue
        gu = linear(h, wc, sc, f"e{{i}}_gate_up", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
        a = silu_mul(gu[:, :, :mo.FE], gu[:, :, mo.FE:], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = linear(a, wd, sd, f"e{{i}}_down", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(mo.FE))
        y = ax.dsl.mul(d, r[:, i : i + 1].reshape(1, 1, 1), output_dtype=ax.fp32)
        acc = y if acc is None else ax.dsl.add(acc, y, output_dtype=ax.fp32)
    part = shm_tensor((CORES, 1, H), ax.fp32, name="part_e")
    ax.dsl.move(r=part[c : c + 1], x=acc.reshape(1, 1, H))
    axd.pub(part)
    axd.sub(part)
    if c == 0:
        tot = ax.dsl.cast(ax.dsl.load(base), dtype=ax.fp32)
        for j in range(CORES):
            tot = ax.dsl.add(tot, part[j : j + 1], output_dtype=ax.fp32)
        ax.dsl.store(ax.dsl.cast(tot, dtype=ax.bf16), r=out)
    return out
"""

    _fn = "<moe_experts_gu_npu>"
    _src = _experts_src(mo.TOPK)
    import linecache
    linecache.cache[_fn] = (len(_src), None, _src.splitlines(True), _fn)
    exec(compile(_src, _fn, "exec"), globals())


# the output projection (decode): the GEMV of ggml/gguf_gemv_kernels.py, LM_C / CORES weight rows per core
LM_C = int(os.environ.get("LM_C", 31040))  # rows per call (the vocabulary in calls of LM_C)


def mul_mat_bf16(x, w, out):
    c = ax.program_id(0)
    rows = slice(c * (LM_C // CORES), (c + 1) * (LM_C // CORES))
    o = linear(ax.dsl.load(x), w[rows], None, "mm", output_dtype=ax.fp32, oc_tile_size=q.oc_tile(w.shape[1]))
    ax.dsl.store(o, r=out[:, rows])
    return out


def mul_mat_mx8(x, w, s, out):
    c = ax.program_id(0)
    rows = slice(c * (LM_C // CORES), (c + 1) * (LM_C // CORES))
    o = linear(ax.dsl.load(x), w[rows], s[rows], "mm", output_dtype=ax.fp32, oc_tile_size=256)
    ax.dsl.store(o, r=out[:, rows])
    return out


if MOE and P == 1:
    def experts_x(h2, base, rw, ws):
        """layer i's routed experts on the cores (q35_moe.moe_experts_gu: K / CORES experts each, fp32 partials summed
        through shared OCM): layer i + 1's input x = base + their weighted sum, on chip on every core"""
        c = ax.program_id(0)
        h, r = ax.dsl.load(h2), ax.dsl.load(rw)
        acc = None
        for i, (wc, sc, wd, sd) in enumerate(ws):
            if i // (len(ws) // CORES) != c:
                continue
            gu = linear(h, wc, sc, f"e{i}_gate_up", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
            a = silu_mul(gu[:, :, : mo.FE], gu[:, :, mo.FE :], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
            d = linear(a, wd, sd, f"e{i}_down", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(mo.FE))
            y = ax.dsl.mul(d, r[:, i : i + 1].reshape(1, 1, 1), output_dtype=ax.fp32)
            acc = y if acc is None else ax.dsl.add(acc, y, output_dtype=ax.fp32)
        part = shm_tensor((CORES, 1, H), ax.fp32, name="part_e")
        ax.dsl.move(r=part[c : c + 1], x=acc.reshape(1, 1, H))
        axd.pub(part)
        axd.sub(part)
        tot = ax.dsl.cast(ax.dsl.load(base), dtype=ax.fp32)
        for j in range(CORES):
            tot = ax.dsl.add(tot, part[j : j + 1], output_dtype=ax.fp32)
        return ax.dsl.cast(tot, dtype=ax.bf16)

    def _merged_src(kind):
        """layer i's experts and layer i + 1's body (kind gdn / attn) in one kernel: the experts kernel's inputs, then
        the layer kernel's without x"""
        layer = (gdn_layer_moe_mx if MX else gdn_layer_moe) if kind == "gdn" else (
            attn_layer_moe_mx if MX else attn_layer_moe)
        lin = [n for n in order(layer) if n not in ("x",) and not n.endswith("_out") and n not in ("out", "kc", "vc")]
        louts = {"gdn": ["out", "h2_out", "logits_out", "state_out", "conv_out"],
                 "attn": ["out", "kc", "vc", "h2_out", "logits_out"]}[kind]
        lpar = [n for n in order(layer) if n != "x"]  # the layer's parameters in order (outputs included)
        ex = [f"w_c{i}, s_c{i}, w_d{i}, s_d{i}" for i in range(mo.TOPK)]
        ws = ", ".join(f"(w_c{i}, s_c{i}, w_d{i}, s_d{i})" for i in range(mo.TOPK))
        call = ", ".join(lpar)
        body = "gdn_layer_moe" if kind == "gdn" else "attn_layer_moe"
        return f"""
def experts_{kind}(h2, base, rw, {", ".join(ex)}, {call}):
    S.merged_body("{kind}", h2, base, rw, [{ws}], dict({", ".join(f"{n}={n}" for n in lpar)}))
    return {", ".join(louts)}
"""

    def merged_body(kind, h2, base, rw, ws, a):
        if MX:
            mb.SCALES.clear()
            for w in (mb.GDN_W if kind == "gdn" else mb.ATTN_W):
                mb.SCALES[id(a[w])] = a[mb.scale_name(w)]
        xt = experts_x(h2, base, rw, ws)
        moe = dict(w_router=a["w_router"], w_shg=a["w_shg"], h2_out=a["h2_out"], logits_out=a["logits_out"])
        if kind == "gdn":
            gdn_body(None, a["ln1"], a["w_qkv"], a["w_z"], a["w_b"], a["w_a"], a["conv_w"], a["a_log"], a["dt_bias"],
                     a["norm_w"], a["w_out"], a["ln2"], a["w_gate"], a["w_up"], a["w_down"], a["state_in"],
                     a["conv_in"], a["out"], a["state_out"], a["conv_out"], moe, xt=xt)
        else:
            attn_body(None, a["ln1"], a["w_q"], a["w_k"], a["w_v"], a["q_norm"], a["k_norm"], a["cos"], a["sin"],
                      a["kc"], a["vc"], a["pos"], a["nblk"], a["mask"], a["w_o"], a["ln2"], a["w_gate"], a["w_up"],
                      a["w_down"], a["out"], moe, xt=xt)

    def merged_kernel(kind):
        """the merged kernel as a function of a module in BUILD (the builder reads its source)"""
        import importlib
        src = "import q35_layer_mc as S\n" + _merged_src(kind)
        os.makedirs(BUILD, exist_ok=True)
        open(f"{BUILD}/merged_{kind}.py", "w").write(src)
        sys.path.insert(0, BUILD)
        return getattr(importlib.import_module(f"merged_{kind}"), f"experts_{kind}")


if MOE and P == 1:
    def moe_full(c, x1, ln2, w_gate, w_up, w_down, out, w_router, w_shg, e_gw, e_gs, e_dw, e_ds):
        """the MoE after the layer's attention / delta-net half, all on the NPU: the shared expert (F / CORES channels
        per core), the router logits (NE / CORES per core) and the shared-expert partials meet in shared OCM; every
        core takes softmax and top-K of the logits (the same arithmetic on each), then its K / CORES experts, picked out
        of the layer's whole expert tensors (e_gw / e_gs gate-up [NE, 2 FE, H] MXFP4, e_dw / e_ds down [NE, H, FE]) by
        index_select; the experts' partials meet again, core 0 stores out = base + their weighted sum"""
        K, NE, FE, ne = mo.TOPK, mo.NE, mo.FE, mo.NE // CORES
        fs, rs = slice(c * FC, (c + 1) * FC), slice(c * ne, (c + 1) * ne)
        h2 = rmsnorm(x1, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
        act = silu_mul(q.proj(h2, q.slice_w(w_gate, rows=fs), T("sh_gate")), q.proj(h2, q.slice_w(w_up, rows=fs), T("sh_up")),
                       activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = q.proj(act, wcols(w_down, fs), T("sh_down"), output_dtype=ax.fp32)
        part = shm_tensor((CORES, 1, H), ax.fp32, name=T("part_m"))
        ax.dsl.move(r=part[c : c + 1], x=d.reshape(1, 1, H))
        lg = shm_tensor((1, NE), ax.fp32, name=T("logits_all"))
        ax.dsl.move(r=lg[:, rs], x=linear(h2, w_router[rs], None, T("router"), output_dtype=ax.fp32,
                                          oc_tile_size=ne).reshape(1, ne))
        axd.pub(part)
        axd.pub(lg)
        axd.sub(part)
        axd.sub(lg)
        sh = part[0:1]
        for j in range(1, CORES):
            sh = ax.dsl.add(sh, part[j : j + 1], output_dtype=ax.fp32)
        g = ax.dsl.sum(ax.dsl.mul(h2, ax.dsl.load(w_shg).reshape(1, 1, H), output_dtype=ax.fp32), dim=-1, keepdim=True,
                       output_dtype=ax.fp32)
        base = ax.dsl.add(x1, ax.dsl.mul(sh, ax.dsl.sigmoid(g, output_dtype=ax.fp32), output_dtype=ax.fp32),
                          output_dtype=ax.bf16)  # as the layer kernel's out: bf16
        probs = ax.dsl.softmax(lg, dim=-1, output_dtype=ax.fp32)
        vals = ax.dsl.create_tensor((1, K), ax.fp32, name=T("topk_vals"))
        ids = ax.dsl.create_tensor((1, K), ax.int32, name=T("topk_ids"))
        ax.dsl.topk(probs, k=K, dim=-1, r_values=vals, r_indices=ids)
        rw = ax.dsl.div(vals, ax.dsl.sum(vals, dim=-1, keepdim=True, output_dtype=ax.fp32), output_dtype=ax.fp32)
        h = h2
        acc = None
        for i in range(c * (K // CORES), (c + 1) * (K // CORES)):
            e = ids[:, i : i + 1].reshape(1)
            wc = ax.dsl.index_select(e_gw, e, dim=0).reshape(2 * FE, H)
            sc = ax.dsl.index_select(e_gs, e, dim=0).reshape(2 * FE, H // 32)
            wd = ax.dsl.index_select(e_dw, e, dim=0).reshape(H, FE)
            sd = ax.dsl.index_select(e_ds, e, dim=0).reshape(H, FE // 32)
            gu = linear(h, wc, sc, T(f"e{i}_gate_up"), output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
            a = silu_mul(gu[:, :, :FE], gu[:, :, FE:], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
            y = linear(a, wd, sd, T(f"e{i}_down"), output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(FE))
            y = ax.dsl.mul(y, rw[:, i : i + 1].reshape(1, 1, 1), output_dtype=ax.fp32)
            acc = y if acc is None else ax.dsl.add(acc, y, output_dtype=ax.fp32)
        pe = shm_tensor((CORES, 1, H), ax.fp32, name=T("part_e"))
        ax.dsl.move(r=pe[c : c + 1], x=acc.reshape(1, 1, H))
        axd.pub(pe)
        axd.sub(pe)
        tot = ax.dsl.cast(base, dtype=ax.fp32)
        for j in range(CORES):
            tot = ax.dsl.add(tot, pe[j : j + 1], output_dtype=ax.fp32)
        x2 = ax.dsl.cast(tot, dtype=ax.bf16)  # on every core: the next layer's input
        if c == 0 and out is not None:
            ax.dsl.store(x2, r=out)
        return x2

    def gdn_layer_full_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out,
                          s_out, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, e_gw, e_gs, e_dw,
                          e_ds, state_in, conv_in, out, state_out, conv_out):
        mb.SCALES.clear()
        for w, sc in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate),
                      (w_up, s_up), (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        gdn_body(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                 state_in, conv_in, out, state_out, conv_out,
                 dict(w_router=w_router, w_shg=w_shg, e_gw=e_gw, e_gs=e_gs, e_dw=e_dw, e_ds=e_ds))
        return out, state_out, conv_out

    def attn_layer_full_mx(x, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o,
                           s_o, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, e_gw, e_gs, e_dw,
                           e_ds, out):
        mb.SCALES.clear()
        for w, sc in [(w_q, s_q), (w_k, s_k), (w_v, s_v), (w_o, s_o), (w_gate, s_gate), (w_up, s_up),
                      (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        attn_body(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                  w_down, out, dict(w_router=w_router, w_shg=w_shg, e_gw=e_gw, e_gs=e_gs, e_dw=e_dw, e_ds=e_ds))
        return out, kc, vc

    if NV4:
        gdn_layer_full_mx, attn_layer_full_mx = mb.nv_twin(gdn_layer_full_mx), mb.nv_twin(attn_layer_full_mx)


def order(fn):
    return list(fn.__code__.co_varnames[: fn.__code__.co_argcount])


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def run(k, cores, case, names):
    ax.init(chip_num=1, core_num=cores)
    if MX:
        mb.SCALES.clear()
    k[cores](*[case[n] for n in names])


def refs():
    """the single-core kernels the four-core ones are checked against (and must match in IO): gdn, attn"""
    if P > 1:
        if MOE:
            import q35_moe_prefill as mp
            return (mp.gdn_layer_prefill_moe_mx if MX else mp.gdn_layer_prefill_moe), mp.attn_layer_prefill_moe
        import q35_prefill as qp
        return ((qp.gdn_layer_prefill_mx, qp.attn_layer_prefill_mx) if MX else
                (qp.gdn_layer_prefill, qp.attn_layer_prefill))
    if MOE:
        return (mo.gdn_layer_moe_mx, mo.attn_layer_moe_mx) if MX else (mo.gdn_layer_moe, mo.attn_layer_moe)
    return (mb.gdn_layer_mx, mb.attn_layer_mx) if MX else (q.gdn_layer, q.attn_layer)


TK_FN = {}


def with_topk(fn):
    """fn (a MoE prefill layer kernel) with the router's top-K outputs topk_v [P, K] f32 / topk_i [P, K] int32 after its
    own (finish_pre writes them through TK); same name"""
    import importlib
    import inspect
    names_ = order(fn)
    ret = [ln.strip() for ln in inspect.getsource(fn).splitlines() if ln.strip().startswith("return ")][-1]
    TK_FN[fn.__name__] = fn
    src = (f"import q35_layer_mc as S\n\n\ndef {fn.__name__}({', '.join(names_ + ['topk_v', 'topk_i'])}):\n"
           f"    S.TK.update(v=topk_v, i=topk_i)\n    S.TK_FN[{fn.__name__!r}]({', '.join(names_)})\n    S.TK.clear()\n"
           f"    {ret}, topk_v, topk_i\n")
    mod = f"tkvar_{fn.__name__}"
    os.makedirs(BUILD, exist_ok=True)
    open(f"{BUILD}/{mod}.py", "w").write(src)
    if BUILD not in sys.path:
        sys.path.insert(0, BUILD)
    importlib.invalidate_caches()
    return getattr(importlib.import_module(mod), fn.__name__)


def mine():
    """this script's kernels: gdn, attn (MoE prefill attention in bf16, as q35_moe_prefill)"""
    if MOE and P > 1 and TOPK_OUT:
        if "tk" not in TK_FN:
            TK_FN["tk"] = (with_topk(gdn_layer_moe_mx if MX else gdn_layer_moe), with_topk(attn_layer_moe))
        return TK_FN["tk"]
    if MOE:
        return (gdn_layer_moe_mx if MX else gdn_layer_moe), (attn_layer_moe_mx if MX and P == 1 else attn_layer_moe)
    return (gdn_layer_mx, attn_layer_mx) if MX else (gdn_layer, attn_layer)


def names(kind):
    kp = "qwen35moe" if MOE else q.KP
    pre = f"_prefill{P}" if P > 1 else ""
    tk = "_tk" if MOE and P > 1 and TOPK_OUT else ""
    if kind == "gdn":
        return f"{kp}_gdn{pre}{SFX}{tk}"
    return f"{kp}_attn{pre}{'' if MOE and P > 1 else SFX}{tk}_kvc_c{LMAX}"


def tokens(W, t0):
    emb = W("embed_tokens.weight")
    return emb[1000 + 7 * t0 : 1000 + 7 * t0 + 7 * P : 7].reshape(1, P, H).contiguous()


def rope_rows(t0):
    cs, sn = zip(*(q.rope_cos_sin(t0 + t) for t in range(P)))
    return torch.cat(cs), torch.cat(sn)  # [P, ROT/2]


def mask_rows(t0):
    col = torch.arange(LMAX).reshape(LMAX // BC, 1, BC)
    lim = (t0 + torch.arange(P)).reshape(1, P, 1)
    m = torch.where(col <= lim, 0.0, -3e38).bfloat16()
    return m if P > 1 else m.reshape(LMAX // BC, BC)


def moe_outs():
    if not MOE:
        return {}
    o = dict(h2_out=torch.zeros((1, P, H), dtype=torch.bfloat16), logits_out=torch.zeros((1, P, mo.NE)))
    if P > 1 and TOPK_OUT:
        o.update(topk_v=torch.zeros((P, mo.TOPK)), topk_i=torch.zeros((P, mo.TOPK), dtype=torch.int32))
    return o


def check_gdn(W, out_dir):
    single, mc = refs()[0], mine()[0]
    names_ = order(mc) if NOREF else order(single)  # (NOREF: the four-core kernel may add outputs, e.g. top-K)
    assert NOREF or names_ == order(mc), (names_, order(mc))
    a = q.gdn_args(W, 0)
    a = mx4_args(a, mb.GDN_W) if MX4 else mb.mx_args(a, mb.GDN_W)[0] if MX else a
    k1, k4 = kern(single, "gdn1"), kern(mc, "gdn4")
    st = {r: torch.zeros((NV, DK, DV)) for r in ("1", "4")}
    cv = {r: torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16) for r in ("1", "4")}
    worst = 1.0
    for step in range(STEPS):
        x = tokens(W, step * P)
        res = {}
        for r, k, n in (("1", k1, 1), ("4", k4, CORES))[1 if NOREF else 0 :]:
            case = dict(a, x=x, state_in=st[r], conv_in=cv[r], out=torch.zeros((1, P, H), dtype=torch.bfloat16),
                        state_out=torch.zeros_like(st[r]), conv_out=torch.zeros_like(cv[r]), **moe_outs())
            os.chdir(f"{BUILD}/gdn{r}")
            run(k, n, case, names_)
            st[r], cv[r], res[r] = case["state_out"], case["conv_out"], case
        if NOREF:  # no single-core kernel of this shape (35B GDN prefill P=256): finite outputs only
            fin = bool(res["4"]["out"].float().isfinite().all())
            logger.info(f"{names('gdn')} npu{CORES} step {step}: no reference, out finite {fin}")
            worst = min(worst, 1.0 if fin else 0.0)
            continue
        co, cs = cos(res["4"]["out"], res["1"]["out"]), cos(st["4"], st["1"])
        if MOE:
            co = min(co, *(cos(res["4"][n], res["1"][n]) for n in ("h2_out", "logits_out")))
        neq = int((cv["4"].view(torch.int16) != cv["1"].view(torch.int16)).sum())
        same = bool(torch.equal(res["4"]["out"].view(torch.int16), res["1"]["out"].view(torch.int16)))
        logger.info(f"{names('gdn')} npu{CORES} step {step}: out cos {co:.6f} (bitwise {same}), state cos {cs:.6f}, "
                    f"conv elements differing {neq}")
        worst = min(worst, co, cs)
    path = max(glob.glob(f"{BUILD}/gdn4/cache/{mc.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, names("gdn"), CORES)
    shutil.copy(path, dst)
    board_io.fix(dst)  # FP4 inputs (NVFP4) as bytes: the board runtime misreads FLOAT4E2M1 ones
    logger.info(f"wrote {dst} (worst cos {worst:.6f})")


def check_attn(W, out_dir):
    single, mc = refs()[1], mine()[1]
    names_ = order(mc) if NOREF else order(single)
    assert NOREF or names_ == order(mc), (names_, order(mc))
    li = q.LAYERS.index("full_attention")
    a = q.attn_args(W, li)
    a = mb.mx_args(a, mb.ATTN_W)[0] if MX and not (MOE and P > 1) else a
    k1, k4 = kern(single, "attn1"), kern(mc, "attn4")
    kv = {r: [torch.zeros((LMAX, HKV * D), dtype=torch.half) for _ in range(2)] for r in ("1", "4")}
    worst = 1.0
    for step in range(STEPS):
        t0 = step * P
        x = tokens(W, t0)
        cs, sn = rope_rows(t0)
        res = {}
        for r, k, n in (("1", k1, 1), ("4", k4, CORES))[1 if NOREF else 0 :]:
            kc, vc = kv[r]
            case = dict(a, x=x, cos=cs, sin=sn, kc=kc, vc=vc, pos=torch.tensor([t0], dtype=torch.int32),
                        nblk=torch.tensor([(t0 + P - 1) // BC + 1], dtype=torch.int32), mask=mask_rows(t0),
                        out=torch.zeros((1, P, H), dtype=torch.bfloat16), **moe_outs())
            q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
            os.chdir(f"{BUILD}/attn{r}")
            run(k, n, case, names_)
            q.RESIDENT.clear()
            res[r] = case
        if SAVE:  # this step's four-core outputs and the KV cache after it (other chunk sizes compare by token)
            torch.save({n: res["4"][n].clone() for n in ("out", *moe_outs())} | dict(kc=kv["4"][0].clone(),
                       vc=kv["4"][1].clone(), t0=t0), f"{SAVE}_{step}.pt")
        if NOREF:  # no single-core kernel of this shape (35B attention prefill P=128 at c16384): finite outputs only
            fin = bool(res["4"]["out"].float().isfinite().all())
            logger.info(f"{names('attn')} npu{CORES} step {step}: no reference, out finite {fin}")
            worst = min(worst, 1.0 if fin else 0.0)
            continue
        co = cos(res["4"]["out"], res["1"]["out"])
        if MOE:
            co = min(co, *(cos(res["4"][n], res["1"][n]) for n in ("h2_out", "logits_out")))
        rows = slice(t0, t0 + P)
        ck, cv = cos(kv["4"][0][rows], kv["1"][0][rows]), cos(kv["4"][1][rows], kv["1"][1][rows])
        same = bool(torch.equal(res["4"]["out"].view(torch.int16), res["1"]["out"].view(torch.int16)))
        logger.info(f"{names('attn')} npu{CORES} step {step}: out cos {co:.6f} (bitwise {same}), K cos {ck:.6f}, "
                    f"V cos {cv:.6f}")
        worst = min(worst, co, ck, cv)
    path = max(glob.glob(f"{BUILD}/attn4/cache/{mc.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, names("attn"), CORES)
    shutil.copy(path, dst)
    board_io.fix(dst)  # FP4 inputs (NVFP4) as bytes: the board runtime misreads FLOAT4E2M1 ones
    logger.info(f"wrote {dst} (worst cos {worst:.6f})")


def check_experts(out_dir):
    """the experts kernel against q35_moe's single-core one on random MXFP4 experts"""
    torch.manual_seed(0)
    K, FE = mo.TOPK, mo.FE
    case = dict(h2=torch.randn((1, 1, H)).bfloat16(), base=(torch.randn((1, 1, H)) * 0.5).bfloat16(),
                rw=torch.softmax(torch.randn((1, K)), -1).float())
    for i in range(K):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        case[f"w_c{i}"], case[f"s_c{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        case[f"w_d{i}"], case[f"s_d{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
    names = order(mo.moe_experts_gu)
    assert names == order(moe_experts_gu)
    res = {}
    for r, fn, n in (("1", mo.moe_experts_gu, 1), ("4", moe_experts_gu, CORES)):
        case["out"] = torch.zeros((1, 1, H), dtype=torch.bfloat16)
        run(kern(fn, f"experts{r}"), n, case, names)
        res[r] = case["out"].clone()
    logger.info(f"experts{K}_gu npu{CORES}: out cos vs single-core {cos(res['4'], res['1']):.6f}")
    path = max(glob.glob(f"{BUILD}/experts4/cache/moe_experts_gu*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, f"qwen35moe_experts{K}_gu_mxfp4", CORES)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst}")


def check_lm(out_dir):
    torch.manual_seed(0)
    x, w = torch.randn((1, H), dtype=torch.bfloat16), (torch.randn((LM_C, H)) * H**-0.5).bfloat16()
    out = torch.zeros((1, LM_C))
    if MX:
        w8, s8 = mb.quant_mxfp8(w)
        k, args, name = kern(mul_mat_mx8, "lm"), (x, w8, s8, out), f"mul_mat_mx8_n{LM_C}_k{H}"
        ref = x.float() @ mb.dequant(w8, s8).T
    else:
        k, args, name = kern(mul_mat_bf16, "lm"), (x, w, out), f"mul_mat_bf16_n{LM_C}_k{H}"
        ref = x.float() @ w.float().T
    ax.init(chip_num=1, core_num=CORES)
    k[CORES](*args)
    logger.info(f"{name} npu{CORES}: cos vs torch {cos(out, ref):.6f}")
    path = max(glob.glob(f"{BUILD}/lm/cache/mul_mat_*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, name, CORES)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst}")


def check_merged(W, out_dir, kind):
    """experts + layer in one call against the experts kernel then the layer kernel (both four-core), same inputs"""
    torch.manual_seed(0)
    K, FE = mo.TOPK, mo.FE
    ex = dict(h2=torch.randn((1, 1, H)).bfloat16(), base=(torch.randn((1, 1, H)) * 0.5).bfloat16(),
              rw=torch.softmax(torch.randn((1, K)), -1).float())
    for i in range(K):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        ex[f"w_c{i}"], ex[f"s_c{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        ex[f"w_d{i}"], ex[f"s_d{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
    if kind == "gdn":
        a = q.gdn_args(W, 0)
        a = mb.mx_args(a, mb.GDN_W)[0] if MX else a
        layer = gdn_layer_moe_mx if MX else gdn_layer_moe
        io = lambda: dict(state_in=torch.zeros((NV, DK, DV)),  # noqa: E731
                          conv_in=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16),
                          state_out=torch.zeros((NV, DK, DV)),
                          conv_out=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16))
    else:
        a = q.attn_args(W, q.LAYERS.index("full_attention"))
        a = mb.mx_args(a, mb.ATTN_W)[0] if MX else a
        layer = attn_layer_moe_mx if MX else attn_layer_moe
        cs, sn = q.rope_cos_sin(0)
        io = lambda: dict(cos=cs, sin=sn, kc=torch.zeros((LMAX, HKV * D), dtype=torch.half),  # noqa: E731
                          vc=torch.zeros((LMAX, HKV * D), dtype=torch.half), pos=torch.tensor([0], dtype=torch.int32),
                          nblk=torch.tensor([1], dtype=torch.int32), mask=mask_rows(0))
    outs = lambda: dict(out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **moe_outs())  # noqa: E731
    # reference: the experts kernel, then the layer kernel on its output
    e = dict(ex, out=torch.zeros((1, 1, H), dtype=torch.bfloat16))
    run(kern(moe_experts_gu, "mx_experts"), CORES, e, order(moe_experts_gu))
    ref = dict(a, x=e["out"], **io(), **outs())
    if kind == "attn":
        q.RESIDENT.update(kc=ref["kc"].clone(), vc=ref["vc"].clone())
    run(kern(layer, f"mx_{kind}"), CORES, ref, order(layer))
    q.RESIDENT.clear()
    fn = merged_kernel(kind)
    got = dict(ex, **a, **io(), **outs())
    if kind == "attn":
        q.RESIDENT.update(kc=got["kc"].clone(), vc=got["vc"].clone())
    run(kern(fn, f"merged_{kind}"), CORES, got, order(fn))
    q.RESIDENT.clear()
    names_ = ["out", "h2_out", "logits_out"] + (["state_out"] if kind == "gdn" else ["kc"])
    cs_ = {n: cos(got[n], ref[n]) for n in names_}
    logger.info(f"experts + {kind} npu{CORES} vs the two kernels: " + ", ".join(f"{n} {v:.6f}" for n, v in cs_.items()))
    path = max(glob.glob(f"{BUILD}/merged_{kind}/cache/experts_{kind}.*.compiled.axmodel"), key=os.path.getmtime)
    name = f"qwen35moe_experts{K}_gu_gdn{SFX}" if kind == "gdn" else f"qwen35moe_experts{K}_gu_attn{SFX}_kvc_c{LMAX}"
    dst = board_io.kernel_file(out_dir, name, CORES)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst}")


def check_full(W, out_dir, kind):
    """the whole MoE layer in one call (routing and experts on the NPU) against the layer kernel, the host's routing
    (q35_moe.route) and the experts kernel, on the real layer weights and random MXFP4 experts (all NE of them)"""
    torch.manual_seed(0)
    K, FE, NE = mo.TOPK, mo.FE, mo.NE
    gw, gs, dw, ds = [], [], [], []
    for e in range(NE):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk), gs.append(sc)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk), ds.append(sc)
    ex = dict(e_gw=torch.stack(gw).view(torch.float4_e2m1fn_x2), e_gs=torch.stack(gs).view(torch.float8_e8m0fnu),
              e_dw=torch.stack(dw).view(torch.float4_e2m1fn_x2), e_ds=torch.stack(ds).view(torch.float8_e8m0fnu))
    x = tokens(W, 0)
    if kind == "gdn":
        a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0]
        io = lambda: dict(state_in=torch.zeros((NV, DK, DV)),  # noqa: E731
                          conv_in=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16),
                          state_out=torch.zeros((NV, DK, DV)),
                          conv_out=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16))
        layer, full = gdn_layer_moe_mx, gdn_layer_full_mx
    else:
        a = mb.mx_args(q.attn_args(W, q.LAYERS.index("full_attention")), mb.ATTN_W)[0]
        cs, sn = q.rope_cos_sin(0)
        io = lambda: dict(cos=cs, sin=sn, kc=torch.zeros((LMAX, HKV * D), dtype=torch.half),  # noqa: E731
                          vc=torch.zeros((LMAX, HKV * D), dtype=torch.half), pos=torch.tensor([0], dtype=torch.int32),
                          nblk=torch.tensor([1], dtype=torch.int32), mask=mask_rows(0))
        layer, full = attn_layer_moe_mx, attn_layer_full_mx
    ref = dict(a, x=x, out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **io(), **moe_outs())
    run(kern(layer, f"fl_{kind}"), CORES, ref, order(layer))
    ids, rw = mo.route(ref["logits_out"])
    e = dict(h2=ref["h2_out"], base=ref["out"], rw=rw.float(), out=torch.zeros((1, 1, H), dtype=torch.bfloat16))
    for i, k in enumerate(ids):
        e[f"w_c{i}"], e[f"s_c{i}"] = ex["e_gw"][k].contiguous(), ex["e_gs"][k].contiguous()
        e[f"w_d{i}"], e[f"s_d{i}"] = ex["e_dw"][k].contiguous(), ex["e_ds"][k].contiguous()
    run(kern(moe_experts_gu, "fl_experts"), CORES, e, order(moe_experts_gu))
    got = dict(a, **ex, x=x, out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **io())
    run(kern(full, f"full_{kind}"), CORES, got, order(full))
    logger.info(f"{kind} layer + MoE in one call npu{CORES}: experts {sorted(ids)}, out cos vs layer + host routing "
                f"+ experts kernel {cos(got['out'], e['out']):.6f}")
    path = max(glob.glob(f"{BUILD}/full_{kind}/cache/{full.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    name = f"qwen35moe_layer_gdn{SFX}" if kind == "gdn" else f"qwen35moe_layer_attn{SFX}_kvc_c{LMAX}"
    dst = board_io.kernel_file(out_dir, name, CORES)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst}: {board_io.fix(dst)[:3]}")


# ---------------------------------------------------------------- packed whole MoE layers (the layout of line A's
# q35_layer_pk.py: one input per dtype, members at ALIGN-byte offsets in the order of the kernel's inputs, a .layout
# file next to the kernel for the backend; the expert planes stay inputs of their own)
PK_ALIGN = 64
PK_PACKS = ({"pw": torch.float4_e2m1fn_x2, "ps": torch.float8_e4m3fn, "pf": torch.float32, "pb": torch.bfloat16} if NV4 else
            {"pw": torch.float8_e4m3fn, "ps": torch.float8_e8m0fnu, "pf": torch.float32, "pb": torch.bfloat16})
PK_RAW = (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2)  # packed as bytes


def pk_lps(p):
    """logical elements per stored element of pack p: 2 for FP4 (the DSL addresses E2M1 pairs by element)"""
    return 2 if PK_PACKS[p] == torch.float4_e2m1fn_x2 else 1
PK_SPEC = {}


PK_NOBLK = [False]  # prefill packs: no column blocks (the prefill bodies slice their weights themselves)


def pk_blocks(n):
    """the column blocks (element ranges, in storage order: core-major) of a member split by columns over the cores"""
    if PK_NOBLK[0]:
        return None
    if n in ("w_out", "s_out"):  # GDN out projection: per core its REP strips of KHC value heads
        bl = [((vo * NK + c * KHC) * DV, (vo * NK + c * KHC + KHC) * DV) for c in range(CORES) for vo in range(REP)]
    elif n in ("w_o", "s_o"):  # attention o projection: the core's query heads
        bl = [(c * QH * D, (c + 1) * QH * D) for c in range(CORES)]
    elif n in ("w_down", "s_down"):  # MLP / shared expert down: the core's channels
        bl = [(c * FC, (c + 1) * FC) for c in range(CORES)]
    else:
        return None
    gs = mb.NVG if NV4 else 32  # scale blocks: one scale per group of input channels
    return bl if n.startswith("w_") else [(a // gs, b // gs) for a, b in bl]


def pk_layout(kind, a, names):
    spec, used = {}, {p: 0 for p in PK_PACKS}
    for n in names:
        t = a.get(n)
        if not torch.is_tensor(t) or n == "x" or n.startswith("e_"):
            continue
        p = next(k for k, dt in PK_PACKS.items() if dt == t.dtype)
        es = t.element_size()
        off = (used[p] * es + PK_ALIGN - 1) // PK_ALIGN * PK_ALIGN // es
        spec[n] = (p, off, tuple(t.shape), pk_blocks(n))
        used[p] = off + t.numel()
    PK_SPEC[kind] = spec
    return {p: (used[p] * torch.tensor([], dtype=dt).element_size() + PK_ALIGN - 1) // PK_ALIGN * PK_ALIGN //
            torch.tensor([], dtype=dt).element_size() for p, dt in PK_PACKS.items()}


def pk_packed(kind, a, sizes):
    """the packs of layer args a: each member at its offset (stored elements), column blocks one after the other"""
    pk = {p: torch.zeros(max(1, sizes[p]), dtype=torch.uint8 if dt in PK_RAW else dt) for p, dt in PK_PACKS.items()}
    for n, (p, off, shape, bl) in PK_SPEC[kind].items():
        t, lps = a[n], pk_lps(p)
        if bl:  # the column blocks one after the other, each [rows, columns] contiguous (FP4: columns in pairs)
            t = torch.cat([t[:, x // lps : y // lps].reshape(-1) for x, y in bl])
        t = t.reshape(-1)
        if t.dtype in PK_RAW:
            t = t.view(torch.uint8)
        pk[p][off : off + t.numel()] = t
    return {p: (pk[p].view(dt) if dt in PK_RAW else pk[p]) for p, dt in PK_PACKS.items()}


def pk_member(src, p, off, shape, bl):
    """a member of a pack (src: pack name -> kernel input) in the DSL's logical elements: a tensor, or the column
    blocks of a column-split weight as Cols"""
    lps = pk_lps(p)
    if bl:
        rows, blocks, o = shape[0], {}, off
        for x, y in bl:
            n = rows * (y - x) // lps
            blocks[(x, y)] = src[p][o * lps : (o + n) * lps].reshape(rows, y - x)
            o += n
        return Cols(blocks)
    numel = 1
    for d in shape:
        numel *= d
    return src[p][off * lps : (off + numel) * lps].reshape(*shape[:-1], shape[-1] * lps)


def pk_register(m):
    """the scales of the unpacked weights of m (name -> member): s_<n> for w_<n>, NVFP4 also t_<n> (scale2)"""
    mb.SCALES.clear()
    mb.SCALES2.clear()
    gs = mb.NVG if NV4 else 32
    for n in list(m):
        if not (n.startswith("w_") and "s_" + n[2:] in m):
            continue
        w, sc, s2 = m[n], m["s_" + n[2:]], m.get("t_" + n[2:])
        for blk, bs in ([(b, sc.blocks[(x // gs, y // gs)]) for (x, y), b in w.blocks.items()] if isinstance(w, Cols)
                        else [(w, sc)]):
            mb.SCALES[id(blk)] = bs
            if s2 is not None:
                mb.SCALES2[id(blk)] = s2


def pk_unpack(kind, pw, ps, pf, pb):
    src = dict(pw=pw, ps=ps, pf=pf, pb=pb)
    m = {n: pk_member(src, p, off, shape, bl) for n, (p, off, shape, bl) in PK_SPEC[kind].items()}
    pk_register(m)
    return m


def pk_write_layout(kind, dst, sizes):
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in PK_PACKS.items()}
    with open(dst[: -len(".axmodel")] + ".layout", "w") as f:
        for p in PK_PACKS:
            if any(v[0] == p for v in PK_SPEC[kind].values()):
                f.write(f"{p} - 0 {sizes[p] * es[p]}\n")
        for n, (p, off, shape, bl) in PK_SPEC[kind].items():
            numel = 1
            for d in shape:
                numel *= d
            # column blocks: rows, bytes per source row, then each block's byte range in a row (stored block by block)
            lps = pk_lps(p)
            extra = "" if not bl else f" {shape[0]} {shape[1] * es[p]} " + ",".join(
                f"{x // lps * es[p]}:{y // lps * es[p]}" for x, y in bl)
            f.write(f"{n} {p} {off * es[p]} {numel * es[p]}{extra}\n")


if MOE and P == 1:
    def gdn_layer_full_pk(x, pw, ps, pf, pb, e_gw, e_gs, e_dw, e_ds, state_in, conv_in, out, state_out, conv_out):
        m = pk_unpack("gdn", pw, ps, pf, pb)
        gdn_body(x, m["ln1"], m["w_qkv"], m["w_z"], m["w_b"], m["w_a"], m["conv_w"], m["a_log"], m["dt_bias"],
                 m["norm_w"], m["w_out"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], state_in, conv_in, out,
                 state_out, conv_out, dict(w_router=m["w_router"], w_shg=m["w_shg"], e_gw=e_gw, e_gs=e_gs, e_dw=e_dw,
                                           e_ds=e_ds))
        return out, state_out, conv_out

    def attn_layer_full_pk(x, pw, ps, pf, pb, e_gw, e_gs, e_dw, e_ds, cos, sin, kc, vc, pos, nblk, mask, out):
        m = pk_unpack("attn", pw, ps, pf, pb)
        attn_body(x, m["ln1"], m["w_q"], m["w_k"], m["w_v"], m["q_norm"], m["k_norm"], cos, sin, kc, vc, pos, nblk,
                  mask, m["w_o"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], out,
                  dict(w_router=m["w_router"], w_shg=m["w_shg"], e_gw=e_gw, e_gs=e_gs, e_dw=e_dw, e_ds=e_ds))
        return out, kc, vc


def check_full_pk(W, out_dir, kind):
    """the packed whole layer against the unpacked one (same inputs, random MXFP4 experts)"""
    torch.manual_seed(0)
    FE, NE = mo.FE, mo.NE
    gw, gs, dw, ds = [], [], [], []
    for e in range(NE):
        pk_, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk_), gs.append(sc)
        pk_, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk_), ds.append(sc)
    ex = dict(e_gw=torch.stack(gw).view(torch.float4_e2m1fn_x2), e_gs=torch.stack(gs).view(torch.float8_e8m0fnu),
              e_dw=torch.stack(dw).view(torch.float4_e2m1fn_x2), e_ds=torch.stack(ds).view(torch.float8_e8m0fnu))
    x = tokens(W, 0)
    if kind == "gdn":
        a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0]
        io = lambda: dict(state_in=torch.zeros((NV, DK, DV)),  # noqa: E731
                          conv_in=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16),
                          state_out=torch.zeros((NV, DK, DV)),
                          conv_out=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16))
        full, fpk = gdn_layer_full_mx, gdn_layer_full_pk
    else:
        a = mb.mx_args(q.attn_args(W, q.LAYERS.index("full_attention")), mb.ATTN_W)[0]
        cs, sn = q.rope_cos_sin(0)
        io = lambda: dict(cos=cs, sin=sn, kc=torch.zeros((LMAX, HKV * D), dtype=torch.half),  # noqa: E731
                          vc=torch.zeros((LMAX, HKV * D), dtype=torch.half), pos=torch.tensor([0], dtype=torch.int32),
                          nblk=torch.tensor([1], dtype=torch.int32), mask=mask_rows(0))
        full, fpk = attn_layer_full_mx, attn_layer_full_pk
    a["w_shg"] = a["w_shg"].bfloat16()
    sizes = pk_layout(kind, a, order(full))
    ref = dict(a, **ex, x=x, out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **io())
    run(kern(full, f"fu_{kind}"), CORES, ref, order(full))
    got = dict(pk_packed(kind, a, sizes), **ex, x=x, out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **io())
    run(kern(fpk, f"fpk_{kind}"), CORES, got, order(fpk))
    same = bool(torch.equal(got["out"].view(torch.int16), ref["out"].view(torch.int16)))
    logger.info(f"{kind} whole layer packed npu{CORES}: out cos vs unpacked {cos(got['out'], ref['out']):.6f} "
                f"(bitwise {same}), packs {sizes}")
    path = max(glob.glob(f"{BUILD}/fpk_{kind}/cache/{fpk.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    name = f"qwen35moe_layer_gdn{SFX}_pkc" if kind == "gdn" else f"qwen35moe_layer_attn{SFX}_pkc_kvc_c{LMAX}"
    dst = board_io.kernel_file(out_dir, name, CORES)
    shutil.copy(path, dst)
    board_io.fix(dst)
    pk_write_layout(kind, dst, sizes)
    logger.info(f"wrote {dst} and its layout")


PK_FN = {}  # packed prefill kernels: the unpacked kernel each wraps


def pk_variant(fn, key, a):
    """a packed version of kernel fn: fn's inputs that a holds (the layer's weights) replaced by the packs pw / ps / pf /
    pb (q35_layer_mc's layout `key`), the others (activations, state, KV, rope, mask) and the outputs kept by name"""
    import importlib
    import inspect
    names = order(fn)
    rest = [n for n in names if not torch.is_tensor(a.get(n))]
    ins = [n for n in rest if n not in ("out", "state_out", "conv_out", "h2_out", "logits_out", "topk_v", "topk_i")]
    outs = [n for n in rest if n not in ins]
    ret = [ln.strip() for ln in inspect.getsource(fn).splitlines() if ln.strip().startswith("return ")][-1]
    PK_FN[key] = fn
    call = ", ".join(f"{n}=m[{n!r}]" if torch.is_tensor(a.get(n)) else f"{n}={n}" for n in names)
    src = (f"import q35_layer_mc as S\n\n\ndef {fn.__name__}_pk({', '.join(ins + ['pw', 'ps', 'pf', 'pb'] + outs)}):\n"
           f"    m = S.pk_unpack({key!r}, pw, ps, pf, pb)\n    S.PK_FN[{key!r}]({call})\n    {ret}\n")
    mod = f"pkvar_{key}"
    os.makedirs(BUILD, exist_ok=True)
    open(f"{BUILD}/{mod}.py", "w").write(src)
    if BUILD not in sys.path:
        sys.path.insert(0, BUILD)
    importlib.invalidate_caches()
    return getattr(importlib.import_module(mod), f"{fn.__name__}_pk")


def check_pre_pk(W, out_dir, kind):
    """the packed prefill kernel (four cores, this PCHUNK) against the unpacked one (mine()): the same inputs, the
    layer's weights in the packs (no column blocks); written as <name>_pk with its .layout"""
    torch.manual_seed(0)
    mc = mine()[0 if kind == "gdn" else 1]
    if kind == "gdn":
        a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0] if MX else q.gdn_args(W, 0)
        io = dict(state_in=torch.randn((NV, DK, DV)) * 0.1,
                  conv_in=(torch.randn((1, CONV - 1, 2 * KD + VD)) * 0.1).bfloat16(), state_out=torch.zeros((NV, DK, DV)),
                  conv_out=torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16))
    else:
        li = q.LAYERS.index("full_attention")
        a = q.attn_args(W, li)
        a = mb.mx_args(a, mb.ATTN_W)[0] if MX and not (MOE and P > 1) else a
        cs, sn = rope_rows(0)
        io = dict(cos=cs, sin=sn, kc=torch.zeros((LMAX, HKV * D), dtype=torch.half),
                  vc=torch.zeros((LMAX, HKV * D), dtype=torch.half), pos=torch.tensor([0], dtype=torch.int32),
                  nblk=torch.tensor([(P - 1) // BC + 1], dtype=torch.int32), mask=mask_rows(0))
    if "w_shg" in a:
        a["w_shg"] = a["w_shg"].bfloat16()
    key = f"pre{kind}"
    PK_NOBLK[0] = True
    sizes = pk_layout(key, a, order(mc))
    fpk = pk_variant(mc, key, a)
    base = dict(x=tokens(W, 0), out=torch.zeros((1, P, H), dtype=torch.bfloat16), **moe_outs())
    ref = dict(a, **base, **{n: t.clone() for n, t in io.items()})
    got = dict(pk_packed(key, a, sizes), **{n: t.clone() for n, t in base.items()}, **{n: t.clone() for n, t in io.items()})
    for case, fn, sub in ((ref, mc, f"pre_{kind}"), (got, fpk, f"prepk_{kind}")):
        if kind == "attn":
            q.RESIDENT.update(kc=case["kc"].clone(), vc=case["vc"].clone())
        run(kern(fn, sub), CORES, case, order(fn))
        q.RESIDENT.clear()
    outs = [n for n in ("out", "h2_out", "logits_out", "state_out", "conv_out", "kc", "vc", "topk_v", "topk_i")
            if n in ref]
    if TOPK_OUT:  # the kernel's top-K against the host's rule on its own logits: descending, ties to the lower expert
        lg = ref["logits_out"].reshape(P, mo.NE)
        sv, si = torch.sort(lg, dim=-1, descending=True, stable=True)
        iv = bool(torch.equal(ref["topk_i"], si[:, : mo.TOPK].to(torch.int32)))
        vv = bool(torch.equal(ref["topk_v"].view(torch.int32), sv[:, : mo.TOPK].contiguous().view(torch.int32)))
        logger.info(f"{names(kind)} top-K vs the host rule on its logits: ids bitwise {iv}, values bitwise {vv}")
    same = {n: bool(torch.equal(got[n].contiguous().view(torch.uint8), ref[n].contiguous().view(torch.uint8))) for n in outs}
    logger.info(f"{names(kind)} packed npu{CORES}: bitwise vs unpacked {same}, packs {sizes}")
    path = max(glob.glob(f"{BUILD}/prepk_{kind}/cache/{fpk.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, names(kind) + "_pk", CORES)
    shutil.copy(path, dst)
    board_io.fix(dst)
    pk_write_layout(key, dst, sizes)
    PK_NOBLK[0] = False
    logger.info(f"wrote {dst} and its layout")


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(q.MODEL)
    only = os.environ.get("ONLY", "gdn,attn,experts" if MOE else "gdn,attn").split(",")
    for kind in ("gdn", "attn"):
        if f"x{kind}" in only and MOE and P == 1:
            check_merged(W, out_dir, kind)
        if f"full{kind}" in only and MOE and P == 1:
            check_full(W, out_dir, kind)
        if f"pk{kind}" in only and MOE and P == 1:
            check_full_pk(W, out_dir, kind)
    for kind in ("gdn", "attn"):
        if f"pkpre{kind}" in only and P > 1 and (kind == "attn" or q.HYBRID):
            check_pre_pk(W, out_dir, kind)
    if "lm" in only:
        check_lm(out_dir)
    if "experts" in only and MOE and P == 1:
        check_experts(out_dir)
    if "gdn" in only and q.HYBRID:
        check_gdn(W, out_dir)
    if "attn" in only:
        check_attn(W, out_dir)


if __name__ == "__main__":
    sys.modules["q35_layer_mc"] = sys.modules["__main__"]  # generated kernels import this module: the same PK_SPEC
    main()
