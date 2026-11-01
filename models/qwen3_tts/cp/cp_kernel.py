"""Qwen3-TTS code predictor: one frame, one kernel, runtime top-k.

The 16 positions run as a runtime AXDSL range (position 0 is one static seed
pass), rather than sixteen statically unrolled copies of the five-layer stack.
The codebook heads are loaded in dynamic row tiles and sampled on chip.  The
sampler always asks the hardware for 50 candidates; a runtime ``top_k`` masks
the tail and only swaps the first two candidates when ``top_k > 1``.

Reuse: q3tts_cp_step.py, q35_group.py, q35_layer_mc.py, q35_layer_pk.py and
q35_mx_build.py from the frozen 4acb7f6 reference are imported read-only.  The
layer arithmetic is therefore the same code that built cp.frame.k1/k50.
"""

from __future__ import annotations

import os
import sys

import ml_dtypes
import numpy as np

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl import util as axu
from axnn.dsl.kernels.utils import rmsnorm, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.linear import linear


REF_SRC = os.environ.get(
    "CP_REF_SRC", os.path.dirname(os.path.abspath(__file__))
)
if REF_SRC not in sys.path:
    sys.path.insert(0, REF_SRC)

os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("KVL", "cell")
os.environ.setdefault("LMAX", "16")
os.environ.setdefault("BC", "16")
os.environ.setdefault("KVX", "1")
os.environ.setdefault("DEC_SPLIT", "0")
os.environ.setdefault("LMHEAD", "0")
os.environ.setdefault("FIRST", "0")

import q3tts_cp_step as S  # noqa: E402
import q35_group as Qg  # noqa: E402
import q35_layer_mc as L  # noqa: E402
import q35_layer_pk as K  # noqa: E402
import q35_mx_build as mb  # noqa: E402


Qg_g, Lq, Kq, mbq, qq = S.Qg, S.L, S.K, S.mb, S.q
NL, H, HT, V, CORES, PROJ = S.NL, S.H, S.HT, S.V, S.CORES, S.PROJ
EPS = S.EPS
NP = 16
TOPK_MAX = 50
NB = Lq.LMAX // Lq.BC
HEAD_TILE = int(os.environ.get("HEAD_TILE", "64"))
STATIC_PYRUN = os.environ.get(
    "CP_STATIC_PYRUN",
    "1" if os.environ.get("RUN_MODE", "").upper() == "PYRUN" else "0",
) == "1"


def attn_layer_ocm(c, xt, m, cs, pn, mask, kc, vc, tag):
    """q35_group.attn_layer for OCM-selected cs / pn / mask.

    The frozen function calls ``ax.dsl.load`` on those operands, which only
    accepts DDR.  Here the runtime position loop has already selected them into
    OCM, so pass them through directly.  Every other operation is identical.
    """
    QH, QW, D, G_, HKV, ROT = Lq.QH, Lq.QW, Lq.D, Lq.G, Lq.HKV, Lq.ROT
    if cs.memory_type == ax.MemoryType.DDR:
        cos_t, sin_t = ax.dsl.load(cs[0:1]), ax.dsl.load(cs[1:2])
    else:
        cos_t, sin_t = cs[0:1], cs[1:2]
    h = rmsnorm(xt, ax.dsl.load(m["ln1"]).reshape(1, 1, H), eps=EPS)
    qg = qq.proj(h, qq.slice_w(m["w_q"], rows=slice(c * QH * QW, (c + 1) * QH * QW)), tag + "attn_q").reshape(1, QH, QW)
    qh = rmsnorm(qg[:, :, 0:D], ax.dsl.load(m["q_norm"]).reshape(1, 1, D), eps=EPS)
    qh = axnn_rope(qh, cos_t, sin_t, rotary_dim=ROT)
    hk0, hk1 = (c * QH) // G_, ((c + 1) * QH - 1) // G_
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / np.sqrt(D))
    attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, ax.MemoryType.OCM, name=tag + "attn")
    pn0 = ax.dsl.load(pn[0:1]) if pn.memory_type == ax.MemoryType.DDR else pn[0:1]
    p = ax.dsl.cast(pn0, dtype=ax.uint32)
    HC, QD = HKV // CORES, QH * D
    ks = slice(c * HC * D, (c + 1) * HC * D)
    kc_ = rmsnorm(
        qq.proj(h, qq.slice_w(m["w_k"], rows=ks), tag + "attn_k").reshape(1, HC, D),
        ax.dsl.load(m["k_norm"]).reshape(1, 1, D),
        eps=EPS,
    )
    kc_ = axnn_rope(kc_, cos_t, sin_t, rotary_dim=ROT)
    vc_ = qq.proj(h, qq.slice_w(m["w_v"], rows=ks), tag + "attn_v").reshape(1, HC, D)
    qs = ax.dsl.mul(qh.reshape(QH, D), scale, output_dtype=ax.bf16)
    W_ = QD if Lq.DEC_SPLIT else 0
    ex = Lq.shm_tensor((CORES, W_ + 2 * HC * D), ax.bf16, name=tag + "qkv_all")
    o0 = QD if Lq.DEC_SPLIT else 0
    if Lq.DEC_SPLIT:
        ax.dsl.move(r=ex[c : c + 1, 0:QD], x=qs.reshape(1, QD))
    ax.dsl.move(r=ex[c : c + 1, o0 : o0 + HC * D], x=kc_.reshape(1, HC * D))
    ax.dsl.move(r=ex[c : c + 1, o0 + HC * D : o0 + 2 * HC * D], x=vc_.reshape(1, HC * D))
    axd.pub(ex)
    axd.sub(ex)
    k = ax.dsl.create_tensor((CORES, HC * D), ax.bf16, ax.MemoryType.OCM, name=tag + "k_row")
    v = ax.dsl.create_tensor((CORES, HC * D), ax.bf16, ax.MemoryType.OCM, name=tag + "v_row")
    ax.dsl.move(r=k, x=ex[:, o0 : o0 + HC * D])
    ax.dsl.move(r=v, x=ex[:, o0 + HC * D : o0 + 2 * HC * D])
    zero, kc, vc = kv_write_update(kc, vc, k.reshape(1, HKV, D), v.reshape(1, HKV, D), p)
    zero = zero.reshape(1, 1)
    if Lq.DEC_SPLIT:
        qa = ax.dsl.create_tensor((CORES, QD), ax.bf16, ax.MemoryType.OCM, name=tag + "q_heads")
        ax.dsl.move(r=qa, x=ex[:, 0:QD])
        qa = qa.reshape(Lq.HQ, D)
        qa = ax.dsl.add(qa, ax.dsl.cast(zero, dtype=ax.bf16), output_dtype=ax.bf16)
        Qg.attention_split_qa(c, qa, kc, vc, pn[1:2], mask, attn, tag)
    j = 0
    for hk in range(hk0, (hk1 + 1) if not Lq.DEC_SPLIT else hk0):
        j1 = min(QH, (hk + 1) * G_ - c * QH)
        qs = ax.dsl.mul(qh[:, j:j1, :].reshape(j1 - j, D), scale, output_dtype=ax.bf16)
        qs = ax.dsl.add(qs, ax.dsl.cast(zero, dtype=ax.bf16), output_dtype=ax.bf16)
        attention_ocm(hk, qs, kc, vc, pn[1:2], mask, attn, j, tag)
        j = j1
    gated = attn.reshape(1, 1, QH * D)
    o = qq.proj(gated, qq.slice_w(m["w_o"], cols=slice(c * QH * D, (c + 1) * QH * D)), tag + "attn_o", output_dtype=ax.fp32)
    return Qg.mlp(c, Qg.reduce(c, xt, o, tag + "a"), m, tag), kc, vc


def axnn_rope(x, cos, sin, *, rotary_dim):
    from axnn.dsl.kernels.utils.rope import rope

    return rope(x, cos, sin, rotary_dim=rotary_dim)


def kv_write_update(kc, vc, k, v, p):
    """Write one K / V row and return the updated loop-carried caches.

    The frozen ``q35_layer_mc.kv_write`` relies on the target view being
    mutated in place.  PYRUN's ``select_scatter`` instead returns a new array,
    so carrying that return value is required for the same graph to evaluate
    correctly in both PYRUN and CRUN.
    """
    idx = ax.dsl.cast(p[0:1], dtype=ax.int32)
    kc2 = ax.dsl.select_scatter(
        ax.dsl.cast(k.reshape(1, Lq.HKV * Lq.D), dtype=ax.fp16),
        idx,
        dim=0,
        r=kc.reshape(Lq.LMAX, Lq.HKV * Lq.D),
    ).reshape(Lq.LMAX, Lq.HKV * Lq.D)
    vc2 = ax.dsl.select_scatter(
        ax.dsl.cast(v.reshape(1, Lq.HKV * Lq.D), dtype=ax.fp16),
        idx,
        dim=0,
        r=vc.reshape(Lq.LMAX, Lq.HKV * Lq.D),
    ).reshape(Lq.LMAX, Lq.HKV * Lq.D)
    kb = ax.dsl.index_select(kc2, idx, dim=0)
    vb = ax.dsl.index_select(vc2, idx, dim=0)
    back = ax.dsl.add(
        ax.dsl.cast(kb[:, 0:1], dtype=ax.fp32),
        ax.dsl.cast(vb[:, 0:1], dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    zero = ax.dsl.mul(
        back.reshape(1),
        ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=0.0),
        output_dtype=ax.fp32,
    )
    return zero, kc2, vc2


def attention_ocm(hk, qh, kc, vc, nblk, mask, attn, j0, tag):
    """q35_layer_mc.attention with an OCM nblk (the frozen one calls load)."""
    n = qh.shape[0]
    BC, D = Lq.BC, Lq.D
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix=tag + f"range_start{j0}")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix=tag + f"range_step{j0}")
    range_stop = nblk
    oi = ax.dsl.zeros(shape=(n, D), dtype=ax.bf16)
    li = ax.dsl.zeros(shape=(n,), dtype=ax.fp32)
    mi = ax.dsl.full(shape=(n, 1), dtype=ax.bf16, fill_value=-3e38)
    zero_col = ax.dsl.zeros(shape=(n, 1), dtype=ax.bf16)
    index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

    def block(index, kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1):
        kj, vj = Lq.kv_block(kc, index, hk), Lq.kv_block(vc, index, hk)
        mj = Lq.mask_block(mask, index, 1)
        sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
        sj = ax.dsl.add(sj, mj, output_dtype=ax.bf16)
        m_s = ax.dsl.concat(mi, sj, zero_col, dim=-1)
        m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
        m_s = ax.dsl.sub(m_s, m_next)
        pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
        p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
        lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
        ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(n)
        ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
        loop_li = ax.dsl.add(ratio_li, lij, output_dtype=ax.fp32)
        oij = pj @ vj
        ratio_oi = ax.dsl.mul(ratio.reshape(n, 1), oi, output_dtype=ax.fp32)
        loop_oi = ax.dsl.add(ratio_oi, oij, output_dtype=ax.bf16)
        loop_mi = ax.dsl.mul(m_s[:, -1:], minus_1)
        return (kc, vc, mask, qh, loop_mi, loop_li, loop_oi, block_stride, kv_block_shape, mask_block_shape, zero_col,
                minus_1)

    # LMAX == BC == 16, so every frame position sees exactly one pending KV
    # block.  Keep this single block static in both PYRUN and CRUN: a nested
    # runtime Range is unnecessary and SGL PYRUN cannot carry 2-D Range state.
    index = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix=tag + "block0")
    (_, _, _, _, mi, li, oi, _, _, _, _, _) = block(
        index, kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape,
        mask_block_shape, zero_col, minus_1,
    )
    ax.dsl.div(oi, li.reshape(n, 1), output_dtype=ax.bf16, r=attn[:, j0 : j0 + n, :].reshape(n, D))


def _icmp(op, x, y):
    """Integer compare through the public AXDSL elementwise op."""
    return axu.op_convert(
        "elemwise_int",
        {"op": op, "output_dtype": ax.int32},
        {"x": x, "y": y},
        {"r": None},
    )


def _row_pack(pack, offset, shape):
    """Load one contiguous member from a packed input into OCM."""
    return ax.dsl.dynamic_ld(
        pack,
        ax.dsl.full((1,), ax.int32, int(offset)),
        ax.dsl.full((1,), ax.int32, int(np.prod(shape))),
        dim=(0,),
        output_shape=(int(np.prod(shape)),),
    ).reshape(*shape)


def _store_back(src, dst):
    """Materialise a loop-carried value into an output (load first if in DDR)."""
    if src.memory_type == ax.MemoryType.DDR:
        src = ax.dsl.load(src)
    ax.dsl.store(src, r=dst)


def project(c, xe, pm, tag):
    """Talker width -> predictor width (the 1.7B model); identity for 0.6B."""
    if not PROJ:
        return xe
    rs = slice(c * (H // CORES), (c + 1) * (H // CORES))
    pr = linear(
        xe,
        pm["w_pin"][rs],
        None,
        tag + "proj_in",
        output_dtype=ax.fp32,
        oc_tile_size=H // CORES,
    )
    pr = ax.dsl.add(
        pr,
        ax.dsl.load(pm["b_pin"][rs]).reshape(1, 1, H // CORES),
        output_dtype=ax.fp32,
    )
    sh = Lq.shm_tensor((1, 1, H), ax.bf16, tag + "proj_all")
    ax.dsl.move(r=sh[:, :, rs], x=ax.dsl.cast(pr, dtype=ax.bf16))
    axd.pub(sh)
    axd.sub(sh)
    xt = ax.dsl.create_tensor((1, 1, H), ax.bf16, ax.MemoryType.OCM, name=tag + "x_in")
    ax.dsl.move(r=xt, x=sh)
    return xt


def sample(lg, draw, top_k, tag):
    """top-k sampling with a runtime k in 1..50.

    The hardware top-k is fixed at 50.  Candidate positions ``j >= top_k`` are
    masked to -3e38 before softmax; for ``top_k == 1`` the first two candidates
    are *not* swapped, matching ggml's top_k for k == 1.  For ``top_k == 50``
    this is exactly the frozen k50 arithmetic: 50 candidates, first two swapped.
    """
    v, i = ax.dsl.topk(lg, k=TOPK_MAX, dim=-1)
    fi = ax.dsl.cast(i, dtype=ax.fp32)
    swap = _icmp("gt", top_k, ax.dsl.full((1,), ax.int32, 1))

    vs = ax.dsl.create_tensor((1, TOPK_MAX), ax.fp32, ax.MemoryType.OCM, name=tag + "vals")
    js = ax.dsl.create_tensor((1, TOPK_MAX), ax.fp32, ax.MemoryType.OCM, name=tag + "ids")
    # A runtime first-two swap without AxWhere (whose broadcast form produces
    # an invalid TENG cycle config in this toolchain).  The indices pick the
    # original fp32 candidates verbatim, so k=50 is bit-identical to the old
    # kernel and k=1 leaves the order untouched.
    pair = ax.dsl.concat(
        swap,
        ax.dsl.sub(ax.dsl.full((1,), ax.int32, 1), swap, output_dtype=ax.int32),
        dim=0,
    ).reshape(2)
    ax.dsl.move(r=vs[:, 0:2], x=ax.dsl.index_select(v, pair, dim=1).reshape(1, 2))
    ax.dsl.move(r=js[:, 0:2], x=ax.dsl.index_select(fi, pair, dim=1).reshape(1, 2))
    ax.dsl.move(r=vs[:, 2:TOPK_MAX], x=v[:, 2:TOPK_MAX])
    ax.dsl.move(r=js[:, 2:TOPK_MAX], x=fi[:, 2:TOPK_MAX])

    pos = ax.dsl.from_numpy(np.arange(TOPK_MAX, dtype=np.int32).reshape(1, TOPK_MAX))
    kk = ax.dsl.add(top_k.reshape(1, 1), ax.dsl.zeros((1, TOPK_MAX), ax.int32), output_dtype=ax.int32)
    valid = _icmp("lt", pos, kk)
    vf = ax.dsl.cast(valid, dtype=ax.fp32)
    addmask = ax.dsl.mul(
        ax.dsl.sub(vf, ax.dsl.full((1, 1), ax.fp32, 1.0), output_dtype=ax.fp32),
        ax.dsl.full((1, TOPK_MAX), ax.fp32, 3e38),
        output_dtype=ax.fp32,
    )
    vals = ax.dsl.add(vs, addmask, output_dtype=ax.fp32)

    p = ax.dsl.softmax(vals, dim=-1, output_dtype=ax.fp32)
    cdf = ax.dsl.cumsum(p, dim=-1, output_dtype=ax.fp32)
    cross = ax.dsl.cast(
        ax.dsl.ceil(ax.dsl.clip(ax.dsl.sub(cdf, draw, output_dtype=ax.fp32), min=0.0, max=1.0)),
        dtype=ax.fp32,
    )
    prev = ax.dsl.create_tensor((1, TOPK_MAX), ax.fp32, ax.MemoryType.OCM, name=tag + "prev")
    ax.dsl.move(r=prev[:, 0:1], x=ax.dsl.full((1, 1), ax.fp32, 0.0))
    ax.dsl.move(r=prev[:, 1:TOPK_MAX], x=cross[:, 0 : TOPK_MAX - 1])
    onehot = ax.dsl.sub(cross, prev, output_dtype=ax.fp32)
    idf = ax.dsl.sum(
        ax.dsl.mul(onehot, js, output_dtype=ax.fp32),
        dim=-1,
        keepdim=True,
        output_dtype=ax.fp32,
    )
    return ax.dsl.cast(idf, dtype=ax.int32)


def head_logits(c, hn, heads, p, tag):
    """The core's V/CORES head rows for position ``p``, as [1,V] through shared OCM."""
    local = ax.dsl.create_tensor((V // CORES,), ax.fp32, ax.MemoryType.OCM, name=tag + "lg_local")
    base = ax.dsl.add(
        ax.dsl.mul(ax.dsl.sub(p, ax.dsl.full((1,), ax.int32, 1)), ax.dsl.full((1,), ax.int32, V), output_dtype=ax.int32),
        ax.dsl.full((1,), ax.int32, c * (V // CORES)),
        output_dtype=ax.int32,
    )
    for off in range(0, V // CORES, HEAD_TILE):
        src = ax.dsl.add(base, ax.dsl.full((1,), ax.int32, off), output_dtype=ax.int32)
        w = ax.dsl.dynamic_ld(
            heads,
            src,
            ax.dsl.full((1,), ax.int32, HEAD_TILE),
            dim=(0,),
            output_shape=(HEAD_TILE, H),
        )
        lg = linear(hn, w, None, f"{tag}head{off}", output_dtype=ax.fp32, oc_tile_size=HEAD_TILE)
        ax.dsl.move(r=local[off : off + HEAD_TILE], x=lg.reshape(HEAD_TILE))

    sh = Lq.shm_tensor((CORES, V // CORES), ax.fp32, tag + "lg_all")
    ax.dsl.move(r=sh[c : c + 1], x=local.reshape(1, V // CORES))
    axd.pub(sh)
    axd.sub(sh)
    full = ax.dsl.create_tensor((1, V), ax.fp32, ax.MemoryType.OCM, name=tag + "lg_full")
    for k in range(CORES):
        ax.dsl.move(r=full[:, k * (V // CORES) : (k + 1) * (V // CORES)], x=sh[k : k + 1])
    return local, full


def frame_body(e0, e1, packs, heads_lo, heads_hi, embd, cs16, pn16, mask16, rnd, top_k, codes, lgs, io):
    c = ax.program_id(0)
    mbq.SCALES.clear()
    Kq.BLK.clear()
    pm = Qg.members(packs, "P")
    norm = ax.dsl.load(pm["norm"]).reshape(1, 1, H)
    top_k_ocm = ax.dsl.load(top_k)
    rows = slice(c * (V // CORES), (c + 1) * (V // CORES))

    # AxRange operands may not alias a kernel output, and a kernel output
    # cannot be read before its first write.  The ABI outputs are write-only:
    # codes[0] stays zero, every other code / logit is written below, and
    # KV row p is written before position p reads it.  Keep this mutable state
    # in private OCM and write the ABI outputs once at the end.  The KV
    # therefore also stays on chip across all 16 positions.
    codes_w = ax.dsl.zeros((NP,), codes.dtype, ax.MemoryType.OCM)
    lgs_w = ax.dsl.zeros((NP - 1, V // CORES), lgs.dtype, ax.MemoryType.OCM)
    kv_w = [
        ax.dsl.zeros(t.shape, t.dtype, ax.MemoryType.OCM)
        for pair in io
        for t in pair
    ]

    # Position 0 only seeds the predictor KV.  Keeping it static makes the
    # runtime body uniform (always a head + a draw + a next embedding).
    x_row = ax.dsl.load(e0)
    xt = project(c, x_row, pm, "p0_")
    for j in range(NL):
        xt, kv_w[2 * j], kv_w[2 * j + 1] = attn_layer_ocm(
            c, xt, Qg.members(packs, j), cs16[0:2], pn16[0:2], mask16[0:NB],
            kv_w[2 * j], kv_w[2 * j + 1], f"p0_L{j}_",
        )

    x_row = ax.dsl.load(e1)
    half = Lq.ROT // 2
    one = ax.dsl.full((1,), ax.int32, 1)

    def run_positions(state, heads, pos_base, count):
        """One runtime loop over a contiguous group of positions.

        The 15 head tables are split at position 8 so each AxDynamicLd source
        stays inside the 64 MiB SDMA window.  ``pos_base`` is the number of
        global positions already processed by earlier groups.
        """
        start = ax.dsl.full((1,), ax.int32, pos_base + 1)
        stop = ax.dsl.full((1,), ax.int32, pos_base + count + 1)

        def body(p, x_row, codes, lgs, *kv):
            tag = "pl_"
            kv = list(kv)
            cs = ax.dsl.index_select(cs16.reshape(NP, 2, half), p, dim=0).reshape(2, half)
            pn = ax.dsl.index_select(pn16.reshape(NP, 2), p, dim=0).reshape(2)
            msk = ax.dsl.index_select(mask16.reshape(NP, NB, Lq.BC), p, dim=0).reshape(NB, Lq.BC)

            xt = project(c, x_row, pm, tag)
            for j in range(NL):
                xt, kv[2 * j], kv[2 * j + 1] = attn_layer_ocm(
                    c, xt, Qg.members(packs, j), cs, pn, msk,
                    kv[2 * j], kv[2 * j + 1], f"{tag}L{j}_",
                )

            hn = rmsnorm(xt, norm, eps=EPS)
            p_head = ax.dsl.sub(p, ax.dsl.full((1,), ax.int32, pos_base), output_dtype=ax.int32)
            local, full = head_logits(c, hn, heads, p_head, tag)
            lgs = ax.dsl.select_scatter(
                local.reshape(1, V // CORES), ax.dsl.sub(p, one), dim=0, r=lgs
            )

            draw = ax.dsl.index_select(rnd, ax.dsl.sub(p, one), dim=1).reshape(1, 1)
            code = sample(full, draw, top_k_ocm, tag)
            codes = ax.dsl.select_scatter(code.reshape(1), p, dim=0, r=codes)

            # This position's sampled code selects the next position's embedding.
            # p == 15 still computes a valid (unused) table-14 row.
            flat = ax.dsl.add(
                ax.dsl.mul(ax.dsl.sub(p, one), ax.dsl.full((1,), ax.int32, V), output_dtype=ax.int32),
                code,
                output_dtype=ax.int32,
            )
            nxt = ax.dsl.index_select(embd, flat, dim=0).reshape(1, 1, HT)
            return (nxt, codes, lgs, *kv)

        if STATIC_PYRUN:
            for pv in range(pos_base + 1, pos_base + count + 1):
                p = ax.dsl.from_numpy(np.array([pv], dtype=np.int32), prefix=f"pos{pv}")
                state = body(p, *state)
            return state
        return ax.dsl.range(start, start, stop, one, body, *state)

    state = (x_row, codes_w, lgs_w, *kv_w)
    state = run_positions(state, heads_lo, 0, 8)
    state = run_positions(state, heads_hi, 8, NP - 1 - 8)
    x_row = state[0]
    codes_f, lgs_f = state[1], state[2]
    kv_f = state[3:]
    ax.dsl.store(codes_f.reshape(1, NP), r=codes[c : c + 1, :])
    ax.dsl.store(lgs_f, r=lgs[:, rows])
    for j in range(NL):
        ax.dsl.store(kv_f[2 * j], r=io[j][0])
        ax.dsl.store(kv_f[2 * j + 1], r=io[j][1])


def kernel_source():
    outs = [f"{kv}{j}" for j in range(NL) for kv in ("kc", "vc")]
    packs = list(Qg.PDT)
    params = ", ".join(
        ["e0", "e1"] + packs + ["heads_lo", "heads_hi", "embd", "cs16", "pn16", "mask16", "rnd", "top_k", "codes", "lgs"] + outs
    )
    io = ", ".join(f"(kc{j}, vc{j})" for j in range(NL))
    return (
        "import cp_kernel as F\n\n\n"
        f"def cp_frame({params}):\n"
        f"    F.frame_body(e0, e1, dict({', '.join(f'{p}={p}' for p in packs)}), heads_lo, heads_hi, embd, cs16, pn16, mask16, "
        f"rnd, top_k, codes, lgs, [{io}])\n"
        f"    return {', '.join(['codes', 'lgs'] + outs)}\n"
    )
