"""The Qwen3.5 vision encoder (27 layers of 1152, 16 heads of 72; Qwen3.8-27B / 35B-A3B) in ONE four-core kernel:
patch embedding, every layer, the 2 x 2 merge and projector, for N patches (the host pads an image to N).

Each core keeps a whole copy of the hidden state x [N, H] in its own DDR (written only by itself), so no core reads
DDR another core wrote in the same call; cores exchange data only through shared OCM (pub / sub):
  pre    by heads: every core LayerNorm1 of all N rows (cheap), q / k / v of its NH / 4 heads, RoPE -> its own DDR
  attn   by heads: its heads over its own q / k / v (as q35_vit.vit_attn)
  post   by rows, in rounds: round r, core c's tile = its rows c * RC + r * MT ..; every core puts its heads'
         attention rows of the four tiles of the round into shared OCM, each core takes its tile's 16 heads and runs
         o projection + residual + LayerNorm2 + MLP + residual on it; the four new x tiles go through shared OCM into
         every core's copy
  embed  by rows (rounds as post), merge by rows from the core's own copy.

Built as qwen35_vit_full<N>_h<H>_npu4.axmodel. VIT_L=<l>: the first l layers only (simulator checks), output xo.
Usage: MODEL=<hf dir> N=3072 [GH=.. GW=..] [VIT_L=2] python vit_full.py [kernel_dir]
"""
import glob
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CORES", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

import axnn.dsl as ax  # noqa: E402
from axnn.dsl import parallel as axd  # noqa: E402
from axnn.dsl.kernels.utils.activation import gelu_tanh  # noqa: E402
from axnn.dsl.kernels.utils.rope import rope  # noqa: E402
import board_io  # noqa: E402
import q35_vit as V  # noqa: E402

N, H, NH, HD, F, OUT, CORES = V.N, V.H, V.NH, V.HD, V.F, V.OUT, V.CORES
RC, NHC, MT, QB, BC, CP, MERGE = V.RC, V.NHC, V.MT, V.QB, V.BC, V.CP, V.MERGE
HC = NHC * HD  # a core's attention width
assert CORES == 4 and N % (CORES * MT) == 0
ROUNDS = RC // MT
LAYERS = int(os.environ.get("VIT_L", 0)) or None  # None: the config's (all)
# 0: bf16 online softmax / output (as the first 27B build); 1: the output accumulator in fp32; 2: the softmax
# running max / exp in fp32 too (the reference computes softmax in fp32 and casts to bf16)
AF32 = int(os.environ.get("VIT_ATTN_FP32", "0"))
# online: one pass, unnormalized probabilities (the 27B build); ref: two passes, the full fp32 softmax normalized
# before the bf16 probabilities, as transformers' eager attention
AMODE = os.environ.get("VIT_ATTN_MODE", "online")

ORDER = ["px", "pos", "cos", "sin", "kmask", "nblk", "w_patch", "b_patch",
         "ln1_w", "ln1_b", "w_qkv", "b_qkv", "w_o", "b_o", "ln2_w", "ln2_b", "w_up", "b_up", "w_down", "b_down",
         "post_w", "post_b", "w0", "b0", "w2", "b2", "xr", "qo", "ko", "vo", "ao", "out", "xo"]


def at(t, *idx):
    """t[i, j, ..] for integer indices (the DSL takes slices only)"""
    for i in idx:
        t = t[i : i + 1].reshape(*t.shape[1:])
    return t


def shm(shape, dtype, name):
    return ax.dsl.create_tensor(shape, dtype, ax.MemoryType.SHARED_OCM, name=name)


def tile_rows(k, r):
    """core k's tile of round r"""
    return slice(k * RC + r * MT, k * RC + (r + 1) * MT)


def vit_full(px, pos, cos, sin, kmask, nblk, w_patch, b_patch, ln1_w, ln1_b, w_qkv, b_qkv, w_o, b_o, ln2_w, ln2_b,
             w_up, b_up, w_down, b_down, post_w, post_b, w0, b0, w2, b2, xr, qo, ko, vo, ao, out, xo):
    c = ax.program_id(0)
    L = ln1_w.shape[0]
    h0 = c * NHC
    X, Q, K, Vv, A = at(xr, c), at(qo, c), at(ko, c), at(vo, c), at(ao, c)  # this core's own DDR
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=HD**-0.5)

    def exchange_x(r, new, tag):
        """this core's new x tile of round r -> every core's copy (through shared OCM)"""
        sx = shm((CORES, MT, H), ax.bf16, f"{tag}_sx")
        ax.dsl.move(x=new, r=at(sx, c))
        axd.pub(sx)
        axd.sub(sx)
        for k in range(CORES):
            ax.dsl.store(at(sx, k), r=X[tile_rows(k, r)])

    # embed by rows, rounds
    for r in range(ROUNDS):
        rows = tile_rows(c, r)
        e = V.proj(ax.dsl.load(px[rows]).reshape(1, MT, CP), w_patch, b_patch, f"e{r}", output_dtype=ax.bf16)
        e = ax.dsl.add(e, ax.dsl.load(pos[rows]).reshape(1, MT, H), output_dtype=ax.bf16).reshape(MT, H)
        exchange_x(r, e, f"e{r}")

    for l in range(L):
        t = f"l{l}_"
        # pre: all rows, this core's heads
        for i in range(N // MT):
            rows = slice(i * MT, (i + 1) * MT)
            h = V.layernorm(ax.dsl.load(X[rows]).reshape(1, MT, H), at(ln1_w, l), at(ln1_b, l))
            cs, sn = ax.dsl.load(cos[rows]), ax.dsl.load(sin[rows])

            def part(p, nm):  # q / k / v of this core's heads: rows p * H + h0 * HD .. of w_qkv
                a, b = p * H + h0 * HD, p * H + (h0 + NHC) * HD
                return V.proj(h, at(w_qkv, l)[a:b], at(b_qkv, l)[a:b], f"{t}{nm}{i}")
            qh = rope(part(0, "q").reshape(MT, NHC, HD), cs, sn)
            kh = rope(part(1, "k").reshape(MT, NHC, HD), cs, sn)
            qh_store = qh if AMODE == "ref" else ax.dsl.mul(qh, scale, output_dtype=ax.bf16)
            ax.dsl.store(qh_store, r=Q[rows].reshape(MT, NHC, HD))
            ax.dsl.store(kh, r=K[rows].reshape(MT, NHC, HD))
            ax.dsl.store(part(2, "v").reshape(MT, HC), r=Vv[rows])
        # attention: this core's heads over its own q / k / v
        if AMODE == "ref":
            attn_heads_ref(Q, K, Vv, nblk, kmask, A, t)
        else:
            attn_heads(Q, K, Vv, nblk, kmask, A, t)
        # post: rounds of four row tiles, one per core
        for r in range(ROUNDS):
            sa = shm((CORES, CORES, MT, HC), ax.bf16, f"{t}r{r}_sa")  # [owner k][source core][rows][its heads]
            for k in range(CORES):
                ax.dsl.move(x=ax.dsl.load(A[tile_rows(k, r)]), r=at(sa, k, c))
            axd.pub(sa)
            axd.sub(sa)
            att = ax.dsl.create_tensor((MT, H), ax.bf16, name=f"{t}r{r}_att")
            for s in range(CORES):
                ax.dsl.move(x=at(sa, c, s), r=att[:, s * HC : (s + 1) * HC])
            rows = tile_rows(c, r)
            xt = ax.dsl.load(X[rows]).reshape(1, MT, H)
            a = V.proj(att.reshape(1, MT, H), at(w_o, l), at(b_o, l), f"{t}o{r}", output_dtype=ax.bf16)
            x1 = ax.dsl.add(xt, a, output_dtype=ax.bf16)
            u = gelu_tanh(V.proj(V.layernorm(x1, at(ln2_w, l), at(ln2_b, l)), at(w_up, l), at(b_up, l), f"{t}up{r}", output_dtype=ax.bf16),
                          output_dtype=ax.bf16)
            d = V.proj(u, at(w_down, l), at(b_down, l), f"{t}dn{r}", output_dtype=ax.bf16)
            exchange_x(r, ax.dsl.add(x1, d, output_dtype=ax.bf16).reshape(MT, H), f"{t}r{r}")

    # the final hidden state (own rows) and the merge + projector
    G = MERGE * MERGE
    for r in range(ROUNDS):
        rows = tile_rows(c, r)
        xt = ax.dsl.load(X[rows])
        ax.dsl.store(xt, r=xo[rows])
        h = V.layernorm(xt.reshape(1, MT, H), post_w, post_b).reshape(1, MT // G, G * H)
        u = ax.dsl.gelu(V.proj(h, w0, b0, f"m0_{r}", output_dtype=ax.bf16), output_dtype=ax.bf16)
        y = V.proj(u, w2, b2, f"m2_{r}", output_dtype=ax.bf16)
        ax.dsl.store(ax.dsl.cast(y, dtype=ax.fp32).reshape(MT // G, OUT), r=out[rows.start // G : rows.stop // G])
    return out, xo


def attn_heads(qs, kc, vc, nblk, kmask, out, tag):
    """q35_vit.vit_attn on this core's own [N, NHC * HD] q / k / v (heads 0 .. NHC - 1 of its slice)"""
    import ml_dtypes
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=np.float32 if AF32 else ml_dtypes.bfloat16, fill_value=-1))
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix=f"{tag}range_step")
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix=f"{tag}range_start")
    range_stop = ax.dsl.load(nblk)
    zero_col = ax.dsl.zeros(shape=(QB, 1), dtype=ax.fp32 if AF32 == 2 else ax.bf16)

    def body(index, kc, vc, kmask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, *heads):
        t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
        kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HC))
        vb = ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HC))
        mq = ax.dsl.dynamic_ld(kmask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                               output_shape=(1, QB, BC)).reshape(QB, BC)
        for j in range(NHC):
            qh, mi, li, oi = heads[4 * j : 4 * j + 4]
            kj, vj = kb[:, j * HD : (j + 1) * HD], vb[:, j * HD : (j + 1) * HD]
            m_s = ax.dsl.create_tensor(shape=(QB, 1 + BC + 1), dtype=ax.fp32 if AF32 == 2 else ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            ax.dsl.add(sj, mq, output_dtype=m_s.dtype, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s, output_dtype=m_s.dtype)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            if AF32:
                pj = ax.dsl.cast(p32, dtype=ax.bf16)  # the reference's fp32 softmax -> bf16
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(QB)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=li, output_dtype=ax.fp32)
            oij = pj @ vj if not AF32 else ax.dsl.matmul(pj, vj, output_dtype=ax.fp32)
            ratio_oi = ax.dsl.mul(ratio.reshape(QB, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=oi, output_dtype=oi.dtype)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=mi)
        return (kc, vc, kmask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, *heads)

    for b in range(N // QB):
        rows = slice(b * QB, (b + 1) * QB)
        qa = ax.dsl.load(qs[rows]).reshape(QB, NHC, HD)
        heads = []
        for j in range(NHC):
            qh = ax.dsl.create_tensor((QB, HD), ax.bf16, name=f"{tag}q{b}_{j}")
            ax.dsl.move(x=qa[:, j : j + 1, :].reshape(QB, HD), r=qh)
            heads += [qh, ax.dsl.full(shape=(QB, 1), dtype=ax.fp32 if AF32 == 2 else ax.bf16, fill_value=-3e38),
                      ax.dsl.zeros(shape=(QB,), dtype=ax.fp32),
                      ax.dsl.zeros(shape=(QB, HD), dtype=ax.fp32 if AF32 else ax.bf16)]
        index = ax.dsl.create_tensor((1,), ax.int32, name=f"{tag}index{b}")
        ax.dsl.move(x=range_start, r=index)
        res = ax.dsl.range(index, range_start, range_stop, range_step, body, kc, vc, kmask, block_stride,
                           kv_block_shape, mask_block_shape, zero_col, minus_1, *heads)
        heads = res[8:]
        o = ax.dsl.create_tensor((QB, HC), ax.bf16, name=f"{tag}o{b}")
        for j in range(NHC):
            _, _, li, oi = heads[4 * j : 4 * j + 4]
            ax.dsl.div(oi, li.reshape(QB, 1), r=o[:, j * HD : (j + 1) * HD])
        ax.dsl.store(o, r=out[rows])


def attn_heads_ref(qs, kc, vc, nblk, kmask, out, tag):
    """transformers' eager attention: scores rounded to bf16 after the scaling, a full fp32 softmax (two passes over
    the key blocks) normalized before the bf16 probabilities, then a fp32 accumulation of the bf16 PV blocks"""
    import ml_dtypes
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=np.float32, fill_value=-1))
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix=f"{tag}range_step")
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix=f"{tag}range_start")
    range_stop = ax.dsl.load(nblk)
    zero_col = ax.dsl.zeros(shape=(QB, 1), dtype=ax.fp32)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=HD**-0.5)

    def body1(index, kc, kmask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, scale, *heads):
        t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
        kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HC))
        mq = ax.dsl.dynamic_ld(kmask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                               output_shape=(1, QB, BC)).reshape(QB, BC)
        for j in range(NHC):
            qh, mi, li = heads[3 * j : 3 * j + 3]
            kj = kb[:, j * HD : (j + 1) * HD]
            m_s = ax.dsl.create_tensor(shape=(QB, 1 + BC + 1), dtype=ax.fp32)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            sj = ax.dsl.mul(sj, scale, output_dtype=ax.bf16)  # the reference scales the bf16 scores
            ax.dsl.add(sj, mq, output_dtype=ax.fp32, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(QB)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=li, output_dtype=ax.fp32)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=mi)  # the new running max
        return (kc, kmask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, scale, *heads)

    def body2(index, kc, vc, kmask, block_stride, kv_block_shape, mask_block_shape, scale, *heads):
        t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
        kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HC))
        vb = ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HC))
        mq = ax.dsl.dynamic_ld(kmask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                               output_shape=(1, QB, BC)).reshape(QB, BC)
        for j in range(NHC):
            qh, mi, li, oi = heads[4 * j : 4 * j + 4]
            kj, vj = kb[:, j * HD : (j + 1) * HD], vb[:, j * HD : (j + 1) * HD]
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            sj = ax.dsl.mul(sj, scale, output_dtype=ax.bf16)
            s = ax.dsl.add(sj, mq, output_dtype=ax.fp32)
            s = ax.dsl.sub(s, mi, output_dtype=ax.fp32)
            p32 = ax.dsl.exp(s, output_dtype=ax.fp32)
            pn = ax.dsl.div(p32, li.reshape(QB, 1), output_dtype=ax.fp32)
            pj = ax.dsl.cast(pn, dtype=ax.bf16)
            oij = ax.dsl.matmul(pj, vj, output_dtype=ax.fp32)
            ax.dsl.add(oij, oi, r=oi, output_dtype=ax.fp32)
        return (kc, vc, kmask, block_stride, kv_block_shape, mask_block_shape, scale, *heads)

    for b in range(N // QB):
        rows = slice(b * QB, (b + 1) * QB)
        qa = ax.dsl.load(qs[rows]).reshape(QB, NHC, HD)
        heads = []
        for j in range(NHC):
            qh = ax.dsl.create_tensor((QB, HD), ax.bf16, name=f"{tag}q{b}_{j}")
            ax.dsl.move(x=qa[:, j : j + 1, :].reshape(QB, HD), r=qh)
            heads += [qh, ax.dsl.full(shape=(QB, 1), dtype=ax.fp32, fill_value=-3e38),
                      ax.dsl.zeros(shape=(QB,), dtype=ax.fp32)]
        index = ax.dsl.create_tensor((1,), ax.int32, name=f"{tag}index{b}")
        ax.dsl.move(x=range_start, r=index)
        res = ax.dsl.range(index, range_start, range_stop, range_step, body1, kc, kmask, block_stride,
                           kv_block_shape, mask_block_shape, zero_col, minus_1, scale, *heads)
        heads = res[8:]
        heads2 = []
        for j in range(NHC):
            qh, mi, li = heads[3 * j : 3 * j + 3]
            heads2 += [qh, mi, li, ax.dsl.zeros(shape=(QB, HD), dtype=ax.fp32)]
        index = ax.dsl.create_tensor((1,), ax.int32, name=f"{tag}index2{b}")
        ax.dsl.move(x=range_start, r=index)
        res = ax.dsl.range(index, range_start, range_stop, range_step, body2, kc, vc, kmask, block_stride,
                           kv_block_shape, mask_block_shape, scale, *heads2)
        heads = res[7:]
        o = ax.dsl.create_tensor((QB, HC), ax.bf16, name=f"{tag}o{b}")
        for j in range(NHC):
            oi = heads[4 * j + 3]
            ax.dsl.move(x=ax.dsl.cast(oi, dtype=ax.bf16), r=o[:, j * HD : (j + 1) * HD])
        ax.dsl.store(o, r=out[rows])


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(V.BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=CORES)
    torch.manual_seed(0)
    vm = V.load_visual()
    if os.environ.get("REF_BF16") == "1":  # the kernel's activations are bf16 at each operation boundary
        vm = vm.to(torch.bfloat16)
    L = LAYERS or len(vm.blocks)
    TPS, PS = V.TPS, V.PS
    image = os.environ.get("IMAGE")
    if image:  # a real image: the processor's patches, resized to exactly N (gh x gw), as the board gives them
        from PIL import Image
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(V.MODEL)
        proc.size = {"shortest_edge": N * PS * PS, "longest_edge": N * PS * PS}
        proc_out = proc(images=Image.open(image).convert("RGB"), return_tensors="pt")
        grid = proc_out["image_grid_thw"]
        gh, gw = int(grid[0, 1]), int(grid[0, 2])
        n = gh * gw
        assert int(grid[0, 0]) == 1 and n == N, (grid, N)
        pixels = proc_out["pixel_values"]
    else:
        gh, gw = int(os.environ.get("GH", 16)), int(os.environ.get("GW", N // int(os.environ.get("GH", 16))))
        n = gh * gw
        assert n <= N and gh % MERGE == 0 and gw % MERGE == 0
        grid = torch.tensor([[1, gh, gw]])
        pixels = torch.rand((n, 3 * TPS * PS * PS)) * 2 - 1
        pixels = pixels.view(n, 3, TPS, PS * PS)
        pixels[:, :, 1] = pixels[:, :, 0]
        pixels = pixels.reshape(n, -1)
    if os.environ.get("REF_BF16") == "1":
        pixels = pixels.to(torch.bfloat16)
    assert gh % MERGE == 0 and gw % MERGE == 0
    rec = {}
    hooks = [vm.blocks[i].register_forward_hook(lambda m, a, o, i=i: rec.__setitem__(i, o.detach())) for i in range(len(vm.blocks))]
    hooks.append(vm.blocks[0].register_forward_pre_hook(lambda m, a, kw: rec.__setitem__("x0", a[0].detach()), with_kwargs=True))
    hooks.append(vm.blocks[0].register_forward_pre_hook(lambda m, a, kw: rec.__setitem__("pe", kw["position_embeddings"]), with_kwargs=True))
    with torch.no_grad():
        ref = vm(pixels, grid_thw=grid).pooler_output
    for hk in hooks:
        hk.remove()
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    pad = lambda t: torch.cat([t, torch.zeros((N - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)]) if t.shape[0] < N else t  # noqa: E731
    w = vm.patch_embed.proj.weight.detach()
    frame = pixels.view(n, 3, TPS, PS * PS)[:, :, 0].reshape(n, CP)
    pos = rec["x0"] - torch.nn.functional.linear(frame, w.sum(2).reshape(H, CP), vm.patch_embed.proj.bias.detach())
    cs, sn = rec["pe"]
    kmask = torch.zeros((N,), dtype=torch.float32)
    kmask[n:] = -3e38
    blocks = vm.blocks[:L]
    st = lambda f: bf(torch.stack([f(b).detach() for b in blocks]))  # noqa: E731
    mg = vm.merger
    a = dict(px=bf(pad(frame)), pos=bf(pad(pos)), cos=bf(pad(cs[:, : HD // 2])), sin=bf(pad(sn[:, : HD // 2])),
             kmask=bf(kmask.reshape(N // BC, 1, BC).expand(N // BC, QB, BC)), nblk=torch.tensor([-(-n // BC)], dtype=torch.int32),
             w_patch=bf(w.sum(2).reshape(H, CP)), b_patch=bf(vm.patch_embed.proj.bias.detach()),
             ln1_w=st(lambda b: b.norm1.weight), ln1_b=st(lambda b: b.norm1.bias), w_qkv=st(lambda b: b.attn.qkv.weight),
             b_qkv=st(lambda b: b.attn.qkv.bias), w_o=st(lambda b: b.attn.proj.weight), b_o=st(lambda b: b.attn.proj.bias),
             ln2_w=st(lambda b: b.norm2.weight), ln2_b=st(lambda b: b.norm2.bias), w_up=st(lambda b: b.mlp.linear_fc1.weight),
             b_up=st(lambda b: b.mlp.linear_fc1.bias), w_down=st(lambda b: b.mlp.linear_fc2.weight),
             b_down=st(lambda b: b.mlp.linear_fc2.bias), post_w=bf(mg.norm.weight.detach()), post_b=bf(mg.norm.bias.detach()),
             w0=bf(mg.linear_fc1.weight.detach()), b0=bf(mg.linear_fc1.bias.detach()), w2=bf(mg.linear_fc2.weight.detach()),
             b2=bf(mg.linear_fc2.bias.detach()),
             xr=torch.zeros((CORES, N, H), dtype=torch.bfloat16), qo=torch.zeros((CORES, N, HC), dtype=torch.bfloat16),
             ko=torch.zeros((CORES, N, HC), dtype=torch.bfloat16), vo=torch.zeros((CORES, N, HC), dtype=torch.bfloat16),
             ao=torch.zeros((CORES, N, HC), dtype=torch.bfloat16),
             out=torch.zeros((N // (MERGE * MERGE), OUT), dtype=torch.float32), xo=torch.zeros((N, H), dtype=torch.bfloat16))
    sub = f"full{N}_l{L}"
    os.makedirs(f"{V.BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{V.BUILD}/{sub}")
    rm = ax.RunMode(os.environ.get("RUN_MODE", "CRUN"))
    k = ax.builder(run_mode=rm, enable_cache=rm != ax.RunMode.PYRUN, dump_path=f"{V.BUILD}/{sub}")(vit_full)
    k[CORES](*[a[x] for x in ORDER])
    def row_cos(y, r):
        c = torch.nn.functional.cosine_similarity(y.float(), r.float(), dim=-1)
        return c.mean().item(), c.min().item()

    xm, xn = row_cos(a["xo"][:n], rec[L - 1])
    xc = torch.nn.functional.cosine_similarity(a["xo"][:n].float(), rec[L - 1].float(), dim=-1)
    msg = (f"vit_full N={N} L={L} (n={n}): x after layer {L - 1} vs HF cos "
           f"{V.cos(a['xo'][:n], rec[L - 1]):.6f} (rows {xm:.6f}/{xn:.6f} at {int(xc.argmin())})")
    if L == len(vm.blocks):
        po = a["out"][: n // (MERGE * MERGE)]
        om, on = row_cos(po, ref)
        msg += f", out vs HF cos {V.cos(po, ref):.6f} (rows {om:.6f}/{on:.6f})"
        msg += (f"; ACCEPT {'PASS' if xm >= 0.9995 and xn >= 0.999 and om >= 0.9995 and on >= 0.999 else 'FAIL'}")
        torch.save({"xo": a["xo"][:n].cpu(), "x_ref": rec[L - 1].cpu(), "out": po.cpu(), "out_ref": ref.cpu()},
                   f"{V.BUILD}/check_vit_{N}_{L}.pt")
    else:
        torch.save({"xo": a["xo"][:n].cpu(), "x_ref": rec[L - 1].cpu()}, f"{V.BUILD}/check_vit_{N}_{L}.pt")
    logger.info(msg)
    if L == len(vm.blocks):
        assert xm >= 0.9995 and xn >= 0.999 and om >= 0.9995 and on >= 0.999, (xm, xn, om, on)
    if out_dir and rm == ax.RunMode.CRUN:
        (path,) = glob.glob(f"{V.BUILD}/{sub}/cache/vit_full.*.compiled.axmodel")
        name = os.environ.get("KNAME", f"qwen35_9b_vit{N}")
        dst = board_io.kernel_file(out_dir, name, CORES)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
