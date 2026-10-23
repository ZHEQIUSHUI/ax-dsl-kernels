"""The Qwen3-TTS code predictor's NL layers in one kernel per position (q3tts_cp.py) on CORES NPU cores, the cores
synchronized between the halves of each layer through shared OCM (tensor pub / sub):

  per layer, on core c: rmsnorm of the whole hidden state (every core); its HQ / CORES query heads and HKV / CORES KV
  heads (q / k / v rows, K / V columns of the resident cache row pos), their attention, the o-projection partial of
  their columns -> shared part_a[c] (fp32); pub / sub; x1 = x + sum_c part_a[c] (every core); rmsnorm, its F / CORES
  MLP channels (gate / up rows, down columns) -> shared part_m[c]; pub / sub; x2 = x1 + sum_c part_m[c]. Core 0
  stores the last layer's result.

Same IO as q3tts_cp.py (qwen3tts_cp_stack<NL>[_mx8]_c<LMAX>), written as _npu<CORES>. Checked on the simulator
against NL calls of the single-layer kernel for positions 0 .. STEPS-1 (out, K / V).

Usage: MODEL=<Qwen3-0.6B dir> KVL=cell LMAX=16 BC=16 NL=5 CORES=4 [WFMT=mx8] [STEPS=2] python q3tts_cp_mc.py <dir>
"""
import glob
import importlib
import math
import os
import shutil
import sys

os.environ.setdefault("KVL", "cell")
os.environ.setdefault("LMAX", "16")
os.environ.setdefault("BC", "16")

import ml_dtypes
import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import rmsnorm, silu_mul
from axnn.dsl.kernels.utils.rope import rope
import board_io
import q35_step as q
import q3tts_cp as cp

CORES = int(os.environ.get("CORES", 4))
NL, MX, STEPS = cp.NL, cp.MX, cp.STEPS
mb = cp.mb if MX else None
H, D, HQ, HKV, G, F, EPS, BC = q.H, q.D, q.HQ, q.HKV, q.G, q.F, q.EPS, q.BC
QH, KH, FC = HQ // CORES, HKV // CORES, F // CORES  # query heads, KV heads, MLP channels per core
assert HQ % CORES == 0 and HKV % CORES == 0 and F % CORES == 0 and QH == KH * G and q.KVL == "cell" and not q.GATE
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q3tts_cp_mc"))


def attention(c, qh_all, kc, vc, nblk, mask, tag):
    """the core's KH KV heads over the resident cache: [1, QH, D]"""
    attn = ax.dsl.create_tensor((1, QH, D), ax.bf16, ax.MemoryType.OCM, name=tag + "attn")
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_stop = ax.dsl.load(nblk)
    for j in range(KH):
        hk = c * KH + j  # the KV head (cache columns)
        qh = ax.dsl.mul(qh_all[:, j * G : (j + 1) * G, :].reshape(G, D), scale, output_dtype=ax.bf16)
        oi = ax.dsl.zeros(shape=(G, D), dtype=ax.bf16)
        li = ax.dsl.zeros(shape=(G,), dtype=ax.fp32)
        mi = ax.dsl.full(shape=(G, 1), dtype=ax.bf16, fill_value=-3e38)
        zero_col = ax.dsl.zeros(shape=(G, 1), dtype=ax.bf16)
        index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

        def body(index, kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col,
                 minus_1):
            loop_mi, loop_li, loop_oi = mi, li, oi
            t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
            kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D))
            vb = ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D))
            kj = ax.dsl.cast(kb[:, hk * D : (hk + 1) * D], dtype=ax.bf16)
            vj = ax.dsl.cast(vb[:, hk * D : (hk + 1) * D], dtype=ax.bf16)
            mj = ax.dsl.dynamic_ld(mask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                                   output_shape=(1, BC))
            m_s = ax.dsl.create_tensor(shape=(G, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            ax.dsl.add(sj, mj, output_dtype=ax.bf16, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(G)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
            oij = pj @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(G, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
            return (kc, vc, mask, qh, loop_mi, loop_li, loop_oi, block_stride, kv_block_shape, mask_block_shape,
                    zero_col, minus_1)

        (kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1) = ax.dsl.range(
            index, range_start, range_stop, range_step, body,
            kc, vc, mask, qh, mi, li, oi, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1,
        )
        ax.dsl.div(oi, li.reshape(G, 1), output_dtype=ax.bf16, r=attn[:, j * G : (j + 1) * G, :].reshape(G, D))
    return attn


def reduce(c, x, part, partial):
    """x + the cores' partials: the core's partial into shared part[c], published, then all of them summed"""
    ax.dsl.move(r=part[c : c + 1], x=partial.reshape(1, 1, H))
    axd.pub(part)
    axd.sub(part)
    acc = part[0:1]
    for j in range(1, CORES):
        acc = ax.dsl.add(acc, part[j : j + 1], output_dtype=ax.fp32)
    # the projection rounded to bf16, then added to x: as the single-core layer (a bf16 projection output)
    return ax.dsl.add(x, ax.dsl.cast(acc, dtype=ax.bf16), output_dtype=ax.bf16)


def layer(c, xt, l, cos_t, sin_t, kc, vc, pos, nblk, mask, tag):
    qs, ks, fs = slice(c * QH * D, (c + 1) * QH * D), slice(c * KH * D, (c + 1) * KH * D), slice(c * FC, (c + 1) * FC)
    h = rmsnorm(xt, ax.dsl.load(l["ln1"]).reshape(1, 1, H), eps=EPS)
    qh = rmsnorm(q.proj(h, q.slice_w(l["w_q"], rows=qs), tag + "attn_q").reshape(1, QH, D),
                 ax.dsl.load(l["q_norm"]).reshape(1, 1, D), eps=EPS)
    k = rmsnorm(q.proj(h, q.slice_w(l["w_k"], rows=ks), tag + "attn_k").reshape(1, KH, D),
                ax.dsl.load(l["k_norm"]).reshape(1, 1, D), eps=EPS)
    v = q.proj(h, q.slice_w(l["w_v"], rows=ks), tag + "attn_v").reshape(1, KH, D)
    qh = rope(qh, cos_t, sin_t, rotary_dim=q.ROT)
    k = rope(k, cos_t, sin_t, rotary_dim=q.ROT)
    # the new token's K / V: the core's columns of cache row pos
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    dst = ax.dsl.concat(p, ax.dsl.from_numpy(np.array([c], dtype=np.uint32)), dim=0)  # (cell, core)
    src0 = ax.dsl.from_numpy(np.zeros(shape=(2,), dtype=np.uint32))
    blk = ax.dsl.from_numpy(np.array([1, 1], dtype=np.uint32))
    shape = (q.LMAX, CORES, KH * D)  # the cache rows as one block of KH heads per core
    ax.dsl.dynamic_st(ax.dsl.cast(k.reshape(1, 1, KH * D), dtype=ax.fp16), src0, dst, blk, dim=(0, 1),
                      output_shape=shape, r=kc.reshape(*shape))
    ax.dsl.dynamic_st(ax.dsl.cast(v.reshape(1, 1, KH * D), dtype=ax.fp16), src0, dst, blk, dim=(0, 1),
                      output_shape=shape, r=vc.reshape(*shape))
    attn = attention(c, qh, kc, vc, nblk, mask, tag)
    o = q.proj(attn.reshape(1, 1, QH * D), q.slice_w(l["w_o"], cols=qs), tag + "attn_o", output_dtype=ax.fp32)
    part_a = ax.dsl.create_tensor((CORES, 1, H), ax.fp32, ax.MemoryType.SHARED_OCM, name=tag + "part_a")
    x1 = reduce(c, xt, part_a, o)
    h2 = rmsnorm(x1, ax.dsl.load(l["ln2"]).reshape(1, 1, H), eps=EPS)
    act = silu_mul(q.proj(h2, q.slice_w(l["w_gate"], rows=fs), tag + "mlp_gate"),
                   q.proj(h2, q.slice_w(l["w_up"], rows=fs), tag + "mlp_up"), activation_output_dtype=ax.fp32,
                   output_dtype=ax.bf16)
    d = q.proj(act, q.slice_w(l["w_down"], cols=fs), tag + "mlp_down", output_dtype=ax.fp32)
    part_m = ax.dsl.create_tensor((CORES, 1, H), ax.fp32, ax.MemoryType.SHARED_OCM, name=tag + "part_m")
    return reduce(c, x1, part_m, d)


def body(a):
    """the kernel: a = its arguments by name"""
    c = ax.program_id(0)
    if MX:
        mb.SCALES.clear()
        for i in range(NL):
            for w in mb.ATTN_W:
                mb.SCALES[id(a[f"{w}_{i}"])] = a[f"{mb.scale_name(w)}_{i}"]
    xt = ax.dsl.load(a["x"])
    cos_t, sin_t = ax.dsl.load(a["cos"]), ax.dsl.load(a["sin"])
    for i in range(NL):
        l = {w: a[f"{w}_{i}"] for w in cp.LAYER_W}
        xt = layer(c, xt, l, cos_t, sin_t, a[f"kc_{i}"], a[f"vc_{i}"], a["pos"], a["nblk"], a["mask"], f"l{i}_")
    if c == 0:
        ax.dsl.store(xt, r=a["out"])


def kernel_function():
    ins = ["x", "cos", "sin", "pos", "nblk", "mask"] + [f"{n}_{i}" for i in range(NL) for n in cp.layer_inputs()]
    outs = ["out"] + [f"{kv}_{i}" for i in range(NL) for kv in ("kc", "vc")]
    src = (f"import q3tts_cp_mc as S\n\n\ndef cp_stack({', '.join(ins + outs)}):\n    S.body(locals())\n"
           f"    return {', '.join(outs)}\n")
    os.makedirs(BUILD, exist_ok=True)
    open(f"{BUILD}/cp_stack_mc_kernel.py", "w").write(src)
    sys.path.insert(0, BUILD)
    return importlib.import_module("cp_stack_mc_kernel").cp_stack, ins, outs


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(q.MODEL)
    fn, ins, outs = kernel_function()
    args = []
    for i in range(NL):
        a = q.attn_args(W, i)
        args.append(mb.mx_args(a, mb.ATTN_W)[0] if MX else a)
    single = mb.attn_layer_mx if MX else q.attn_layer
    single_order = mb.order(mb.attn_layer_mx) if MX else q.ATTN_ORDER
    for d in ("stack", "single"):
        os.makedirs(f"{BUILD}/{d}", exist_ok=True)
    os.chdir(f"{BUILD}/stack")
    k_stack = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/stack")(fn)
    os.chdir(f"{BUILD}/single")
    k_single = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/single")(single)

    kv = lambda: torch.zeros((q.LMAX, HKV * D), dtype=torch.half)  # noqa: E731
    kc_s, vc_s = [kv() for _ in range(NL)], [kv() for _ in range(NL)]
    kc_r, vc_r = [kv() for _ in range(NL)], [kv() for _ in range(NL)]
    emb = W("embed_tokens.weight")
    worst = 1.0
    for pos in range(STEPS):
        x = emb[1000 + 7 * pos].reshape(1, 1, H).contiguous()
        cs, sn = q.rope_cos_sin(pos)
        m = torch.full((q.LMAX,), -3e38)
        m[: pos + 1] = 0
        common = dict(cos=cs, sin=sn, pos=torch.tensor([pos], dtype=torch.int32),
                      nblk=torch.tensor([pos // BC + 1], dtype=torch.int32), mask=m.reshape(q.LMAX // BC, BC).bfloat16())
        h = x
        ax.init(chip_num=1, core_num=1)
        for i in range(NL):
            out = torch.zeros((1, 1, H), dtype=torch.bfloat16)
            case = dict(args[i], x=h, kc=kc_r[i], vc=vc_r[i], out=out, **common)
            q.RESIDENT.update(kc=kc_r[i].clone(), vc=vc_r[i].clone())
            if MX:
                mb.SCALES.clear()
            os.chdir(f"{BUILD}/single")
            k_single[1](*[case[n] for n in single_order])
            q.RESIDENT.clear()
            h = out
        out = torch.zeros((1, 1, H), dtype=torch.bfloat16)
        case = dict(x=x, out=out, **common)
        for i in range(NL):
            case.update({f"{n}_{i}": args[i][n] for n in cp.layer_inputs()})
            case[f"kc_{i}"], case[f"vc_{i}"] = kc_s[i], vc_s[i]
            q.RESIDENT.update({f"kc_{i}": kc_s[i].clone(), f"vc_{i}": vc_s[i].clone()})
        ax.init(chip_num=1, core_num=CORES)
        os.chdir(f"{BUILD}/stack")
        k_stack[CORES](*[case[n] for n in ins + outs])
        q.RESIDENT.clear()
        co = cp.cos(out, h)
        ck = min(cp.cos(kc_s[i][pos], kc_r[i][pos]) for i in range(NL))
        cv = min(cp.cos(vc_s[i][pos], vc_r[i][pos]) for i in range(NL))
        logger.info(f"{cp.NAME} npu{CORES} pos {pos}: out cos {co:.6f}, K cos min {ck:.6f}, V cos min {cv:.6f}")
        worst = min(worst, co, ck, cv)
    path = max(glob.glob(f"{BUILD}/stack/cache/cp_stack.*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, cp.NAME, CORES)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst} (worst cos vs the single-layer calls {worst:.6f})")


if __name__ == "__main__":
    main()
