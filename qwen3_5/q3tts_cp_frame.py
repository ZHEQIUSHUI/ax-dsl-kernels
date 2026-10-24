"""A whole frame of the Qwen3-TTS code predictor in one four-core kernel: its 16 positions (position 0 the talker's
hidden state, position 1 code 0's embedding, then each sampled code's embedding) through q3tts_cp_step's projection and
layers, and at positions 1 .. 15 the codebook head and the sampling of the next code on chip, as mtmd's graph samples:
the K largest logits (ties: the lower vocabulary id first), the first two swapped when K > 1 (ggml's top_k leaves them
so), softmax and cumulative sum in f32, the first candidate whose sum exceeds the position's uniform draw. The sampled
code's row of the next codebook's embedding table is loaded by that code and is the next position's input.

Inputs: e0 / e1 [1, 1, HT] bf16 (the talker's hidden state, code 0's embedding), the step kernel's packs, heads [15 * V,
H] bf16 (the codebook heads), embd [15 * V, HT] bf16 (the codebooks' embedding tables; table g embeds code g + 1), cs16
[16 * 2, ROT / 2] (cos | sin rows per position), pn16 [16 * 2] (pos, nblk per position), mask16 [16 * LMAX / BC, BC],
rnd [1, 15] f32 (the positions' draws, mtmd's order); outputs codes [CORES, 16] int32 (every core's row, codes 1 .. 15
at their positions), lgs [15, V] f32 (the heads' logits), kc<j> / vc<j> (the resident KV). Every product over the whole
K; each core projects its KV heads and reads back only its own writes (as the step kernel). Written as
qwen3tts_cp_frame<NL>_mx8_h<HT>_k<K>_c<LMAX>_npu<C> with the step kernel's .layout. Checked on the simulator: the codes
against the sampling of the kernel's own logits in torch, the logits against torch on the dequantized weights (teacher
forced by the kernel's codes).

Usage: MODEL=<predictor cfg dir> HT=2048 TOPK=50 LMAX=16 BC=16 CORES=4 python q3tts_cp_frame.py <dir>
"""
import glob
import math
import os
import sys

import torch
from loguru import logger

import q3tts_cp_step as S

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import rmsnorm
from axnn.dsl.kernels.utils.linear import linear

Qg, L, K, mb, q = S.Qg, S.L, S.K, S.mb, S.q
NL, H, HT, V, CORES, PROJ = S.NL, S.H, S.HT, S.V, S.CORES, S.PROJ
NP = 16  # positions per frame
TOPK = int(os.environ.get("TOPK", 50))
NB = L.LMAX // L.BC  # mask rows per position


def project(c, xe, pm, t):
    """the talker-wide row to the predictor's width: each core its H / CORES rows (+ bias), the row through shared OCM"""
    if not PROJ:
        return xe
    rs = slice(c * (H // CORES), (c + 1) * (H // CORES))
    pr = linear(xe, pm["w_pin"][rs], None, t + "proj_in", output_dtype=ax.fp32, oc_tile_size=H // CORES)
    pr = ax.dsl.add(pr, ax.dsl.load(pm["b_pin"][rs]).reshape(1, 1, H // CORES), output_dtype=ax.fp32)
    sh = L.shm_tensor((1, 1, H), ax.bf16, t + "proj_all")
    ax.dsl.move(r=sh[:, :, rs], x=ax.dsl.cast(pr, dtype=ax.bf16))
    axd.pub(sh)
    axd.sub(sh)
    xt = ax.dsl.create_tensor((1, 1, H), ax.bf16, ax.MemoryType.OCM, name=t + "x_in")
    ax.dsl.move(r=xt, x=sh)
    return xt


def sample(lg, r, t):
    """the next code from the full logits [1, V] f32 and the draw r [1, 1] f32, as mtmd's do_sampling: an int32 [1, 1]"""
    v, i = ax.dsl.topk(lg, k=TOPK, dim=-1)
    if TOPK > 1:  # ggml's top_k swaps its first two
        vs = ax.dsl.create_tensor((1, TOPK), ax.fp32, ax.MemoryType.OCM, name=t + "vals")
        js = ax.dsl.create_tensor((1, TOPK), ax.fp32, ax.MemoryType.OCM, name=t + "ids")
        fi = ax.dsl.cast(i, dtype=ax.fp32)
        ax.dsl.move(r=vs[:, 0:1], x=v[:, 1:2])
        ax.dsl.move(r=vs[:, 1:2], x=v[:, 0:1])
        ax.dsl.move(r=js[:, 0:1], x=fi[:, 1:2])
        ax.dsl.move(r=js[:, 1:2], x=fi[:, 0:1])
        if TOPK > 2:
            ax.dsl.move(r=vs[:, 2:TOPK], x=v[:, 2:TOPK])
            ax.dsl.move(r=js[:, 2:TOPK], x=fi[:, 2:TOPK])
        v, fi = vs, js
    else:
        fi = ax.dsl.cast(i, dtype=ax.fp32)
    p = ax.dsl.softmax(v, dim=-1, output_dtype=ax.fp32)
    cdf = ax.dsl.cumsum(p, dim=-1, output_dtype=ax.fp32)
    # crossed[k] = cdf[k] > r (ggml's step); the first crossing as a one-hot row picks the candidate's id
    cross = ax.dsl.cast(ax.dsl.ceil(ax.dsl.clip(ax.dsl.sub(cdf, r, output_dtype=ax.fp32), min=0.0, max=1.0)),
                        dtype=ax.fp32)
    prev = ax.dsl.create_tensor((1, TOPK), ax.fp32, ax.MemoryType.OCM, name=t + "prev")
    ax.dsl.move(r=prev[:, 0:1], x=ax.dsl.full((1, 1), ax.fp32, 0.0))
    if TOPK > 1:
        ax.dsl.move(r=prev[:, 1:TOPK], x=cross[:, 0 : TOPK - 1])
    oh = ax.dsl.sub(cross, prev, output_dtype=ax.fp32)
    idf = ax.dsl.sum(ax.dsl.mul(oh, fi, output_dtype=ax.fp32), dim=-1, keepdim=True, output_dtype=ax.fp32)
    return ax.dsl.cast(idf, dtype=ax.int32)


def frame_body(e0, e1, packs, heads, embd, cs16, pn16, mask16, rnd, codes, lgs, io):
    c = ax.program_id(0)
    mb.SCALES.clear()
    K.BLK.clear()
    pm = Qg.members(packs, "P")
    norm = ax.dsl.load(pm["norm"]).reshape(1, 1, H)
    draws = ax.dsl.load(rnd)  # [1, 15]
    rows = slice(c * (V // CORES), (c + 1) * (V // CORES))
    x_row = ax.dsl.load(e0)
    for p in range(NP):
        t = f"p{p}_"
        xt = project(c, x_row, pm, t)
        for j in range(NL):
            m = Qg.members(packs, j)
            xt = Qg.attn_layer(c, xt, m, cs16[2 * p : 2 * p + 2], pn16[2 * p : 2 * p + 2], mask16[p * NB : (p + 1) * NB],
                               io[j][0], io[j][1], f"{t}L{j}_")
        if p == 0:  # position 0 only seeds the KV
            x_row = ax.dsl.load(e1)
            continue
        hn = rmsnorm(xt, norm, eps=S.EPS)
        lg = linear(hn, heads[(p - 1) * V : p * V][rows], None, t + "head", output_dtype=ax.fp32,
                    oc_tile_size=min(256, V // CORES))
        ax.dsl.store(lg.reshape(1, V // CORES), r=lgs[p - 1 : p, rows])
        sh = L.shm_tensor((CORES, V // CORES), ax.fp32, t + "lg_all")
        ax.dsl.move(r=sh[c : c + 1], x=lg.reshape(1, V // CORES))
        axd.pub(sh)
        axd.sub(sh)
        full = ax.dsl.create_tensor((1, V), ax.fp32, ax.MemoryType.OCM, name=t + "lg_full")
        for k in range(CORES):
            ax.dsl.move(r=full[:, k * (V // CORES) : (k + 1) * (V // CORES)], x=sh[k : k + 1])
        code = sample(full, draws[:, p - 1 : p], t)
        if p == NP - 1:  # every shared OCM tensor of the frame read once more before its last output (KEEP_SHM)
            code = L.keep_shm(code)
        ax.dsl.store(code, r=codes[c : c + 1, p : p + 1])
        if p < NP - 1:  # the code's row of its codebook's table: the next position's input
            x_row = ax.dsl.dynamic_ld(embd[(p - 1) * V : p * V], code.reshape(1), ax.dsl.full((1,), ax.int32, 1), dim=(0,),
                                      output_shape=(1, HT)).reshape(1, 1, HT)


def kernel_source():
    outs = [f"{kv}{j}" for j in range(NL) for kv in ("kc", "vc")]
    packs = list(Qg.PDT)
    params = ", ".join(["e0", "e1"] + packs + ["heads", "embd", "cs16", "pn16", "mask16", "rnd", "codes", "lgs"] + outs)
    io = ", ".join(f"(kc{j}, vc{j})" for j in range(NL))
    return (f"import q3tts_cp_frame as F\n\n\ndef cp_frame({params}):\n"
            f"    F.frame_body(e0, e1, dict({', '.join(f'{p}={p}' for p in packs)}), heads, embd, cs16, pn16, mask16, rnd, "
            f"codes, lgs, [{io}])\n"
            f"    return {', '.join(['codes', 'lgs'] + outs)}\n")


def ref_sample(lg, r):
    """mtmd's do_sampling on f32 logits [V] (top_k, top_p 1): the candidate index and the sampled id"""
    order = torch.sort(lg, descending=True, stable=True).indices[:TOPK]
    if TOPK > 1:
        order = torch.cat([order[1:2], order[0:1], order[2:]])
    cdf = torch.softmax(lg[order], 0).cumsum(0)
    n = int((cdf - r > 0).sum())
    return TOPK - n, int(order[TOPK - n]) if n > 0 else -1


def main():
    sys.modules["q3tts_cp_frame"] = sys.modules[__name__]
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    pk, sizes, es, raw, extra = S.build_packs()
    LMAX, BC, HKV, D = L.LMAX, L.BC, L.HKV, L.D
    heads = (torch.randn((NP - 1, V, H)) * H**-0.5 * 4).bfloat16()  # (wide logits: a sharp distribution)
    embd = (torch.randn((NP - 1, V, HT)) * 0.5).bfloat16()
    build = os.path.abspath(os.environ.get("DUMP", "out_q3tts_cp_frame"))
    os.makedirs(f"{build}/k", exist_ok=True)
    with open(f"{build}/k/cp_frame_kernel.py", "w") as f:
        f.write(kernel_source())
    sys.path.insert(0, f"{build}/k")
    fn = __import__("cp_frame_kernel").cp_frame
    os.chdir(f"{build}/k")
    PERF = os.environ.get("PERF") == "1"
    kk = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=not PERF, dump_dsl_perf=PERF, dump_path=f"{build}/k")(fn)
    cs16, pn16, mask16 = [], [], []
    for pos in range(NP):
        cs_, sn_ = L.rope_rows(pos)
        cs16.append(torch.cat([cs_, sn_]))
        pn16.append(torch.tensor([pos, pos // BC + 1], dtype=torch.int32))
        mrow = torch.full((LMAX,), -3e38)
        mrow[: pos + 1] = 0
        mask16.append(mrow.reshape(NB, BC))
    worst, bad = 1.0, 0
    for frame in range(int(os.environ.get("FRAMES", 1))):
        e0 = (torch.randn((1, 1, HT)) * 0.5).bfloat16()
        e1 = (torch.randn((1, 1, HT)) * 0.5).bfloat16()
        rnd = torch.rand((1, NP - 1))
        case = dict(pk, e0=e0, e1=e1, heads=heads.reshape(-1, H), embd=embd.reshape(-1, HT), cs16=torch.cat(cs16).bfloat16(),
                    pn16=torch.cat(pn16), mask16=torch.cat(mask16).bfloat16(), rnd=rnd,
                    codes=torch.zeros((CORES, NP), dtype=torch.int32), lgs=torch.zeros((NP - 1, V)))
        kv = [[torch.zeros((LMAX, HKV * D), dtype=torch.half) for _ in range(2)] for _ in range(NL)]
        for j in range(NL):
            case[f"kc{j}"], case[f"vc{j}"] = kv[j]
            q.RESIDENT[f"kc{j}"], q.RESIDENT[f"vc{j}"] = kv[j][0].clone(), kv[j][1].clone()
        ax.init(chip_num=1, core_num=CORES)
        mb.SCALES.clear()
        kk[CORES](*[case[n] for n in L.order(fn)])
        q.RESIDENT.clear()
        codes, lgs = case["codes"], case["lgs"]
        same_rows = all(bool((codes[k] == codes[0]).all()) for k in range(CORES))
        # the kernel's codes against the sampling of its own logits; its logits against torch, teacher forced
        kref = [[torch.zeros((LMAX, HKV * D)) for _ in range(2)] for _ in range(NL)]
        x_row = e0.float().reshape(HT)
        for pos in range(NP):
            x = x_row
            if PROJ:
                x = (extra["w_pin"].float() @ x + extra["b_pin"]).bfloat16().float()
            x = S.ref_layers(x, pos, raw, kref)
            if pos == 0:
                x_row = e1.float().reshape(HT)
                continue
            ref = heads[pos - 1].float() @ S.rms(x, extra["norm"])
            got = lgs[pos - 1]
            co = float(torch.nn.functional.cosine_similarity(got, ref, 0))
            ci, cid = ref_sample(got, float(rnd[0, pos - 1]))
            code = int(codes[0, pos])
            if code != cid:
                bad += 1
                logger.warning(f"frame {frame} pos {pos}: kernel code {code}, the sampling of its logits {cid} (index {ci})")
            logger.info(f"cp frame {frame} pos {pos}: logits cos vs torch {co:.6f}, code {code} ({'same' if code == cid else 'DIFFERS'})")
            worst = min(worst, co)
            x_row = embd[pos - 1, code].float()  # (teacher forced by the kernel's code)
        logger.info(f"frame {frame}: core rows {'the same' if same_rows else 'DIFFER'}")
        bad += 0 if same_rows else 1
    path = max(glob.glob(f"{build}/k/cache/cp_frame.*.compiled.axmodel"), key=os.path.getmtime)
    name = f"qwen3tts_cp_frame{NL}{K.FMT}_h{HT}_k{TOPK}_c{LMAX}"
    dst = S.write_kernel(path, out_dir, name, sizes, es)
    logger.info(f"wrote {dst} and its layout (worst logits cos {worst:.6f}, {bad} codes / rows differ)")


if __name__ == "__main__":
    main()
