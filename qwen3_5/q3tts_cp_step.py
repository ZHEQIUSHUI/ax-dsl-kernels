"""One step of the Qwen3-TTS code predictor in one four-core kernel: the input projection (talker hidden -> predictor
hidden, + bias; HT == H without one), the NL layers as q35_group's attention layers (packed + column-blocked MXFP8
parameters, the KV heads split over the cores and exchanged through shared OCM, every core writing the whole K / V
row and reading only its own writes), the final norm and one codebook head -> logits [1, V] f32. One call per
predictor position instead of the projection GEMV, the layer stack and the head GEMV.

Inputs: e [1, 1, HT] bf16 (the step's embedding row), the packs (members "L<j>.<name>", "LP.w_pin" / "LP.b_pin" /
"LP.norm"), w_head [V, H] bf16 (this step's codebook head, bound per call), cs [2, ROT/2], pn [2] (pos, nblk), mask
[LMAX / BC, BC]; outputs logits [1, V], kc<j> / vc<j> (the predictor's resident KV, in place). Written as
qwen3tts_cp_step<NL>_mx8_h<HT>_c<LMAX>_npu<C> with a .layout. Checked on the simulator against torch on the dequantized
weights.

Usage: MODEL=<predictor cfg dir> HT=2048 KVL=cell LMAX=16 BC=16 CORES=4 python q3tts_cp_step.py <dir>
"""
import glob
import math
import os
import shutil
import sys

os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("KVL", "cell")
os.environ.setdefault("LMAX", "16")
os.environ.setdefault("BC", "16")
os.environ["KVX"] = "1"
os.environ["DEC_SPLIT"] = "0"
os.environ["LMHEAD"] = "0"
os.environ.setdefault("FIRST", "0")

import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils import rmsnorm
from axnn.dsl.kernels.utils.linear import linear
import board_io
import q35_step as q

os.environ.setdefault("G", str(len(q.LAYERS)))
import q35_group as Qg
import q35_layer_mc as L
import q35_layer_pk as K
import q35_mx_build as mb

NL, H, CORES, EPS = Qg.G, L.H, L.CORES, L.EPS
HT = int(os.environ.get("HT", H))  # the talker's hidden size (the projection's input)
V = int(q.cfg["vocab_size"])
PROJ = HT != H
assert V % CORES == 0 and H % CORES == 0 and Qg.KINDS == ["attn"] * NL


def step_body(e, packs, w_head, cs, pn, mask, logits, io):
    c = ax.program_id(0)
    mb.SCALES.clear()
    K.BLK.clear()
    xe = ax.dsl.load(e)
    if PROJ:  # each core its H / CORES rows, the row through shared OCM
        pm = Qg.members(packs, "P")
        rs = slice(c * (H // CORES), (c + 1) * (H // CORES))
        pr = linear(xe, pm["w_pin"][rs], None, "proj_in", output_dtype=ax.fp32, oc_tile_size=H // CORES)
        pr = ax.dsl.add(pr, ax.dsl.load(pm["b_pin"][rs]).reshape(1, 1, H // CORES), output_dtype=ax.fp32)
        sh = L.shm_tensor((1, 1, H), ax.bf16, "proj_all")
        ax.dsl.move(r=sh[:, :, rs], x=ax.dsl.cast(pr, dtype=ax.bf16))
        axd.pub(sh)
        axd.sub(sh)
        xt = ax.dsl.create_tensor((1, 1, H), ax.bf16, ax.MemoryType.OCM, name="x_in")
        ax.dsl.move(r=xt, x=sh)
    else:
        xt = xe
    for j in range(NL):
        m = Qg.members(packs, j)
        xt = Qg.attn_layer(c, xt, m, cs, pn, mask, io[j][0], io[j][1], f"L{j}_")
    pm = Qg.members(packs, "P")
    hn = rmsnorm(xt, ax.dsl.load(pm["norm"]).reshape(1, 1, H), eps=EPS)
    rows = slice(c * (V // CORES), (c + 1) * (V // CORES))
    lg = linear(hn, w_head[rows], None, "cp_head", output_dtype=ax.fp32, oc_tile_size=min(256, V // CORES))
    lg = L.keep_shm(lg)  # every shared OCM tensor of the call read once more before the output (KEEP_SHM)
    ax.dsl.store(lg.reshape(1, V // CORES), r=logits[:, rows])


def kernel_source():
    outs = [f"{kv}{j}" for j in range(NL) for kv in ("kc", "vc")]
    packs = list(Qg.PDT)
    params = ", ".join(["e"] + packs + ["w_head", "cs", "pn", "mask", "logits"] + outs)
    io = ", ".join(f"(kc{j}, vc{j})" for j in range(NL))
    return (f"import q3tts_cp_step as S\n\n\ndef cp_step({params}):\n"
            f"    S.step_body(e, dict({', '.join(f'{p}={p}' for p in packs)}), w_head, cs, pn, mask, logits, [{io}])\n"
            f"    return {', '.join(['logits'] + outs)}\n"), outs


def dequant_w(w, s):
    """an MXFP8 (e4m3 + e8m0 per 32) or bf16 weight as f32"""
    if s is None:
        return w.float()
    return mb.dequant(w, s).float()


def build_packs():
    """the predictor's layers (random MXFP8 weights of the config's shapes), the projection and the final norm in
    their packs: (packs, pack sizes, element sizes, the layers' raw args, the extra weights)"""
    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    args, raw = [], []
    for j in range(NL):
        a = mb.mx_args(q.attn_args(W, j), mb.ATTN_W)[0]
        raw.append(a)
        for n in K.BW:
            if n in a:
                Qg.FULL[(j, n)] = a[n].shape[1]
        K.FULL_COLS["attn"] = {n: Qg.FULL[(j, n)] for n in K.BW if n in a}
        args.append(K.block_args(a))
    extra = dict(norm=(torch.randn(H) * 0.1 + 1).float())
    if PROJ:
        extra["w_pin"] = (torch.randn((H, HT)) * HT**-0.5).bfloat16()
        extra["b_pin"] = (torch.randn(H) * 0.02).float()
    order = L.order(mb.attn_layer_mx)
    items = [(f"L{j}.{n}", j, args[j][n]) for j in range(NL) for n in order if torch.is_tensor(args[j].get(n)) and n != "x"]
    items += [(f"LP.{n}", NL - 1, extra[n]) for n in ("w_pin", "b_pin", "norm") if n in extra]
    used = {p: 0 for p in K.PACKS}
    for key, j, tns in items:
        p = K.pack_of(tns)
        es_ = tns.element_size()
        off = (used[p] * es_ + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es_
        Qg.SPEC[key] = (p, off, tuple(tns.shape))
        used[p] = off + tns.numel()
    Qg.PDT.update({p: K.PACKS[p] for p in used if used[p]})
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in Qg.PDT.items()}
    sizes = {p: max(1, (used[p] * es[p] + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es[p]) for p in Qg.PDT}
    pk = {p: torch.zeros(sizes[p], dtype=torch.uint8 if es[p] == 1 else dt) for p, dt in Qg.PDT.items()}
    for key, (p, off, shape) in Qg.SPEC.items():
        lj, n = key.split(".", 1)
        tns = (extra[n] if lj == "LP" else args[int(lj[1:])][n]).reshape(-1)
        tns = tns.view(torch.uint8) if es[p] == 1 else tns
        pk[p][off : off + tns.numel()] = tns
    pk = {p: (pk[p].view(dt) if es[p] == 1 else pk[p]) for p, dt in Qg.PDT.items()}
    return pk, sizes, es, raw, extra


def rms(v, w):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS) * w


def ref_layers(x, pos, raw, kref):
    """torch reference of the layers at position pos on the dequantized weights (f32), x [H]; kref the K / V rows"""
    LMAX, BC, HKV, D = L.LMAX, L.BC, L.HKV, L.D
    cs_, sn_ = L.rope_rows(pos)
    cos_f, sin_f = cs_.float().reshape(-1), sn_.float().reshape(-1)

    def rot(t):  # NEOX rope over the whole head (ROT = D)
        h = t.shape[-1] // 2
        return torch.cat([t[..., :h] * cos_f - t[..., h:] * sin_f, t[..., h:] * cos_f + t[..., :h] * sin_f], -1)

    for j in range(NL):
        a = raw[j]
        wq, wk, wv = (dequant_w(a[f"w_{n}"], a.get(f"s_{n}")) for n in ("q", "k", "v"))
        wo = dequant_w(a["w_o"], a.get("s_o"))
        h = rms(x, a["ln1"].float())
        qh = rot(rms((wq @ h).reshape(L.HQ, D), a["q_norm"].float()))
        kh = rot(rms((wk @ h).reshape(HKV, D), a["k_norm"].float()))
        vh = (wv @ h).reshape(HKV, D)
        kref[j][0][pos], kref[j][1][pos] = kh.reshape(-1), vh.reshape(-1)
        kk_ = kref[j][0][: pos + 1].reshape(pos + 1, HKV, D)
        vv_ = kref[j][1][: pos + 1].reshape(pos + 1, HKV, D)
        att = []
        for hq in range(L.HQ):
            hk = hq // L.G
            sc = (kk_[:, hk] @ qh[hq]) / math.sqrt(D)
            att.append(torch.softmax(sc, 0) @ vv_[:, hk])
        x1 = x + wo @ torch.cat(att)
        h2 = rms(x1, a["ln2"].float())
        wg, wu, wd = (dequant_w(a[f"w_{n}"], a.get(f"s_{n}")) for n in ("gate", "up", "down"))
        g = wg @ h2
        x = x1 + wd @ (g * torch.sigmoid(g) * (wu @ h2))
    return x


def write_kernel(path, out_dir, name, sizes, es):
    """the compiled kernel as <name>_npu<CORES>.axmodel in out_dir with its .layout: "<pack> - 0 <bytes>" per pack,
    "<member> <pack> <offset> <bytes> [blk <rows> <cols> <bw>]" per member"""
    dst = board_io.kernel_file(out_dir, name, CORES)
    shutil.copy(path, dst)
    with open(dst[: -len(".axmodel")] + ".layout", "w") as f:
        for p in Qg.PDT:
            f.write(f"{p} - 0 {sizes[p] * es[p]}\n")
        for key, (p, off, shape) in Qg.SPEC.items():
            lj, n = key.split(".", 1)
            numel = int(np.prod(shape))
            w = n if n.startswith("w_") else "w_" + n[2:]
            blk = ""
            if lj != "LP" and w in K.BW and (int(lj[1:]), w) in Qg.FULL:
                gs = 1 if n == w else mb.GROUP
                cols, bw = Qg.FULL[(int(lj[1:]), w)] // gs, K.BW[w] // gs
                blk = f" blk {numel // cols} {cols} {bw}"
            f.write(f"{key} {p} {off * es[p]} {numel * es[p]}{blk}\n")
    return dst


def main():
    sys.modules["q3tts_cp_step"] = sys.modules[__name__]
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    pk, sizes, es, raw, extra = build_packs()
    w_head = (torch.randn((V, H)) * H**-0.5).bfloat16()
    src, outs = kernel_source()
    build = os.path.abspath(os.environ.get("DUMP", "out_q3tts_cp_step"))
    os.makedirs(f"{build}/k", exist_ok=True)
    with open(f"{build}/k/cp_step_kernel.py", "w") as f:
        f.write(src)
    sys.path.insert(0, f"{build}/k")
    fn = __import__("cp_step_kernel").cp_step
    os.chdir(f"{build}/k")
    PERF = os.environ.get("PERF") == "1"  # simulator trace of the kernel (DUMP_DSL_PERF artifacts in <DUMP>/k)
    kk = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=not PERF, dump_dsl_perf=PERF, dump_path=f"{build}/k")(fn)
    LMAX, BC, HKV, D = L.LMAX, L.BC, L.HKV, L.D
    kv = [[torch.zeros((LMAX, HKV * D), dtype=torch.half) for _ in range(2)] for _ in range(NL)]
    kref = [[torch.zeros((LMAX, HKV * D)) for _ in range(2)] for _ in range(NL)]
    worst = 1.0
    for pos in range(int(os.environ.get("STEPS", 3))):
        e = (torch.randn((1, 1, HT)) * 0.5).bfloat16()
        cs_, sn_ = L.rope_rows(pos)
        mrow = torch.full((LMAX,), -3e38)
        mrow[: pos + 1] = 0
        case = dict(pk, e=e, w_head=w_head, cs=torch.cat([cs_, sn_]), pn=torch.tensor([pos, pos // BC + 1], dtype=torch.int32),
                    mask=mrow.reshape(LMAX // BC, BC).bfloat16(), logits=torch.zeros((1, V)))
        for j in range(NL):
            case[f"kc{j}"], case[f"vc{j}"] = kv[j]
            q.RESIDENT[f"kc{j}"], q.RESIDENT[f"vc{j}"] = kv[j][0].clone(), kv[j][1].clone()
        ax.init(chip_num=1, core_num=CORES)
        mb.SCALES.clear()
        kk[CORES](*[case[n] for n in L.order(fn)])
        q.RESIDENT.clear()
        # torch reference on the dequantized weights (f32)
        x = e.float().reshape(H if not PROJ else HT)
        if PROJ:
            x = (extra["w_pin"].float() @ x + extra["b_pin"]).bfloat16().float()
        x = ref_layers(x, pos, raw, kref)
        ref = w_head.float() @ rms(x, extra["norm"])
        co = float(torch.nn.functional.cosine_similarity(case["logits"].reshape(-1), ref, 0))
        top_same = int(case["logits"].reshape(-1).argmax()) == int(ref.argmax())
        logger.info(f"cp step pos {pos}: logits cos vs torch {co:.6f}, argmax same {top_same}")
        worst = min(worst, co)
    path = max(glob.glob(f"{build}/k/cache/cp_step.*.compiled.axmodel"), key=os.path.getmtime)
    name = f"qwen3tts_cp_step{NL}{K.FMT}_h{HT}_c{LMAX}"  # (HT: the talker's width, the 1.7B's 2048 or the 0.6B's 1024)
    dst = write_kernel(path, out_dir, name, sizes, es)
    logger.info(f"wrote {dst} and its layout (worst logits cos {worst:.6f})")


if __name__ == "__main__":
    main()
