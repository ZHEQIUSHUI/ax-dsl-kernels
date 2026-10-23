"""One four-core decode layer kernel for both layer kinds of a Qwen3.5 model (MXFP8, packed parameters): the runtime
input `sel` picks the gated delta net layer (0) or the attention layer (1) through ax.dsl.switch, so every layer is a
call of the same handle and the backend can queue all of them asynchronously (the runtime refuses another handle's
submission while one is in flight).

Inputs: the packs of q35_layer_pk (each sized for the larger of the two layouts; a layer binds its kind's members at
their offsets), sel int32 [1], the gdn state / conv, the attention's cos / sin / pos / nblk / mask; outputs out,
state_out / conv_out (in place), kc / vc. A layer of one kind binds the other kind's IO to any buffers of their sizes
(not touched). Layout file: "kind member pack offset bytes" lines, "- pack - 0 bytes" for the packs.

Usage: MODEL=<hf dir> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=mx8 python q35_layer_sw.py <dir>
"""
import glob
import os
import shutil
import sys

os.environ.setdefault("WFMT", "mx8")

import torch
from loguru import logger

import axnn.dsl as ax
import board_io
import q35_layer_mc as L
import q35_layer_pk as K
import q35_mx_build as mb
import q35_step as q

assert L.MX and not L.MOE and L.P == 1 and q.HYBRID


def layer_mx_sw(x, pw, ps, pf, pb, sel, state_in, conv_in, cos, sin, pos, nblk, mask, out, state_out, conv_out, kc,
                vc):
    core = ax.program_id(0)  # the switch's branch graphs have no grid index: the bodies get the core id from here
    pid, ct = ax.program_id, ax.dsl.create_tensor
    # nor shared OCM of their own: the bodies' shared partial buffers (part_a, part_m) are created here
    shared = {n: ct((L.CORES, 1, L.H), ax.fp32, ax.MemoryType.SHARED_OCM, name=n) for n in ("part_a", "part_m")}

    def create(shape, dtype, mem=None, name=None, **kw):
        if mem == ax.MemoryType.SHARED_OCM:
            return shared[name]
        return ct(shape, dtype, mem, name=name, **kw) if mem is not None else ct(shape, dtype, name=name, **kw)

    def fixed_core(fn):
        def run(*a):
            ax.program_id = lambda axis: core
            ax.dsl.create_tensor = create
            try:
                return fn(*a)
            finally:
                ax.program_id, ax.dsl.create_tensor = pid, ct
        return run

    @fixed_core
    def gdn_branch(o, so, co, k_, v_):
        m = K.unpack("gdn", pw, ps, pf, pb)
        L.gdn_body(x, m["ln1"], m["w_qkv"], m["w_z"], m["w_b"], m["w_a"], m["conv_w"], m["a_log"], m["dt_bias"],
                   m["norm_w"], m["w_out"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], state_in, conv_in, o, so,
                   co)
        return o, so, co, k_, v_

    @fixed_core
    def attn_branch(o, so, co, k_, v_):
        m = K.unpack("attn", pw, ps, pf)
        L.attn_body(x, m["ln1"], m["w_q"], m["w_k"], m["w_v"], m["q_norm"], m["k_norm"], cos, sin, k_, v_, pos, nblk,
                    mask, m["w_o"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], o)
        return o, so, co, k_, v_

    idx = ax.dsl.load(sel)
    out, state_out, conv_out, kc, vc = ax.dsl.switch(idx, {0: gdn_branch, 1: attn_branch}, out, state_out, conv_out,
                                                     kc, vc)
    return out, state_out, conv_out, kc, vc


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(q.MODEL)
    ag = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0]
    aa = mb.mx_args(q.attn_args(W, q.LAYERS.index("full_attention")), mb.ATTN_W)[0]
    K.FULL_COLS["gdn"] = {n: ag[n].shape[1] for n in K.BW if n in ag}
    K.FULL_COLS["attn"] = {n: aa[n].shape[1] for n in K.BW if n in aa}
    sg = K.layout("gdn", K.block_args(ag), L.order(mb.gdn_layer_mx))
    sa = K.layout("attn", K.block_args(aa), L.order(mb.attn_layer_mx))
    sizes = {p: max(sg[p], sa[p]) for p in K.PACKS}
    pg, pa = K.packed(K.block_args(ag), "gdn", sizes), K.packed(K.block_args(aa), "attn", sizes)
    P, H, LMAX, BC = 1, L.H, L.LMAX, L.BC
    kg, ka = L.kern(mb.gdn_layer_mx, "sw_gdn1"), L.kern(mb.attn_layer_mx, "sw_attn1")
    ksw = L.kern(layer_mx_sw, "sw4")
    st = {r: torch.zeros((L.NV, L.DK, L.DV)) for r in ("1", "4")}
    cv = {r: torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16) for r in ("1", "4")}
    kv = {r: [torch.zeros((LMAX, L.HKV * L.D), dtype=torch.half) for _ in range(2)] for r in ("1", "4")}
    worst = 1.0
    for step in range(L.STEPS):
        x = L.tokens(W, step)
        cs, sn = L.rope_rows(step)
        att = dict(cos=cs, sin=sn, pos=torch.tensor([step], dtype=torch.int32),
                   nblk=torch.tensor([step // BC + 1], dtype=torch.int32), mask=L.mask_rows(step))
        for kind in ("gdn", "attn"):
            res = {}
            for r in ("1", "4"):
                o = torch.zeros((1, P, H), dtype=torch.bfloat16)
                so, co = torch.zeros_like(st[r]), torch.zeros_like(cv[r])
                if r == "1":
                    case = dict(ag if kind == "gdn" else aa, x=x, out=o)
                    if kind == "gdn":
                        case.update(state_in=st[r], conv_in=cv[r], state_out=so, conv_out=co)
                    else:
                        case.update(att, kc=kv[r][0], vc=kv[r][1])
                        q.RESIDENT.update(kc=kv[r][0].clone(), vc=kv[r][1].clone())
                    os.chdir(f"{L.BUILD}/sw_{kind}1")
                    L.run(kg if kind == "gdn" else ka, 1, case, L.order(mb.gdn_layer_mx if kind == "gdn" else
                                                                      mb.attn_layer_mx))
                else:
                    case = dict(pg if kind == "gdn" else pa, x=x, out=o, sel=torch.tensor([0 if kind == "gdn" else 1],
                                                                                            dtype=torch.int32),
                                state_in=st[r], conv_in=cv[r], state_out=so, conv_out=co, kc=kv[r][0], vc=kv[r][1],
                                **att)
                    q.RESIDENT.update(kc=kv[r][0].clone(), vc=kv[r][1].clone())
                    os.chdir(f"{L.BUILD}/sw4")
                    L.run(ksw, L.CORES, case, L.order(layer_mx_sw))
                q.RESIDENT.clear()
                if kind == "gdn":
                    st[r], cv[r] = so, co
                res[r] = o
            c = L.cos(res["4"], res["1"])
            logger.info(f"switch kernel step {step} {kind}: out cos {c:.6f}" +
                        (f", state cos {L.cos(st['4'], st['1']):.6f}" if kind == "gdn" else
                         f", K cos {L.cos(kv['4'][0][step:step + 1], kv['1'][0][step:step + 1]):.6f}"))
            worst = min(worst, c)
    path = max(glob.glob(f"{L.BUILD}/sw4/cache/layer_mx_sw.*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, f"{q.KP}_layer_mx8_sw_kvc_c{LMAX}", L.CORES)
    shutil.copy(path, dst)
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in K.PACKS.items()}
    with open(dst[: -len(".axmodel")] + ".layout", "w") as f:
        for p in K.PACKS:
            f.write(f"- {p} - 0 {sizes[p] * es[p]}\n")
        for kind in ("gdn", "attn"):
            for line in K.layout_lines(kind, sizes, prefix=kind + " "):
                f.write(line + "\n")
    logger.info(f"wrote {dst} and its layout (worst cos {worst:.6f})")


if __name__ == "__main__":
    main()
