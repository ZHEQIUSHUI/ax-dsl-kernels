"""Several consecutive decode layers of a Qwen3.5 MoE model (35B-A3B) in one four-core kernel: the whole-layer bodies of
q35_layer_mc.py (the layer, the router, top-K and the experts picked out of the layer's planar expert tensors, all on
the NPU) one after another, each layer's output on chip as the next one's input, every named tensor of layer j
prefixed L<j>_ (its own shared OCM slots). One call per group of G layers instead of G.

Inputs: x [1, 1, H] (the group's input); the packs of all G layers as q35_layer_mc's packed layers (pw fp8, ps e8m0,
pf f32, pb bf16; layer j's members "L<j>.<name>", column-split weights in per-core blocks); per layer j its expert
planes e_gw<j> / e_gs<j> / e_dw<j> / e_ds<j>; per gated delta net layer j its recurrent state st<j> and conv state
cv<j>, per attention layer kc<j> / vc<j> (read and written in place); cos, sin, pos, nblk, mask for the attention
layers. Output out (the group's output). Written as qwen35moe_group<G>_<kinds>_mx8_kvc_c<LMAX>_npu<C> (kinds g / a per
layer) with a .layout file as q35_layer_mc's packed kernels.

Checked on the simulator against the layers' whole-layer kernels called one after another (layers FIRST .., real
weights, random MXFP4 experts). Usage: MODEL=<35B dir> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=mx8 G=4 [FIRST=0]
[NOREF=1] python q35_moe_group.py <dir>
"""
import glob
import importlib
import os
import shutil
import sys

os.environ.setdefault("WFMT", "mx8")  # or nv4: NVFP4 projections

import torch
from loguru import logger

import axnn.dsl as ax
import board_io
import q35_layer_mc as L
import q35_mx_build as mb
import q35_step as q

assert L.MX and L.MOE and L.P == 1
mo = L.mo
G = int(os.environ.get("G", 4))
FIRST = int(os.environ.get("FIRST", 0))
NOREF = os.environ.get("NOREF") == "1"
KINDS = ["gdn" if q.LAYERS[FIRST + j] == "linear_attention" else "attn" for j in range(G)]
ANY_ATTN = "attn" in KINDS
H, NV, DK, DV, KD, VD, CONV = L.H, L.NV, L.DK, L.DV, L.KD, L.VD, L.CONV
SPEC = {}  # "L<j>.<name>" -> (pack, element offset, shape, column blocks)
LAYER_FN = {"gdn": L.gdn_layer_full_mx, "attn": L.attn_layer_full_mx}  # the per-layer kernels (names, reference)
SKIP = ("x", "out", "state_in", "conv_in", "state_out", "conv_out", "kc", "vc", "cos", "sin", "pos", "nblk", "mask")


def layout(args):
    """layer j's parameters (args[j]) placed in the packs, in the order of its whole-layer kernel's inputs"""
    used = {p: 0 for p in L.PK_PACKS}
    for j, kind in enumerate(KINDS):
        for n in L.order(LAYER_FN[kind]):
            t = args[j].get(n)
            if n in SKIP or n.startswith("e_") or not torch.is_tensor(t):
                continue
            p = next(k for k, dt in L.PK_PACKS.items() if dt == t.dtype)
            es = t.element_size()
            off = (used[p] * es + L.PK_ALIGN - 1) // L.PK_ALIGN * L.PK_ALIGN // es
            SPEC[f"L{j}.{n}"] = (p, off, tuple(t.shape), L.pk_blocks(n))
            used[p] = off + t.numel()
    return {p: (used[p] * torch.tensor([], dtype=dt).element_size() + L.PK_ALIGN - 1) // L.PK_ALIGN * L.PK_ALIGN //
            torch.tensor([], dtype=dt).element_size() for p, dt in L.PK_PACKS.items()}


def packed(args, sizes):
    flat = {f"L{j}.{n}": t for j in range(G) for n, t in args[j].items()}
    L.PK_SPEC["group"] = SPEC
    return L.pk_packed("group", flat, sizes)


def unpack(pw, ps, pf, pb):
    """every layer's members (the column-split ones as L.Cols), their scales (and NVFP4 scale2) registered"""
    L.PK_SPEC["group"] = SPEC
    src = dict(pw=pw, ps=ps, pf=pf, pb=pb)
    m = [{} for _ in range(G)]
    for key, (p, off, shape, bl) in SPEC.items():
        j, n = key.split(".", 1)
        m[int(j[1:])][n] = L.pk_member(src, p, off, shape, bl)
    regs = []
    for j in range(G):  # pk_register clears the registries: collect each layer's, then register them all
        L.pk_register(m[j])
        regs.append((dict(mb.SCALES), dict(mb.SCALES2)))
    mb.SCALES.clear()
    mb.SCALES2.clear()
    for sc, s2 in regs:
        mb.SCALES.update(sc)
        mb.SCALES2.update(s2)
    return m


def body(a):
    """the kernel: a = its arguments by name"""
    m = unpack(a["pw"], a["ps"], a["pf"], a["pb"])
    xt = ax.dsl.load(a["x"])
    for j, kind in enumerate(KINDS):
        L.PFX[0] = f"L{j}_"
        w = m[j]
        moe = dict(w_router=w["w_router"], w_shg=w["w_shg"], e_gw=a[f"e_gw{j}"], e_gs=a[f"e_gs{j}"],
                   e_dw=a[f"e_dw{j}"], e_ds=a[f"e_ds{j}"])
        out = a["out"] if j == G - 1 else None
        if kind == "gdn":
            st, cv = a[f"st{j}"], a[f"cv{j}"]
            xt = L.gdn_body(None, w["ln1"], w["w_qkv"], w["w_z"], w["w_b"], w["w_a"], w["conv_w"], w["a_log"],
                            w["dt_bias"], w["norm_w"], w["w_out"], w["ln2"], w["w_gate"], w["w_up"], w["w_down"], st,
                            cv, out, st, cv, moe, xt=xt)
        else:
            xt = L.attn_body(None, w["ln1"], w["w_q"], w["w_k"], w["w_v"], w["q_norm"], w["k_norm"], a["cos"],
                             a["sin"], a[f"kc{j}"], a[f"vc{j}"], a["pos"], a["nblk"], a["mask"], w["w_o"], w["ln2"],
                             w["w_gate"], w["w_up"], w["w_down"], out, moe, xt=xt)
    L.PFX[0] = ""


def kernel_function():
    ins = ["x", "pw", "ps", "pf", "pb"] + [f"{e}{j}" for j in range(G) for e in ("e_gw", "e_gs", "e_dw", "e_ds")]
    if ANY_ATTN:
        ins += ["cos", "sin", "pos", "nblk", "mask"]
    states = [n for j, kind in enumerate(KINDS) for n in ((f"st{j}", f"cv{j}") if kind == "gdn" else (f"kc{j}", f"vc{j}"))]
    outs = ["out"] + states
    src = (f"import q35_moe_group as S\n\n\ndef moe_group({', '.join(ins + outs)}):\n    S.body(locals())\n"
           f"    return {', '.join(outs)}\n")
    os.makedirs(L.BUILD, exist_ok=True)
    open(f"{L.BUILD}/moe_group_kernel.py", "w").write(src)
    sys.path.insert(0, L.BUILD)
    return importlib.import_module("moe_group_kernel").moe_group, ins + outs


def experts():
    """random MXFP4 experts of every layer: the planes"""
    FE, NE = mo.FE, mo.NE
    gw, gs, dw, ds = [], [], [], []
    for e in range(NE):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk), gs.append(sc)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk), ds.append(sc)
    return dict(e_gw=torch.stack(gw).view(torch.float4_e2m1fn_x2), e_gs=torch.stack(gs).view(torch.float8_e8m0fnu),
                e_dw=torch.stack(dw).view(torch.float4_e2m1fn_x2), e_ds=torch.stack(ds).view(torch.float8_e8m0fnu))


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    args, ex = [], []
    for j, kind in enumerate(KINDS):
        il = FIRST + j
        a = q.gdn_args(W, il) if kind == "gdn" else q.attn_args(W, il)
        a = mb.mx_args(a, mb.GDN_W if kind == "gdn" else mb.ATTN_W)[0]
        a["w_shg"] = a["w_shg"].bfloat16()
        args.append(a)
        ex.append(experts() if j < 2 or not NOREF else ex[j % 2])
    sizes = layout(args)
    fn, names = kernel_function()
    x = L.tokens(W, 0)
    cs, sn = q.rope_cos_sin(0)
    att = dict(cos=cs, sin=sn, pos=torch.tensor([0], dtype=torch.int32), nblk=torch.tensor([1], dtype=torch.int32),
               mask=L.mask_rows(0))
    st = lambda: torch.zeros((NV, DK, DV))  # noqa: E731
    cv = lambda: torch.zeros((1, CONV - 1, 2 * KD + VD), dtype=torch.bfloat16)  # noqa: E731
    kv = lambda: torch.zeros((L.LMAX, L.HKV * L.D), dtype=torch.half)  # noqa: E731
    case = dict(packed(args, sizes), x=x, out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **(att if ANY_ATTN else {}))
    for j, kind in enumerate(KINDS):
        case.update({f"{k}{j}": v for k, v in ex[j].items()})
        case.update({f"st{j}": st(), f"cv{j}": cv()} if kind == "gdn" else {f"kc{j}": kv(), f"vc{j}": kv()})
    q.RESIDENT.update({n: case[n].clone() for n in case if n[:2] in ("st", "cv", "kc", "vc")})
    L.run(L.kern(fn, f"group{G}"), L.CORES, case, names)
    q.RESIDENT.clear()
    if not NOREF:  # the layers one after another, each its whole-layer kernel
        h = x
        for j, kind in enumerate(KINDS):
            fnl = LAYER_FN[kind]
            r = dict(args[j], **ex[j], x=h, out=torch.zeros((1, 1, H), dtype=torch.bfloat16))
            r.update(dict(state_in=st(), conv_in=cv(), state_out=st(), conv_out=cv()) if kind == "gdn" else
                     dict(att, kc=kv(), vc=kv()))
            if kind == "attn":
                q.RESIDENT.update(kc=r["kc"].clone(), vc=r["vc"].clone())
            L.run(L.kern(fnl, f"ref_{kind}"), L.CORES, r, L.order(fnl))
            q.RESIDENT.clear()
            h = r["out"]
        logger.info(f"group of {G} ({''.join(k[0] for k in KINDS)}) npu{L.CORES}: out cos vs the layers one by one "
                    f"{L.cos(case['out'], h):.6f} (bitwise {bool(torch.equal(case['out'].view(torch.int16), h.view(torch.int16)))})")
    else:
        logger.info(f"group of {G}: no reference, out finite {bool(case['out'].float().isfinite().all())}")
    path = max(glob.glob(f"{L.BUILD}/group{G}/cache/moe_group.*.compiled.axmodel"), key=os.path.getmtime)
    name = f"qwen35moe_group{G}_{''.join(k[0] for k in KINDS)}{L.SFX}_kvc_c{L.LMAX}"
    dst = board_io.kernel_file(out_dir, name, L.CORES)
    shutil.copy(path, dst)
    board_io.fix(dst)
    L.PK_SPEC["group"] = SPEC
    L.pk_write_layout("group", dst, sizes)
    logger.info(f"wrote {dst} and its layout (packs {sizes})")


if __name__ == "__main__":
    sys.modules["q35_moe_group"] = sys.modules["__main__"]  # the generated kernel imports this module: the same SPEC
    main()
