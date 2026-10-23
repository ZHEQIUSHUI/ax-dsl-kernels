"""Board check of a packed whole MoE layer kernel (q35_layer_mc gdn_layer_full_pk / attn_layer_full_pk, WFMT=mx8 or
nv4): the inputs the simulator check uses (layer 0's real weights packed, random MXFP4 experts) written as <IO name>.bin
next to the compiled kernel, with the simulator's outputs (sim_<name>.bin); after `axrun model.axmodel <dir>` on the board,
`--cmp <dir>` prints cos / max error per output. Usage: MODEL=<35B> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=nv4
KIND=gdn python layer_board.py <dir> | python layer_board.py --cmp <dir>"""
import glob
import os
import shutil
import sys

import numpy as np
import torch

if sys.argv[1] != "--cmp":
    import board_io
    import q35_layer_mc as L
    import q35_step as q
    mb, mo = L.mb, L.mo


def raw(t):
    t = t.contiguous()
    return (t.view(torch.uint8) if t.dtype != torch.uint8 else t).numpy().tobytes()


def build(d, kind):
    torch.manual_seed(0)
    W = q.Weights(q.MODEL)
    FE, NE, H = mo.FE, mo.NE, L.H
    gw, gs, dw, ds = [], [], [], []
    for _ in range(NE):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk), gs.append(sc)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk), ds.append(sc)
    ex = dict(e_gw=torch.stack(gw).view(torch.float4_e2m1fn_x2), e_gs=torch.stack(gs).view(torch.float8_e8m0fnu),
              e_dw=torch.stack(dw).view(torch.float4_e2m1fn_x2), e_ds=torch.stack(ds).view(torch.float8_e8m0fnu))
    x = L.tokens(W, 0)
    if kind == "gdn":
        a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0]
        io = dict(state_in=torch.randn((L.NV, L.DK, L.DV)) * 0.1,
                  conv_in=(torch.randn((1, L.CONV - 1, 2 * L.KD + L.VD)) * 0.1).bfloat16(),
                  state_out=torch.zeros((L.NV, L.DK, L.DV)),
                  conv_out=torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16))
        full, fpk = L.gdn_layer_full_mx, L.gdn_layer_full_pk
    else:
        a = mb.mx_args(q.attn_args(W, q.LAYERS.index("full_attention")), mb.ATTN_W)[0]
        cs, sn = q.rope_cos_sin(5)
        kv = lambda: (torch.randn((L.LMAX, L.HKV * L.D)) * 0.5).half()  # noqa: E731
        io = dict(cos=cs, sin=sn, kc=kv(), vc=kv(), pos=torch.tensor([5], dtype=torch.int32),
                  nblk=torch.tensor([1], dtype=torch.int32), mask=L.mask_rows(5))
        full, fpk = L.attn_layer_full_mx, L.attn_layer_full_pk
    a["w_shg"] = a["w_shg"].bfloat16()
    sizes = L.pk_layout(kind, a, L.order(full))
    got = dict(L.pk_packed(kind, a, sizes), **ex, x=x, out=torch.zeros((1, 1, H), dtype=torch.bfloat16), **io)
    ins = {n: got[n].clone() for n in L.order(fpk)}  # before the run (in-place state / KV)
    if kind == "attn":
        q.RESIDENT.update(kc=got["kc"].clone(), vc=got["vc"].clone())
    L.run(L.kern(fpk, f"lb_{kind}"), L.CORES, got, L.order(fpk))
    q.RESIDENT.clear()
    os.makedirs(d, exist_ok=True)
    path = max(glob.glob(f"{L.BUILD}/lb_{kind}/cache/{fpk.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    shutil.copy(path, f"{d}/model.axmodel")
    board_io.fix(f"{d}/model.axmodel")
    for n, t in ins.items():
        open(f"{d}/{n}.bin", "wb").write(raw(t))
    for n in ("out", "state_out", "conv_out", "kc", "vc"):
        if n in got:
            open(f"{d}/sim_{n}.bin", "wb").write(raw(got[n]))
    print(f"wrote {d}: inputs {list(ins)}")


def build_gdn_mine(d):
    """the four-core GDN layer kernel of this PCHUNK (P > 1: prefill) with plain MX / NVFP4 inputs, as check_gdn"""
    W = q.Weights(q.MODEL)
    mc = L.mine()[0]
    a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0] if L.MX else q.gdn_args(W, 0)
    case = dict(a, x=L.tokens(W, 0), state_in=torch.randn((L.NV, L.DK, L.DV)) * 0.1,
                conv_in=(torch.randn((1, L.CONV - 1, 2 * L.KD + L.VD)) * 0.1).bfloat16(),
                out=torch.zeros((1, L.P, L.H), dtype=torch.bfloat16), state_out=torch.zeros((L.NV, L.DK, L.DV)),
                conv_out=torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16), **L.moe_outs())
    ins = {n: case[n].clone() for n in L.order(mc) if n not in ("out", "state_out", "conv_out", "h2_out", "logits_out")}
    L.run(L.kern(mc, "lb_gdnp"), L.CORES, case, L.order(mc))
    os.makedirs(d, exist_ok=True)
    path = max(glob.glob(f"{L.BUILD}/lb_gdnp/cache/{mc.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    shutil.copy(path, f"{d}/model.axmodel")
    board_io.fix(f"{d}/model.axmodel")
    for n, t in ins.items():
        open(f"{d}/{n}.bin", "wb").write(raw(t))
    for n in ("out", "state_out", "conv_out", "h2_out", "logits_out"):
        if n in case:
            open(f"{d}/sim_{n}.bin", "wb").write(raw(case[n]))
    print(f"wrote {d}: inputs {list(ins)}")


def cmp(d):
    for f in sorted(glob.glob(f"{d}/sim_*.bin")):
        n = os.path.basename(f)[4:-4]
        b = f"{d}/out_{n}.bin"
        if not os.path.exists(b):
            print(f"{n}: no board output")
            continue
        dt = (np.float32 if n in ("state_out", "logits_out") else np.uint16 if n in ("out", "conv_out", "h2_out")
              else np.float16)
        s, g = np.fromfile(f, dtype=dt), np.fromfile(b, dtype=dt)
        if dt == np.uint16:  # bf16
            s, g = (s.astype(np.uint32) << 16).view(np.float32), (g.astype(np.uint32) << 16).view(np.float32)
        s, g = s.astype(np.float64), g.astype(np.float64)
        c = float(np.dot(s, g) / (np.linalg.norm(s) * np.linalg.norm(g) + 1e-30))
        print(f"{n}: board vs simulator cos {c:.6f}, max abs err {np.abs(s - g).max():.4g} (sim max {np.abs(s).max():.4g}),"
              f" bitwise {bool((s == g).all())}")


if __name__ == "__main__":
    if sys.argv[1] == "--cmp":
        cmp(os.path.abspath(sys.argv[2]))
    else:
        kind = os.environ.get("KIND", "gdn")
        if kind == "gdnmine":
            build_gdn_mine(os.path.abspath(sys.argv[1]))
        else:
            build(os.path.abspath(sys.argv[1]), kind)
