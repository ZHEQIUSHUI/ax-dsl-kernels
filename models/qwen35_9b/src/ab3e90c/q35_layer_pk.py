"""q35_layer_mc's four-core decode layer kernels (MXFP8, P = 1) with the layer's parameters packed into one input per
dtype instead of one input per tensor: the runtime spends about 7 us per kernel IO on every call (a fused layer kernel
had 25 - 28 of them), while one more load from an input costs next to nothing.

  pw   fp8 e4m3   the w_<x> elements          ps   e8m0   the s_<x> scales
  pf   f32        norms, a_log, dt_bias        pb   bf16   conv_w

each member at an offset rounded up to ALIGN bytes, in the order of the single-core kernel's inputs. The layout is written
next to the kernel as <file>.layout ("member pack offset bytes" per line) and read by the backend, which copies the
converted weights into the packs once. Kernels: <KP>_gdn[_prefill<P>]_mx8_pk_npu<C>, <KP>_attn[_prefill<P>]_mx8_pk_kvc_c<LMAX>_npu<C>
(P from the environment, 1: decode).

Usage: MODEL=<hf dir> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=mx8 [ONLY=gdn|attn] python q35_layer_pk.py <dir>
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
import q35_mx_build as mb
import q35_step as q

assert L.MX and not L.MOE
ALIGN = 64
PACKS = {"pw": torch.float8_e4m3fn, "ps": torch.float8_e8m0fnu, "pf": torch.float32, "pb": torch.bfloat16}
# WFMT4=1: the layer weights in MXFP4 (E2M1 pairs, low nibble first, an E8M0 scale per 32: the 35B experts' format) in
# the pack p4 (torch counts a byte per element pair, the kernel an element per value)
# WFMT=nv4: every projection in NVFP4 instead (q35_mx_build.NV4): elements in p4, E4M3 scales in ps, the FP32 tensor
# scale2 t_<x> in pf; pw stays an (empty) input of another dtype
NVF = mb.NV4
MX4 = os.environ.get("WFMT4", "0") == "1" or NVF
if MX4:
    PACKS = {"p4": torch.float4_e2m1fn_x2, **PACKS}
if NVF:
    PACKS.update(pw=torch.float8_e5m2, ps=torch.float8_e4m3fn)
BYTE_TYPES = (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2, torch.float8_e5m2)
SG = mb.NVG if NVF else mb.GROUP  # inputs per scale


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quant_mxfp4(w):
    """[N, K] -> (E2M1 pairs [N, K / 2] uint8, low nibble first; E8M0 [N, K / 32]) as q35_moe.quant_mxfp4 (ggml's)"""
    n, k = w.shape
    g = w.float().reshape(n, k // mb.GROUP, mb.GROUP)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2, torch.zeros_like(amax))
    x = (g / torch.exp2(e)).reshape(n, k)
    code = ((x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1) | ((x < 0).long() << 3)).to(torch.uint8)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


# MIX=1 (with WFMT4=1): only the MLP (w_gate, w_up, w_down) in MXFP4, the rest MXFP8 (kernels _mx4m)
MIX = os.environ.get("MIX", "0") == "1"
FMT = "_nv4" if NVF else "_mx4m" if MX4 and MIX else "_mx4" if MX4 else "_mx8"
MLP_W = ("w_gate", "w_up", "w_down")


def mx4_args(a, names):
    """the weights of names (bf16 / f32) as MXFP4 w_<x> (fp4 pairs) and s_<x> (e8m0); MIX: the MLP's only, the
    others MXFP8"""
    if MIX:
        a = mb.mx_args(a, [n for n in names if n not in MLP_W])[0]
        names = [n for n in names if n in MLP_W]
    a = dict(a)
    for n in names:
        if NVF and n in a and torch.is_tensor(a[n]):
            a[n], a["s_" + n[2:]], a["t_" + n[2:]] = mb.quant_nvfp4(a[n].float())
            continue
        if n in a and torch.is_tensor(a[n]):
            pk, sc = quant_mxfp4(a[n].float())
            a[n], a["s_" + n[2:]] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
    return a


def dsl_dims(p, off, shape):
    """offset and shape in the kernel's elements (an fp4 byte is two)"""
    if p == "p4":
        return 2 * off, tuple(shape[:-1]) + (2 * shape[-1],)
    return off, tuple(shape)
SPEC = {}  # kind -> {member: (pack, element offset, shape)}, set by layout() before the kernel is built
# BLOCK=1 (decode): the weights the cores slice by columns (w_out / w_o, w_down: each core reads a strip of every row)
# stored as their column strips one after another ([strips * rows, strip width]), so each core reads one contiguous
# block; q.slice_w maps a column strip to its rows
BLOCK = os.environ.get("BLOCK", "0") == "1" and L.P == 1
BW = {"w_down": L.FC, "w_out": L.KHC * L.DV, "w_o": L.QH * L.D}  # strip widths (the cores' column slices)
BLK = {}  # id(blocked weight) -> (rows, strip width)


def blocked(t, bw):
    """[R, C] -> its column strips of width bw stacked: [C / bw * R, bw] (fp4: bytes, bw / 2)"""
    if t.dtype == torch.float4_e2m1fn_x2:
        u = t.view(torch.uint8)
        return blocked(u, bw // 2).view(torch.float4_e2m1fn_x2)
    return torch.cat([t[:, j * bw : (j + 1) * bw] for j in range(t.shape[1] // bw)], dim=0).contiguous()


def block_args(a):
    if not BLOCK:
        return a
    a = dict(a)
    for n, bw in BW.items():
        if n in a:
            a[n] = blocked(a[n], bw)
            a["s_" + n[2:]] = blocked(a["s_" + n[2:]], bw // SG)
    return a


_slice_w = q.slice_w


def slice_blocked(w, rows=slice(None), cols=slice(None)):
    b = BLK.get(id(w))
    if b is None or cols == slice(None):
        return _slice_w(w, rows, cols)
    R, bw = b
    j = cols.start // bw
    assert cols.stop - cols.start == bw and cols.start % bw == 0 and rows == slice(None), (cols, bw)
    ws = w[j * R : (j + 1) * R]
    s = mb.SCALES.get(id(w))
    if s is not None:
        mb.SCALES[id(ws)] = s[j * R : (j + 1) * R]
    if id(w) in mb.SCALES2:
        mb.SCALES2[id(ws)] = mb.SCALES2[id(w)]
    return ws


q.slice_w = slice_blocked


FULL_COLS = {}  # kind -> {blocked member: its column count before blocking}


def pack_of(t):
    return next(p for p, dt in PACKS.items() if dt == t.dtype)


def layout(kind, a, names):
    """the members of kind's kernel (its weight inputs in order) placed in the packs; SPEC[kind] and the pack sizes"""
    spec, used = {}, {p: 0 for p in PACKS}
    for n in names:
        t = a.get(n)
        if not torch.is_tensor(t) or n in ("x",):
            continue
        p = pack_of(t)
        es = t.element_size()
        off = (used[p] * es + ALIGN - 1) // ALIGN * ALIGN // es
        spec[n] = (p, off, tuple(t.shape))
        used[p] = off + t.numel()
    SPEC[kind] = spec
    return {p: (used[p] * torch.tensor([], dtype=dt).element_size() + ALIGN - 1) // ALIGN * ALIGN //
            torch.tensor([], dtype=dt).element_size() for p, dt in PACKS.items()}


def packed(a, kind, sizes):
    """the pack tensors of the members in a"""
    pk = {p: torch.zeros(max(1, sizes[p]), dtype=torch.uint8 if dt in BYTE_TYPES else dt) for p, dt in PACKS.items()}
    for n, (p, off, shape) in SPEC[kind].items():
        t = a[n].reshape(-1)
        if t.dtype in BYTE_TYPES:
            t = t.view(torch.uint8)
        pk[p][off : off + t.numel()] = t
    return {p: (pk[p].view(dt) if dt in BYTE_TYPES else pk[p]) for p, dt in PACKS.items()}


def unpack(kind, pw, ps, pf, pb=None, p4=None):
    src = dict(pw=pw, ps=ps, pf=pf, pb=pb, p4=p4)
    m = {}
    for n, (p, off, shape) in SPEC[kind].items():
        off, shape = dsl_dims(p, off, shape)
        numel = 1
        for s in shape:
            numel *= s
        m[n] = src[p][off : off + numel].reshape(*shape)
    mb.SCALES.clear()
    mb.SCALES2.clear()
    BLK.clear()
    for n in list(m):
        if n.startswith("w_") and "s_" + n[2:] in m:
            mb.SCALES[id(m[n])] = m["s_" + n[2:]]
            if "t_" + n[2:] in m:
                mb.SCALES2[id(m[n])] = m["t_" + n[2:]]
    for n in list(m):
        if BLOCK and n in BW:
            BLK[id(m[n])] = (m[n].shape[0] // (FULL_COLS[kind][n] // BW[n]), BW[n])
    return m


def gdn_layer_mx4_pk(x, p4, pw, ps, pf, pb, state_in, conv_in, out, state_out, conv_out):
    m = unpack("gdn", pw, ps, pf, pb, p4)
    L.gdn_body(x, m["ln1"], m["w_qkv"], m["w_z"], m["w_b"], m["w_a"], m["conv_w"], m["a_log"], m["dt_bias"],
               m["norm_w"], m["w_out"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], state_in, conv_in, out,
               state_out, conv_out)
    return out, state_out, conv_out


def attn_layer_mx4_pk(x, p4, pw, ps, pf, cos, sin, kc, vc, pos, nblk, mask, out):
    m = unpack("attn", pw, ps, pf, None, p4)
    L.attn_body(x, m["ln1"], m["w_q"], m["w_k"], m["w_v"], m["q_norm"], m["k_norm"], cos, sin, kc, vc, pos, nblk, mask,
                m["w_o"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], out)
    return out, kc, vc


def gdn_layer_mx_pk(x, pw, ps, pf, pb, state_in, conv_in, out, state_out, conv_out):
    m = unpack("gdn", pw, ps, pf, pb)
    L.gdn_body(x, m["ln1"], m["w_qkv"], m["w_z"], m["w_b"], m["w_a"], m["conv_w"], m["a_log"], m["dt_bias"],
               m["norm_w"], m["w_out"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], state_in, conv_in, out,
               state_out, conv_out)
    return out, state_out, conv_out


def attn_layer_mx_pk(x, pw, ps, pf, cos, sin, kc, vc, pos, nblk, mask, out):  # no bf16 members
    m = unpack("attn", pw, ps, pf)
    L.attn_body(x, m["ln1"], m["w_q"], m["w_k"], m["w_v"], m["q_norm"], m["k_norm"], cos, sin, kc, vc, pos, nblk, mask,
                m["w_o"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], out)
    return out, kc, vc


def blk_info(kind, n):
    """" blk <rows> <cols> <strip width>" (elements, before blocking) of a blocked member, else """""
    w = n if n.startswith("w_") else "w_" + n[2:]
    if n.startswith("t_"):  # an NVFP4 scale2: one float, not blocked
        return ""
    if not BLOCK or w not in BW or w not in FULL_COLS.get(kind, {}):
        return ""
    g = 1 if n == w else SG
    cols, bw = FULL_COLS[kind][w] // g, BW[w] // g
    if SPEC[kind][n][0] == "p4":  # the backend copies bytes: two fp4 values per byte
        cols, bw = cols // 2, bw // 2
    rows = SPEC[kind][n][2][0] * SPEC[kind][n][2][1] // cols
    return f" blk {rows} {cols} {bw}"


def layout_lines(kind, sizes, prefix=""):
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in PACKS.items()}
    lines = []
    for n, (p, off, shape) in SPEC[kind].items():
        numel = 1
        for s in shape:
            numel *= s
        lines.append(f"{prefix}{n} {p} {off * es[p]} {numel * es[p]}{blk_info(kind, n)}")
    return lines


# IOMIN=1 (decode): fewer IO: the recurrent state and conv state as one output each (read and written in place, as
# the attention's kc / vc), cos / sin as one input cs [2, ROT/2], pos / nblk as one input pn [2]
IOMIN = os.environ.get("IOMIN", "0") == "1" and L.P == 1


def gdn_layer_mx_pk2(x, pw, ps, pf, pb, out, state, conv):
    m = unpack("gdn", pw, ps, pf, pb)
    L.gdn_body(x, m["ln1"], m["w_qkv"], m["w_z"], m["w_b"], m["w_a"], m["conv_w"], m["a_log"], m["dt_bias"],
               m["norm_w"], m["w_out"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], state, conv, out, state, conv)
    return out, state, conv


def attn_layer_mx_pk2(x, pw, ps, pf, cs, pn, mask, out, kc, vc):
    m = unpack("attn", pw, ps, pf)
    L.attn_body(x, m["ln1"], m["w_q"], m["w_k"], m["w_v"], m["q_norm"], m["k_norm"], cs[0:1], cs[1:2], kc, vc,
                pn[0:1], pn[1:2], mask, m["w_o"], m["ln2"], m["w_gate"], m["w_up"], m["w_down"], out)
    return out, kc, vc


def write_layout(kind, dst, sizes):
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in PACKS.items()}
    with open(dst[: -len(".axmodel")] + ".layout", "w") as f:
        for p in PACKS:
            if any(v[0] == p for v in SPEC[kind].values()):
                f.write(f"{p} - 0 {sizes[p] * es[p]}\n")  # the pack itself: its size in bytes
        for line in layout_lines(kind, sizes):
            f.write(line + "\n")


def check(kind, W, out_dir):
    single = L.refs()[0 if kind == "gdn" else 1]  # the single-core kernel (decode or prefill P) of the same IO
    if NVF:  # its NVFP4 twin: t_<x> after each s_<x>
        single = mb.nv_twin(single)
    names = L.order(single)
    conv = (lambda a, names: mx4_args(a, names)) if MX4 else (lambda a, names: mb.mx_args(a, names)[0])
    if kind == "gdn":
        a = conv(q.gdn_args(W, 0), mb.GDN_W)
    else:
        a = conv(q.attn_args(W, q.LAYERS.index("full_attention")), mb.ATTN_W)
    if MX4:
        FULL_COLS[kind] = {n: a[n].shape[1] * (2 if a[n].dtype == torch.float4_e2m1fn_x2 else 1) for n in BW if n in a}
    else:
        FULL_COLS[kind] = {n: a[n].shape[1] for n in BW if n in a}
    ab = block_args(a)  # a stays the single-core reference's
    sizes = layout(kind, ab, names)
    pk = packed(ab, kind, sizes)
    if kind == "attn":
        pk.pop("pb")
    fn = (gdn_layer_mx_pk2 if IOMIN else gdn_layer_mx4_pk if MX4 else gdn_layer_mx_pk) if kind == "gdn" else \
        (attn_layer_mx_pk2 if IOMIN else attn_layer_mx4_pk if MX4 else attn_layer_mx_pk)
    if os.environ.get("PERF"):  # one call of the packed kernel with the simulator trace (DUMP_DSL_PERF artifacts)
        sub = f"{L.BUILD}/{kind}4pk_perf"
        os.makedirs(sub, exist_ok=True)
        os.chdir(sub)
        k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=False, dump_dsl_perf=True, dump_path=sub)(fn)
        case = dict(pk, x=L.tokens(W, 0), out=torch.zeros((1, L.P, L.H), dtype=torch.bfloat16))  # positions 0 .. P-1
        if kind == "gdn":
            st = torch.zeros((L.NV, L.DK, L.DV))
            cv = torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16)
            case.update(state_in=st, conv_in=cv, state_out=torch.zeros_like(st), conv_out=torch.zeros_like(cv))
        else:
            kc, vc = (torch.zeros((L.LMAX, L.HKV * L.D), dtype=torch.half) for _ in range(2))
            cs, sn = L.rope_rows(0)
            case.update(cos=cs, sin=sn, kc=kc, vc=vc, pos=torch.tensor([0], dtype=torch.int32),
                        nblk=torch.tensor([(L.P - 1) // L.BC + 1], dtype=torch.int32), mask=L.mask_rows(0))
            q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
        L.run(k, L.CORES, case, L.order(fn))
        logger.info(f"perf artifacts in {sub}")
        return
    k1, k4 = L.kern(single, f"{kind}1"), L.kern(fn, f"{kind}4pk")
    worst, P, H, LMAX, BC = 1.0, L.P, L.H, L.LMAX, L.BC
    st = {r: torch.zeros((L.NV, L.DK, L.DV)) for r in ("1", "4")}
    cv = {r: torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16) for r in ("1", "4")}
    kv = {r: [torch.zeros((LMAX, L.HKV * L.D), dtype=torch.half) for _ in range(2)] for r in ("1", "4")}
    for step in range(L.STEPS):
        t0 = step * P + int(os.environ.get("T0", 0))  # the first position checked
        x = L.tokens(W, t0)
        res = {}
        for r, k, n, args in (("1", k1, 1, a), ("4", k4, L.CORES, pk)):
            case = dict(args, x=x, out=torch.zeros((1, P, H), dtype=torch.bfloat16))
            if kind == "gdn" and IOMIN and r == "4":
                case.update(state=st[r].clone(), conv=cv[r].clone())
                q.RESIDENT.update(state=st[r].clone(), conv=cv[r].clone())  # in place: the simulator's outputs start zeroed
            elif kind == "gdn":
                case.update(state_in=st[r], conv_in=cv[r], state_out=torch.zeros_like(st[r]),
                            conv_out=torch.zeros_like(cv[r]))
            else:
                cs, sn = L.rope_rows(t0)
                case.update(cos=cs, sin=sn, kc=kv[r][0], vc=kv[r][1], pos=torch.tensor([t0], dtype=torch.int32),
                            nblk=torch.tensor([(t0 + P - 1) // BC + 1], dtype=torch.int32), mask=L.mask_rows(t0))
                if IOMIN and r == "4":
                    case.update(cs=torch.cat([cs, sn]), pn=torch.tensor([t0, (t0 + P - 1) // BC + 1], dtype=torch.int32))
                q.RESIDENT.update(kc=kv[r][0].clone(), vc=kv[r][1].clone())
            os.chdir(f"{L.BUILD}/{kind}{'1' if r == '1' else '4pk'}")
            L.run(k, n, case, L.order(single) if r == "1" else L.order(fn))
            q.RESIDENT.clear()
            if kind == "gdn":
                st[r], cv[r] = (case["state"], case["conv"]) if IOMIN and r == "4" else (case["state_out"], case["conv_out"])
            res[r] = case
        c = L.cos(res["4"]["out"], res["1"]["out"])
        extra = f"state cos {L.cos(st['4'], st['1']):.6f}" if kind == "gdn" else \
            f"K cos {L.cos(kv['4'][0][t0:t0 + P], kv['1'][0][t0:t0 + P]):.6f}"
        logger.info(f"{kind} packed npu{L.CORES} step {step}: out cos {c:.6f}, {extra}")
        worst = min(worst, c)
    path = max(glob.glob(f"{L.BUILD}/{kind}4pk/cache/{fn.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    pre = f"_prefill{P}" if P > 1 else ""
    pk = "_pk2" if IOMIN else "_pk"
    name = f"{q.KP}_gdn{pre}{FMT}{pk}" if kind == "gdn" else f"{q.KP}_attn{pre}{FMT}{pk}_kvc_c{LMAX}"  # layout: blocking
    # the single-core kernel too (the backend binds the layer through it; any weight format)
    p1 = max(glob.glob(f"{L.BUILD}/{kind}1/cache/{single.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
    n1 = f"{q.KP}_gdn{pre}{FMT}" if kind == "gdn" else f"{q.KP}_attn{pre}{FMT}_kvc_c{LMAX}"
    shutil.copy(p1, board_io.kernel_file(out_dir, n1, 1))
    dst = board_io.kernel_file(out_dir, name, L.CORES)
    shutil.copy(path, dst)
    write_layout(kind, dst, sizes)
    logger.info(f"wrote {dst} and its layout (worst cos {worst:.6f}, packs {sizes})")


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    W = q.Weights(q.MODEL)
    only = os.environ.get("ONLY", "gdn,attn").split(",")
    if "gdn" in only and q.HYBRID:
        check("gdn", W, out_dir)
    if "attn" in only:
        check("attn", W, out_dir)


if __name__ == "__main__":
    main()
