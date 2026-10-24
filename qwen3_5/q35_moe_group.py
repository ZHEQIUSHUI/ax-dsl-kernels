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
# POISON=<byte> (debugging): the simulator's DDR filled with that byte instead of zeros before the inputs are placed, so
# a read of DDR the call never wrote (a swapped-out tensor read before its write) shows as huge values
if os.environ.get("POISON") or os.environ.get("OCM_POISON"):
    import numpy as _np
    import axnn.backend.ax8860.simulator as _sim

    class _NpPoison:
        def __getattr__(self, n):
            return getattr(_np, n)

        def full(self, shape, fill, dtype=None, *a, **kw):  # OCM_POISON=<byte>: local and shared OCM (sim: 0xA5)
            if os.environ.get("OCM_POISON") and fill == 0xA5 and _np.dtype(dtype) == _np.uint8:
                print(f"OCM_POISON: simulator OCM of {shape} filled", flush=True)
                return _np.full(shape, int(os.environ["OCM_POISON"], 0), _np.uint8)
            return _np.full(shape, fill, dtype, *a, **kw)

        def zeros(self, shape, dtype=float, *a, **kw):
            if os.environ.get("POISON") and _np.dtype(dtype) == _np.uint8 and isinstance(shape, tuple) and len(shape) == 1 and shape[0] >= (1 << 20):
                print(f"POISON: simulator memory of {shape[0]} bytes filled", flush=True)
                return _np.full(shape, int(os.environ["POISON"], 0), _np.uint8)
            return _np.zeros(shape, dtype, *a, **kw)

    _sim.np = _NpPoison()
# SHM_NOREUSE=1 (debugging): axnn's shared OCM / shared DDR-swap allocator never frees (no address reused within a
# call); SHM_REPORT=1: log the shared tensors, their sizes and which of them live in DDR swap
if os.environ.get("SHM_NOREUSE") == "1":  # the onepass builder's shared OCM allocator (builder_onepass._Allocator)
    import axnn.dsl.builder_onepass as _bo

    class _ShmNoFree(_bo._Allocator):
        def free(self, *a, **kw):
            if getattr(self, "addr_space", None) == "shmem":
                return None
            return super().free(*a, **kw)

    _bo._Allocator = _ShmNoFree
if False:
    import axnn.yasched.multicore_pass.jobs_allocation as _ja

    if os.environ.get("SHM_NOREUSE") == "1":
        class _NoFree(_ja._Allocator):
            def free(self, *a, **kw):
                pass

        class _NoFreeDDR(_ja._DDRAlloc):
            def free(self, *a, **kw):
                pass

        _ja._Allocator, _ja._DDRAlloc = _NoFree, _NoFreeDDR
    _alloc_shmem = _ja.allocate_shmem

    def _report(states, shmem_tsizes, ddr_tensors, env, graph_results):
        r = _alloc_shmem(states, shmem_tsizes, ddr_tensors, env, graph_results)
        ind = sorted(n for n in shmem_tsizes if n in ddr_tensors)
        logger.info(f"SHM: {len(shmem_tsizes)} shared tensors, {sum(shmem_tsizes.values())} bytes; in DDR swap "
                    f"{len(ind)}: {ind[:40]}; shared DDR {r[1]} bytes")
        return r

    _ja.allocate_shmem = _report
BOARD = os.environ.get("BOARD")
KVDBG = os.environ.get("KVDBG") == "1"  # debugging: an output kv_dbg [G, CORES, 2, HKV * D], each core's K / V rows  # a directory: the case's inputs, the simulator's outputs and the kernel for axrun
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


# KEEP_SHM=1: every shared OCM tensor of the call kept until its end. axnn hands a shared tensor's address to a later
# one once its consumers ran, and pub / sub orders only write before read: a fast core could write the next layer's
# slot over one a slow core still reads. Core 0 reads the first element of each and multiplies out by clip(sum, 1, 1)
# (exactly 1 for any finite or infinite sum: out bitwise unchanged)
KEEP_SHM = os.environ.get("KEEP_SHM") == "1"


SHM_NOREUSE = os.environ.get("SHM_NOREUSE") == "1"


def body(a):
    """the kernel: a = its arguments by name"""
    assert SHM_NOREUSE in (False, True)  # (in the kernel's cache key: the allocator patch changes the build)
    shm, create = [], ax.dsl.create_tensor
    if KEEP_SHM:
        def create_kept(shape, dtype, mem=None, *args, **kw):
            t = create(shape, dtype, mem, *args, **kw)
            if mem == ax.MemoryType.SHARED_OCM:
                shm.append(t)
            return t
        ax.dsl.create_tensor = create_kept
    try:
        body_layers(a)
    finally:
        ax.dsl.create_tensor = create
    if KEEP_SHM and ax.program_id(0) == 0:
        acc = None
        for t in shm:
            v = ax.dsl.cast(t[tuple(slice(0, 1) for _ in t.shape)], dtype=ax.fp32).reshape(1, 1, 1)
            acc = v if acc is None else ax.dsl.add(acc, v, output_dtype=ax.fp32)
        one = ax.dsl.clip(acc, min=1.0, max=1.0)
        o = ax.dsl.load(a["out"])
        ax.dsl.store(ax.dsl.mul(o, one, output_dtype=ax.bf16), r=a["out"])


# LAYER_BARRIER=1: before each layer after the first, every core publishes a value of its previous layer's output and
# waits for all of them, and the layer's input depends on that (x * clip(sum, 1, 1): bitwise x): no core writes the
# next layer's shared OCM while another may still read the previous layer's (shared addresses are reused)
LAYER_BARRIER = os.environ.get("LAYER_BARRIER") == "1"


def layer_barrier(xt, j):
    return ax.dsl.mul(xt, L.shm_barrier(ax.program_id(0), [xt]), output_dtype=ax.bf16)


def body_layers(a):
    m = unpack(a["pw"], a["ps"], a["pf"], a["pb"])
    if LAYER_BARRIER:  # per layer: before it, between its halves, before its experts' partials
        L.BAR = [ax.dsl.create_tensor((3 * G, L.CORES, 1, 1), ax.fp32, ax.MemoryType.SHARED_OCM, name="shm_bar"), 0]
    xt = ax.dsl.load(a["x"])
    for j, kind in enumerate(KINDS):
        L.PFX[0] = f"L{j}_"
        if LAYER_BARRIER and j > 0:
            xt = layer_barrier(xt, j)
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
            if KVDBG:
                L.KV_DBG = (a["kv_dbg"], j)
            xt = L.attn_body(None, w["ln1"], w["w_q"], w["w_k"], w["w_v"], w["q_norm"], w["k_norm"], a["cos"],
                             a["sin"], a[f"kc{j}"], a[f"vc{j}"], a["pos"], a["nblk"], a["mask"], w["w_o"], w["ln2"],
                             w["w_gate"], w["w_up"], w["w_down"], out, moe, xt=xt)
    L.PFX[0] = ""
    if LAYER_BARRIER:  # BAR kept to the end of the call (core 0: out * clip(its first element, 1, 1))
        bar = L.BAR[0]
        L.BAR = None
        if ax.program_id(0) == 0:
            one = ax.dsl.clip(bar[0:1, 0:1].reshape(1, 1, 1), min=1.0, max=1.0)
            ax.dsl.store(ax.dsl.mul(ax.dsl.load(a["out"]), one, output_dtype=ax.bf16), r=a["out"])


def kernel_function():
    ins = ["x", "pw", "ps", "pf", "pb"] + [f"{e}{j}" for j in range(G) for e in ("e_gw", "e_gs", "e_dw", "e_ds")]
    if ANY_ATTN:
        ins += ["cos", "sin", "pos", "nblk", "mask"]
    states = [n for j, kind in enumerate(KINDS) for n in ((f"st{j}", f"cv{j}") if kind == "gdn" else (f"kc{j}", f"vc{j}"))]
    outs = ["out"] + states + (["kv_dbg"] if KVDBG else [])
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
    if KVDBG:
        case["kv_dbg"] = torch.zeros((G, L.CORES, 2, L.HKV * L.D), dtype=torch.bfloat16)
    for j, kind in enumerate(KINDS):
        case.update({f"{k}{j}": v for k, v in ex[j].items()})
        case.update({f"st{j}": st(), f"cv{j}": cv()} if kind == "gdn" else {f"kc{j}": kv(), f"vc{j}": kv()})
    q.RESIDENT.update({n: case[n].clone() for n in case if n[:2] in ("st", "cv", "kc", "vc")})
    before = {n: case[n].clone() for n in names} if BOARD else None
    L.run(L.kern(fn, f"group{G}"), L.CORES, case, names)
    q.RESIDENT.clear()
    if BOARD:  # the case for the board: <IO name>.bin before the run, sim_<name>.bin after it, model.axmodel
        os.makedirs(BOARD, exist_ok=True)
        raw = lambda t: (t.contiguous().view(torch.uint8) if t.dtype != torch.uint8 else t.contiguous()).numpy().tobytes()  # noqa: E731
        for n, t in before.items():
            open(f"{BOARD}/{n}.bin", "wb").write(raw(t))
        for n in names:
            if n in ("out", "kv_dbg") or n[:2] in ("st", "cv", "kc", "vc"):
                open(f"{BOARD}/sim_{n}.bin", "wb").write(raw(case[n]))
        path = max(glob.glob(f"{L.BUILD}/group{G}/cache/moe_group.*.compiled.axmodel"), key=os.path.getmtime)
        shutil.copy(path, f"{BOARD}/model.axmodel")
        board_io.fix(f"{BOARD}/model.axmodel")
        logger.info(f"wrote the board case {BOARD}")
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
