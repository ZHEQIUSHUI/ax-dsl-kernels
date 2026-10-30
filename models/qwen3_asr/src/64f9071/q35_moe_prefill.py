"""Qwen3.5 MoE (35B-A3B) prefill kernels: P tokens per call.

  qwen35moe_gdn_prefill<P> / qwen35moe_attn_prefill<P>_kvc_c<LMAX>
      q35_prefill's GDN / attention layer on P tokens up to the MLP, then the MoE's dense part as the decode kernels
      (q35_moe.moe_pre on P rows): outputs out (base) [1, P, H], h2_out [1, P, H], logits_out [1, P, NE] f32
  qwen35moe_slots<S>_g<G>_gu_mxfp4
      the routed experts of P tokens grouped by expert: S slots, each one expert (w_c<s> gate-up [2 FE, H] and w_d<s>
      [H, FE] MXFP4 with s_* E8M0 scales, the layout of qwen35moe_experts<K>_gu_mxfp4) on up to G token rows:
      out[s, g] = rw[s, g] * down_s(silu(gate_s x[s, g]) * up_s x[s, g])   x [S, G, H] bf16, rw [S, G] f32,
      out [S, G, H] f32 (SLOT_OUT=bf16: bf16, named _bo: half the bytes for the host to read back). The host gathers
      each slot's rows from h2, binds the slot's expert weights, and adds the rows
      of out to the tokens' base (unused rows: rw 0)

Checks on the simulator: the layer kernels on P tokens against the decode kernels (q35_moe) token by token; the slots
kernel against torch on the dequantized weights. Usage:
MODEL=<hf dir> KVL=cell LMAX=4096 PCHUNK=64 [NODEC=1] [ONLY=slots|layers] [SLOTS=32 SLOT_G=8]
python q35_moe_prefill.py [kernel_dir]
"""
import glob
import linecache
import os
import shutil
import sys

import torch
from loguru import logger

os.environ.setdefault("KVL", "cell")

import axnn.dsl as ax
from axnn.dsl.kernels.utils import silu_mul
from axnn.dsl.kernels.utils.linear import linear

import board_io
import q35_moe as mo  # patches q35_step's MLP with the MoE dense part (moe_pre), before q35_prefill uses it
import q35_prefill as qp

q = mo.q
H, NE, FE, P, LMAX, BC = mo.H, mo.NE, mo.FE, qp.P, qp.LMAX, qp.BC
S, G = int(os.environ.get("SLOTS", 32)), int(os.environ.get("SLOT_G", 8))
OUT_BF = os.environ.get("SLOT_OUT") == "bf16"  # the slots kernel's out in bf16 (qwen35moe_slots..._bo)
CORES = int(os.environ.get("CORES", 1))  # slots kernel: S / CORES slots per core, no synchronization (_npu<CORES>)
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_moe_prefill"))
NODEC = os.environ.get("NODEC") == "1"


def gdn_layer_prefill_moe(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up,
                          w_down, w_router, w_shg, state_in, conv_in, out, h2_out, logits_out, state_out, conv_out):
    mo.MOE.update(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out)
    qp.gdn_layer_prefill(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up,
                         w_down, state_in, conv_in, out, state_out, conv_out)
    return out, h2_out, logits_out, state_out, conv_out


def attn_layer_prefill_moe(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate,
                           w_up, w_down, w_router, w_shg, out, h2_out, logits_out):
    mo.MOE.update(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out)
    qp.attn_layer_prefill(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate,
                          w_up, w_down, out)
    return out, kc, vc, h2_out, logits_out


if mo.WFMT == "mx8":  # MXFP8 projections (q35_mx_build via q35_moe); router and gate stay bf16
    mb = mo.mb

    def gdn_layer_prefill_moe_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w,
                                 w_out, s_out, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg,
                                 state_in, conv_in, out, h2_out, logits_out, state_out, conv_out):
        mb.SCALES.clear()
        for w, sc in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate),
                      (w_up, s_up), (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        gdn_layer_prefill_moe(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up,
                              w_down, w_router, w_shg, state_in, conv_in, out, h2_out, logits_out, state_out, conv_out)
        return out, h2_out, logits_out, state_out, conv_out


def _slots_src(n):
    ins = ", ".join(f"w_c{i}, s_c{i}, w_d{i}, s_d{i}" for i in range(n))
    return f"""
def moe_slots(x, rw, {ins}, out):
    c = ax.program_id(0)
    ws = [{", ".join(f"(w_c{i}, s_c{i}, w_d{i}, s_d{i})" for i in range(n))}]
    for i, (wc, sc, wd, sd) in enumerate(ws):
        if i // (S // CORES) != c:  # the core's slots
            continue
        h = ax.dsl.load(x[i : i + 1])  # [1, G, H] bf16
        r = ax.dsl.load(rw[i : i + 1])  # [1, G] f32
        gu = linear(h, wc, sc, f"s{{i}}_gate_up", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
        a = silu_mul(gu[:, :, :FE], gu[:, :, FE:], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = linear(a, wd, sd, f"s{{i}}_down", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(FE))
        ax.dsl.store(ax.dsl.mul(d, r.reshape(1, G, 1), output_dtype=OUT_DT), r=out[i : i + 1])
    return out
"""


OUT_DT = ax.bf16 if OUT_BF else ax.fp32


def moe_slots_pl(x, rw, sid, e_gw, e_gs, e_dw, e_ds, out):
    """moe_slots with the layer's planar expert tensors (all NE experts: e_gw / e_gs gate-up codes / scales
    [NE, 2 FE, H], e_dw / e_ds down [NE, H, FE]) and each slot's expert sid [S] int32, picked by index_select: seven
    IO for any S instead of four per slot"""
    c = ax.program_id(0)
    for i in range(c * (S // CORES), (c + 1) * (S // CORES)):
        e = ax.dsl.load(sid[i : i + 1])
        wc = ax.dsl.index_select(e_gw, e, dim=0).reshape(2 * FE, H)
        sc = ax.dsl.index_select(e_gs, e, dim=0).reshape(2 * FE, H // 32)
        wd = ax.dsl.index_select(e_dw, e, dim=0).reshape(H, FE)
        sd = ax.dsl.index_select(e_ds, e, dim=0).reshape(H, FE // 32)
        h = ax.dsl.load(x[i : i + 1])  # [1, G, H] bf16
        r = ax.dsl.load(rw[i : i + 1])  # [1, G] f32
        gu = linear(h, wc, sc, f"s{i}_gate_up", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
        a = silu_mul(gu[:, :, :FE], gu[:, :, FE:], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = linear(a, wd, sd, f"s{i}_down", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(FE))
        ax.dsl.store(ax.dsl.mul(d, r.reshape(1, G, 1), output_dtype=OUT_DT), r=out[i : i + 1])
    return out


def _define(src, name):
    """as q35_moe._define, in this module (the source refers to its names)"""
    fn = f"<{name}>"
    linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
    exec(compile(src, fn, "exec"), globals())


_define(_slots_src(S), f"moe_slots{S}")  # moe_slots(x, rw, w_c0, s_c0, w_d0, s_d0, ..., out)


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def publish(sub, fn, out_dir, name, cores=1):
    if out_dir:
        path = max(glob.glob(f"{BUILD}/{sub}/cache/{fn.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
        dst = board_io.kernel_file(out_dir, name, cores)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}: {board_io.fix(dst)[:2]} ...")


T_GROUP = int(os.environ.get("T_GROUP", 2048))  # tokens of a group (h2 / base / x rows of the whole group)


def moe_slots_px(h2, tok, rw, e_gw, e_gs, e_dw, e_ds, out):
    """moe_slots_pl reading its rows out of the group's h2 [T, H] by token (tok [S, G] int32, unused rows any token
    with rw 0): the host no longer gathers the rows"""
    c = ax.program_id(0)
    for i in range(c * (S // CORES), (c + 1) * (S // CORES)):
        e = ax.dsl.load(sid_of(tok, i))
        wc = ax.dsl.index_select(e_gw, e, dim=0).reshape(2 * FE, H)
        sc = ax.dsl.index_select(e_gs, e, dim=0).reshape(2 * FE, H // 32)
        wd = ax.dsl.index_select(e_dw, e, dim=0).reshape(H, FE)
        sd = ax.dsl.index_select(e_ds, e, dim=0).reshape(H, FE // 32)
        h = ax.dsl.index_select(h2, ax.dsl.load(tok[i : i + 1, 1:]).reshape(G), dim=0).reshape(1, G, H)  # [1, G, H] bf16
        r = ax.dsl.load(rw[i : i + 1])  # [1, G] f32
        gu = linear(h, wc, sc, f"s{i}_gate_up", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
        a = silu_mul(gu[:, :, :FE], gu[:, :, FE:], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = linear(a, wd, sd, f"s{i}_down", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(FE))
        ax.dsl.store(ax.dsl.mul(d, r.reshape(1, G, 1), output_dtype=OUT_DT), r=out[i : i + 1])
    return out


def sid_of(tok, i):
    return tok[i : i + 1, 0:1].reshape(1)  # tok row i: [expert, G token ids]


RB = int(os.environ.get("COMBINE_ROWS", 32))  # tokens per pass of the combine kernel


def moe_combine(base, loc, sout, out):
    """x = base + the token's K expert results: loc [K, T] int32 rows of sout [R, H] (the slots kernels' outputs of
    the group, weighted already), out [T, H] bf16; the cores take T / CORES tokens each, RB at a time"""
    c = ax.program_id(0)
    TC = T_GROUP // CORES
    K = loc.shape[0]
    for t0 in range(c * TC, (c + 1) * TC, RB):
        acc = ax.dsl.cast(ax.dsl.load(base[t0 : t0 + RB]), dtype=ax.fp32)  # [RB, H]
        for k in range(K):
            ids = ax.dsl.load(loc[k : k + 1, t0 : t0 + RB]).reshape(RB)
            rows = ax.dsl.index_select(sout, ids, dim=0)  # [RB, H]
            acc = ax.dsl.add(acc, ax.dsl.cast(rows, dtype=ax.fp32), output_dtype=ax.fp32)
        ax.dsl.store(ax.dsl.cast(acc, dtype=ax.bf16), r=out[t0 : t0 + RB])
    return out


def slots_px_check(out_dir):
    """the h2-gathering slots kernel and the combine kernel against torch (dequantized experts)"""
    torch.manual_seed(0)
    T, K = T_GROUP, mo.TOPK
    h2 = torch.randn((T, H)).bfloat16()
    gw, gs, dw, ds, dq = [], [], [], [], {}
    for e in range(NE):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk), gs.append(sc)
        pk2, sc2 = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk2), ds.append(sc2)
    ex = dict(e_gw=torch.stack(gw).view(torch.float4_e2m1fn_x2), e_gs=torch.stack(gs).view(torch.float8_e8m0fnu),
              e_dw=torch.stack(dw).view(torch.float4_e2m1fn_x2), e_ds=torch.stack(ds).view(torch.float8_e8m0fnu))
    tok = torch.zeros((S, 1 + G), dtype=torch.int32)
    tok[:, 0] = torch.randint(0, NE, (S,))
    tok[:, 1:] = torch.randint(0, T, (S, G))
    rw = torch.rand((S, G))
    out = torch.zeros((S, G, H), dtype=torch.bfloat16 if OUT_BF else torch.float32)
    ax.init(chip_num=1, core_num=CORES)
    case = dict(h2=h2, tok=tok, rw=rw, out=out, **ex)
    bo = "_bo" if OUT_BF else ""
    kern(moe_slots_px, f"slots{S}_g{G}_px{bo}_npu{CORES}")[CORES](*[case[n] for n in mo.order(moe_slots_px)])
    ref = torch.zeros((S, G, H))
    for i in range(S):
        e = int(tok[i, 0])
        wcd, wdd = mo.dequant_mxfp4(gw[e], gs[e]), mo.dequant_mxfp4(dw[e], ds[e])
        xi = h2[tok[i, 1:].long()].float()
        gu = (xi @ wcd.T).bfloat16().float()
        a = (torch.nn.functional.silu(gu[:, :FE]) * gu[:, FE:]).bfloat16().float()
        ref[i] = rw[i].reshape(G, 1) * (a @ wdd.T)
    logger.info(f"moe slots px S={S} G={G}: out cos vs torch {mo.cos(case['out'], ref):.6f}")
    publish(f"slots{S}_g{G}_px{bo}_npu{CORES}", moe_slots_px, out_dir, f"qwen35moe_slots{S}_g{G}_px_mxfp4{bo}", CORES)
    ax.init(chip_num=1, core_num=1)
    combine_check(out_dir)


def combine_check(out_dir):
    """the combine kernel against torch"""
    torch.manual_seed(1)
    T, K = T_GROUP, mo.TOPK
    ax.init(chip_num=1, core_num=CORES)
    # combine: T tokens, their K rows among R slot rows
    R = int(os.environ.get("COMBINE_R", S * G))  # the group's slot rows (all its slots calls)
    sout = torch.randn((R, H)).bfloat16()
    base = torch.randn((T, H)).bfloat16()
    loc = torch.randint(0, R, (K, T), dtype=torch.int32)
    xo = torch.zeros((T, H), dtype=torch.bfloat16)
    cc = dict(base=base, loc=loc, sout=sout, out=xo)
    kern(moe_combine, f"combine_t{T}_npu{CORES}")[CORES](*[cc[n] for n in mo.order(moe_combine)])
    refc = base.float() + sout.float()[loc.long()].sum(0)
    logger.info(f"moe combine T={T} K={K} R={R}: out cos vs torch {mo.cos(cc['out'], refc):.6f}")
    publish(f"combine_t{T}_npu{CORES}", moe_combine, out_dir, f"qwen35moe_combine{K}_t{T}_r{R}", CORES)
    ax.init(chip_num=1, core_num=1)


def slots_pl_check(out_dir):
    """the planar slots kernel against the per-slot one on the same experts"""
    torch.manual_seed(0)
    x = torch.randn((S, G, H)).bfloat16()
    rw = torch.rand((S, G))
    gw, gs, dw, ds = [], [], [], []
    for e in range(NE):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk), gs.append(sc)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk), ds.append(sc)
    sid = torch.randint(0, NE, (S,), dtype=torch.int32)
    case = dict(x=x, rw=rw, sid=sid, e_gw=torch.stack(gw).view(torch.float4_e2m1fn_x2),
                e_gs=torch.stack(gs).view(torch.float8_e8m0fnu), e_dw=torch.stack(dw).view(torch.float4_e2m1fn_x2),
                e_ds=torch.stack(ds).view(torch.float8_e8m0fnu),
                out=torch.zeros((S, G, H), dtype=torch.bfloat16 if OUT_BF else torch.float32))
    ref = dict(x=x, rw=rw, out=torch.zeros_like(case["out"]))
    for i in range(S):
        e = int(sid[i])
        ref[f"w_c{i}"], ref[f"s_c{i}"] = gw[e].view(torch.float4_e2m1fn_x2), gs[e].view(torch.float8_e8m0fnu)
        ref[f"w_d{i}"], ref[f"s_d{i}"] = dw[e].view(torch.float4_e2m1fn_x2), ds[e].view(torch.float8_e8m0fnu)
    ax.init(chip_num=1, core_num=CORES)
    bo = "_bo" if OUT_BF else ""
    k = kern(moe_slots, f"slots{S}_g{G}{bo}_npu{CORES}")
    k[CORES](*[ref[n] for n in mo.order(moe_slots)])
    kp = kern(moe_slots_pl, f"slots{S}_g{G}_pl{bo}_npu{CORES}")
    kp[CORES](*[case[n] for n in mo.order(moe_slots_pl)])
    ax.init(chip_num=1, core_num=1)
    same = bool(torch.equal(case["out"].view(torch.int16) if OUT_BF else case["out"], ref["out"].view(torch.int16) if OUT_BF else ref["out"]))
    logger.info(f"moe slots planar S={S} G={G}: out cos vs per-slot kernel {mo.cos(case['out'], ref['out']):.6f} "
                f"(bitwise {same})")
    publish(f"slots{S}_g{G}_pl{bo}_npu{CORES}", moe_slots_pl, out_dir, f"qwen35moe_slots{S}_g{G}_pl_mxfp4{bo}", CORES)


def slots_check(out_dir):
    torch.manual_seed(0)
    x = torch.randn((S, G, H)).bfloat16()
    rw = torch.rand((S, G))
    rw[S // 2 :, G // 2 :] = 0  # unused rows
    case = dict(x=x, rw=rw, out=torch.zeros((S, G, H), dtype=torch.bfloat16 if OUT_BF else torch.float32))
    ref = torch.zeros((S, G, H))
    for i in range(S):
        dq = []
        for n, shape in (("c", (2 * FE, H)), ("d", (H, FE))):
            pk, sc = mo.quant_mxfp4(torch.randn(shape) * 0.02)
            case[f"w_{n}{i}"], case[f"s_{n}{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
            dq.append(mo.dequant_mxfp4(pk, sc))
        xi = x[i].float()  # [G, H]
        gu = (xi @ dq[0].T).bfloat16().float()
        a = (torch.nn.functional.silu(gu[:, :FE]) * gu[:, FE:]).bfloat16().float()
        ref[i] = rw[i].reshape(G, 1) * (a @ dq[1].T)
    ax.init(chip_num=1, core_num=CORES)
    k = kern(moe_slots, f"slots{S}_g{G}{'_bo' if OUT_BF else ''}_npu{CORES}")
    k[CORES](*[case[n] for n in mo.order(moe_slots)])
    ax.init(chip_num=1, core_num=1)
    logger.info(f"moe slots S={S} G={G}: out cos vs torch {mo.cos(case['out'], ref):.6f}, unused rows zero "
                f"{bool((case['out'][S // 2 :, G // 2 :] == 0).all())}")
    bo = "_bo" if OUT_BF else ""
    publish(f"slots{S}_g{G}{bo}_npu{CORES}", moe_slots, out_dir, f"qwen35moe_slots{S}_g{G}_gu_mxfp4{bo}", CORES)


def layer_check(W, kind, il, out_dir):
    a = q.gdn_args(W, il) if kind == "gdn" else q.attn_args(W, il)
    mx = mo.WFMT == "mx8"
    if mx:
        assert kind == "gdn", "MXFP8 MoE prefill: the GDN layer only (attention: split kernels)"
        a = mb.mx_args(a, mb.GDN_W)[0]
    torch.manual_seed(0)
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (P,))].reshape(1, P, H).contiguous()
    z = lambda *sh, dt=torch.bfloat16: torch.zeros(sh, dtype=dt)  # noqa: E731
    pre_fn = (gdn_layer_prefill_moe_mx if mx else gdn_layer_prefill_moe) if kind == "gdn" else attn_layer_prefill_moe
    name = f"qwen35moe_gdn_prefill{P}{'_mx8' if mx else ''}" if kind == "gdn" else f"qwen35moe_attn_prefill{P}_kvc_c{LMAX}"
    k_pre = kern(pre_fn, name)
    k_dec = None if NODEC else kern(mo.LAYER_FN[kind], f"dec_{kind}")
    pos0 = 70
    dec = []
    if kind == "gdn":
        st0 = torch.randn((q.NV, q.DK, q.DV)) * 0.02
        cv0 = (torch.randn((1, q.CONV - 1, 2 * q.KD + q.VD)) * 0.1).bfloat16()
        st, cv = st0.clone(), cv0.clone()
        for t in range(0 if NODEC else P):
            case = dict(a, x=xs[:, t : t + 1].contiguous(), state_in=st, conv_in=cv, out=z(1, 1, H), h2_out=z(1, 1, H),
                        logits_out=torch.zeros((1, 1, NE)), state_out=torch.zeros_like(st), conv_out=torch.zeros_like(cv))
            k_dec[1](*[case[n] for n in mo.order(mo.LAYER_FN[kind])])
            dec.append(case)
            st, cv = case["state_out"], case["conv_out"]
        case = dict(a, x=xs, state_in=st0.clone(), conv_in=cv0.clone(), out=z(1, P, H), h2_out=z(1, P, H),
                    logits_out=torch.zeros((1, P, NE)), state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
        k_pre[1](*[case[n] for n in mo.order(pre_fn)])
        extra = "" if NODEC else f", state cos {mo.cos(case['state_out'], st):.6f}"
    else:
        kc0 = (torch.randn((LMAX, q.HKV * q.D)) * 0.5).half()
        vc0 = (torch.randn((LMAX, q.HKV * q.D)) * 0.5).half()
        kc, vc = kc0.clone(), vc0.clone()
        for t in range(0 if NODEC else P):
            p = pos0 + t
            c, sn = q.rope_cos_sin(p)
            case = dict(a, x=xs[:, t : t + 1].contiguous(), cos=c, sin=sn, kc=kc, vc=vc,
                        pos=torch.tensor([p], dtype=torch.int32), nblk=torch.tensor([p // BC + 1], dtype=torch.int32),
                        mask=qp.decode_mask(p), out=z(1, 1, H), h2_out=z(1, 1, H), logits_out=torch.zeros((1, 1, NE)))
            q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
            k_dec[1](*[case[n] for n in mo.order(mo.LAYER_FN[kind])])
            q.RESIDENT.clear()
            dec.append(case)
            kc, vc = case["kc"], case["vc"]
        cs = [q.rope_cos_sin(pos0 + t) for t in range(P)]
        case = dict(a, x=xs, cos=torch.cat([c for c, _ in cs]), sin=torch.cat([s for _, s in cs]), kc=kc0.clone(),
                    vc=vc0.clone(), pos=torch.tensor([pos0], dtype=torch.int32),
                    nblk=torch.tensor([(pos0 + P - 1) // BC + 1], dtype=torch.int32), mask=qp.prefill_mask(pos0),
                    out=z(1, P, H), h2_out=z(1, P, H), logits_out=torch.zeros((1, P, NE)))
        q.RESIDENT.update(kc=kc0.clone(), vc=vc0.clone())
        k_pre[1](*[case[n] for n in mo.order(pre_fn)])
        q.RESIDENT.clear()
        extra = ""
    if NODEC:
        logger.info(f"{kind} prefill P={P}: built, outputs finite "
                    f"{all(bool(case[n].float().isfinite().all()) for n in ('out', 'h2_out', 'logits_out'))}")
    else:
        worst = {n: min(mo.cos(case[n][:, t], dec[t][n][:, 0]) for t in range(P)) for n in ("out", "h2_out", "logits_out")}
        logger.info(f"{kind} prefill P={P} vs decode: min cos {', '.join(f'{n} {v:.6f}' for n, v in worst.items())}{extra}")
    publish(name, pre_fn, out_dir, name)


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    only = os.environ.get("ONLY")
    if only == "slots_pl":
        slots_pl_check(out_dir)
        return
    if only == "slots_px":
        slots_px_check(out_dir)
        return
    if only == "combine":
        combine_check(out_dir)
        return
    if only != "layers":
        slots_check(out_dir)
    if only == "slots":
        return
    W = q.Weights(q.MODEL)
    kinds = os.environ.get("KINDS", "gdn,attn").split(",")
    if "gdn" in kinds:
        layer_check(W, "gdn", int(os.environ.get("GDN_LAYER", 0)), out_dir)
    if "attn" in kinds:
        layer_check(W, "attn", int(os.environ.get("ATTN_LAYER", 3)), out_dir)


if __name__ == "__main__":
    main()
