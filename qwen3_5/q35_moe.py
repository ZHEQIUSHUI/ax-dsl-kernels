"""Qwen3.5 MoE (35B-A3B and up): single-core decode kernels, every weight a runtime input. Two calls per layer:

  qwen35moe_gdn / qwen35moe_attn_kvc_c<LMAX>   q35_step's GDN / attention layer up to the MLP, then the MoE's dense part:
                               h2 = RMSNorm(x1); out = x1 + sigmoid(h2 . w_shg) * shared_expert(h2) (base);
                               logits = h2 @ w_router.T [1, 1, NE] f32 (bf16 router); outputs out, h2_out, logits_out
  qwen35moe_experts<K>_mxfp4   out = base + sum_i rw[i] * down_i(silu(gate_i h2) * up_i h2), i < K (top-K experts)

The host takes softmax / top-K / renormalization of the logits in between (as transformers' Qwen3_5MoeTopKRouter).

h2 [1, 1, H] bf16 (the post-attention-norm hidden state), base [1, 1, H] bf16 (residual plus the gated shared expert),
rw [1, K] f32 (the renormalized routing weights). The K experts' weights are inputs of their own: w_g<i> / w_u<i> [FE, H]
and w_d<i> [H, FE] MXFP4 (FP4 E2M1, two per byte, low nibble first) with s_* E8M0 scales per 32 inputs. The host picks
the experts and binds their addresses for each call: one layer's experts (256 x 3 MB) are far beyond what one kernel
input may index at run time.

Checks on the simulator: the experts kernel against torch on the dequantized weights; one GDN and one attention layer
(both kernels, NDEC tokens) against transformers' Qwen3_5MoeDecoderLayer with the real weights, its experts replaced by
their MXFP4 round trip. Usage:
MODEL=<hf dir> KVL=cell LMAX=4096 [WFMT=mx8] [NDEC=2 GDN_LAYER=0 ATTN_LAYER=3] python q35_moe.py [kernel_dir]
"""
import glob
import linecache
import os
import shutil
import sys

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_mul
from axnn.dsl.kernels.utils.linear import linear

os.environ.setdefault("KVL", "cell")  # before q35_step: the llama.cpp KV cache layout

import board_io
import q35_step as q

cfg = q.cfg
H, NE, TOPK, FE = q.H, cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
GROUP = 32
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_moe"))
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quant_mxfp4(w):
    """[N, K] -> (packed FP4 [N, K / 2] uint8, low nibble first; E8M0 scale [N, K / 32] uint8), as ggml's MXFP4:
    scale 2^(floor(log2(amax)) - 2) per group, elements rounded to the nearest E2M1 value"""
    n, k = w.shape
    g = w.float().reshape(n, k // GROUP, GROUP)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2, torch.zeros_like(amax))
    x = (g / torch.exp2(e)).reshape(n, k)
    code = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1) | ((x < 0).long() << 3)
    code = code.to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def dequant_mxfp4(packed, scale):
    code = torch.stack([packed & 15, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    v = E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    return (v.reshape(v.shape[0], -1, GROUP) * torch.exp2(scale.float() - 127).unsqueeze(-1)).reshape(v.shape[0], -1)


def oc_tile_fp4(cols):
    return max(16, min(1024, (256 * 1024) // (cols // 2)))  # fp4 tiles around 256 KB


def _experts_src(k):
    ins = ", ".join(f"w_g{i}, s_g{i}, w_u{i}, s_u{i}, w_d{i}, s_d{i}" for i in range(k))
    return f"""
def moe_experts(h2, base, rw, {ins}, out):
    h = ax.dsl.load(h2)  # [1, 1, H] bf16
    r = ax.dsl.load(rw)  # [1, K] f32
    acc = ax.dsl.cast(ax.dsl.load(base), dtype=ax.fp32)
    ws = [{", ".join(f"(w_g{i}, s_g{i}, w_u{i}, s_u{i}, w_d{i}, s_d{i})" for i in range(k))}]
    for i, (wg, sg, wu, su, wd, sd) in enumerate(ws):
        g = linear(h, wg, sg, f"e{{i}}_gate", output_dtype=ax.bf16, oc_tile_size=oc_tile_fp4(H))
        u = linear(h, wu, su, f"e{{i}}_up", output_dtype=ax.bf16, oc_tile_size=oc_tile_fp4(H))
        a = silu_mul(g, u, activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = linear(a, wd, sd, f"e{{i}}_down", output_dtype=ax.fp32, oc_tile_size=oc_tile_fp4(FE))
        acc = ax.dsl.add(acc, ax.dsl.mul(d, r[:, i : i + 1].reshape(1, 1, 1), output_dtype=ax.fp32), output_dtype=ax.fp32)
    ax.dsl.store(ax.dsl.cast(acc, dtype=ax.bf16), r=out)
    return out
"""


def _define(src, name):
    """exec generated kernel source, registered in linecache: the builder reads a kernel's source (inspect)"""
    fn = f"<{name}>"
    linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
    exec(compile(src, fn, "exec"), globals())


def _experts_gu_src(k):
    """as _experts_src with each expert's gate and up rows in one weight: w_c<i> [2 FE, H] (gate rows, then up), one
    projection per expert instead of two (the board pays ~25 us per small projection)"""
    ins = ", ".join(f"w_c{i}, s_c{i}, w_d{i}, s_d{i}" for i in range(k))
    return f"""
def moe_experts_gu(h2, base, rw, {ins}, out):
    h = ax.dsl.load(h2)  # [1, 1, H] bf16
    r = ax.dsl.load(rw)  # [1, K] f32
    acc = ax.dsl.cast(ax.dsl.load(base), dtype=ax.fp32)
    ws = [{", ".join(f"(w_c{i}, s_c{i}, w_d{i}, s_d{i})" for i in range(k))}]
    for i, (wc, sc, wd, sd) in enumerate(ws):
        gu = linear(h, wc, sc, f"e{{i}}_gate_up", output_dtype=ax.bf16, oc_tile_size=oc_tile_fp4(H))
        a = silu_mul(gu[:, :, :FE], gu[:, :, FE:], activation_output_dtype=ax.fp32, output_dtype=ax.bf16)
        d = linear(a, wd, sd, f"e{{i}}_down", output_dtype=ax.fp32, oc_tile_size=oc_tile_fp4(FE))
        acc = ax.dsl.add(acc, ax.dsl.mul(d, r[:, i : i + 1].reshape(1, 1, 1), output_dtype=ax.fp32), output_dtype=ax.fp32)
    ax.dsl.store(ax.dsl.cast(acc, dtype=ax.bf16), r=out)
    return out
"""


_define(_experts_src(TOPK), f"moe_experts{TOPK}")  # moe_experts(h2, base, rw, w_g0, s_g0, ..., out)
_define(_experts_gu_src(TOPK), f"moe_experts_gu{TOPK}")  # moe_experts_gu(h2, base, rw, w_c0, s_c0, w_d0, ..., out)


MOE = {}  # router / shared-expert-gate weights and the extra outputs of the layer kernel being built


def moe_pre(x1, ln2, w_gate, w_up, w_down, out):
    """replaces q35_step's mlp_residual: the shared expert (w_gate / w_up / w_down) with its sigmoid gate on the
    residual (out), plus the post-attention-norm hidden state and the router logits for the experts kernel"""
    h2 = rmsnorm(x1, ax.dsl.load(ln2).reshape(1, 1, H), eps=q.EPS)
    act = silu_mul(q.proj(h2, w_gate, "sh_gate"), q.proj(h2, w_up, "sh_up"), activation_output_dtype=ax.fp32,
                   output_dtype=ax.bf16)
    sh = q.proj(act, w_down, "sh_down", output_dtype=ax.fp32)
    g = ax.dsl.sum(ax.dsl.mul(h2, ax.dsl.load(MOE["w_shg"]).reshape(1, 1, H), output_dtype=ax.fp32), dim=-1,
                   keepdim=True, output_dtype=ax.fp32)
    base = ax.dsl.add(x1, ax.dsl.mul(sh, ax.dsl.sigmoid(g, output_dtype=ax.fp32), output_dtype=ax.fp32),
                      output_dtype=ax.bf16)
    ax.dsl.store(base, r=out)
    ax.dsl.store(h2, r=MOE["h2_out"])
    logits = linear(h2, MOE["w_router"], None, "router", output_dtype=ax.fp32, oc_tile_size=NE)
    ax.dsl.store(logits, r=MOE["logits_out"])


q.mlp_residual = moe_pre  # gdn_layer / attn_layer call it by name


def moe_mlp_args(W, p):
    m = p + "mlp."
    return dict(
        ln2=q.one_plus(W(p + "post_attention_layernorm.weight")),
        w_gate=W(m + "shared_expert.gate_proj.weight"),
        w_up=W(m + "shared_expert.up_proj.weight"),
        w_down=W(m + "shared_expert.down_proj.weight"),
        w_router=W(m + "gate.weight"),
        w_shg=W(m + "shared_expert_gate.weight").reshape(H).contiguous(),
    )


q.mlp_args = moe_mlp_args  # gdn_args / attn_args take the MoE's dense weights


def gdn_layer_moe(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                  w_router, w_shg, state_in, conv_in, out, h2_out, logits_out, state_out, conv_out):
    MOE.update(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out)
    q.gdn_layer(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                state_in, conv_in, out, state_out, conv_out)
    return out, h2_out, logits_out, state_out, conv_out


def attn_layer_moe(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                   w_down, w_router, w_shg, out, h2_out, logits_out):
    MOE.update(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out)
    q.attn_layer(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                 w_down, out)
    return out, kc, vc, h2_out, logits_out


WFMT = os.environ.get("WFMT", "bf16")
if WFMT == "mx8":
    import q35_mx_build as mb  # MXFP8 projections (scales looked up per weight); router and gate stay bf16

    def gdn_layer_moe_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out,
                         s_out, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, state_in, conv_in,
                         out, h2_out, logits_out, state_out, conv_out):
        mb.SCALES.clear()
        for w, sc in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate),
                      (w_up, s_up), (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        gdn_layer_moe(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                      w_router, w_shg, state_in, conv_in, out, h2_out, logits_out, state_out, conv_out)
        return out, h2_out, logits_out, state_out, conv_out

    def attn_layer_moe_mx(x, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o,
                          s_o, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg, out, h2_out,
                          logits_out):
        mb.SCALES.clear()
        for w, sc in [(w_q, s_q), (w_k, s_k), (w_v, s_v), (w_o, s_o), (w_gate, s_gate), (w_up, s_up),
                      (w_down, s_down)]:
            mb.SCALES[id(w)] = sc
        attn_layer_moe(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate,
                       w_up, w_down, w_router, w_shg, out, h2_out, logits_out)
        return out, kc, vc, h2_out, logits_out


def order(fn):
    return list(fn.__code__.co_varnames[: fn.__code__.co_argcount])


LAYER_FN = ({"gdn": gdn_layer_moe_mx, "attn": attn_layer_moe_mx} if WFMT == "mx8" else
            {"gdn": gdn_layer_moe, "attn": attn_layer_moe})


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def experts_check(out_dir):
    torch.manual_seed(0)
    h2 = (torch.randn((1, 1, H))).bfloat16()
    base = (torch.randn((1, 1, H)) * 0.5).bfloat16()
    rw = torch.softmax(torch.randn((1, TOPK)), -1)
    case, ref = dict(h2=h2, base=base, rw=rw.float()), base.float().reshape(H).clone()
    for i in range(TOPK):
        wg, wu, wd = (torch.randn((FE, H)) * 0.02, torch.randn((FE, H)) * 0.02, torch.randn((H, FE)) * 0.02)
        dq = []
        for n, w in (("g", wg), ("u", wu), ("d", wd)):
            pk, sc = quant_mxfp4(w)
            case[f"w_{n}{i}"], case[f"s_{n}{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
            dq.append(dequant_mxfp4(pk, sc))
        x = h2.float().reshape(H)
        g, u = (dq[0] @ x).bfloat16().float(), (dq[1] @ x).bfloat16().float()
        a = (torch.nn.functional.silu(g) * u).bfloat16().float()
        ref += rw[0, i] * (dq[2] @ a)
    case["out"] = torch.zeros((1, 1, H), dtype=torch.bfloat16)
    # the gate-up kernel on the same experts: w_c<i> = gate rows then up rows (packed codes and scales alike)
    for i in range(TOPK):
        for n in ("w", "s"):
            g, u = case[f"{n}_g{i}"].view(torch.uint8), case[f"{n}_u{i}"].view(torch.uint8)
            c = torch.cat([g, u])
            case[f"{n}_c{i}"] = c.view(torch.float4_e2m1fn_x2) if n == "w" else c.view(torch.float8_e8m0fnu)
    for fn, sub, name in ((moe_experts, f"experts{TOPK}", f"qwen35moe_experts{TOPK}_mxfp4"),
                          (moe_experts_gu, f"experts_gu{TOPK}", f"qwen35moe_experts{TOPK}_gu_mxfp4")):
        case["out"] = torch.zeros((1, 1, H), dtype=torch.bfloat16)
        k = kern(fn, sub)
        k[1](*[case[n] for n in order(fn)])
        logger.info(f"{fn.__name__} K={TOPK} FE={FE} H={H}: out cos vs torch {cos(case['out'].float(), ref):.6f}")
        if out_dir:
            path = max(glob.glob(f"{BUILD}/{sub}/cache/{fn.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
            dst = f"{out_dir}/{name}.axmodel"
            shutil.copy(path, dst)
            logger.info(f"wrote {dst}: {board_io.fix(dst)[:2]} ...")


NDEC = int(os.environ.get("NDEC", 2))


def decode_mask(pos):
    m = torch.full((q.LMAX,), -3e38)
    m[: pos + 1] = 0
    return m.reshape(q.LMAX // q.BC, q.BC).bfloat16()


class Experts:
    """one layer's routed experts as the kernels get them: per expert gate / up / down in MXFP4 (+ their dequantized
    values for the transformers reference)"""

    def __init__(self, W, il):
        p = f"layers.{il}.mlp.experts."
        gu, dn = W(p + "gate_up_proj"), W(p + "down_proj")  # [NE, 2 FE, H], [NE, H, FE]
        self.mx, self.gu, self.dn = [], torch.empty_like(gu), torch.empty_like(dn)
        for e in range(NE):
            ent = {}
            for n, w in (("g", gu[e, :FE]), ("u", gu[e, FE:]), ("d", dn[e])):
                ent[n] = quant_mxfp4(w)
            self.mx.append(ent)
            self.gu[e] = torch.cat([dequant_mxfp4(*ent["g"]), dequant_mxfp4(*ent["u"])]).to(gu.dtype)
            self.dn[e] = dequant_mxfp4(*ent["d"]).to(dn.dtype)

    def args(self, ids):
        a = {}
        for i, e in enumerate(ids):
            for n in ("g", "u", "d"):
                pk, sc = self.mx[e][n]
                a[f"w_{n}{i}"], a[f"s_{n}{i}"] = pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)
        return a


def route(logits):
    """transformers' Qwen3_5MoeTopKRouter on the kernel's logits: softmax, top-K, renormalized"""
    pr = torch.softmax(logits.float().reshape(NE), -1)
    v, ids = torch.topk(pr, TOPK)
    return ids.tolist(), (v / v.sum()).reshape(1, TOPK)


def hf_layer(W, il, ex):
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (Qwen3_5MoeDecoderLayer,
                                                                       Qwen3_5MoeTextRotaryEmbedding)

    hc = AutoConfig.from_pretrained(q.MODEL).text_config
    hc._attn_implementation = "eager"
    layer = Qwen3_5MoeDecoderLayer(hc, il).float().eval()
    pre = f"model.language_model.layers.{il}."
    sd = {k[len(pre) :]: W.raw(k).float() for k in W.where if k.startswith(pre)}
    sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = ex.gu.float(), ex.dn.float()  # the MXFP4 round trip
    layer.load_state_dict(sd, strict=True)
    return layer, Qwen3_5MoeTextRotaryEmbedding(hc)


def hf_out(layer, rot, x):
    P = x.shape[1]
    pos = torch.arange(P)[None]
    mask = torch.full((1, 1, P, P), float("-inf")).triu(1) if layer.layer_type == "full_attention" else None
    logits = []
    hook = layer.mlp.gate.register_forward_hook(lambda m, i, o: logits.append(o[0]))
    with torch.no_grad():
        y = layer(x, position_embeddings=rot(x, pos), attention_mask=mask, position_ids=pos)
    hook.remove()
    return (y[0] if isinstance(y, tuple) else y), logits[0].reshape(P, NE)


def layer_check(W, kind, il, k_layer, k_exp, order_l, args, out_dir, name):
    ex = Experts(W, il)
    layer, rot = hf_layer(W, il, ex)
    torch.manual_seed(0)
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (NDEC,))].reshape(1, NDEC, H).contiguous()
    ref, ref_logits = hf_out(layer, rot, xs.float())
    z = lambda *sh, dt=torch.bfloat16: torch.zeros(sh, dtype=dt)  # noqa: E731
    st = dict(state_in=torch.zeros((q.NV, q.DK, q.DV)), conv_in=z(1, q.CONV - 1, 2 * q.KD + q.VD))
    kc, vc = z(q.LMAX, q.HKV * q.D, dt=torch.half), z(q.LMAX, q.HKV * q.D, dt=torch.half)
    for t in range(NDEC):
        case = dict(args, x=xs[:, t : t + 1].contiguous(), out=z(1, 1, H), h2_out=z(1, 1, H),
                    logits_out=torch.zeros((1, 1, NE)))
        if kind == "gdn":
            case.update(st, state_out=torch.zeros_like(st["state_in"]), conv_out=torch.zeros_like(st["conv_in"]))
        else:
            c, sn = q.rope_cos_sin(t)
            case.update(cos=c, sin=sn, kc=kc, vc=vc, pos=torch.tensor([t], dtype=torch.int32),
                        nblk=torch.tensor([t // q.BC + 1], dtype=torch.int32), mask=decode_mask(t))
            q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
        k_layer[1](*[case[n] for n in order_l])
        q.RESIDENT.clear()
        if kind == "gdn":
            st = dict(state_in=case["state_out"], conv_in=case["conv_out"])
        else:
            kc, vc = case["kc"], case["vc"]
        ids, rw = route(case["logits_out"])
        hf_ids = torch.topk(torch.softmax(ref_logits[t].float(), -1), TOPK).indices.tolist()
        ec = dict(ex.args(ids), h2=case["h2_out"], base=case["out"], rw=rw.float(), out=z(1, 1, H))
        k_exp[1](*[ec[n] for n in order(moe_experts)])
        logger.info(f"{kind} layer {il} token {t}: logits cos {cos(case['logits_out'], ref_logits[t]):.6f}, experts "
                    f"{'same as' if sorted(ids) == sorted(hf_ids) else 'differ from'} HF, out cos vs HF "
                    f"{cos(ec['out'].float(), ref[:, t]):.6f}")
    if out_dir:
        path = max(glob.glob(f"{BUILD}/{name}/cache/{LAYER_FN[kind].__name__}.*.compiled.axmodel"), key=os.path.getmtime)
        shutil.copy(path, f"{out_dir}/{name}.axmodel")
        logger.info(f"wrote {out_dir}/{name}.axmodel: {board_io.fix(f'{out_dir}/{name}.axmodel')}")


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    experts_check(out_dir)
    if os.environ.get("ONLY") == "experts":
        return
    k_exp = kern(moe_experts, f"experts{TOPK}")
    W = q.Weights(q.MODEL)
    mx = WFMT == "mx8"
    sfx = "_mx8" if mx else ""
    for kind in ("gdn", "attn"):
        il = int(os.environ.get("GDN_LAYER" if kind == "gdn" else "ATTN_LAYER", 0 if kind == "gdn" else 3))
        a = q.gdn_args(W, il) if kind == "gdn" else q.attn_args(W, il)
        if mx:
            a = mb.mx_args(a, mb.GDN_W if kind == "gdn" else mb.ATTN_W)[0]
        name = f"qwen35moe_gdn{sfx}" if kind == "gdn" else f"qwen35moe_attn{sfx}_kvc_c{q.LMAX}"
        fn = LAYER_FN[kind]
        layer_check(W, kind, il, kern(fn, name), k_exp, order(fn), a, out_dir, name)


if __name__ == "__main__":
    main()
