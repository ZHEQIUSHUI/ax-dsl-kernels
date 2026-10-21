"""Qwen3.5 prefill layers on CORES NPU cores without synchronization between them: each call splits its work over the
cores (projections by output channel or by head, the delta rule by head), and the projections whose input is split
(out / o / down) leave one fp32 partial result per core. The residual stream passes between calls as a base plus the
CORES partials of the previous call: x = base + sum_c part[c], computed by every core, stored by core 0.

  mlp_mc<P>   x = base + sum(part); x2 = x + down(silu(gate(h)) * up(h)), h = rmsnorm(x): gate / up rows and the
              matching down columns per core -> base_out = x, part_out[c] = the core's down partial
  gdn_mc<P>   x = base + sum(part); the delta-net half of the layer for NV / CORES value heads per core (their q / k / v
              / z rows, conv channels, state rows; b / a for all heads) -> base_out = x, part_out[c] = the core's
              out-projection partial (its heads' columns); then mlp_mc. P = 1 (decode): the recurrent step per head
              instead of the chunked delta rule
  attn_head_mc<P>   x = base + sum(part) -> base_out = x; per core HQ / CORES query heads (q pre-scaled, gate) and,
              on cores below HKV, KV head c (k / v f16 rows) -> as q35_flash attn_fd_head
  mul_mat_bf16_n<C>_k<K>   the GEMV of ggml/gguf_gemv_kernels.py (x [1, K] bf16, w [C, K] -> out [1, C]
              f32), C / CORES weight rows per core: the output projection in decode
  attn_tail_mc<P>   the core's query heads of the attention output, sigmoid gate, o-projection partial -> part_out[c];
              then mlp_mc with base = the head's base_out

Written as qwen35_<kernel><P>_npu<CORES>.axmodel (mul_mat_bf16_n<C>_k<K>_npu<CORES>.axmodel).
Check on the simulator against torch (fp32 reference). Usage: MODEL=<hf dir> P=64 CORES=4 python q35_mc.py [kernel_dir]
"""
import glob
import os
import shutil
import sys

os.environ.setdefault("KVL", "cell")

import torch
from loguru import logger

import math

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_gate, silu_mul, sigmoid_mul
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.utils.rope import rope
from axnn.dsl.kernels.qwen_gdn_utils import (chunk_prefix_sum_fp32, gdn_gate, gdn_recurrent_step, gdn_stable_chunk,
                                             l2norm_lastdim)
import board_io
import q35_step as q

P = int(os.environ.get("P", 64))
CORES = int(os.environ.get("CORES", 4))
H, F, EPS = q.H, q.F, q.EPS
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_mc"))
FC = F // CORES  # MLP rows per core
NK, NV, DK, DV, KD, VD, CONV = q.NK, q.NV, q.DK, q.DV, q.KD, q.VD, q.CONV
HQ, HKV, D, G, ROT = q.HQ, q.HKV, q.D, q.G, q.ROT
NH = NV // CORES  # delta-net heads per core
QH = HQ // CORES  # query heads per core
CHUNK = min(P, int(os.environ.get("CHUNK", 32)))
NP = int(os.environ.get("NP", 8))  # partial attention slots the tail merges (segments x segment-kernel cores)


def residual_in(base, part):
    """x = base + sum_c part[c] (fp32 sum), bf16 [1, P, H]; core 0 stores it to base_out by the caller"""
    acc = ax.dsl.cast(ax.dsl.load(base), dtype=ax.fp32)
    for c in range(CORES):
        acc = ax.dsl.add(acc, ax.dsl.load(part[c : c + 1]), output_dtype=ax.fp32)
    return ax.dsl.cast(acc, dtype=ax.bf16)


def mlp_mc(base, part, ln2, w_gate, w_up, w_down, base_out, part_out):
    c = ax.program_id(0)
    x = residual_in(base, part)  # [1, P, H]
    if c == 0:
        ax.dsl.store(x, r=base_out)
    h = rmsnorm(x, ax.dsl.load(ln2).reshape(1, 1, H), eps=EPS)
    rows = slice(c * FC, (c + 1) * FC)
    act = silu_mul(q.proj(h, w_gate[rows], "mlp_gate"), q.proj(h, w_up[rows], "mlp_up"),
                   activation_output_dtype=ax.fp32, output_dtype=ax.bf16)  # [1, P, FC]
    down = q.linear(act, w_down[:, rows], None, "mlp_down", output_dtype=ax.fp32, oc_tile_size=q.oc_tile(FC))
    ax.dsl.store(down, r=part_out[c : c + 1])
    return base_out, part_out


def gdn_mc(base, part, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, state_in, conv_in,
           base_out, part_out, state_out, conv_out):
    assert NV == NK and NV % CORES == 0 and (P == 1 or P % CHUNK == 0)
    c = ax.program_id(0)
    x = residual_in(base, part)
    if c == 0:
        ax.dsl.store(x, r=base_out)
    h = rmsnorm(x, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    hs = slice(c * NH, (c + 1) * NH)
    # the core's channels of q, k, v (conv runs per channel), each [1, P, NH * D*]
    mixed = []
    for i, (off, w) in enumerate(((0, DK), (KD, DK), (2 * KD, DV))):
        ch = slice(off + c * NH * w, off + (c + 1) * NH * w)
        pr = q.proj(h, w_qkv[ch], f"gdn_qkv{i}")
        mixed.append(stateful_runtime_causal_depthwise_conv1d_silu(pr, conv_w[ch], conv_in[:, :, ch],
                                                                   conv_out[:, :, ch], CONV, f"gdn_conv{i}"))
    qm = mixed[0].reshape(1, P, NH, DK)
    km = mixed[1].reshape(1, P, NH, DK)
    vm = mixed[2].reshape(1, P, NH, DV)
    z = q.proj(h, w_z[c * NH * DV : (c + 1) * NH * DV], "gdn_z").reshape(1, P, NH, DV)
    b = q.proj(h, w_b, "gdn_b")[:, :, hs]  # all heads (16 rows), then the core's
    a = q.proj(h, w_a, "gdn_a")[:, :, hs]
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    if P == 1:  # decode: the recurrent step per head, as q35_step.gdn_layer
        core = ax.dsl.create_tensor((1, 1, NH, DV), ax.bf16, name="gdn_core")
        a_log_t, dt_bias_t = ax.dsl.load(a_log[hs]), ax.dsl.load(dt_bias[hs])
        for j in range(NH):
            q_tok = ax.dsl.mul(l2norm_lastdim(qm[:, :, j : j + 1, :].reshape(1, DK)), q_scale, output_dtype=ax.bf16)
            k_tok = l2norm_lastdim(km[:, :, j : j + 1, :].reshape(1, DK))
            g, beta = gdn_gate(a[:, :, j : j + 1].reshape(1, 1), b[:, :, j : j + 1].reshape(1, 1), a_log_t[j : j + 1],
                               dt_bias_t[j : j + 1])
            vh = c * NH + j
            st = ax.dsl.load(state_in[vh : vh + 1]).reshape(DK, DV)
            o_h, st = gdn_recurrent_step(q_tok, k_tok, vm[:, :, j : j + 1, :].reshape(1, DV), g, beta, st, ax.bf16)
            ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, :, j : j + 1, :])
            ax.dsl.store(x=st.reshape(1, DK, DV), r=state_out[vh : vh + 1])
        gdn_out(core, z, norm_w, w_out, part_out, c)
        return base_out, part_out, state_out, conv_out
    qn = ax.dsl.mul(l2norm_lastdim(qm), q_scale, output_dtype=ax.bf16)
    kn = l2norm_lastdim(km)
    g_all, beta_all = gdn_gate(a, b, ax.dsl.load(a_log[hs]), ax.dsl.load(dt_bias[hs]))  # [1, P, NH]
    prefix_all = chunk_prefix_sum_fp32(g_all, CHUNK, "gdn_prefix_g")
    exp_prefix_all = ax.dsl.exp(prefix_all, output_dtype=ax.fp32)
    core = ax.dsl.create_tensor((1, P, NH, DV), ax.bf16, name="gdn_core")
    st = ax.dsl.load(state_in[hs]).reshape(NH, DK, DV)
    for c0 in range(0, P, CHUNK):
        cs = slice(c0, c0 + CHUNK)
        q_tile = ax.dsl.transpose(qn[:, cs], perm=(0, 2, 1, 3)).reshape(NH, CHUNK, DK)
        k_tile = ax.dsl.transpose(kn[:, cs], perm=(0, 2, 1, 3)).reshape(NH, CHUNK, DK)
        v_tile = ax.dsl.transpose(vm[:, cs], perm=(0, 2, 1, 3)).reshape(NH, CHUNK, DV)
        prefix_g = ax.dsl.transpose(prefix_all[:, cs], perm=(0, 2, 1)).reshape(NH, CHUNK, 1)
        exp_prefix_g = ax.dsl.transpose(exp_prefix_all[:, cs], perm=(0, 2, 1)).reshape(NH, CHUNK, 1)
        beta = ax.dsl.transpose(beta_all[:, cs], perm=(0, 2, 1)).reshape(NH, CHUNK, 1)
        out_tile, st = gdn_stable_chunk(q_tile, k_tile, v_tile, beta, st, prefix_g, exp_prefix_g, f"gdn_c{c0}",
                                        solver="triton")
        ax.dsl.move(x=ax.dsl.transpose(out_tile.reshape(1, NH, CHUNK, DV), perm=(0, 2, 1, 3)), r=core[:, cs])
    ax.dsl.store(x=st.reshape(NH, DK, DV), r=state_out[hs])
    gdn_out(core, z, norm_w, w_out, part_out, c)
    return base_out, part_out, state_out, conv_out


def gdn_out(core, z, norm_w, w_out, part_out, c):
    """gated RMSNorm of the core's heads, their out-projection partial -> part_out[c]"""
    normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, P, NH * DV)
    o = q.linear(gated, w_out[:, c * NH * DV : (c + 1) * NH * DV], None, "gdn_out", output_dtype=ax.fp32,
                 oc_tile_size=q.oc_tile(NH * DV))
    ax.dsl.store(o, r=part_out[c : c + 1])


def attn_head_mc(base, part, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, base_out, q_out, gate_out, k_out, v_out):
    c = ax.program_id(0)
    x = residual_in(base, part)
    if c == 0:
        ax.dsl.store(x, r=base_out)
    h = rmsnorm(x, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)  # [P, ROT/2]
    qg = q.proj(h, w_q[c * QH * 2 * D : (c + 1) * QH * 2 * D], "attn_q").reshape(P, QH, 2 * D)  # [q | gate]
    qh = rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS)
    qh = rope(qh, cos_t, sin_t, rotary_dim=ROT)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    ax.dsl.store(ax.dsl.mul(qh, scale, output_dtype=ax.bf16), r=q_out[:, c * QH : (c + 1) * QH])
    ax.dsl.store(qg[:, :, D : 2 * D], r=gate_out[:, c * QH : (c + 1) * QH])
    if c < HKV:
        k = rmsnorm(q.proj(h, w_k[c * D : (c + 1) * D], "attn_k").reshape(P, 1, D),
                    ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
        k = rope(k, cos_t, sin_t, rotary_dim=ROT)
        v = q.proj(h, w_v[c * D : (c + 1) * D], "attn_v").reshape(P, D)
        ax.dsl.store(ax.dsl.cast(k.reshape(P, D), dtype=ax.fp16), r=k_out[:, c * D : (c + 1) * D])
        ax.dsl.store(ax.dsl.cast(v, dtype=ax.fp16), r=v_out[:, c * D : (c + 1) * D])
    return base_out, q_out, gate_out, k_out, v_out


def attn_tail_mc(m_all, l_all, o_all, gate, w_o, part_out):
    """The NP partial attention results (segment kernel outputs [NP][HKV][G * P] rows g * P + t; a slot without
    data has m = -3e38) merged for the core's query heads: m = max m_i, att = sum e^(m_i - m) o_i / sum e^(m_i - m)
    l_i; then the gate and the o-projection partial. A row that sees no cell of a slot has m_i near -3e38 there,
    so its weight is 0."""
    c = ax.program_id(0)
    hs = slice(c * QH, (c + 1) * QH)
    hk, g0 = (c * QH) // G, (c * QH) % G  # KV head, first of its query heads in the group
    rows = slice(g0 * P, (g0 + QH) * P)
    mi = ax.dsl.cast(ax.dsl.load(m_all[:, hk : hk + 1, rows]), dtype=ax.fp32).reshape(NP, QH * P, 1)
    li = ax.dsl.load(l_all[:, hk : hk + 1, rows]).reshape(NP, QH * P, 1)
    oi = ax.dsl.load(o_all[:, hk : hk + 1, rows]).reshape(NP, QH * P, D)
    mx = ax.dsl.max(mi, dim=0, keepdim=True)  # [1, QH * P, 1]
    w = ax.dsl.exp(ax.dsl.sub(mi, mx, output_dtype=ax.fp32), output_dtype=ax.fp32)  # [NP, QH * P, 1]
    lsum = ax.dsl.sum(ax.dsl.mul(w, li, output_dtype=ax.fp32), dim=0, output_dtype=ax.fp32)  # [QH * P, 1]
    osum = ax.dsl.sum(ax.dsl.mul(w, oi, output_dtype=ax.fp32), dim=0, output_dtype=ax.fp32)  # [QH * P, D]
    att = ax.dsl.div(osum, lsum, output_dtype=ax.bf16).reshape(QH, P, D)
    att = ax.dsl.transpose(att, perm=(1, 0, 2))  # [P, QH, D]
    gated = sigmoid_mul(att, ax.dsl.load(gate[:, hs]), activation_output_dtype=ax.fp32,
                        output_dtype=ax.bf16).reshape(1, P, QH * D)
    o = q.linear(gated, w_o[:, c * QH * D : (c + 1) * QH * D], None, "attn_o", output_dtype=ax.fp32,
                 oc_tile_size=q.oc_tile(QH * D))
    ax.dsl.store(o, r=part_out[c : c + 1])
    return part_out


LM_C, LM_K = int(os.environ.get("LM_C", 31040)), H  # output projection rows per call (the vocabulary in 8 calls)


def mul_mat_bf16(x, w, out):
    c = ax.program_id(0)
    rows = slice(c * (LM_C // CORES), (c + 1) * (LM_C // CORES))
    o = q.linear(ax.dsl.load(x), w[rows], None, "mm", output_dtype=ax.fp32, oc_tile_size=q.oc_tile(w.shape[1]))
    ax.dsl.store(o, r=out[:, rows])
    return out


MLP_ORDER = ["base", "part", "ln2", "w_gate", "w_up", "w_down", "base_out", "part_out"]
GDN_ORDER = ["base", "part", "ln1", "w_qkv", "w_z", "w_b", "w_a", "conv_w", "a_log", "dt_bias", "norm_w", "w_out",
             "state_in", "conv_in", "base_out", "part_out", "state_out", "conv_out"]
HEAD_ORDER = ["base", "part", "ln1", "w_q", "w_k", "w_v", "q_norm", "k_norm", "cos", "sin", "base_out", "q_out",
              "gate_out", "k_out", "v_out"]
TAIL_ORDER = ["m_all", "l_all", "o_all", "gate", "w_o", "part_out"]


def ref_mlp(a):
    x = (a["base"].float() + a["part"].float().sum(0, keepdim=True)).bfloat16().float()
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * a["ln2"].float()
    g, u = h @ a["w_gate"].float().T, h @ a["w_up"].float().T
    return x, x + (torch.nn.functional.silu(g) * u) @ a["w_down"].float().T


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    os.environ["PCHUNK"] = os.environ["R"] = str(P)
    import q35_flash as fl  # noqa: E402  (R = P from the environment)
    import q35_prefill as pf  # noqa: E402

    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    emb = W("embed_tokens.weight")
    tests = os.environ.get("TEST", "mlp,gdn,head,tail").split(",")
    base = emb[torch.randint(0, 200000, (P,))].reshape(1, P, H).contiguous()
    part = (torch.randn((CORES, P, H)) * 0.02).float()
    x_in = (base.float() + part.sum(0, keepdim=True)).bfloat16()  # the residual both versions see
    zeros = lambda *sh, dt=torch.bfloat16: torch.zeros(sh, dtype=dt)  # noqa: E731
    built = []

    def run(fn, sub, args, order, cores):
        ax.init(chip_num=1, core_num=cores)
        k = kern(fn, sub)
        k[cores](*[args[n] for n in order])
        if cores > 1:
            built.append((sub, fn.__name__, f"qwen35_{fn.__name__}{P}"))

    if "lm" in tests:
        a = dict(x=torch.randn((1, LM_K), dtype=torch.bfloat16), w=(torch.randn((LM_C, LM_K)) * LM_K**-0.5).bfloat16(),
                 out=zeros(1, LM_C, dt=torch.float32))
        ax.init(chip_num=1, core_num=CORES)
        k = kern(mul_mat_bf16, f"lm_npu{CORES}")
        k[CORES](a["x"], a["w"], a["out"])
        logger.info(f"mul_mat_bf16 n{LM_C} k{LM_K} npu{CORES}: cos {cos(a['out'], a['x'].float() @ a['w'].float().T):.6f}")
        built.append((f"lm_npu{CORES}", "mul_mat_bf16", f"mul_mat_bf16_n{LM_C}_k{LM_K}"))

    if "mlp" in tests:
        a = dict(q.mlp_args(W, "layers.0."), base=base, part=part, base_out=zeros(1, P, H),
                 part_out=zeros(CORES, P, H, dt=torch.float32))
        run(mlp_mc, f"mlp{P}_npu{CORES}", a, MLP_ORDER, CORES)
        x_ref, x2_ref = ref_mlp(a)
        x2 = a["base_out"].float() + a["part_out"].sum(0, keepdim=True)
        logger.info(f"mlp_mc P={P} npu{CORES}: base cos {cos(a['base_out'], x_ref):.6f}, out cos {cos(x2, x2_ref):.6f}, "
                    f"delta cos {cos(x2 - a['base_out'].float(), x2_ref - x_ref):.6f}")

    if "gdn" in tests:  # gdn_mc + mlp_mc vs the single-core prefill layer
        st0 = torch.randn((NV, DK, DV)) * 0.02
        cv0 = (torch.randn((1, CONV - 1, 2 * KD + VD)) * 0.1).bfloat16()
        g = q.gdn_args(W, 0)
        ga = {k: v for k, v in g.items() if k not in ("ln2", "w_gate", "w_up", "w_down")}
        a = dict(ga, base=base, part=part, state_in=st0.clone(), conv_in=cv0.clone(), base_out=zeros(1, P, H),
                 part_out=zeros(CORES, P, H, dt=torch.float32), state_out=torch.zeros_like(st0),
                 conv_out=torch.zeros_like(cv0))
        run(gdn_mc, f"gdn{P}_npu{CORES}", a, GDN_ORDER, CORES)
        m = dict(q.mlp_args(W, "layers.0."), base=a["base_out"], part=a["part_out"], base_out=zeros(1, P, H),
                 part_out=zeros(CORES, P, H, dt=torch.float32))
        run(mlp_mc, f"mlp{P}_npu{CORES}", m, MLP_ORDER, CORES)
        out = m["base_out"].float() + m["part_out"].sum(0, keepdim=True)
        r = dict(g, x=x_in, state_in=st0.clone(), conv_in=cv0.clone(), out=zeros(1, P, H),
                 state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
        if P == 1:
            run(q.gdn_layer, "gdn_ref1", r, q.GDN_ORDER, 1)
        else:
            run(pf.gdn_layer_prefill, f"gdn_ref{P}", r, pf.GDN_ORDER, 1)
        logger.info(f"gdn_mc P={P} npu{CORES} vs the single-core layer: out cos {cos(out, r['out']):.6f}, state cos "
                    f"{cos(a['state_out'], r['state_out']):.6f}, conv equal {torch.equal(a['conv_out'], r['conv_out'])}")

    if "head" in tests or "tail" in tests:  # vs q35_flash head (R = P) and torch
        att = dict(q.attn_args(W, 3))
        posn = torch.arange(1000, 1000 + P)
        cs = [q.rope_cos_sin(int(p)) for p in posn]
        cos_t, sin_t = torch.cat([c for c, _ in cs]), torch.cat([s_ for _, s_ in cs])
        outs = lambda: dict(q_out=zeros(P, HQ, D), gate_out=zeros(P, HQ, D), k_out=zeros(P, HKV * D, dt=torch.half),  # noqa: E731
                            v_out=zeros(P, HKV * D, dt=torch.half))
        if "head" in tests:
            a = dict(att, base=base, part=part, cos=cos_t, sin=sin_t, base_out=zeros(1, P, H), **outs())
            run(attn_head_mc, f"head{P}_npu{CORES}", a, HEAD_ORDER, CORES)
            r = dict(att, x=x_in, cos=cos_t, sin=sin_t, **outs())
            run(fl.attn_fd_head, f"head_ref{P}", r, fl.HEAD_ORDER, 1)
            logger.info(f"attn_head_mc P={P} npu{CORES}: " + ", ".join(
                f"{n} cos {cos(a[n], r[n]):.6f}" for n in ("q_out", "gate_out", "k_out", "v_out")) +
                f", base equal {torch.equal(a['base_out'], x_in)}")
        if "tail" in tests:
            # scores over NC cells split into NP - 1 slots (the last one empty), rows hk, g * P + t
            NC = 128 * NP
            sc = torch.randn((HKV, G * P, NC)) * 2
            vv = torch.randn((HKV, NC, D)) * 0.5
            sc[:, :, NC // 2 : NC // 2 + 64] = -3e38  # one slot sees nothing
            ma, la, oa = zeros(NP, HKV, G * P, 1), zeros(NP, HKV, G * P, dt=torch.float32), zeros(NP, HKV, G * P, D)
            ma[:] = -3e38
            per = NC // (NP - 1) // 64 * 64
            for i in range(NP - 1):
                cs_ = slice(i * per, NC if i == NP - 2 else (i + 1) * per)
                mm = sc[:, :, cs_].max(-1, keepdim=True).values
                pp = torch.exp(sc[:, :, cs_] - mm)
                ma[i], la[i], oa[i] = mm.bfloat16(), pp.sum(-1), (pp @ vv[:, cs_]).bfloat16()
            ref_att = torch.softmax(sc, -1) @ vv  # [HKV, G * P, D]
            at = ref_att.reshape(HKV, G, P, D).permute(2, 0, 1, 3).reshape(P, HQ, D)
            gt = (torch.randn((P, HQ, D))).bfloat16()
            a = dict(att, m_all=ma, l_all=la, o_all=oa, gate=gt, part_out=zeros(CORES, P, H, dt=torch.float32))
            run(attn_tail_mc, f"tail{P}_npu{CORES}", a, TAIL_ORDER, CORES)
            ref = (torch.sigmoid(gt.float()) * at.float()).reshape(1, P, HQ * D) @ att["w_o"].float().T
            logger.info(f"attn_tail_mc P={P} npu{CORES}: o cos {cos(a['part_out'].sum(0, keepdim=True), ref):.6f}")

    if out_dir:
        for sub, fn, name in built:
            (path,) = glob.glob(f"{BUILD}/{sub}/cache/{fn}.*.compiled.axmodel")
            shutil.copy(path, board_io.kernel_file(out_dir, name, CORES))
            logger.info(f"wrote {board_io.kernel_file(out_dir, name, CORES)}")


if __name__ == "__main__":
    main()
