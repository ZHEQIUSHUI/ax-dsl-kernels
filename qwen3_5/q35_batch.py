"""Qwen3.5 batched decode kernels: one token of each of B sequences per call, so every weight is read once per B tokens.

  gdn_layer_batch    projections and MLP as B-row matrix products; per sequence its own causal-conv state and
                     delta-rule state: rows b of state_in [B, NV, DK, DV] f32 and conv_in [B, 2*(CONV-1), C] bf16, i.e.
                     B consecutive rows of the llama.cpp recurrent cache (its f32 conv rows hold the bf16 state in the
                     first half), updated in place
  attn_layer_batch   q35_prefill's attention with P = B independent tokens (SCATTER): K/V row b to cell pos[b], RoPE
                     per token, mask [LMAX/BC, B, BC] (token b's KQ mask row), one block loop per KV head for all B
  mul_mat_bf16_batch output projection of B rows: x [1, B, K] bf16, w [C, K] bf16 -> out [1, B, C] f32

Check on the simulator against the decode kernels run once per sequence (own state, own cells). Writes
qwen35_gdn_batch<B>.axmodel, qwen35_attn_batch<B>_kvc_c<LMAX>.axmodel and mul_mat_bf16_b<B>_n<C>_k<H>.axmodel into
kernel_dir. ONLY=attn builds the attention kernel alone (other contexts).

Usage: MODEL=<hf dir> KVL=cell LMAX=4096 BC=64 B=4 python q35_batch.py [kernel_dir]
"""
import glob
import os
import shutil
import sys

B = int(os.environ.get("B", 4))
os.environ["PCHUNK"] = str(B)
os.environ.setdefault("KVL", "cell")

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_gate
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.qwen_gdn_utils import gdn_gate, gdn_recurrent_step, l2norm_lastdim
import q35_step as q
import q35_prefill as pf

H, D, HQ, HKV = q.H, q.D, q.HQ, q.HKV
NK, NV, DK, DV, KD, VD, CONV, EPS = q.NK, q.NV, q.DK, q.DV, q.KD, q.VD, q.CONV, q.EPS
LMAX, BC = q.LMAX, q.BC
C = 2 * KD + VD  # conv channels
HEAD_C = int(os.environ.get("HEAD_C", 31040))  # output projection rows per call
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_batch"))
assert q.KVL == "cell", "batched decode runs on the llama.cpp KV layout (KVL=cell)"
pf.SCATTER = True


def gdn_layer_batch(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                    state_in, conv_in, out, state_out, conv_out):
    xt = ax.dsl.load(x)  # [1, B, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qkv = q.proj(h, w_qkv, "gdn_qkv")  # [1, B, C]
    z = q.proj(h, w_z, "gdn_z").reshape(1, B, NV, DV)
    b = q.proj(h, w_b, "gdn_b")
    a = q.proj(h, w_a, "gdn_a")

    # causal conv per sequence (as the decode kernel's, seq_len 1): state rows 0 .. CONV-2 of each conv_in row
    cin = ax.dsl.load(conv_in)  # [B, 2*(CONV-1), C]
    conv_input = ax.dsl.concat(cin[:, 0 : CONV - 1, :], qkv.reshape(B, 1, C), dim=1)  # [B, CONV, C]
    ax.dsl.store(ax.dsl.concat(conv_input[:, 1:CONV, :], cin[:, CONV - 1 :, :], dim=1), r=conv_out)
    conv_output = ax.dsl.create_tensor((B, 1, C), ax.bf16, name="gdn_conv_output")
    for wo in range(CONV):
        tap = ax.dsl.load(conv_w[:, CONV - 1 - wo : CONV - wo]).reshape(1, 1, C)
        product = ax.dsl.mul(conv_input[:, wo : wo + 1, :], tap, output_dtype=ax.bf16)
        ax.dsl.move(product, r=conv_output, op=None if wo == 0 else "add")
    mixed = ax.dsl.silu(conv_output, output_dtype=ax.bf16, r=conv_output).reshape(1, B, C)
    qm = mixed[:, :, 0:KD].reshape(1, B, NK, DK)
    km = mixed[:, :, KD : 2 * KD].reshape(1, B, NK, DK)
    vm = mixed[:, :, 2 * KD :].reshape(1, B, NV, DV)

    a_log_t = ax.dsl.load(a_log)
    dt_bias_t = ax.dsl.load(dt_bias)
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    core = ax.dsl.create_tensor((1, B, NV, DV), ax.bf16, name="gdn_core")
    rep = NV // NK
    for t in range(B):  # sequence t: its own state row
        for kh in range(NK):
            q_tok = ax.dsl.mul(l2norm_lastdim(qm[:, t : t + 1, kh : kh + 1, :].reshape(1, DK)), q_scale,
                               output_dtype=ax.bf16)
            k_tok = l2norm_lastdim(km[:, t : t + 1, kh : kh + 1, :].reshape(1, DK))
            for vo in range(rep):
                vh = kh * rep + vo
                g, beta = gdn_gate(a[:, t : t + 1, vh : vh + 1].reshape(1, 1), b[:, t : t + 1, vh : vh + 1].reshape(1, 1),
                                   a_log_t[vh : vh + 1], dt_bias_t[vh : vh + 1])
                st = ax.dsl.load(state_in[t : t + 1, vh : vh + 1]).reshape(DK, DV)
                o_h, st = gdn_recurrent_step(q_tok, k_tok, vm[:, t : t + 1, vh : vh + 1, :].reshape(1, DV), g, beta,
                                             st, ax.bf16)
                ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, t : t + 1, vh : vh + 1, :])
                ax.dsl.store(x=st.reshape(1, 1, DK, DV), r=state_out[t : t + 1, vh : vh + 1])

    normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, B, VD)
    x1 = ax.dsl.add(xt, q.proj(gated, w_out, "gdn_out"), output_dtype=ax.bf16)
    q.mlp_residual(x1, ln2, w_gate, w_up, w_down, out)
    return out, state_out, conv_out


def attn_layer_batch(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                     w_down, out):
    pf.attn_layer_prefill(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate,
                          w_up, w_down, out)
    return out, kc, vc


def mul_mat_bf16_batch(x, w, out):
    o = linear(ax.dsl.load(x), w, None, "mm", output_dtype=ax.fp32, oc_tile_size=q.oc_tile(w.shape[1]))
    ax.dsl.store(o, r=out)
    return out


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def gdn_check(W, xs):
    a = q.gdn_args(W, 0)
    st0 = torch.randn((B, NV, DK, DV)) * 0.02
    cv0 = (torch.randn((B, 2 * (CONV - 1), C)) * 0.1).bfloat16()
    k_dec, k_bat = kern(q.gdn_layer, "gdn_dec"), kern(gdn_layer_batch, f"gdn_batch{B}")
    dec_out, dec_st, dec_cv = [], [], []
    for t in range(B):
        case = dict(a, x=xs[:, t : t + 1].contiguous(), state_in=st0[t].clone(),
                    conv_in=cv0[t : t + 1, : CONV - 1].contiguous(), out=torch.zeros((1, 1, H), dtype=torch.bfloat16),
                    state_out=torch.zeros((NV, DK, DV)), conv_out=torch.zeros((1, CONV - 1, C), dtype=torch.bfloat16))
        k_dec[1](*[case[n] for n in q.GDN_ORDER])
        dec_out.append(case["out"].float())
        dec_st.append(case["state_out"])
        dec_cv.append(case["conv_out"])
    case = dict(a, x=xs, state_in=st0.clone(), conv_in=cv0.clone(), out=torch.zeros((1, B, H), dtype=torch.bfloat16),
                state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
    k_bat[1](*[case[n] for n in q.GDN_ORDER])
    bo = case["out"].float()
    logger.info(f"gdn batch B={B}: out min cos {min(cos(bo[:, t], dec_out[t]) for t in range(B)):.6f} "
                f"bitwise {all(torch.equal(bo[:, t : t + 1], dec_out[t]) for t in range(B))} "
                f"state min cos {min(cos(case['state_out'][t], dec_st[t]) for t in range(B)):.6f} "
                f"conv equal {all(torch.equal(case['conv_out'][t : t + 1, : CONV - 1], dec_cv[t]) for t in range(B))} "
                f"conv tail kept {torch.equal(case['conv_out'][:, CONV - 1 :], cv0[:, CONV - 1 :])}")


def attn_check(W, xs):
    """B sequences interleaved in one KV cache: sequence t owns the cells c < C0 with c % B == t, plus its new cell
    C0 + t; its position is its number of earlier cells."""
    a = q.attn_args(W, 3)
    C0 = 4 * BC + 7
    kc0 = (torch.randn((LMAX, HKV * D)) * 0.5).half()
    vc0 = (torch.randn((LMAX, HKV * D)) * 0.5).half()
    cells = [C0 + t for t in range(B)]
    posn = [len(range(t, C0, B)) for t in range(B)]

    def mask_row(t):
        m = torch.full((LMAX,), -3e38)
        m[list(range(t, C0, B)) + [cells[t]]] = 0
        return m

    nblk = max(cells) // BC + 1
    k_dec, k_bat = kern(q.attn_layer, f"attn_dec_kvc_c{LMAX}"), kern(attn_layer_batch, f"attn_batch{B}_kvc_c{LMAX}")
    kc, vc, dec_out = kc0.clone(), vc0.clone(), []
    for t in range(B):
        c, s = q.rope_cos_sin(posn[t])
        case = dict(a, x=xs[:, t : t + 1].contiguous(), cos=c, sin=s, kc=kc, vc=vc,
                    pos=torch.tensor([cells[t]], dtype=torch.int32), nblk=torch.tensor([nblk], dtype=torch.int32),
                    mask=mask_row(t).reshape(LMAX // BC, BC).bfloat16(), out=torch.zeros((1, 1, H), dtype=torch.bfloat16))
        q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
        k_dec[1](*[case[n] for n in q.ATTN_ORDER])
        q.RESIDENT.clear()
        dec_out.append(case["out"].float())
        kc, vc = case["kc"], case["vc"]
    cs = [q.rope_cos_sin(p) for p in posn]
    mask = torch.stack([mask_row(t).reshape(LMAX // BC, BC) for t in range(B)], dim=1).bfloat16()  # [LMAX/BC, B, BC]
    case = dict(a, x=xs, cos=torch.cat([c for c, _ in cs]), sin=torch.cat([s for _, s in cs]), kc=kc0.clone(),
                vc=vc0.clone(), pos=torch.tensor(cells, dtype=torch.int32), nblk=torch.tensor([nblk], dtype=torch.int32),
                mask=mask.contiguous(), out=torch.zeros((1, B, H), dtype=torch.bfloat16))
    q.RESIDENT.update(kc=kc0.clone(), vc=vc0.clone())
    k_bat[1](*[case[n] for n in q.ATTN_ORDER])
    q.RESIDENT.clear()
    bo = case["out"].float()
    logger.info(f"attn batch B={B} LMAX={LMAX}: out min cos {min(cos(bo[:, t], dec_out[t]) for t in range(B)):.6f} "
                f"kv equal {torch.equal(case['kc'], kc) and torch.equal(case['vc'], vc)}")


def head_check(W, xs):
    w = W("embed_tokens.weight")[:HEAD_C].contiguous()
    k = kern(mul_mat_bf16_batch, f"mm_b{B}")
    case = dict(x=xs, w=w, out=torch.zeros((1, B, HEAD_C)))
    k[1](case["x"], case["w"], case["out"])
    ref = xs.float().reshape(B, H) @ w.float().T
    logger.info(f"mul_mat_bf16 B={B} n{HEAD_C}: cos {cos(case['out'].reshape(B, -1), ref):.6f}")


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None  # before the chdirs
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (B,))].reshape(1, B, H).contiguous()
    only = os.environ.get("ONLY")
    if only != "attn":
        gdn_check(W, xs)
        head_check(W, xs)
    attn_check(W, xs)
    if out_dir:
        built = [("attn", f"attn_batch{B}_kvc_c{LMAX}", "attn_layer_batch", f"qwen35_attn_batch{B}_kvc_c{LMAX}")]
        if only != "attn":
            built += [("gdn", f"gdn_batch{B}", "gdn_layer_batch", f"qwen35_gdn_batch{B}"),
                      ("mm", f"mm_b{B}", "mul_mat_bf16_batch", f"mul_mat_bf16_b{B}_n{HEAD_C}_k{H}")]
        for _, sub, fn, dst in built:
            (path,) = glob.glob(f"{BUILD}/{sub}/cache/{fn}.*.compiled.axmodel")
            shutil.copy(path, f"{out_dir}/{dst}.axmodel")
            logger.info(f"wrote {out_dir}/{dst}.axmodel")


if __name__ == "__main__":
    main()
