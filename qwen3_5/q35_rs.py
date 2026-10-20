"""Qwen3.5 GDN prefill layer for speculative-decoding verification with recurrent-state snapshots (llama.cpp n_rs_seq):
P tokens (the delta rule token by token, as q35_prefill GDN=seq) and, besides the final state / conv state (slot 0,
in place as the prefill kernel), the state and the conv state after each earlier token of the chunk:

  state_rs<j> [1, NV, DK, DV] f32, conv_rs<j> [1, CONV - 1, C] bf16, j = 1 .. P - 1: after token P - 1 - j

so a rejected draft rolls back to slot j without recomputation. The host binds each snapshot output to its plane of
the llama.cpp recurrent cache (slot j at row j * mem_size + the sequence's row). Kernel
qwen35_gdn_prefill<P>[_mx8]_rs.

Check on the simulator against the decode kernel run P times: outputs, final state, every snapshot. Usage:
MODEL=<hf dir> KVL=cell PCHUNK=4 [WFMT=mx8] [QW_RANDOM=1] python q35_rs.py [kernel_dir]
"""
import glob
import linecache
import os
import shutil
import sys

os.environ["GDN"] = "seq"
os.environ.setdefault("KVL", "cell")

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, silu_gate
from axnn.dsl.kernels.utils.causal_conv import stateful_runtime_causal_depthwise_conv1d_silu
from axnn.dsl.kernels.qwen_gdn_utils import gdn_gate, gdn_recurrent_step, l2norm_lastdim

import board_io
import q35_prefill as qp

q = qp.q
P, H, NK, NV, DK, DV, KD, VD, EPS = qp.P, qp.H, qp.NK, qp.NV, qp.DK, qp.DV, qp.KD, qp.VD, qp.EPS
C = 2 * KD + VD
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_rs"))
WFMT = os.environ.get("WFMT", "bf16")


def gdn_rs_impl(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                state_in, conv_in, out, state_out, conv_out, state_rs, conv_rs):
    """q35_prefill.gdn_layer_prefill (GDN=seq) plus the snapshots: state_rs[j - 1] / conv_rs[j - 1] after token P-1-j"""
    xt = ax.dsl.load(x)  # [1, P, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qkv = q.proj(h, w_qkv, "gdn_qkv")  # [1, P, C] pre-conv rows
    z = q.proj(h, w_z, "gdn_z").reshape(1, P, NV, DV)
    b = q.proj(h, w_b, "gdn_b")
    a = q.proj(h, w_a, "gdn_a")
    # conv snapshots: the CONV - 1 rows of [conv_in | qkv] ending at token t
    cin = ax.dsl.load(conv_in)  # [1, CONV - 1, C]
    W = q.CONV - 1
    for j in range(1, P):
        t = P - 1 - j
        for r in range(W):
            k = t + 1 + r  # row of [conv_in (W rows) | qkv (P rows)]
            src = cin[:, k : k + 1, :] if k < W else qkv[:, k - W : k - W + 1, :]
            ax.dsl.store(x=src, r=conv_rs[j - 1][:, r : r + 1, :])
    mixed = stateful_runtime_causal_depthwise_conv1d_silu(qkv, conv_w, conv_in, conv_out, q.CONV, "gdn_conv")
    qm = mixed[:, :, 0:KD].reshape(1, P, NK, DK)
    km = mixed[:, :, KD : 2 * KD].reshape(1, P, NK, DK)
    vm = mixed[:, :, 2 * KD :].reshape(1, P, NV, DV)

    a_log_t = ax.dsl.load(a_log)
    dt_bias_t = ax.dsl.load(dt_bias)
    q_scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=DK**-0.5)
    core = ax.dsl.create_tensor((1, P, NV, DV), ax.bf16, name="gdn_core")
    rep = NV // NK
    for kh in range(NK):
        for vo in range(rep):
            vh = vo * NK + kh if q.VORDER == "tiled" else kh * rep + vo
            st = ax.dsl.load(state_in[vh : vh + 1]).reshape(DK, DV)
            for t in range(P):
                q_tok = ax.dsl.mul(l2norm_lastdim(qm[:, t : t + 1, kh : kh + 1, :].reshape(1, DK)), q_scale,
                                   output_dtype=ax.bf16)
                k_tok = l2norm_lastdim(km[:, t : t + 1, kh : kh + 1, :].reshape(1, DK))
                g, beta = gdn_gate(a[:, t : t + 1, vh : vh + 1].reshape(1, 1), b[:, t : t + 1, vh : vh + 1].reshape(1, 1),
                                   a_log_t[vh : vh + 1], dt_bias_t[vh : vh + 1])
                o_h, st = gdn_recurrent_step(q_tok, k_tok, vm[:, t : t + 1, vh : vh + 1, :].reshape(1, DV), g, beta, st,
                                             ax.bf16)
                ax.dsl.move(x=o_h.reshape(1, 1, 1, DV), r=core[:, t : t + 1, vh : vh + 1, :])
                if t < P - 1:  # snapshot slot P - 1 - t
                    ax.dsl.store(x=st.reshape(1, DK, DV), r=state_rs[P - 2 - t][vh : vh + 1])
            ax.dsl.store(x=st.reshape(1, DK, DV), r=state_out[vh : vh + 1])

    normed = rmsnorm(core, ax.dsl.load(norm_w).reshape(1, 1, 1, DV), eps=EPS)
    gated = silu_gate(normed, z, activation_output_dtype=ax.fp32, output_dtype=ax.bf16).reshape(1, P, VD)
    x1 = ax.dsl.add(xt, q.proj(gated, w_out, "gdn_out"), output_dtype=ax.bf16)
    q.mlp_residual(x1, ln2, w_gate, w_up, w_down, out)


def _define(src, name):
    fn = f"<{name}>"
    linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
    exec(compile(src, fn, "exec"), globals())
    return globals()[name]


def _wrapper_src(mx):
    srs = ", ".join(f"state_rs{j}" for j in range(1, P))
    crs = ", ".join(f"conv_rs{j}" for j in range(1, P))
    if mx:
        ws = ("w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out, s_out, ln2, w_gate, "
              "s_gate, w_up, s_up, w_down, s_down")
        pre = ("    mb.SCALES.clear()\n"
               "    for w, sc in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate),"
               " (w_up, s_up), (w_down, s_down)]:\n"
               "        mb.SCALES[id(w)] = sc\n")
    else:
        ws = "w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down"
        pre = ""
    name = f"gdn_rs{'_mx' if mx else ''}"
    return name, f"""
def {name}(x, ln1, {ws}, state_in, conv_in, out, state_out, conv_out, {srs}, {crs}):
{pre}    gdn_rs_impl(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                state_in, conv_in, out, state_out, conv_out, [{srs}], [{crs}])
    return out, state_out, conv_out, {srs}, {crs}
"""


if WFMT == "mx8":
    import q35_mx_build as mb
name, src = _wrapper_src(WFMT == "mx8")
gdn_rs = _define(src, name)


def order(fn):
    return list(fn.__code__.co_varnames[: fn.__code__.co_argcount])


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    mx = WFMT == "mx8"
    a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0] if mx else q.gdn_args(W, 0)
    dec = mb.gdn_layer_mx if mx else q.gdn_layer
    torch.manual_seed(0)
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (P,))].reshape(1, P, H).contiguous()
    st0 = torch.randn((NV, DK, DV)) * 0.02
    cv0 = (torch.randn((1, q.CONV - 1, C)) * 0.1).bfloat16()

    def kern(fn, sub):
        os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
        os.chdir(f"{BUILD}/{sub}")
        return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)

    # reference: the decode kernel token by token
    k_dec = kern(dec, "dec")
    st, cv, outs, sts, cvs = st0.clone(), cv0.clone(), [], [], []
    for t in range(P):
        case = dict(a, x=xs[:, t : t + 1].contiguous(), state_in=st, conv_in=cv,
                    out=torch.zeros((1, 1, H), dtype=torch.bfloat16), state_out=torch.zeros_like(st),
                    conv_out=torch.zeros_like(cv))
        k_dec[1](*[case[n] for n in (mb.order(dec) if mx else q.GDN_ORDER)])
        outs.append(case["out"].float())
        st, cv = case["state_out"], case["conv_out"]
        sts.append(st.clone())
        cvs.append(cv.clone())
    sub = f"rs{P}{'_mx8' if mx else ''}"
    k = kern(gdn_rs, sub)
    case = dict(a, x=xs, state_in=st0.clone(), conv_in=cv0.clone(), out=torch.zeros((1, P, H), dtype=torch.bfloat16),
                state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
    for j in range(1, P):
        case[f"state_rs{j}"] = torch.zeros_like(st0)
        case[f"conv_rs{j}"] = torch.zeros_like(cv0)
    k[1](*[case[n] for n in order(gdn_rs)])
    po = case["out"].float()
    worst = min(qp.cos(po[:, t], outs[t]) for t in range(P))
    snap = [(qp.cos(case[f"state_rs{j}"], sts[P - 1 - j]), torch.equal(case[f"conv_rs{j}"], cvs[P - 1 - j]))
            for j in range(1, P)]
    logger.info(f"gdn rs P={P}: out min cos {worst:.6f}, final state cos {qp.cos(case['state_out'], st):.6f} conv equal "
                f"{torch.equal(case['conv_out'], cv)}, snapshots (state cos, conv equal) {snap}")
    if out_dir:
        path = max(glob.glob(f"{BUILD}/{sub}/cache/{gdn_rs.__name__}.*.compiled.axmodel"), key=os.path.getmtime)
        dst = f"{out_dir}/qwen35_gdn_prefill{P}{'_mx8' if mx else ''}_rs.axmodel"
        shutil.copy(path, dst)
        logger.info(f"wrote {dst} {board_io.fix(dst)[:1]}")


if __name__ == "__main__":
    main()
