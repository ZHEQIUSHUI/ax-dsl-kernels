"""Qwen3.5 MoE (35B-A3B) attention split for long contexts (flash decoding, as q35_flash.py): R tokens per call.

  qwen35_attn_fd_head<R>[_mx8], qwen35_attn_fd_seg<R>_c<S>   q35_flash's head and segment kernels (no MoE part)
  qwen35moe_attn_fd_tail<R>[_mx8]   attention output (the host's merge of the segments) -> sigmoid gate -> o_proj ->
                                    the MoE's dense part (q35_moe.moe_pre): outputs out (base), h2_out, logits_out

Check on the simulator (CHECK=1, S <= 16384): R tokens at cells in segment 1, history in segments 0 and 1: head,
segments, merge and tail against the fused MoE attention prefill kernel (q35_moe_prefill, scattered cells) with
LMAX = 2 S. Usage:
MODEL=<hf dir> R=4 S=2048 BC=64 [WFMT=mx8] python q35_moe_flash.py [kernel_dir]   (CHECK=0: compile only, S = 32768)
"""
import glob
import os
import shutil
import sys

R = int(os.environ.get("R", 4))
S = int(os.environ.get("S", 4096))
CHECK = os.environ.get("CHECK", "1") == "1" and 2 * S <= 32768
# as q35_flash sets them, before q35_moe imports q35_step
os.environ["PCHUNK"] = str(R)
os.environ["KVL"] = "cell"
os.environ["LMAX"] = str(2 * S if CHECK else S)

import torch
from loguru import logger

import axnn.dsl as ax

import board_io
import q35_moe as mo  # patches q35_step's MLP with the MoE dense part
import q35_flash as fl
import q35_moe_prefill as mp

q = mo.q
H, D, HQ, HKV, G, BC, NE = q.H, q.D, q.HQ, q.HKV, q.G, q.BC, mo.NE
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_moe_flash"))


def attn_fd_tail_moe(x, attn, gate, w_o, ln2, w_gate, w_up, w_down, w_router, w_shg, out, h2_out, logits_out):
    mo.MOE.update(w_router=w_router, w_shg=w_shg, h2_out=h2_out, logits_out=logits_out)
    fl.attn_fd_tail(x, attn, gate, w_o, ln2, w_gate, w_up, w_down, out)
    return out, h2_out, logits_out


if fl.WFMT == "mx8":
    mb = fl.mb

    def attn_fd_tail_moe_mx(x, attn, gate, w_o, s_o, ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, w_router, w_shg,
                            out, h2_out, logits_out):
        mb.SCALES.clear()
        for w, s in ((w_o, s_o), (w_gate, s_gate), (w_up, s_up), (w_down, s_down)):
            mb.SCALES[id(w)] = s
        attn_fd_tail_moe(x, attn, gate, w_o, ln2, w_gate, w_up, w_down, w_router, w_shg, out, h2_out, logits_out)
        return out, h2_out, logits_out


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    fl.BUILD = BUILD
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    mx = fl.WFMT == "mx8"
    a, ref_a = mb.mx_args(q.attn_args(W, 3), mb.ATTN_W) if mx else (q.attn_args(W, 3), None)
    ref_a = ref_a or a
    sfx = "_mx8" if mx else ""
    head_fn = fl.attn_fd_head_mx if mx else fl.attn_fd_head
    tail_fn = attn_fd_tail_moe_mx if mx else attn_fd_tail_moe
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (R,))].reshape(1, R, H).contiguous()
    NC = 2 * S if CHECK else S
    C0 = (S + 5) if CHECK else 70
    cells = [C0 + t for t in range(R)]
    posn = [C0 - C0 // 3 + t for t in range(R)]

    def mask_row(t):  # history: cells c < C0 with c % 3 != 0; new tokens causal
        m = torch.full((NC,), -3e38)
        m[[c for c in range(C0) if c % 3]] = 0
        m[C0 : C0 + t + 1] = 0
        return m

    kc0 = (torch.randn((NC, HKV * D)) * 0.5).half()
    vc0 = (torch.randn((NC, HKV * D)) * 0.5).half()
    cs = [q.rope_cos_sin(p) for p in posn]
    cos_t, sin_t = torch.cat([c for c, _ in cs]), torch.cat([s for _, s in cs])

    k_head = fl.kern(head_fn, f"head{R}{sfx}")
    hc = dict(a, x=xs, cos=cos_t, sin=sin_t, q_out=torch.zeros((R, HQ, D), dtype=torch.bfloat16),
              gate_out=torch.zeros((R, HQ, D), dtype=torch.bfloat16), k_out=torch.zeros((R, HKV * D), dtype=torch.half),
              v_out=torch.zeros((R, HKV * D), dtype=torch.half))
    k_head[1](*[hc[n] for n in mb.order(head_fn)] if mx else [hc[n] for n in fl.HEAD_ORDER])
    kc, vc = kc0.clone(), vc0.clone()
    kc[cells], vc[cells] = hc["k_out"], hc["v_out"]

    seg = f"seg{R}_c{S}"
    k_seg = fl.kern(fl.attn_fd_seg, seg)
    parts, seen = [], []
    for g in range(NC // S):
        rows = torch.stack([mask_row(t)[g * S : (g + 1) * S] for t in range(R)])
        vis = rows == 0
        if not vis.any():
            continue
        nb = int(vis.any(0).nonzero().max()) // BC + 1
        sc = dict(qs=hc["q_out"], kc=kc[g * S : (g + 1) * S].contiguous(), vc=vc[g * S : (g + 1) * S].contiguous(),
                  nblk=torch.tensor([nb], dtype=torch.int32),
                  mask=rows.reshape(R, S // BC, BC).permute(1, 0, 2).bfloat16().contiguous(),
                  m_out=torch.zeros((HKV, G * R, 1), dtype=torch.bfloat16), l_out=torch.zeros((HKV, G * R)),
                  o_out=torch.zeros((HKV, G * R, D), dtype=torch.bfloat16))
        k_seg[1](*[sc[n] for n in fl.SEG_ORDER])
        parts.append((sc["m_out"], sc["l_out"], sc["o_out"]))
        seen.append(vis.any(1).tolist())
    att = fl.merge(parts, seen)

    k_tail = fl.kern(tail_fn, f"tail{R}{sfx}")
    z = lambda *sh: torch.zeros(sh, dtype=torch.bfloat16)  # noqa: E731
    tc = dict(a, x=xs, attn=att, gate=hc["gate_out"], out=z(1, R, H), h2_out=z(1, R, H),
              logits_out=torch.zeros((1, R, NE)))
    k_tail[1](*[tc[n] for n in mo.order(tail_fn)])
    logger.info(f"moe flash R={R} S={S}: {len(parts)} segment(s), outputs finite "
                f"{all(bool(tc[n].float().isfinite().all()) for n in ('out', 'h2_out', 'logits_out'))}")

    if CHECK:  # the fused MoE attention prefill kernel over both segments (bf16 weights: the dequantized MXFP8)
        k_ref = fl.kern(mp.attn_layer_prefill_moe, f"ref{R}_c{NC}")
        mask = torch.stack([mask_row(t).reshape(NC // BC, BC) for t in range(R)], dim=1).bfloat16()
        if mx:
            mb.SCALES.clear()
        rc = dict(ref_a, x=xs, cos=cos_t, sin=sin_t, kc=kc0.clone(), vc=vc0.clone(),
                  pos=torch.tensor(cells, dtype=torch.int32), nblk=torch.tensor([max(cells) // BC + 1], dtype=torch.int32),
                  mask=mask.contiguous(), out=z(1, R, H), h2_out=z(1, R, H), logits_out=torch.zeros((1, R, NE)))
        q.RESIDENT.update(kc=kc0.clone(), vc=vc0.clone())
        k_ref[1](*[rc[n] for n in mo.order(mp.attn_layer_prefill_moe)])
        q.RESIDENT.clear()
        worst = {n: min(fl.cos(tc[n][:, t].float(), rc[n][:, t].float()) for t in range(R))
                 for n in ("out", "h2_out", "logits_out")}
        logger.info(f"moe flash R={R} S={S} vs fused LMAX={NC}: min cos "
                    f"{', '.join(f'{n} {v:.6f}' for n, v in worst.items())}, kv equal "
                    f"{torch.equal(rc['kc'], kc) and torch.equal(rc['vc'], vc)}")

    if out_dir:
        kernels = [(f"head{R}{sfx}", head_fn.__name__, f"qwen35_attn_fd_head{R}{sfx}"),
                   (f"tail{R}{sfx}", tail_fn.__name__, f"qwen35moe_attn_fd_tail{R}{sfx}")]
        if not CHECK:
            kernels.append((seg, "attn_fd_seg", f"qwen35_attn_fd_{seg}"))
        for sub, fn, dst in kernels:
            path = max(glob.glob(f"{BUILD}/{sub}/cache/{fn}.*.compiled.axmodel"), key=os.path.getmtime)
            shutil.copy(path, f"{out_dir}/{dst}.axmodel")
            logger.info(f"wrote {out_dir}/{dst}.axmodel {board_io.fix(f'{out_dir}/{dst}.axmodel')[:1]}")


if __name__ == "__main__":
    main()
