"""Qwen3.5 attention layer split for contexts beyond one KV IO (flash decoding): R tokens per call (decode R = 1,
batched decode R = B, prefill R = P), any number of KV segments of S cells.

  attn_fd_head<R>      norm, q / k / v projections, q / k norm, RoPE: q (pre-scaled by 1/sqrt(D)) and gate
                       [R, HQ, D] bf16, the new K / V rows [R, HKV*D] f16 (the host writes them to their cells)
  attn_fd_seg<R>_c<S>  partial attention of the R tokens over one KV segment [S, HKV*D] f16 (the host binds the
                       segment's address in the llama.cpp KV cache): online softmax over the runtime block count with
                       mask [S/BC, R, BC], each block of BC cells loaded once for all KV heads (BC from env: larger
                       blocks, fewer loop iterations); outputs per KV head and row g*R + t the running max m (bf16),
                       the sum l (f32) and the unnormalized output o (bf16)
  attn_fd_tail<R>      attention output (the host's merge of the segments) -> sigmoid gate -> o_proj -> MLP

Merging segments i: m = max m_i, l = sum exp(m_i - m) l_i, o = sum exp(m_i - m) o_i / l, skipping segments where a
row sees no cell. With one segment the result is the fused kernel's.

Check on the simulator (S <= 16384): R tokens at cells in segment 1, history in segments 0 and 1, against the fused
attention kernel (q35_prefill, SCATTER) with LMAX = 2 S. Usage:
MODEL=<hf dir> R=4 S=4096 BC=64 python q35_flash.py [kernel_dir]   (CHECK=0: compile only, e.g. S = 32768)
"""
import glob
import math
import os
import shutil
import sys

R = int(os.environ.get("R", 4))
CORES = int(os.environ.get("CORES", 1))  # segment kernel: cores, each scanning its own run of blocks
S = int(os.environ.get("S", 4096))
CHECK = os.environ.get("CHECK", "1") == "1" and 2 * S <= 32768
os.environ["PCHUNK"] = str(R)
os.environ["KVL"] = "cell"
os.environ["LMAX"] = str(2 * S if CHECK else S)

import ml_dtypes
import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm, sigmoid_mul
from axnn.dsl.kernels.utils.rope import rope
import q35_step as q
import q35_prefill as pf

H, D, HQ, HKV, G, ROT, EPS, BC = q.H, q.D, q.HQ, q.HKV, q.G, q.ROT, q.EPS, q.BC
GR = G * R
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_flash"))
pf.SCATTER = True


def attn_fd_head(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, q_out, gate_out, k_out, v_out):
    xt = ax.dsl.load(x)  # [1, R, H]
    h = rmsnorm(xt, ax.dsl.load(ln1).reshape(1, 1, H), eps=EPS)
    qg = q.proj(h, w_q, "attn_q").reshape(R, HQ, 2 * D)  # per head [q | gate]
    qh_all = rmsnorm(qg[:, :, 0:D], ax.dsl.load(q_norm).reshape(1, 1, D), eps=EPS)
    k = rmsnorm(q.proj(h, w_k, "attn_k").reshape(R, HKV, D), ax.dsl.load(k_norm).reshape(1, 1, D), eps=EPS)
    v = q.proj(h, w_v, "attn_v").reshape(R, HKV, D)
    cos_t, sin_t = ax.dsl.load(cos), ax.dsl.load(sin)  # [R, ROT/2]
    qh_all = rope(qh_all, cos_t, sin_t, rotary_dim=ROT)
    k = rope(k, cos_t, sin_t, rotary_dim=ROT)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    ax.dsl.store(ax.dsl.mul(qh_all, scale, output_dtype=ax.bf16), r=q_out)
    ax.dsl.store(qg[:, :, D : 2 * D], r=gate_out)
    ax.dsl.store(ax.dsl.cast(k.reshape(R, HKV * D), dtype=ax.fp16), r=k_out)
    ax.dsl.store(ax.dsl.cast(v.reshape(R, HKV * D), dtype=ax.fp16), r=v_out)
    return q_out, gate_out, k_out, v_out


def attn_fd_seg(qs, kc, vc, nblk, mask, m_out, l_out, o_out):
    """CORES == 1: nblk [1], blocks 0 .. nblk - 1; outputs [HKV, ..]. CORES > 1: nblk [CORES, 2], core c scans blocks
    nblk[c, 0] .. nblk[c, 1] - 1 (at least one; the host ignores a core given no block); outputs [CORES, HKV, ..], one
    partial result per core, merged by the host like the segments. No synchronization between the cores."""
    c = ax.program_id(0)
    qa = ax.dsl.load(qs)  # [R, HQ, D], pre-scaled
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    if CORES == 1:
        range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
        range_stop = ax.dsl.load(nblk)
    else:
        range_start = ax.dsl.load(nblk[c : c + 1, 0:1]).reshape(1)
        range_stop = ax.dsl.load(nblk[c : c + 1, 1:2]).reshape(1)
    zero_col = ax.dsl.zeros(shape=(GR, 1), dtype=ax.bf16)
    heads = []  # per KV head: its G query heads x R tokens, and the running (max, sum, output)
    for hk in range(HKV):
        qh = ax.dsl.create_tensor((GR, D), ax.bf16, name=f"qh{hk}")
        for g in range(G):
            ax.dsl.move(x=qa[:, hk * G + g : hk * G + g + 1, :].reshape(R, D), r=qh[g * R : (g + 1) * R, :])
        heads += [qh, ax.dsl.full(shape=(GR, 1), dtype=ax.bf16, fill_value=-3e38), ax.dsl.zeros(shape=(GR,), dtype=ax.fp32),
                  ax.dsl.zeros(shape=(GR, D), dtype=ax.bf16)]
    index = ax.dsl.create_tensor((1,), ax.int32, name="index")
    ax.dsl.move(x=range_start, r=index)

    def body(index, kc, vc, mask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, *heads):
        # one block of BC cells, both KV heads' columns, loaded and converted once for all heads
        t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
        kb = ax.dsl.cast(ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D)), dtype=ax.bf16)
        vb = ax.dsl.cast(ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, HKV * D)), dtype=ax.bf16)
        mj = ax.dsl.dynamic_ld(mask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                               output_shape=(1, R, BC)).reshape(R, BC)
        for hk in range(HKV):
            qh, mi, li, oi = heads[4 * hk : 4 * hk + 4]
            kj = kb[:, hk * D : (hk + 1) * D]
            vj = vb[:, hk * D : (hk + 1) * D]
            m_s = ax.dsl.create_tensor(shape=(GR, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            for g in range(G):
                ax.dsl.add(sj[g * R : (g + 1) * R, :], mj, output_dtype=ax.bf16, r=m_s[g * R : (g + 1) * R, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(GR)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=li, output_dtype=ax.fp32)
            oij = pj @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(GR, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=mi)
        return (kc, vc, mask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, *heads)

    res = ax.dsl.range(index, range_start, range_stop, range_step, body, kc, vc, mask, block_stride, kv_block_shape,
                       mask_block_shape, zero_col, minus_1, *heads)
    heads = res[8:]
    for hk in range(HKV):
        _, mi, li, oi = heads[4 * hk : 4 * hk + 4]
        if CORES == 1:
            ax.dsl.store(mi.reshape(1, GR, 1), r=m_out[hk : hk + 1])
            ax.dsl.store(li.reshape(1, GR), r=l_out[hk : hk + 1])
            ax.dsl.store(oi.reshape(1, GR, D), r=o_out[hk : hk + 1])
        else:
            ax.dsl.store(mi.reshape(1, 1, GR, 1), r=m_out[c : c + 1, hk : hk + 1])
            ax.dsl.store(li.reshape(1, 1, GR), r=l_out[c : c + 1, hk : hk + 1])
            ax.dsl.store(oi.reshape(1, 1, GR, D), r=o_out[c : c + 1, hk : hk + 1])
    return m_out, l_out, o_out


def attn_fd_tail(x, attn, gate, w_o, ln2, w_gate, w_up, w_down, out):
    xt = ax.dsl.load(x)  # [1, R, H]
    gated = sigmoid_mul(ax.dsl.load(attn), ax.dsl.load(gate), activation_output_dtype=ax.fp32,
                        output_dtype=ax.bf16).reshape(1, R, HQ * D)
    x1 = ax.dsl.add(xt, q.proj(gated, w_o, "attn_o"), output_dtype=ax.bf16)
    q.mlp_residual(x1, ln2, w_gate, w_up, w_down, out)
    return out


HEAD_ORDER = ["x", "ln1", "w_q", "w_k", "w_v", "q_norm", "k_norm", "cos", "sin", "q_out", "gate_out", "k_out", "v_out"]
SEG_ORDER = ["qs", "kc", "vc", "nblk", "mask", "m_out", "l_out", "o_out"]
TAIL_ORDER = ["x", "attn", "gate", "w_o", "ln2", "w_gate", "w_up", "w_down", "out"]


def merge(parts, seen):
    """parts: per segment (m [HKV, GR, 1] bf16, l [HKV, GR] f32, o [HKV, GR, D] bf16); seen[i][t]: token t sees a
    cell of segment i. Returns the attention output [R, HQ, D] bf16, as the backend computes it."""
    M = L = O = None
    for (m, l, o), sn in zip(parts, seen):
        vis = torch.tensor(sn).repeat(G).reshape(1, GR)  # row g*R + t
        m, l, o = m.float().reshape(HKV, GR), l.float(), o.float()
        if M is None:
            M, L, O = torch.where(vis, m, -3e38), torch.where(vis, l, 0.0), torch.where(vis[..., None], o, 0.0)
            continue
        Mn = torch.where(vis, torch.maximum(M, m), M)
        a, b = torch.exp(M - Mn), torch.where(vis, torch.exp(m - Mn), 0.0)
        L, O, M = a * L + b * l, a[..., None] * O + b[..., None] * o, Mn
    att = (O / L[..., None]).reshape(HKV, G, R, D).permute(2, 0, 1, 3).reshape(R, HQ, D)
    return att.bfloat16()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def split_blocks(nb, cores):
    """the host's split of blocks 0 .. nb - 1 over the cores: [start, stop) each, an empty run as [0, 1) (ignored)"""
    per = -(-nb // cores)
    runs = [(min(nb, i * per), min(nb, (i + 1) * per)) for i in range(cores)]
    return [(a, b) if b > a else (0, 1) for a, b in runs], [b > a for a, b in runs]


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None  # before the chdirs
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    a = q.attn_args(W, 3)
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (R,))].reshape(1, R, H).contiguous()
    NC = 2 * S if CHECK else S  # cells of the test cache
    C0 = (S + 5) if CHECK else int(os.environ.get("C0", 70))  # the R new tokens at cells C0 .. (segment 1 checking)
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

    # head
    k_head = kern(attn_fd_head, f"head{R}")
    hc = dict(a, x=xs, cos=cos_t, sin=sin_t, q_out=torch.zeros((R, HQ, D), dtype=torch.bfloat16),
              gate_out=torch.zeros((R, HQ, D), dtype=torch.bfloat16), k_out=torch.zeros((R, HKV * D), dtype=torch.half),
              v_out=torch.zeros((R, HKV * D), dtype=torch.half))
    k_head[1](*[hc[n] for n in HEAD_ORDER])
    kc, vc = kc0.clone(), vc0.clone()
    kc[cells], vc[cells] = hc["k_out"], hc["v_out"]  # the host's write

    # segments
    SEG = f"seg{R}_c{S}" + (f"_x{CORES}" if CORES > 1 else "")
    ax.init(chip_num=1, core_num=CORES)  # the segment kernel only: head, tail and the reference are single-core
    k_seg = kern(attn_fd_seg, SEG)
    parts, seen = [], []
    for g in range(NC // S):
        rows = torch.stack([mask_row(t)[g * S : (g + 1) * S] for t in range(R)])  # [R, S]
        vis = (rows == 0)
        if not vis.any():
            continue
        last = int(vis.any(0).nonzero().max())
        nb = last // BC + 1
        runs, used = split_blocks(nb, CORES)
        lead = (CORES,) if CORES > 1 else ()
        sc = dict(qs=hc["q_out"], kc=kc[g * S : (g + 1) * S].contiguous(), vc=vc[g * S : (g + 1) * S].contiguous(),
                  nblk=torch.tensor(runs, dtype=torch.int32) if CORES > 1 else torch.tensor([nb], dtype=torch.int32),
                  mask=rows.reshape(R, S // BC, BC).permute(1, 0, 2).bfloat16().contiguous(),
                  m_out=torch.zeros(lead + (HKV, GR, 1), dtype=torch.bfloat16), l_out=torch.zeros(lead + (HKV, GR)),
                  o_out=torch.zeros(lead + (HKV, GR, D), dtype=torch.bfloat16))
        k_seg[CORES](*[sc[n] for n in SEG_ORDER])
        if CORES == 1:
            parts.append((sc["m_out"], sc["l_out"], sc["o_out"]))
            seen.append(vis.any(1).tolist())
            continue
        for i, ((a0, a1), u) in enumerate(zip(runs, used)):
            sv = vis[:, a0 * BC : a1 * BC].any(1) & u
            if sv.any():
                parts.append((sc["m_out"][i], sc["l_out"][i], sc["o_out"][i]))
                seen.append(sv.tolist())
    att = merge(parts, seen)
    if os.environ.get("SAVE"):
        torch.save(att, os.environ["SAVE"])
    ax.init(chip_num=1, core_num=1)

    # tail
    k_tail = kern(attn_fd_tail, f"tail{R}")
    tc = dict(a, x=xs, attn=att, gate=hc["gate_out"], out=torch.zeros((1, R, H), dtype=torch.bfloat16))
    k_tail[1](*[tc[n] for n in TAIL_ORDER])
    fo = tc["out"].float()
    logger.info(f"flash R={R} S={S}: {len(parts)} segment(s), out finite {bool(fo.isfinite().all())}")

    if CHECK:  # the fused kernel over both segments
        k_ref = kern(pf.attn_layer_prefill, f"ref{R}_c{NC}")
        mask = torch.stack([mask_row(t).reshape(NC // BC, BC) for t in range(R)], dim=1).bfloat16()
        rc = dict(a, x=xs, cos=cos_t, sin=sin_t, kc=kc0.clone(), vc=vc0.clone(),
                  pos=torch.tensor(cells, dtype=torch.int32), nblk=torch.tensor([max(cells) // BC + 1], dtype=torch.int32),
                  mask=mask.contiguous(), out=torch.zeros((1, R, H), dtype=torch.bfloat16))
        q.RESIDENT.update(kc=kc0.clone(), vc=vc0.clone())
        k_ref[1](*[rc[n] for n in q.ATTN_ORDER])
        q.RESIDENT.clear()
        ro = rc["out"].float()
        logger.info(f"flash R={R} S={S} vs fused LMAX={NC}: out min cos "
                    f"{min(cos(fo[:, t], ro[:, t]) for t in range(R)):.6f} kv equal "
                    f"{torch.equal(rc['kc'], kc) and torch.equal(rc['vc'], vc)}")

    if out_dir:
        kernels = [(f"head{R}", "attn_fd_head", f"qwen35_attn_fd_head{R}"), (SEG, "attn_fd_seg", f"qwen35_attn_fd_{SEG}"),
                   (f"tail{R}", "attn_fd_tail", f"qwen35_attn_fd_tail{R}")]
        for sub, fn, dst in kernels if CORES == 1 else kernels[1:2]:
            (path,) = glob.glob(f"{BUILD}/{sub}/cache/{fn}.*.compiled.axmodel")
            shutil.copy(path, f"{out_dir}/{dst}.axmodel")
            logger.info(f"wrote {out_dir}/{dst}.axmodel")


if __name__ == "__main__":
    main()
