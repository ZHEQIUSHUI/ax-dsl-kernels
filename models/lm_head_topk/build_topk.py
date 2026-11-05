"""Build the fused lm-head + penalties + exact-top-k kernel for dense Qwen3.5/3.8.

One call computes the output projection, applies the host-maintained repetition
state, keeps the exact top-K rows over the vocabulary, and returns the
full-vocabulary max and sum(exp((logit - max) / T)).

The vocabulary is split over four cores.  Each core splits its rows into
CHUNKS blocks, keeps K rows per block, merges those blocks locally, then core 0
merges the four local candidate sets.  Keeping K per block makes the merge
exact: a block can contribute more than K rows to the global top-K only if it
has more than K rows, which the local top-K does not truncate.

The builder is generic over the head format (MXFP4 or MXFP8), hidden size, and
row count.  ROWS=1 keeps the pilot ABI.  ROWS=4 mirrors the MTP output head:
x is [4,H], m is the valid-row count, and every row gets its own top-K, vmax
and sumexp from the same penalty arrays.  Rows >= m are zeroed.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl.kernels.utils.linear import linear

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "qwen35_4b", "whole", "lm4"))
import board_io  # noqa: E402

V = int(os.environ.get("V", "248320"))
H = int(os.environ.get("H", "2048"))
K = int(os.environ.get("TOPK", "1024"))
HEAD = os.environ.get("HEAD", "mx4").lower()
ROWS = int(os.environ.get("ROWS", "1"))
CORES = int(os.environ.get("CORES", "4"))
CHUNKS = int(os.environ.get("CHUNKS", "8"))
REAL = os.environ.get("REAL", "0") == "1"
TILE = int(os.environ.get("TILE", "0")) or max(
    16, min(1024, ((512 if HEAD == "mx8" else 256) * 1024) //
            (H if HEAD == "mx8" else H // 2))
)
TAG = os.environ.get("TAG", "real" if REAL else "synthetic")
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "out")))
DATA = os.path.abspath(os.environ.get(
    "DATA", os.path.join(ROOT, "data" if REAL else "data_synth")))
BUILD = os.path.abspath(os.environ.get("BUILD", os.path.join(ROOT, "build", TAG)))


def positive_mask(x):
    """1.0 above zero, 0.0 below it (logits are O(10), never in (0, 1e-20))."""
    return ax.dsl.clip(ax.dsl.mul(x, ax.dsl.full((1,), ax.fp32, fill_value=1e30),
                                  output_dtype=ax.fp32), min=0.0, max=1.0)


def greater_equal(x, y):
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    zero = ax.dsl.zeros((1,), ax.int32)
    return ax.dsl.minimum(ax.dsl.maximum(x - y + one, zero), one, output_dtype=ax.uint8)


def _linear(x, w, s, rows, name, out):
    return linear(x, w[rows], s[rows], name, output_dtype=ax.fp32,
                  oc_tile_size=TILE, r=out)


def _penalised(raw, bias_c, pos_c, neg_c, sl, name):
    p = ax.dsl.create_tensor(raw.shape, ax.fp32, ax.MemoryType.OCM, name=name)
    pos = positive_mask(raw)
    scale = ax.dsl.add(neg_c[sl],
                       ax.dsl.mul(ax.dsl.sub(pos_c[sl], neg_c[sl], output_dtype=ax.fp32),
                                  pos, output_dtype=ax.fp32),
                       output_dtype=ax.fp32)
    ax.dsl.mul(raw, scale, output_dtype=ax.fp32, r=p)
    ax.dsl.add(p, bias_c[sl], output_dtype=ax.fp32, r=p)
    return p


def lm_head_topk(x, w, s, bias, mul_pos, mul_neg, temp, values, ids, vmax, sumexp):
    core = ax.program_id(0)
    per_core = V // CORES
    chunk_rows = per_core // CHUNKS
    rows0 = core * per_core

    bias_c = ax.dsl.load(bias[rows0:rows0 + per_core])
    pos_c = ax.dsl.load(mul_pos[rows0:rows0 + per_core])
    neg_c = ax.dsl.load(mul_neg[rows0:rows0 + per_core])
    temp_o = ax.dsl.load(temp)

    cand_v = ax.dsl.create_tensor((1, CHUNKS * K), ax.fp32, ax.MemoryType.OCM,
                                  name="topk_chunk_values")
    cand_i = ax.dsl.create_tensor((1, CHUNKS * K), ax.int32, ax.MemoryType.OCM,
                                  name="topk_chunk_ids")
    local_logits = ax.dsl.create_tensor((1, per_core), ax.fp32, ax.MemoryType.OCM,
                                        name="topk_local_logits")

    for c in range(CHUNKS):
        rows = slice(rows0 + c * chunk_rows, rows0 + (c + 1) * chunk_rows)
        raw = ax.dsl.create_tensor((1, chunk_rows), ax.fp32, ax.MemoryType.OCM,
                                   name=f"topk_raw_{c}")
        _linear(ax.dsl.load(x), w, s, rows, f"lm_head_topk_{c}", raw)

        sl = slice(c * chunk_rows, (c + 1) * chunk_rows)
        p = _penalised(raw, bias_c, pos_c, neg_c, sl, f"topk_pen_{c}")

        chunk_pos = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                         name=f"topk_chunk_pos_{c}")
        ax.dsl.topk(p, k=K, dim=-1,
                    r_values=cand_v[:, c * K:(c + 1) * K],
                    r_indices=chunk_pos)
        chunk_off = ax.dsl.full((1, K), ax.int32, fill_value=c * chunk_rows)
        ax.dsl.add(chunk_pos, chunk_off, r=cand_i[:, c * K:(c + 1) * K])
        ax.dsl.move(p, r=local_logits[0:1, c * chunk_rows:(c + 1) * chunk_rows])

    local_v = ax.dsl.create_tensor((1, K), ax.fp32, ax.MemoryType.OCM,
                                   name="topk_local_values")
    local_pos = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                     name="topk_local_pos")
    local_i = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                   name="topk_local_ids")
    ax.dsl.topk(cand_v, k=K, dim=-1, r_values=local_v, r_indices=local_pos)
    ax.dsl.gather(cand_i, local_pos, dim=-1, r=local_i)
    off = ax.dsl.full((1, K), ax.int32, fill_value=rows0)
    ax.dsl.add(local_i, off, r=local_i)

    cross_v = ax.dsl.create_tensor((CORES, K), ax.fp32, ax.MemoryType.SHARED_OCM,
                                   name="topk_cross_values")
    cross_i = ax.dsl.create_tensor((CORES, K), ax.int32, ax.MemoryType.SHARED_OCM,
                                   name="topk_cross_ids")
    ax.dsl.move(local_v, r=cross_v[core:core + 1, :])
    ax.dsl.move(local_i, r=cross_i[core:core + 1, :])

    local_max = ax.dsl.max(local_logits, dim=-1, keepdim=True)
    cross_max = ax.dsl.create_tensor((CORES, 1), ax.fp32, ax.MemoryType.SHARED_OCM,
                                     name="topk_cross_max")
    ax.dsl.move(local_max, r=cross_max[core:core + 1, :])

    axd.pub(cross_v)
    axd.pub(cross_i)
    axd.pub(cross_max)
    axd.sub(cross_v)
    axd.sub(cross_i)
    axd.sub(cross_max)

    gmax = ax.dsl.max(cross_max, dim=0, keepdim=True)
    z = ax.dsl.exp(ax.dsl.mul(ax.dsl.sub(local_logits, gmax, output_dtype=ax.fp32),
                              ax.dsl.reciprocal(temp_o, output_dtype=ax.fp32),
                              output_dtype=ax.fp32), output_dtype=ax.fp32)
    local_sum = ax.dsl.sum(z, dim=-1, keepdim=True, output_dtype=ax.fp32)
    cross_sum = ax.dsl.create_tensor((CORES, 1), ax.fp32, ax.MemoryType.SHARED_OCM,
                                     name="topk_cross_sumexp")
    ax.dsl.move(local_sum, r=cross_sum[core:core + 1, :])
    axd.pub(cross_sum)
    axd.sub(cross_sum)

    if core == 0:
        final_v = ax.dsl.create_tensor((1, K), ax.fp32, ax.MemoryType.OCM,
                                       name="topk_final_values")
        final_i = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                       name="topk_final_ids")
        ax.dsl.topk(cross_v.reshape(1, CORES * K), k=K, dim=-1,
                    r_values=final_v, r_indices=local_pos)
        ax.dsl.gather(cross_i.reshape(1, CORES * K), local_pos, dim=-1, r=final_i)
        ax.dsl.store(final_v, r=values.reshape(1, K))
        ax.dsl.store(final_i, r=ids.reshape(1, K))
        ax.dsl.store(gmax.reshape(1), r=vmax)
        ax.dsl.store(ax.dsl.sum(cross_sum, dim=0, keepdim=True).reshape(1), r=sumexp)
    return values, ids, vmax, sumexp


def lm_head_topk_m4(x, w, s, bias, mul_pos, mul_neg, temp, m, values, ids, vmax, sumexp):
    core = ax.program_id(0)
    per_core = V // CORES
    chunk_rows = per_core // CHUNKS
    rows0 = core * per_core

    bias_c = ax.dsl.load(bias[rows0:rows0 + per_core])
    pos_c = ax.dsl.load(mul_pos[rows0:rows0 + per_core])
    neg_c = ax.dsl.load(mul_neg[rows0:rows0 + per_core])
    temp_o = ax.dsl.load(temp)

    cand_v = [ax.dsl.create_tensor((1, CHUNKS * K), ax.fp32, ax.MemoryType.OCM,
                                   name=f"topk_chunk_values_{r}") for r in range(ROWS)]
    cand_i = [ax.dsl.create_tensor((1, CHUNKS * K), ax.int32, ax.MemoryType.OCM,
                                   name=f"topk_chunk_ids_{r}") for r in range(ROWS)]
    local_logits = [ax.dsl.create_tensor((1, per_core), ax.fp32, ax.MemoryType.OCM,
                                         name=f"topk_local_logits_{r}") for r in range(ROWS)]

    for c in range(CHUNKS):
        rows = slice(rows0 + c * chunk_rows, rows0 + (c + 1) * chunk_rows)
        raw = ax.dsl.create_tensor((ROWS, chunk_rows), ax.fp32, ax.MemoryType.OCM,
                                   name=f"topk_raw_{c}")
        _linear(ax.dsl.load(x), w, s, rows, f"lm_head_topk_{c}", raw)

        sl = slice(c * chunk_rows, (c + 1) * chunk_rows)
        for r in range(ROWS):
            p = _penalised(raw[r:r + 1], bias_c, pos_c, neg_c, sl, f"topk_pen_{c}_{r}")
            chunk_pos = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                             name=f"topk_chunk_pos_{c}_{r}")
            ax.dsl.topk(p, k=K, dim=-1,
                        r_values=cand_v[r][:, c * K:(c + 1) * K],
                        r_indices=chunk_pos)
            chunk_off = ax.dsl.full((1, K), ax.int32, fill_value=c * chunk_rows)
            ax.dsl.add(chunk_pos, chunk_off, r=cand_i[r][:, c * K:(c + 1) * K])
            ax.dsl.move(p, r=local_logits[r][0:1, c * chunk_rows:(c + 1) * chunk_rows])

    cross_v = [ax.dsl.create_tensor((CORES, K), ax.fp32, ax.MemoryType.SHARED_OCM,
                                    name=f"topk_cross_values_{r}") for r in range(ROWS)]
    cross_i = [ax.dsl.create_tensor((CORES, K), ax.int32, ax.MemoryType.SHARED_OCM,
                                    name=f"topk_cross_ids_{r}") for r in range(ROWS)]
    cross_max = [ax.dsl.create_tensor((CORES, 1), ax.fp32, ax.MemoryType.SHARED_OCM,
                                      name=f"topk_cross_max_{r}") for r in range(ROWS)]
    cross_sum = [ax.dsl.create_tensor((CORES, 1), ax.fp32, ax.MemoryType.SHARED_OCM,
                                      name=f"topk_cross_sumexp_{r}") for r in range(ROWS)]

    local_pos = [ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                      name=f"topk_local_pos_{r}") for r in range(ROWS)]
    for r in range(ROWS):
        local_v = ax.dsl.create_tensor((1, K), ax.fp32, ax.MemoryType.OCM,
                                       name=f"topk_local_values_{r}")
        local_i = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                       name=f"topk_local_ids_{r}")
        ax.dsl.topk(cand_v[r], k=K, dim=-1, r_values=local_v, r_indices=local_pos[r])
        ax.dsl.gather(cand_i[r], local_pos[r], dim=-1, r=local_i)
        off = ax.dsl.full((1, K), ax.int32, fill_value=rows0)
        ax.dsl.add(local_i, off, r=local_i)
        ax.dsl.move(local_v, r=cross_v[r][core:core + 1, :])
        ax.dsl.move(local_i, r=cross_i[r][core:core + 1, :])
        local_max = ax.dsl.max(local_logits[r], dim=-1, keepdim=True)
        ax.dsl.move(local_max, r=cross_max[r][core:core + 1, :])

    for r in range(ROWS):
        axd.pub(cross_v[r])
        axd.pub(cross_i[r])
        axd.pub(cross_max[r])
    for r in range(ROWS):
        axd.sub(cross_v[r])
        axd.sub(cross_i[r])
        axd.sub(cross_max[r])

    for r in range(ROWS):
        gmax = ax.dsl.max(cross_max[r], dim=0, keepdim=True)
        z = ax.dsl.exp(ax.dsl.mul(ax.dsl.sub(local_logits[r], gmax, output_dtype=ax.fp32),
                                  ax.dsl.reciprocal(temp_o, output_dtype=ax.fp32),
                                  output_dtype=ax.fp32), output_dtype=ax.fp32)
        local_sum = ax.dsl.sum(z, dim=-1, keepdim=True, output_dtype=ax.fp32)
        ax.dsl.move(local_sum, r=cross_sum[r][core:core + 1, :])
    for r in range(ROWS):
        axd.pub(cross_sum[r])
    for r in range(ROWS):
        axd.sub(cross_sum[r])

    if core == 0:
        idx = ax.dsl.from_numpy(np.arange(1, ROWS + 1, dtype=np.int32))
        valid = greater_equal(ax.dsl.load(m), idx).reshape(ROWS, 1)
        for r in range(ROWS):
            final_v = ax.dsl.create_tensor((1, K), ax.fp32, ax.MemoryType.OCM,
                                           name=f"topk_final_values_{r}")
            final_i = ax.dsl.create_tensor((1, K), ax.int32, ax.MemoryType.OCM,
                                           name=f"topk_final_ids_{r}")
            ax.dsl.topk(cross_v[r].reshape(1, CORES * K), k=K, dim=-1,
                        r_values=final_v, r_indices=local_pos[r])
            ax.dsl.gather(cross_i[r].reshape(1, CORES * K), local_pos[r], dim=-1,
                          r=final_i)
            gmax = ax.dsl.max(cross_max[r], dim=0, keepdim=True)
            total = ax.dsl.sum(cross_sum[r], dim=0, keepdim=True)
            mask_f = ax.dsl.cast(valid[r:r + 1], dtype=ax.fp32)
            mask_i = ax.dsl.cast(valid[r:r + 1], dtype=ax.int32)
            final_v = ax.dsl.mul(final_v, mask_f, output_dtype=ax.fp32)
            final_i = ax.dsl.mul(final_i, mask_i, output_dtype=ax.int32)
            ax.dsl.store(final_v, r=values[r:r + 1, :])
            ax.dsl.store(final_i, r=ids[r:r + 1, :])
            ax.dsl.store(ax.dsl.mul(gmax, mask_f, output_dtype=ax.fp32).reshape(1),
                         r=vmax[r:r + 1])
            ax.dsl.store(ax.dsl.mul(total, mask_f, output_dtype=ax.fp32).reshape(1),
                         r=sumexp[r:r + 1])
    return values, ids, vmax, sumexp


def _penalty_arrays(n):
    bias = torch.zeros(n, dtype=torch.float32)
    mul_pos = torch.ones(n, dtype=torch.float32)
    mul_neg = torch.ones(n, dtype=torch.float32)
    repeat = 1.5
    for tid in (0, 7, 31, 255, 1024, 65535, n // 2, n - 3):
        mul_pos[tid] = 1.0 / repeat
        mul_neg[tid] = repeat
        bias[tid] = -float((tid % 5) + 1)
    return bias, mul_pos, mul_neg


def load_synth():
    torch.manual_seed(1234)
    gen = torch.Generator().manual_seed(1234)
    x = (torch.randn((ROWS, H), generator=gen) * 1.0).bfloat16()
    if ROWS > 2:
        x[2] += 7.0
    if ROWS > 3:
        x[3] -= 7.0
    if HEAD == "mx4":
        w = torch.randint(0, 256, (V, H // 2), generator=gen, dtype=torch.uint8)
    else:
        w = torch.randint(0, 256, (V, H), generator=gen, dtype=torch.uint8)
    s = torch.exp2(torch.randint(-4, 2, (V, H // 32), generator=gen,
                                 dtype=torch.int32).float()).to(torch.uint8) + 120
    bias, mul_pos, mul_neg = _penalty_arrays(V)
    temp = torch.tensor([0.7], dtype=torch.float32)
    return x, w, s, bias, mul_pos, mul_neg, temp


def load_real():
    def rd(name, dtype, shape):
        count = int(np.prod(shape))
        a = np.fromfile(os.path.join(DATA, name), dtype=dtype, count=count)
        return torch.from_numpy(a.copy()).reshape(shape)

    x = rd("x.bin", np.uint16, (ROWS, H)).view(torch.bfloat16)
    w = rd("w.bin", np.uint8, (V, H // 2 if HEAD == "mx4" else H))
    s = rd("s.bin", np.uint8, (V, H // 32))
    bias = rd("bias.bin", np.float32, (V,))
    mul_pos = rd("mul_pos.bin", np.float32, (V,))
    mul_neg = rd("mul_neg.bin", np.float32, (V,))
    temp = rd("temp.bin", np.float32, (1,))
    ref = torch.load(os.path.join(DATA, "ref_topk.pt"))
    ref = {k: (v[:ROWS] if torch.is_tensor(v) and v.ndim and v.shape[0] >= ROWS else v)
           for k, v in ref.items()}
    return x, w, s, bias, mul_pos, mul_neg, temp, ref


def dequant_ref(x, w, s, bias, mul_pos, mul_neg, temp):
    sys.path.insert(0, ROOT)
    from quant import torch_logits
    logits = torch_logits(x, w, s, HEAD)
    logits = logits * torch.where(logits > 0, mul_pos, mul_neg) + bias
    vals, ids = torch.topk(logits, K, dim=-1)
    gmax = logits.max(dim=-1, keepdim=True).values
    sumexp = torch.exp((logits - gmax) / temp[0]).sum(dim=-1, keepdim=True)
    return vals, ids.to(torch.int32), gmax.reshape(ROWS), sumexp.reshape(ROWS), logits


def _report(ref, values, ids, vmax, sumexp, logits):
    ref_v = ref["values"]
    ref_i = ref["ids"]
    ref_max = ref["max"].reshape(ROWS)
    ref_sum = ref["sumexp"].reshape(ROWS)
    ok = True
    for r in range(ROWS):
        got_logits = torch.gather(logits[r:r + 1], 1, ids[r].long().reshape(1, K))
        id_bad = int((ids[r] != ref_i[r]).sum())
        set_bad = int((torch.sort(ids[r]).values != torch.sort(ref_i[r]).values).sum())
        sorted_v_err = float((torch.sort(values[r]).values -
                              torch.sort(ref_v[r]).values).abs().max())
        v_vs_logit = float((values[r] - got_logits[0]).abs().max())
        v_vs_ref = float((values[r] - ref_v[r]).abs().max())
        max_err = float((vmax[r] - ref_max[r]).abs())
        sum_rel = float((sumexp[r] - ref_sum[r]).abs() / ref_sum[r].abs().clamp_min(1e-30))
        print(f"row {r}: ids mismatch {id_bad} (set {set_bad}); "
              f"sorted value err {sorted_v_err:.3e}; value vs logits {v_vs_logit:.3e}; "
              f"value vs ref {v_vs_ref:.3e}; vmax err {max_err:.3e}; "
              f"sumexp rel {sum_rel:.3e}", flush=True)
        ok = ok and set_bad == 0 and sorted_v_err < 5e-6 and sum_rel < 5e-6
    return ok


def main():
    assert ROWS in (1, 4), ROWS
    assert HEAD in ("mx4", "mx8"), HEAD
    assert V % (CORES * CHUNKS) == 0, (V, CORES, CHUNKS)
    assert H % 32 == 0
    assert V // CORES // CHUNKS >= K, "a block must have at least K rows for the chunked top-k to stay exact"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    if REAL:
        x, w, s, bias, mul_pos, mul_neg, temp, ref = load_real()
        logits = ref["logits"].reshape(ROWS, V)
    else:
        x, w, s, bias, mul_pos, mul_neg, temp = load_synth()
        ref_v, ref_i, ref_max, ref_sum, logits = dequant_ref(
            x, w, s, bias, mul_pos, mul_neg, temp)
        ref = {"values": ref_v, "ids": ref_i, "max": ref_max,
               "sumexp": ref_sum, "logits": logits}

    if HEAD == "mx4":
        kw = w.view(torch.float4_e2m1fn_x2)
    else:
        kw = w.view(torch.float8_e4m3fn)
    ks = s.view(torch.float8_e8m0fnu)
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    cache = os.path.join(BUILD, "cache")
    fn = lm_head_topk if ROWS == 1 else lm_head_topk_m4
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, cache_path=cache,
                   dump_path=BUILD)(fn)
    values = torch.zeros(ROWS, K, dtype=torch.float32)
    ids = torch.zeros(ROWS, K, dtype=torch.int32)
    vmax = torch.zeros(ROWS, dtype=torch.float32)
    sumexp = torch.zeros(ROWS, dtype=torch.float32)
    if ROWS == 1:
        k[CORES](x[0:1].contiguous(), kw, ks, bias, mul_pos, mul_neg, temp,
                 values[0], ids[0], vmax[0:1], sumexp[0:1])
        print("run time %.1fs" % (time.time() - t0), flush=True)
        _report(ref, values, ids, vmax, sumexp, logits)
    else:
        for m in range(1, ROWS + 1):
            zero = torch.zeros(ROWS, K, dtype=torch.float32)
            zids = torch.zeros(ROWS, K, dtype=torch.int32)
            zvmax = torch.zeros(ROWS, dtype=torch.float32)
            zsum = torch.zeros(ROWS, dtype=torch.float32)
            k[CORES](x.contiguous(), kw, ks, bias, mul_pos, mul_neg, temp,
                     torch.tensor([m], dtype=torch.int32),
                     zero, zids, zvmax, zsum)
            values[:m] = zero[:m]
            ids[:m] = zids[:m]
            vmax[:m] = zvmax[:m]
            sumexp[:m] = zsum[:m]
            tail_ok = bool((zero[m:] == 0).all() and (zids[m:] == 0).all()
                           and (zvmax[m:] == 0).all() and (zsum[m:] == 0).all())
            print(f"m={m}: invalid tails zero {tail_ok}", flush=True)
        print("run time %.1fs" % (time.time() - t0), flush=True)
        _report(ref, values, ids, vmax, sumexp, logits)

    paths = sorted(glob.glob(os.path.join(BUILD, "cache", "lm_head_topk*.compiled.axmodel")),
                   key=os.path.getmtime)
    assert paths, glob.glob(os.path.join(BUILD, "cache", "*"))
    suffix = "" if ROWS == 1 else f"_m{ROWS}"
    name = f"lm_head_topk_n{V}_k{H}_t{K}_c{CORES}{suffix}"
    dst = board_io.kernel_file(OUT, name, CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)
    json.dump({"V": V, "H": H, "K": K, "head": HEAD, "rows": ROWS,
               "cores": CORES, "chunks": CHUNKS, "tile": TILE, "real": REAL},
              open(os.path.join(OUT, name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
