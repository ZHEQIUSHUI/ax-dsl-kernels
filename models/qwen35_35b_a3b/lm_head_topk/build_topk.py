"""Build and check the fused MXFP4 lm-head + penalty + exact-top-k kernel.

One call computes the output projection, applies the host-maintained repetition
state, keeps the exact top-K rows over the vocabulary, and returns the
full-vocabulary max and sum(exp((logit - max) / T)).

The vocabulary is split over four cores.  Each core splits its rows into
CHUNKS blocks, keeps K rows per block, merges those blocks locally, then core 0
merges the four local candidate sets.  Keeping K per block makes the merge
exact: a block can contribute more than K rows to the global top-K only if it
has more than K rows, which the local top-K does not truncate.
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
HEAD = os.path.join(ROOT, "..", "..", "qwen38_27b", "head")
sys.path.insert(0, HEAD)
sys.path.insert(0, os.path.join(ROOT, "..", "..", "qwen35_4b", "whole", "lm4"))
import board_io  # noqa: E402

V = int(os.environ.get("V", "8192"))
H = int(os.environ.get("H", "2048"))
K = int(os.environ.get("TOPK", "256"))
CORES = int(os.environ.get("CORES", "4"))
CHUNKS = int(os.environ.get("CHUNKS", "8"))
REAL = os.environ.get("REAL", "0") == "1"
TILE = int(os.environ.get("TILE", "0")) or max(16, min(1024, (256 * 1024) // (H // 2)))
TAG = os.environ.get("TAG", "real" if REAL else "synthetic")
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data" if REAL else "data_synth")
BUILD = os.path.join(ROOT, "build", TAG)


def positive_mask(x):
    """1.0 above zero, 0.0 below it (logits are O(10), never in (0, 1e-20))."""
    return ax.dsl.clip(ax.dsl.mul(x, ax.dsl.full((1,), ax.fp32, fill_value=1e30),
                                  output_dtype=ax.fp32), min=0.0, max=1.0)


def lm_head_topk(x, w, s, bias, mul_pos, mul_neg, temp, values, ids, vmax, sumexp):
    core = ax.program_id(0)
    per_core = V // CORES
    chunk_rows = per_core // CHUNKS
    rows0 = core * per_core

    # Host-maintained repetition state for this core's vocabulary shard.  The
    # additive part holds frequency + presence, active marks tokens whose
    # repeat penalty applies, and repeat is llama.cpp's scalar penalty.
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
        linear(ax.dsl.load(x), w[rows], s[rows], f"lm_head_topk_{c}",
               output_dtype=ax.fp32, oc_tile_size=TILE, r=raw)

        p = ax.dsl.create_tensor((1, chunk_rows), ax.fp32, ax.MemoryType.OCM,
                                 name=f"topk_pen_{c}")
        sl = slice(c * chunk_rows, (c + 1) * chunk_rows)
        pos = positive_mask(raw)
        # llama.cpp's repeat penalty is l > 0 ? l / r : l * r.  The host
        # maintains the two multipliers; this selects between them by sign.
        scale = ax.dsl.add(neg_c[sl],
                           ax.dsl.mul(ax.dsl.sub(pos_c[sl], neg_c[sl], output_dtype=ax.fp32),
                                      pos, output_dtype=ax.fp32),
                           output_dtype=ax.fp32)
        ax.dsl.mul(raw, scale, output_dtype=ax.fp32, r=p)
        ax.dsl.add(p, bias_c[sl], output_dtype=ax.fp32, r=p)

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

    # Exact top-K across cores: every core publishes its local K, core 0 merges.
    cross_v = ax.dsl.create_tensor((CORES, K), ax.fp32, ax.MemoryType.SHARED_OCM,
                                   name="topk_cross_values")
    cross_i = ax.dsl.create_tensor((CORES, K), ax.int32, ax.MemoryType.SHARED_OCM,
                                   name="topk_cross_ids")
    ax.dsl.move(local_v, r=cross_v[core:core + 1, :])
    ax.dsl.move(local_i, r=cross_i[core:core + 1, :])

    # Full-vocabulary max: local reduction, one publish, a four-element merge.
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

    # Full-vocabulary normaliser at the requested temperature.  This is the
    # exact denominator for top-p and min-p, not a K-way approximation.
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


def load_synth():
    os.makedirs(DATA, exist_ok=True)
    torch.manual_seed(1234)
    gen = torch.Generator().manual_seed(1234)
    x = (torch.randn((1, H), generator=gen) * 1.0).bfloat16()
    w = torch.randint(0, 256, (V, H // 2), generator=gen, dtype=torch.uint8)
    s = torch.exp2(torch.randint(-4, 2, (V, H // 32), generator=gen,
                                 dtype=torch.int32).float()).to(torch.uint8) + 120
    bias = torch.zeros(V, dtype=torch.float32)
    mul_pos = torch.ones(V, dtype=torch.float32)
    mul_neg = torch.ones(V, dtype=torch.float32)
    repeat = 1.25
    temp = torch.tensor([0.7], dtype=torch.float32)
    for tid in (7, 31, 255, V // 2, V - 3):
        mul_pos[tid] = 1.0 / repeat
        mul_neg[tid] = repeat
        bias[tid] = -0.5 * (int(tid) % 3 + 1)
    return x, w, s, bias, mul_pos, mul_neg, temp


def load_real():
    def rd(name, dtype, shape):
        a = np.fromfile(os.path.join(DATA, name), dtype=dtype)
        return torch.from_numpy(a.copy()).reshape(shape)

    x = rd("x.bin", np.uint16, (1, H)).view(torch.bfloat16)
    w = rd("w.bin", np.uint8, (V, H // 2))
    s = rd("s.bin", np.uint8, (V, H // 32))
    bias = rd("bias.bin", np.float32, (V,))
    mul_pos = rd("mul_pos.bin", np.float32, (V,))
    mul_neg = rd("mul_neg.bin", np.float32, (V,))
    temp = rd("temp.bin", np.float32, (1,))
    ref = torch.load(os.path.join(DATA, "ref_topk.pt"))
    return x, w, s, bias, mul_pos, mul_neg, temp, ref


def dequant_ref(x, w, s, bias, mul_pos, mul_neg, temp):
    from mx4_common import dequantize_mxfp4
    wf = dequantize_mxfp4(w, s)
    logits = x.float() @ wf.T
    logits = logits * torch.where(logits > 0, mul_pos, mul_neg) + bias
    vals, ids = torch.topk(logits, K, dim=-1)
    gmax = logits.max(dim=-1, keepdim=True).values
    sumexp = torch.exp((logits - gmax) / temp[0]).sum(dim=-1, keepdim=True)
    return vals[0], ids[0].to(torch.int32), gmax.reshape(1), sumexp.reshape(1), logits


def main():
    assert V % (CORES * CHUNKS) == 0, (V, CORES, CHUNKS)
    assert H % 32 == 0
    assert V // CORES // CHUNKS >= K, "a block must have at least K rows for the chunked top-k to stay exact"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    if REAL:
        x, w, s, bias, mul_pos, mul_neg, temp, ref = load_real()
        ref_v = ref["values"][:K]
        ref_i = ref["ids"][:K]
        ref_max = ref["max"]
        ref_sum = ref["sumexp"]
        logits = ref["logits"].reshape(1, V)
    else:
        x, w, s, bias, mul_pos, mul_neg, temp = load_synth()
        ref_v, ref_i, ref_max, ref_sum, logits = dequant_ref(x, w, s, bias, mul_pos, mul_neg, temp)
    kw = w.view(torch.float4_e2m1fn_x2)
    ks = s.view(torch.float8_e8m0fnu)
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    cache = os.path.join(BUILD, "cache")
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, cache_path=cache,
                   dump_path=BUILD)(lm_head_topk)
    values = torch.zeros(K, dtype=torch.float32)
    ids = torch.zeros(K, dtype=torch.int32)
    vmax = torch.zeros(1, dtype=torch.float32)
    sumexp = torch.zeros(1, dtype=torch.float32)
    k[CORES](x, kw, ks, bias, mul_pos, mul_neg, temp, values, ids, vmax, sumexp)
    print("run time %.1fs" % (time.time() - t0), flush=True)
    print("max", float(vmax), "ref", float(ref_max), flush=True)
    print("sumexp", float(sumexp), "ref", float(ref_sum), flush=True)
    got_logits = torch.gather(logits, 1, ids.long().reshape(1, K))
    print("ids[0:16]", ids[:16].tolist(), flush=True)
    print("ref ids[0:16]", ref_i[:16].tolist(), flush=True)
    print("value abs err", float((values - got_logits).abs().max()), flush=True)
    print("ref value max abs err", float((values - ref_v).abs().max()), flush=True)
    if not torch.equal(ids, ref_i):
        print("id mismatch count", int((ids != ref_i).sum()), flush=True)

    paths = sorted(glob.glob(os.path.join(BUILD, "cache", "lm_head_topk.*.compiled.axmodel")),
                   key=os.path.getmtime)
    assert paths, glob.glob(os.path.join(BUILD, "cache", "*"))
    name = f"lm_head_topk_n{V}_k{H}_t{K}_c{CORES}"
    dst = board_io.kernel_file(OUT, name, CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)
    json.dump({"V": V, "H": H, "K": K, "cores": CORES, "chunks": CHUNKS,
               "tile": TILE, "real": REAL}, open(os.path.join(OUT, name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
