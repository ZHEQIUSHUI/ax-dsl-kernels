"""Build a gap35-compatible combined final-norm lm-head with greedy argmax.

The RMSNorm and MXFP4 projection below use the same schedule as the embedded
gap35 head. Each vocabulary tile is written to ``out`` unchanged and reduced
to one (max, index) pair. Core 0 then scans the tile pairs in token order.

Inputs: x [ROWS,H] bf16, norm_w [H] f32, w [V,H/2] MXFP4, s [V,H/32] E8M0.
Outputs: out [ROWS,V] f32, ids [ROWS] i32, top_logits [ROWS] f32.
Four-row builds also take m [1] i32 and zero the invalid tail rows.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "qwen35_4b", "whole", "lm4"))
import board_io  # noqa: E402

V = int(os.environ.get("V", "248320"))
H = int(os.environ.get("H", "2048"))
ROWS = int(os.environ.get("ROWS", "1"))
CORES = int(os.environ.get("CORES", "4"))
EPS = float(os.environ.get("EPS", "1e-6"))
TILE_N = int(os.environ.get("TILE_N", "256"))
UNROLL_N = 2
TAG = os.environ.get("TAG", f"gap35_combined_m{ROWS}")
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "out", TAG)))
DATA = os.path.abspath(os.environ.get("DATA", "/data/junda/work/qsmall/argmax/data/qwen35_35b_combined"))
BUILD = os.path.abspath(os.environ.get("BUILD", os.path.join(ROOT, "build", TAG)))


def combined_head(x, norm_w, w, s, ids, top_logits, out, m=None):
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    assert core_num == CORES and ROWS in (1, 4)
    tile_m = ROWS
    shared_m = ROWS
    assert x.shape == (ROWS, H) and w.shape[0] == V
    assert x.shape[-1] % core_num == 0

    shared_mem_type = ax.MemoryType.SHARED_OCM
    hidden_per_core = H // core_num
    hidden_slice = slice(core_id * hidden_per_core, (core_id + 1) * hidden_per_core)
    hidden_size = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=H)
    eps_value = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=EPS)
    total_n_tiles = math.ceil(V / TILE_N)

    normalized_shared = ax.dsl.create_tensor(
        (shared_m, H), ax.bf16, shared_mem_type, name="gap35_combined_normalized"
    )
    tile_max = ax.dsl.create_tensor(
        (ROWS, total_n_tiles), ax.fp32, shared_mem_type, name="gap35_combined_tile_max"
    )
    tile_id = ax.dsl.create_tensor(
        (ROWS, total_n_tiles), ax.int32, shared_mem_type, name="gap35_combined_tile_id"
    )

    # Stage 1: the gap35 RMSNorm, with each core owning one hidden shard.
    a_tile = ax.dsl.load(x)
    norm_weight_tile = ax.dsl.load(norm_w[hidden_slice])
    square = ax.dsl.mul(a_tile[:, hidden_slice], a_tile[:, hidden_slice], output_dtype=ax.fp32)
    partial_sums = ax.dsl.create_tensor(
        (core_num, ROWS), ax.fp32, shared_mem_type, name="gap35_combined_partial_sums"
    )
    ax.dsl.sum(
        square,
        dim=-1,
        keepdim=True,
        output_dtype=ax.fp32,
        r=partial_sums[core_id:core_id + 1, :].reshape(ROWS, 1),
    )
    axd.pub(partial_sums)
    axd.sub(partial_sums)
    square_sum = ax.dsl.sum(partial_sums, dim=0, keepdim=True, output_dtype=ax.fp32)
    square_sum = square_sum.reshape(ROWS, 1)
    variance = ax.dsl.div(square_sum, hidden_size, output_dtype=ax.fp32)
    rms = ax.dsl.sqrt(ax.dsl.add(variance, eps_value, output_dtype=ax.fp32), output_dtype=ax.fp32)
    inv_rms = ax.dsl.reciprocal(rms, output_dtype=ax.fp32)
    normalized = ax.dsl.mul(a_tile[:, hidden_slice], inv_rms, output_dtype=ax.fp32)
    ax.dsl.mul(
        normalized,
        norm_weight_tile,
        output_dtype=ax.bf16,
        r=normalized_shared[:, hidden_slice],
    )
    axd.pub(normalized_shared)
    axd.sub(normalized_shared)

    def reduce_tile(value, tile_idx, n_slice):
        # The AXReduceMax/AxArgMax schedules accept one 256-wide row. Four
        # rows as a batched reduction do not match their vector cycles.
        for r in range(ROWS):
            ax.dsl.max(value[r:r + 1, :], dim=-1, keepdim=True,
                       r=tile_max[r:r + 1, tile_idx:tile_idx + 1])
            ax.dsl.argmax(value[r:r + 1, :], dim=-1, keepdim=True,
                          r=tile_id[r:r + 1, tile_idx:tile_idx + 1])

    # Stage 2: the gap35 projection order. The only addition is the per-tile
    # reduction after the unchanged store of the full logits.
    total_groups = total_n_tiles // UNROLL_N
    for group_idx in range(core_id, total_groups, core_num):
        group_base = group_idx * UNROLL_N * TILE_N
        slices = []
        weights = []
        scales = []
        for u in range(UNROLL_N):
            offset_n = group_base + u * TILE_N
            n_end = min(V, offset_n + TILE_N)
            n_slice = slice(offset_n, n_end)
            slices.append(n_slice)
            weights.append(ax.dsl.load(w[n_slice, :]))
            scales.append(ax.dsl.load(s[n_slice, :]))
        for u in range(UNROLL_N):
            tile_idx = group_idx * UNROLL_N + u
            value = ax.dsl.matmul(
                normalized_shared,
                weights[u],
                w_scale=scales[u],
                transposed=True,
                output_dtype=ax.fp32,
            )
            ax.dsl.store(r=out[:, slices[u]], x=value)
            reduce_tile(value, tile_idx, slices[u])

    for tile_idx in range(total_groups * UNROLL_N, total_n_tiles):
        if tile_idx % core_num != core_id:
            continue
        offset_n = tile_idx * TILE_N
        n_end = min(V, offset_n + TILE_N)
        n_slice = slice(offset_n, n_end)
        value = ax.dsl.matmul(
            normalized_shared,
            ax.dsl.load(w[n_slice, :]),
            w_scale=ax.dsl.load(s[n_slice, :]),
            transposed=True,
            output_dtype=ax.fp32,
        )
        ax.dsl.store(r=out[:, n_slice], x=value)
        reduce_tile(value, tile_idx, n_slice)

    axd.sync_all()

    if core_id == 0:
        final_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                       name="gap35_combined_final_index")
        final_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                       name="gap35_combined_final_value")
        final_p = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                       name="gap35_combined_final_position")
        for r in range(ROWS):
            ax.dsl.max(tile_max[r:r + 1, :], dim=-1, keepdim=True, r=final_v[r:r + 1, :])
            ax.dsl.argmax(tile_max[r:r + 1, :], dim=-1, keepdim=True, r=final_p[r:r + 1, :])
            ax.dsl.gather(tile_id[r:r + 1, :], final_p[r:r + 1, :], dim=-1,
                          r=final_i[r:r + 1, :])
        step = ax.dsl.full((ROWS, 1), ax.int32, fill_value=TILE_N)
        offset = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                      name="gap35_combined_final_offset")
        ax.dsl.mul(final_p, step, r=offset)
        ax.dsl.add(final_i, offset, r=final_i)
        if m is not None:
            idx = ax.dsl.from_numpy(np.arange(1, ROWS + 1, dtype=np.int32))
            one = ax.dsl.full((1,), ax.int32, fill_value=1)
            zero = ax.dsl.zeros((1,), ax.int32)
            valid = ax.dsl.minimum(
                ax.dsl.maximum(ax.dsl.load(m) - idx + one, zero), one, output_dtype=ax.uint8
            ).reshape(ROWS)
            mask_i = ax.dsl.cast(valid, dtype=ax.int32)
            mask_f = ax.dsl.cast(valid, dtype=ax.fp32)
            ax.dsl.store(ax.dsl.mul(final_i.reshape(ROWS), mask_i, output_dtype=ax.int32), r=ids)
            ax.dsl.store(ax.dsl.mul(final_v.reshape(ROWS), mask_f, output_dtype=ax.fp32),
                         r=top_logits)
        else:
            ax.dsl.store(final_i.reshape(ROWS), r=ids)
            ax.dsl.store(final_v.reshape(ROWS), r=top_logits)
    return ids, top_logits, out


def load_bin(name, dtype, shape):
    raw = np.fromfile(os.path.join(DATA, name), dtype=np.uint8)
    if dtype == torch.bfloat16:
        raw = raw.view(np.uint16)
    elif dtype == torch.float32:
        raw = raw.view(np.float32)
    elif dtype == torch.int32:
        raw = raw.view(np.int32)
    return torch.from_numpy(raw.copy()).view(dtype).reshape(shape)


def report(ref, ids, top, out):
    ok = True
    for r in range(ROWS):
        ref_id = int(ref["ids"][r])
        got_id = int(ids[r])
        ref_top = float(ref["logits"][r])
        got_top = float(top[r])
        full_diff = float((out[r].float() - ref["logits_all"][r].float()).abs().max())
        margin = float(ref["top2"][r, 0] - ref["top2"][r, 1])
        print(
            f"row {r}: id ref={ref_id} got={got_id} {'OK' if ref_id == got_id else 'MISMATCH'}; "
            f"logit ref={ref_top:.9g} got={got_top:.9g} diff={abs(got_top-ref_top):.3e}; "
            f"full_logits_maxdiff={full_diff:.3e}; top1-top2={margin:.3e}",
            flush=True,
        )
        ok = ok and ref_id == got_id
    return ok


def main():
    assert ROWS in (1, 4) and CORES == 4
    assert V % TILE_N == 0, (V, TILE_N)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)

    x = load_bin("x.bin", torch.bfloat16, (ROWS, H))
    norm = load_bin("norm.bin", torch.float32, (H,))
    w = load_bin("w.bin", torch.uint8, (V, H // 2))
    s = load_bin("s.bin", torch.uint8, (V, H // 32))
    ref = torch.load(os.path.join(DATA, "ref_argmax.pt"))
    kw = w.view(torch.float4_e2m1fn_x2)
    ks = s.view(torch.float8_e8m0fnu)

    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    k = ax.builder(
        run_mode=ax.RunMode.CRUN,
        enable_cache=True,
        cache_path=os.path.join(BUILD, "cache"),
        dump_path=BUILD,
    )(combined_head)
    ids = torch.zeros(ROWS, dtype=torch.int32)
    top = torch.zeros(ROWS, dtype=torch.float32)
    out = torch.zeros(ROWS, V, dtype=torch.float32)
    if ROWS == 1:
        k[CORES](x, norm, kw, ks, ids, top, out)
    else:
        ms = [int(v) for v in os.environ.get("MCHECK", "1,2,3,4").split(",") if v]
        assert ms and ms[-1] == ROWS
        for m in ms:
            zi = torch.zeros(ROWS, dtype=torch.int32)
            zv = torch.zeros(ROWS, dtype=torch.float32)
            zo = torch.zeros(ROWS, V, dtype=torch.float32)
            k[CORES](x, norm, kw, ks, zi, zv, zo, torch.tensor([m], dtype=torch.int32))
            ids[:m] = zi[:m]
            top[:m] = zv[:m]
            out[:m] = zo[:m]
            ok = bool((zi[m:] == 0).all() and (zv[m:] == 0).all())
            print(f"m={m}: invalid tails zero {ok}", flush=True)
            if not ok:
                raise SystemExit("invalid tail is not zero")

    print("run time %.1fs" % (time.time() - t0), flush=True)
    if not report(ref, ids, top, out):
        raise SystemExit("combined argmax mismatch")

    paths = sorted(glob.glob(os.path.join(BUILD, "cache", "*.compiled.axmodel")), key=os.path.getmtime)
    assert paths, glob.glob(os.path.join(BUILD, "cache", "*"))
    suffix = "" if ROWS == 1 else "_m4"
    name = f"lm_head_combined_gap35_n{V}_k{H}_c{CORES}{suffix}"
    dst = board_io.kernel_file(OUT, name, CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)
    json.dump(
        {"V": V, "H": H, "rows": ROWS, "cores": CORES, "tile_n": TILE_N,
         "unroll_n": UNROLL_N, "eps": EPS, "kind": "combined_gap35"},
        open(os.path.join(OUT, name + ".json"), "w"), indent=2,
    )


if __name__ == "__main__":
    main()
