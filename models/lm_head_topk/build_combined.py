"""Build the combined final-RMSNorm + lm-head kernel: full logits and greedy argmax.

This keeps the upstream four-core RMSNorm+MX GEMM schedule used by the 35B
decode path, writes the complete vocabulary row exactly as the ordinary
lm_head does, and adds a four-core greedy reduction over those logits.  The
reduction is in contiguous vocabulary order, so an exact tie still resolves to
the lowest token id.

The greedy host path reads only ``ids`` and ``top_logits`` and skips the full
logits invalidate; every other path reads ``out`` unchanged.
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
HEAD = os.environ.get("HEAD", "mx4").lower()
ROWS = int(os.environ.get("ROWS", "1"))
CORES = int(os.environ.get("CORES", "4"))
CHUNKS = int(os.environ.get("CHUNKS", "8"))
EPS = float(os.environ.get("EPS", "1e-6"))
TILE_N = int(os.environ.get("TILE_N", "256"))
TILE = int(os.environ.get("TILE", "0")) or max(
    16, min(1024, ((512 if HEAD == "mx8" else 256) * 1024) //
            (H if HEAD == "mx8" else H // 2))
)
TAG = os.environ.get("TAG", "combined")
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "out")))
DATA = os.path.abspath(os.environ.get("DATA", os.path.join(ROOT, "data")))
BUILD = os.path.abspath(os.environ.get("BUILD", os.path.join(ROOT, "build", TAG)))


def _linear(x, w, s, rows, name, out):
    return linear(x, w[rows], s[rows], name, output_dtype=ax.fp32,
                  oc_tile_size=TILE, r=out)


def _rmsnorm(x, weight):
    # The host path is fp32 reduce/divide, fp32 weight multiply, then bf16 for
    # the MX projection. Keep the same rounding boundary here.
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden = ax.dsl.full((1,), ax.fp32, fill_value=H)
    eps = ax.dsl.full((1,), ax.fp32, fill_value=EPS)
    denom = ax.dsl.sqrt(ax.dsl.add(ax.dsl.div(square_sum, hidden), eps))
    normalized = ax.dsl.div(x, denom, output_dtype=ax.fp32)
    return ax.dsl.mul(normalized, weight.reshape(1, H), output_dtype=ax.bf16)


def combined_head(x, norm_w, w, s, ids, top_logits, out):
    core = ax.program_id(0)
    per_core = V // CORES
    chunk_rows = per_core // CHUNKS
    rows0 = core * per_core
    xh = _rmsnorm(ax.dsl.load(x), ax.dsl.load(norm_w))
    chunk_max = ax.dsl.create_tensor((ROWS, CHUNKS), ax.fp32, ax.MemoryType.OCM,
                                     name="combined_chunk_max")
    chunk_id = ax.dsl.create_tensor((ROWS, CHUNKS), ax.int32, ax.MemoryType.OCM,
                                    name="combined_chunk_id")
    for c in range(CHUNKS):
        cols = slice(rows0 + c * chunk_rows, rows0 + (c + 1) * chunk_rows)
        raw = ax.dsl.create_tensor((ROWS, chunk_rows), ax.fp32, ax.MemoryType.OCM,
                                   name=f"combined_raw_{c}")
        _linear(xh, w, s, cols, f"lm_head_combined_{c}", raw)
        ax.dsl.store(r=out[:, cols], x=raw)
        for r in range(ROWS):
            ax.dsl.max(raw[r:r + 1], dim=-1, keepdim=True,
                       r=chunk_max[r:r + 1, c:c + 1])
            ax.dsl.argmax(raw[r:r + 1], dim=-1, keepdim=True,
                          r=chunk_id[r:r + 1, c:c + 1])

    local_pos = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                     name="combined_local_pos")
    local_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                   name="combined_local_value")
    local_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                   name="combined_local_id")
    for r in range(ROWS):
        ax.dsl.max(chunk_max[r:r + 1], dim=-1, keepdim=True, r=local_v[r:r + 1])
        ax.dsl.argmax(chunk_max[r:r + 1], dim=-1, keepdim=True, r=local_pos[r:r + 1])
        ax.dsl.gather(chunk_id[r:r + 1], local_pos[r:r + 1], dim=-1,
                      r=local_i[r:r + 1])
    chunk_step = ax.dsl.full((ROWS, 1), ax.int32, fill_value=chunk_rows)
    chunk_off = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                     name="combined_chunk_offset")
    ax.dsl.mul(local_pos, chunk_step, r=chunk_off)
    ax.dsl.add(local_i, chunk_off, r=local_i)
    off = ax.dsl.full((ROWS, 1), ax.int32, fill_value=rows0)
    ax.dsl.add(local_i, off, r=local_i)

    cross_v = ax.dsl.create_tensor((ROWS, CORES), ax.fp32, ax.MemoryType.SHARED_OCM,
                                   name="combined_cross_values")
    cross_i = ax.dsl.create_tensor((ROWS, CORES), ax.int32, ax.MemoryType.SHARED_OCM,
                                   name="combined_cross_ids")
    ax.dsl.move(local_v, r=cross_v[:, core:core + 1])
    ax.dsl.move(local_i, r=cross_i[:, core:core + 1])
    axd.pub(cross_v)
    axd.pub(cross_i)
    axd.sub(cross_v)
    axd.sub(cross_i)

    if core == 0:
        final_pos = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                         name="combined_final_pos")
        final_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                       name="combined_final_id")
        final_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                       name="combined_final_value")
        for r in range(ROWS):
            ax.dsl.argmax(cross_v[r:r + 1], dim=-1, keepdim=True,
                          r=final_pos[r:r + 1])
            ax.dsl.gather(cross_i[r:r + 1], final_pos[r:r + 1], dim=-1,
                          r=final_i[r:r + 1])
            ax.dsl.max(cross_v[r:r + 1], dim=-1, keepdim=True,
                       r=final_v[r:r + 1])
        ax.dsl.store(final_i.reshape(ROWS), r=ids)
        ax.dsl.store(final_v.reshape(ROWS), r=top_logits)
    return ids, top_logits, out


def lm_head_combined_m4(x, norm_w, w, s, m, ids, top_logits, out):
    combined_head(x, norm_w, w, s, ids, top_logits, out)
    core = ax.program_id(0)
    if core == 0:
        idx = ax.dsl.from_numpy(np.arange(1, ROWS + 1, dtype=np.int32))
        one = ax.dsl.full((1,), ax.int32, fill_value=1)
        zero = ax.dsl.zeros((1,), ax.int32)
        valid = ax.dsl.minimum(ax.dsl.maximum(
            ax.dsl.load(m) - idx + one, zero), one, output_dtype=ax.uint8).reshape(ROWS)
        mask_i = ax.dsl.cast(valid, dtype=ax.int32)
        mask_f = ax.dsl.cast(valid, dtype=ax.fp32)
        ax.dsl.store(ax.dsl.mul(ax.dsl.load(ids), mask_i, output_dtype=ax.int32), r=ids)
        ax.dsl.store(ax.dsl.mul(ax.dsl.load(top_logits), mask_f, output_dtype=ax.fp32), r=top_logits)
    return ids, top_logits, out


def load_bin(name, dtype, shape):
    a = np.fromfile(os.path.join(DATA, name), dtype=np.uint8)
    if dtype == torch.bfloat16:
        a = a.view(np.uint16)
    elif dtype == torch.float32:
        a = a.view(np.float32)
    elif dtype == torch.int32:
        a = a.view(np.int32)
    return torch.from_numpy(a.copy()).view(dtype).reshape(shape)


def report(ref, ids, top, logits):
    ok = True
    all_logits = ref["logits_all"]
    for r in range(ROWS):
        rid = int(ref["ids"][r])
        gid = int(ids[r])
        rl = float(ref["logits"][r])
        gl = float(top[r])
        d = abs(gl - rl)
        full_d = float((logits[r].float() - all_logits[r].float()).abs().max())
        near = float(ref["top2"][r, 0] - ref["top2"][r, 1])
        tie = bool(ref["top2"][r, 0] == ref["top2"][r, 1])
        same = rid == gid
        print(f"row {r}: id ref={rid} got={gid} {'OK' if same else 'MISMATCH'}; "
              f"logit ref={rl:.9g} got={gl:.9g} diff={d:.3e}; full_logits_maxdiff={full_d:.3e}; "
              f"top1-top2={near:.3e} exact_tie={int(tie)}", flush=True)
        if not same:
            for name, i in (("ref", rid), ("got", gid)):
                v = float(all_logits[r, i])
                print(f"  mismatch {name} id={i} logit={v:.9g} delta_vs_refmax={v-rl:.3e}", flush=True)
            ok = False
    return ok


def main():
    assert ROWS in (1, 4), ROWS
    assert HEAD in ("mx4", "mx8"), HEAD
    assert CORES == 4 and V % (CORES * CHUNKS) == 0, (V, CORES, CHUNKS)
    assert H % 32 == 0, H
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)

    x = load_bin("x.bin", torch.bfloat16, (ROWS, H))
    norm = load_bin("norm.bin", torch.float32, (H,))
    w = load_bin("w.bin", torch.uint8, (V, H // 2 if HEAD == "mx4" else H))
    s = load_bin("s.bin", torch.uint8, (V, H // 32))
    ref = torch.load(os.path.join(DATA, "ref_argmax.pt"))
    kw = w.view(torch.float4_e2m1fn_x2) if HEAD == "mx4" else w.view(torch.float8_e4m3fn)
    ks = s.view(torch.float8_e8m0fnu)

    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    fn = combined_head if ROWS == 1 else lm_head_combined_m4
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True,
                   cache_path=os.path.join(BUILD, "cache"), dump_path=BUILD)(fn)
    ids = torch.zeros(ROWS, dtype=torch.int32)
    top = torch.zeros(ROWS, dtype=torch.float32)
    out = torch.zeros(ROWS, V, dtype=torch.float32)
    if ROWS == 1:
        k[CORES](x[0:1].contiguous(), norm, kw, ks, ids[0:1], top[0:1], out[0:1])
    else:
        ms = [int(v) for v in os.environ.get("MCHECK", "1,2,3,4").split(",") if v]
        assert ms and ms[-1] == ROWS, ms
        for m in ms:
            zi = torch.zeros(ROWS, dtype=torch.int32)
            zv = torch.zeros(ROWS, dtype=torch.float32)
            zo = torch.zeros(ROWS, V, dtype=torch.float32)
            k[CORES](x.contiguous(), norm, kw, ks,
                     torch.tensor([m], dtype=torch.int32), zi, zv, zo)
            ids[:m] = zi[:m]
            top[:m] = zv[:m]
            out[:m] = zo[:m]
            tail_ok = bool((zi[m:] == 0).all() and (zv[m:] == 0).all())
            print(f"m={m}: invalid argmax tails zero {tail_ok}", flush=True)
            if not tail_ok:
                raise SystemExit("invalid tail not zero")
    print("run time %.1fs" % (time.time() - t0), flush=True)
    if not report(ref, ids, top, out):
        raise SystemExit("combined argmax mismatch")

    paths = sorted(glob.glob(os.path.join(BUILD, "cache", "*.compiled.axmodel")),
                   key=os.path.getmtime)
    assert paths, glob.glob(os.path.join(BUILD, "cache", "*"))
    suffix = "" if ROWS == 1 else "_m4"
    name = f"lm_head_combined_n{V}_k{H}_c{CORES}{suffix}"
    dst = board_io.kernel_file(OUT, name, CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)
    json.dump({"V": V, "H": H, "head": HEAD, "rows": ROWS, "cores": CORES,
               "chunks": CHUNKS, "tile": TILE, "eps": EPS, "kind": "combined"},
              open(os.path.join(OUT, name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
