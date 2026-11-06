"""Build the fused final-RMSNorm + lm-head + greedy-argmax kernel for Qwen3.5/3.8.

One call takes the last layer's raw hidden rows, applies the model's final
RMSNorm, projects the vocabulary, and returns only each row's greedy token id
and its logit.  This is the K=1 special case of lm_head.topk without penalties,
sumexp or a top-K merge.

The vocabulary is split over four cores.  Every core reduces its CHUNKS blocks
to one (max, index); core 0 reduces the four core pairs.  Both reductions use
AxArgMax's first-index tie rule, and blocks/cores are in increasing token order,
so exact ties resolve to the lowest vocabulary id, matching llama.cpp greedy.
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
REAL = os.environ.get("REAL", "0") == "1"
TILE = int(os.environ.get("TILE", "0")) or max(
    16, min(1024, ((512 if HEAD == "mx8" else 256) * 1024) //
            (H if HEAD == "mx8" else H // 2))
)
MCHECK = [int(v) for v in os.environ.get("MCHECK", "1,2,3,4").split(",") if v]
TAG = os.environ.get("TAG", "real" if REAL else "synthetic")
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "out")))
DATA = os.path.abspath(os.environ.get(
    "DATA", os.path.join(ROOT, "data" if REAL else "data_synth")))
BUILD = os.path.abspath(os.environ.get("BUILD", os.path.join(ROOT, "build", TAG)))


def _linear(x, w, s, rows, name, out):
    return linear(x, w[rows], s[rows], name, output_dtype=ax.fp32,
                  oc_tile_size=TILE, r=out)


def _rmsnorm(x, weight):
    # The host path is fp32 reduce/divide, fp32 weight multiply, then bf16 for
    # the MX projection.  Keep the same rounding boundary here.
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden = ax.dsl.full((1,), ax.fp32, fill_value=H)
    eps = ax.dsl.full((1,), ax.fp32, fill_value=EPS)
    denom = ax.dsl.sqrt(ax.dsl.add(ax.dsl.div(square_sum, hidden), eps))
    normalized = ax.dsl.div(x, denom, output_dtype=ax.fp32)
    return ax.dsl.mul(normalized, weight.reshape(1, H), output_dtype=ax.bf16)


def lm_head_argmax(x_in, norm_w, w, s, ids, logits):
    core = ax.program_id(0)
    per_core = V // CORES
    chunk_rows = per_core // CHUNKS
    rows0 = core * per_core

    xh = _rmsnorm(ax.dsl.load(x_in), ax.dsl.load(norm_w))
    chunk_max = ax.dsl.create_tensor((ROWS, CHUNKS), ax.fp32, ax.MemoryType.OCM,
                                     name="argmax_chunk_max")
    chunk_id = ax.dsl.create_tensor((ROWS, CHUNKS), ax.int32, ax.MemoryType.OCM,
                                    name="argmax_chunk_id")
    for c in range(CHUNKS):
        rows = slice(rows0 + c * chunk_rows, rows0 + (c + 1) * chunk_rows)
        raw = ax.dsl.create_tensor((ROWS, chunk_rows), ax.fp32, ax.MemoryType.OCM,
                                   name=f"argmax_raw_{c}")
        _linear(xh, w, s, rows, f"lm_head_argmax_{c}", raw)
        for r in range(ROWS):
            ax.dsl.max(raw[r:r + 1], dim=-1, keepdim=True,
                       r=chunk_max[r:r + 1, c:c + 1])
            ax.dsl.argmax(raw[r:r + 1], dim=-1, keepdim=True,
                          r=chunk_id[r:r + 1, c:c + 1])

    local_pos = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                     name="argmax_local_pos")
    local_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                   name="argmax_local_value")
    local_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                   name="argmax_local_id")
    for r in range(ROWS):
        ax.dsl.max(chunk_max[r:r + 1], dim=-1, keepdim=True, r=local_v[r:r + 1])
        ax.dsl.argmax(chunk_max[r:r + 1], dim=-1, keepdim=True, r=local_pos[r:r + 1])
        ax.dsl.gather(chunk_id[r:r + 1], local_pos[r:r + 1], dim=-1,
                      r=local_i[r:r + 1])
    # the chunk's first token: the per-chunk argmax is relative to its chunk, the core's to its range
    chunk_step = ax.dsl.full((ROWS, 1), ax.int32, fill_value=chunk_rows)
    chunk_off = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                     name="argmax_chunk_offset")
    ax.dsl.mul(local_pos, chunk_step, r=chunk_off)
    ax.dsl.add(local_i, chunk_off, r=local_i)
    off = ax.dsl.full((ROWS, 1), ax.int32, fill_value=rows0)
    ax.dsl.add(local_i, off, r=local_i)

    cross_v = ax.dsl.create_tensor((ROWS, CORES), ax.fp32, ax.MemoryType.SHARED_OCM,
                                   name="argmax_cross_values")
    cross_i = ax.dsl.create_tensor((ROWS, CORES), ax.int32, ax.MemoryType.SHARED_OCM,
                                   name="argmax_cross_ids")
    ax.dsl.move(local_v, r=cross_v[:, core:core + 1])
    ax.dsl.move(local_i, r=cross_i[:, core:core + 1])
    axd.pub(cross_v)
    axd.pub(cross_i)
    axd.sub(cross_v)
    axd.sub(cross_i)

    if core == 0:
        final_pos = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                         name="argmax_final_pos")
        final_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                       name="argmax_final_id")
        final_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                       name="argmax_final_value")
        for r in range(ROWS):
            ax.dsl.argmax(cross_v[r:r + 1], dim=-1, keepdim=True,
                          r=final_pos[r:r + 1])
            ax.dsl.gather(cross_i[r:r + 1], final_pos[r:r + 1], dim=-1,
                          r=final_i[r:r + 1])
            ax.dsl.max(cross_v[r:r + 1], dim=-1, keepdim=True,
                       r=final_v[r:r + 1])
        ax.dsl.store(final_i.reshape(ROWS), r=ids)
        ax.dsl.store(final_v.reshape(ROWS), r=logits)
    return ids, logits


def lm_head_argmax_m4(x_in, norm_w, w, s, m, ids, logits):
    core = ax.program_id(0)
    per_core = V // CORES
    chunk_rows = per_core // CHUNKS
    rows0 = core * per_core

    xh = _rmsnorm(ax.dsl.load(x_in), ax.dsl.load(norm_w))
    chunk_max = ax.dsl.create_tensor((ROWS, CHUNKS), ax.fp32, ax.MemoryType.OCM,
                                     name="argmax_chunk_max")
    chunk_id = ax.dsl.create_tensor((ROWS, CHUNKS), ax.int32, ax.MemoryType.OCM,
                                    name="argmax_chunk_id")
    for c in range(CHUNKS):
        rows = slice(rows0 + c * chunk_rows, rows0 + (c + 1) * chunk_rows)
        raw = ax.dsl.create_tensor((ROWS, chunk_rows), ax.fp32, ax.MemoryType.OCM,
                                   name=f"argmax_raw_{c}")
        _linear(xh, w, s, rows, f"lm_head_argmax_{c}", raw)
        for r in range(ROWS):
            ax.dsl.max(raw[r:r + 1], dim=-1, keepdim=True,
                       r=chunk_max[r:r + 1, c:c + 1])
            ax.dsl.argmax(raw[r:r + 1], dim=-1, keepdim=True,
                          r=chunk_id[r:r + 1, c:c + 1])

    local_pos = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                     name="argmax_local_pos")
    local_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                   name="argmax_local_value")
    local_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                   name="argmax_local_id")
    for r in range(ROWS):
        ax.dsl.max(chunk_max[r:r + 1], dim=-1, keepdim=True, r=local_v[r:r + 1])
        ax.dsl.argmax(chunk_max[r:r + 1], dim=-1, keepdim=True, r=local_pos[r:r + 1])
        ax.dsl.gather(chunk_id[r:r + 1], local_pos[r:r + 1], dim=-1,
                      r=local_i[r:r + 1])
    chunk_step = ax.dsl.full((ROWS, 1), ax.int32, fill_value=chunk_rows)
    chunk_off = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                     name="argmax_chunk_offset")
    ax.dsl.mul(local_pos, chunk_step, r=chunk_off)
    ax.dsl.add(local_i, chunk_off, r=local_i)
    off = ax.dsl.full((ROWS, 1), ax.int32, fill_value=rows0)
    ax.dsl.add(local_i, off, r=local_i)

    cross_v = ax.dsl.create_tensor((ROWS, CORES), ax.fp32, ax.MemoryType.SHARED_OCM,
                                   name="argmax_cross_values")
    cross_i = ax.dsl.create_tensor((ROWS, CORES), ax.int32, ax.MemoryType.SHARED_OCM,
                                   name="argmax_cross_ids")
    ax.dsl.move(local_v, r=cross_v[:, core:core + 1])
    ax.dsl.move(local_i, r=cross_i[:, core:core + 1])
    axd.pub(cross_v)
    axd.pub(cross_i)
    axd.sub(cross_v)
    axd.sub(cross_i)

    if core == 0:
        final_pos = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                         name="argmax_final_pos")
        final_i = ax.dsl.create_tensor((ROWS, 1), ax.int32, ax.MemoryType.OCM,
                                       name="argmax_final_id")
        final_v = ax.dsl.create_tensor((ROWS, 1), ax.fp32, ax.MemoryType.OCM,
                                       name="argmax_final_value")
        for r in range(ROWS):
            ax.dsl.argmax(cross_v[r:r + 1], dim=-1, keepdim=True,
                          r=final_pos[r:r + 1])
            ax.dsl.gather(cross_i[r:r + 1], final_pos[r:r + 1], dim=-1,
                          r=final_i[r:r + 1])
            ax.dsl.max(cross_v[r:r + 1], dim=-1, keepdim=True,
                       r=final_v[r:r + 1])
        idx = ax.dsl.from_numpy(np.arange(1, ROWS + 1, dtype=np.int32))
        one = ax.dsl.full((1,), ax.int32, fill_value=1)
        zero = ax.dsl.zeros((1,), ax.int32)
        valid = ax.dsl.minimum(ax.dsl.maximum(
            ax.dsl.load(m) - idx + one, zero), one, output_dtype=ax.uint8).reshape(ROWS)
        mask_i = ax.dsl.cast(valid, dtype=ax.int32)
        mask_f = ax.dsl.cast(valid, dtype=ax.fp32)
        ax.dsl.store(ax.dsl.mul(final_i.reshape(ROWS), mask_i, output_dtype=ax.int32), r=ids)
        ax.dsl.store(ax.dsl.mul(final_v.reshape(ROWS), mask_f, output_dtype=ax.fp32), r=logits)
    return ids, logits


def load_bin(name, dtype, shape):
    a = np.fromfile(os.path.join(DATA, name), dtype=np.uint8)
    if dtype == torch.float32:
        a = a.view(np.float32)
    elif dtype == torch.int32:
        a = a.view(np.int32)
    return torch.from_numpy(a.copy()).view(dtype).reshape(shape)


def load_real():
    x = load_bin("x.bin", torch.float32, (ROWS, H))
    norm = load_bin("norm.bin", torch.float32, (H,))
    w = load_bin("w.bin", torch.uint8, (V, H // 2 if HEAD == "mx4" else H))
    s = load_bin("s.bin", torch.uint8, (V, H // 32))
    ref = torch.load(os.path.join(DATA, "ref_argmax.pt"))
    return x, norm, w, s, ref


def make_ref(x, norm, w, s):
    sys.path.insert(0, ROOT)
    from quant import torch_logits
    xf = x.float()
    rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * xf
    xn = (rms * norm).bfloat16()
    logits = torch_logits(xn, w, s, HEAD)
    ids = torch.argmax(logits, dim=-1).to(torch.int32)
    top = torch.gather(logits, 1, ids.long().reshape(ROWS, 1)).reshape(ROWS)
    vals, idx = torch.topk(logits, 2, dim=-1)
    return {"ids": ids, "logits": top, "logits_all": logits,
            "top2": vals, "top2_ids": idx}


def report(ref, ids, logits):
    ok = True
    for r in range(ROWS):
        rid = int(ref["ids"][r])
        gid = int(ids[r])
        rl = float(ref["logits"][r])
        gl = float(logits[r])
        d = abs(gl - rl)
        tie = bool(ref["top2"][r, 0] == ref["top2"][r, 1])
        near = float(ref["top2"][r, 0] - ref["top2"][r, 1])
        same = rid == gid
        print(f"row {r}: id ref={rid} got={gid} {'OK' if same else 'MISMATCH'}; "
              f"logit ref={rl:.9g} got={gl:.9g} diff={d:.3e}; "
              f"top1-top2={near:.3e} exact_tie={int(tie)}", flush=True)
        if not same:
            for name, i in (("ref", rid), ("got", gid)):
                v = float(ref["logits_all"][r, i])
                print(f"  mismatch {name} id={i} logit={v:.9g} delta_vs_refmax={v-rl:.3e}", flush=True)
            ok = False
    return ok


def main():
    assert ROWS in (1, 4), ROWS
    assert HEAD in ("mx4", "mx8"), HEAD
    assert CORES == 4 and V % (CORES * CHUNKS) == 0, (V, CORES, CHUNKS)
    assert H % 32 == 0 and V // CORES // CHUNKS >= 1
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    x, norm, w, s, ref = load_real()
    if HEAD == "mx4":
        kw = w.view(torch.float4_e2m1fn_x2)
    else:
        kw = w.view(torch.float8_e4m3fn)
    ks = s.view(torch.float8_e8m0fnu)
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    fn = lm_head_argmax if ROWS == 1 else lm_head_argmax_m4
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True,
                   cache_path=os.path.join(BUILD, "cache"),
                   dump_path=BUILD)(fn)
    ids = torch.zeros(ROWS, dtype=torch.int32)
    logits = torch.zeros(ROWS, dtype=torch.float32)
    if ROWS == 1:
        k[CORES](x[0:1].contiguous(), norm, kw, ks, ids[0:1], logits[0:1])
    else:
        ms = [m for m in MCHECK if 1 <= m <= ROWS]
        assert ROWS in ms and ms[-1] == ROWS, (MCHECK, ROWS)
        for m in ms:
            zi = torch.zeros(ROWS, dtype=torch.int32)
            zv = torch.zeros(ROWS, dtype=torch.float32)
            k[CORES](x.contiguous(), norm, kw, ks,
                     torch.tensor([m], dtype=torch.int32), zi, zv)
            ids[:m] = zi[:m]
            logits[:m] = zv[:m]
            tail_ok = bool((zi[m:] == 0).all() and (zv[m:] == 0).all())
            print(f"m={m}: invalid tails zero {tail_ok}", flush=True)
            if not tail_ok:
                raise SystemExit("invalid tail not zero")
    print("run time %.1fs" % (time.time() - t0), flush=True)
    ok = report(ref, ids, logits)
    if not ok:
        raise SystemExit("argmax mismatch")

    paths = sorted(glob.glob(os.path.join(BUILD, "cache", "lm_head_argmax*.compiled.axmodel")),
                   key=os.path.getmtime)
    assert paths, glob.glob(os.path.join(BUILD, "cache", "*"))
    suffix = "" if ROWS == 1 else f"_m{ROWS}"
    name = f"lm_head_argmax_n{V}_k{H}_c{CORES}{suffix}"
    dst = board_io.kernel_file(OUT, name, CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)
    json.dump({"V": V, "H": H, "head": HEAD, "rows": ROWS, "cores": CORES,
               "chunks": CHUNKS, "tile": TILE, "eps": EPS, "real": REAL},
              open(os.path.join(OUT, name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
