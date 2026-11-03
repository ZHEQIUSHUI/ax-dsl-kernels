"""Build the 1-row baseline or the 4-row m kernel at the real 27B output-head shape, on CRUN."""
import glob, json, os, shutil, sys, time
import numpy as np
import torch
import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import board_io

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
C = int(os.environ.get("LM_C", "248320"))
K = int(os.environ.get("LM_K", "5120"))
CORES = int(os.environ.get("CORES", "4"))
TILE = int(os.environ.get("LM_TILE", "0")) or max(16, min(1024, (256 * 1024) // (K // 2)))
MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TAG = "m4if" if MODE == 5 else ("m4" if MODE == 4 else "m1")
OUT = os.path.join(ROOT, "out")
BUILD = os.path.join(ROOT, "build", TAG)


def greater_equal(x, y):
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    zero = ax.dsl.zeros((1,), ax.int32)
    return ax.dsl.minimum(ax.dsl.maximum(x - y + one, zero), one, output_dtype=ax.uint8)


def mul_mat_mx4(x, w, s, out):
    c = ax.program_id(0)
    rows = slice(c * (C // CORES), (c + 1) * (C // CORES))
    o = linear(ax.dsl.load(x), w[rows], s[rows], "lm4", output_dtype=ax.fp32, oc_tile_size=TILE)
    ax.dsl.store(o, r=out[:, rows])
    return out


def mul_mat_mx4_m4(x, w, s, m, out):
    c = ax.program_id(0)
    rows = slice(c * (C // CORES), (c + 1) * (C // CORES))
    o = linear(ax.dsl.load(x), w[rows], s[rows], "lm4_m4", output_dtype=ax.fp32, oc_tile_size=TILE)
    idx = ax.dsl.from_numpy(np.arange(1, 5, dtype=np.int32))
    valid = greater_equal(ax.dsl.load(m), idx).reshape(4, 1)
    zero = ax.dsl.zeros((4, C // CORES), ax.fp32)
    o = ax.dsl.where(valid, o, zero, output_dtype=ax.fp32)
    ax.dsl.store(o, r=out[:, rows])
    return out


def mul_mat_mx4_m4_if(x, w, s, m, out):
    c = ax.program_id(0)
    rows = slice(c * (C // CORES), (c + 1) * (C // CORES))
    wr, sr = w[rows], s[rows]
    ml = ax.dsl.load(m)
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    is_one = ax.dsl.cast(ax.dsl.mul(greater_equal(ml, one), greater_equal(one, ml), output_dtype=ax.uint8),
                         dtype=ax.int32)

    def true_body(cond, mv, xv, wv, sv, ov):
        o = linear(xv[0:1], wv, sv, "lm4_m4_if_t1", output_dtype=ax.fp32, oc_tile_size=TILE)
        ax.dsl.store(o, r=ov[0:1, rows])
        return cond, mv, xv, wv, sv, ov

    def false_body(cond, mv, xv, wv, sv, ov):
        o = linear(xv, wv, sv, "lm4_m4_if_t4", output_dtype=ax.fp32, oc_tile_size=TILE)
        idx = ax.dsl.from_numpy(np.arange(1, 5, dtype=np.int32))
        valid = greater_equal(mv, idx).reshape(4, 1)
        o = ax.dsl.where(valid, o, ax.dsl.zeros((4, C // CORES), ax.fp32), output_dtype=ax.fp32)
        ax.dsl.store(o, r=ov[:, rows])
        return cond, mv, xv, wv, sv, ov

    ax.dsl.if_op(is_one, out, true_body, false_body, is_one, ml, ax.dsl.load(x), wr, sr, out)
    return out


def load_bin(name, dtype, shape):
    a = np.fromfile(os.path.join(DATA, name), dtype=np.uint8)
    if dtype == torch.bfloat16:
        a = a.view(np.uint16)
    elif dtype == torch.float32:
        a = a.view(np.float32)
    elif dtype == torch.int32:
        a = a.view(np.int32)
    return torch.from_numpy(a.copy()).view(dtype).reshape(shape)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    w = load_bin("w.bin", torch.uint8, (C, K // 2)).view(torch.float4_e2m1fn_x2)
    s = load_bin("s.bin", torch.uint8, (C, K // 32)).view(torch.float8_e8m0fnu)
    x = load_bin("x.bin", torch.bfloat16, (4, K))
    ref = torch.load(os.path.join(DATA, "ref_torch.pt"))
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    t0 = time.time()
    if MODE == 1:
        k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=BUILD)(mul_mat_mx4)
        outs = torch.zeros(4, C)
        for r in range(4):
            o = torch.zeros(1, C)
            k[CORES](x[r:r + 1].contiguous(), w, s, o)
            outs[r] = o[0]
        name = f"mul_mat_mx4_n{C}_k{K}"
        save_pt = "lm1_outs.pt"
    else:
        fn = mul_mat_mx4_m4_if if MODE == 5 else mul_mat_mx4_m4
        k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=BUILD)(fn)
        outs = torch.zeros(4, 4, C)
        for m in (1, 2, 3, 4):
            o = torch.zeros(4, C)
            k[CORES](x, w, s, torch.tensor([m], dtype=torch.int32), o)
            outs[m - 1] = o
        name = f"mul_mat_mx4_m4_n{C}_k{K}"
        save_pt = "lm4if_outs.pt" if MODE == 5 else "lm4_outs.pt"
    torch.save(outs, os.path.join(BUILD, save_pt))
    print("run time %.1fs" % (time.time() - t0), flush=True)
    pat = "mul_mat_mx4_m4_if.*.compiled.axmodel" if MODE == 5 else (
        "mul_mat_mx4_m4.*.compiled.axmodel" if MODE == 4 else "mul_mat_mx4.*.compiled.axmodel")
    paths = sorted(glob.glob(os.path.join(BUILD, "cache", pat)), key=os.path.getmtime)
    if MODE == 4:
        paths = [p for p in paths if "m4_if" not in os.path.basename(p)]
    elif MODE == 1:
        paths = [p for p in paths if "m4" not in os.path.basename(p)]
    assert paths, glob.glob(os.path.join(BUILD, "cache", "*"))
    dst = board_io.kernel_file(OUT, name, CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)
    # quick numeric summary vs torch
    if MODE == 1:
        for r in range(4):
            a, b = outs[r].float(), ref[r]
            cos = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))
            print(f"row {r}: cos vs torch {cos:.9f}", flush=True)
    else:
        for m in (1, 2, 3, 4):
            vals = []
            for r in range(4):
                a, b = outs[m - 1][r].float(), ref[r]
                vals.append(float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20)))
            print(f"m={m}: cos vs torch {['%.9f' % v for v in vals]}", flush=True)


if __name__ == "__main__":
    main()
