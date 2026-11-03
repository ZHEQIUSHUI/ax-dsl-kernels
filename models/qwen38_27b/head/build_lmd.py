"""The draft head: the 1-row MXFP4 output head over the first LM_C rows (a draft vocabulary's rows, gathered by the
host), on CRUN; numerics against torch on lmh27's data (its first LM_C rows)."""
import glob, os, shutil, sys, time
import numpy as np
import torch
import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
sys.path.insert(0, "/data/junda/work/qsmall/lmh27")
import board_io

DATA = "/data/junda/work/qsmall/lmh27/data"
C = int(os.environ["LM_C"])
K = 5120
FULL = 248320
CORES = 4
TILE = max(16, min(1024, (256 * 1024) // (K // 2)))
ROOT = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "out")
BUILD = os.path.join(ROOT, "build", f"c{C}")


def mul_mat_mx4(x, w, s, out):
    c = ax.program_id(0)
    rows = slice(c * (C // CORES), (c + 1) * (C // CORES))
    o = linear(ax.dsl.load(x), w[rows], s[rows], "lmd", output_dtype=ax.fp32, oc_tile_size=TILE)
    ax.dsl.store(o, r=out[:, rows])
    return out


def load(name, n, dt):
    return np.fromfile(os.path.join(DATA, name), dtype=dt, count=n)


def main():
    assert C % CORES == 0
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    w = torch.from_numpy(load("w.bin", C * K // 2, np.uint8).copy()).view(torch.float4_e2m1fn_x2).reshape(C, K // 2)
    s = torch.from_numpy(load("s.bin", C * K // 32, np.uint8).copy()).view(torch.float8_e8m0fnu).reshape(C, K // 32)
    x = torch.from_numpy(load("x.bin", 4 * K, np.uint16).copy()).view(torch.bfloat16).reshape(4, K)
    ref = torch.load(os.path.join(DATA, "ref_torch.pt"))
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=CORES)
    k = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=BUILD)(mul_mat_mx4)
    t0 = time.time()
    for r in range(2):
        o = torch.zeros(1, C)
        k[CORES](x[r:r + 1].contiguous(), w, s, o)
        a, b = o[0].float(), ref[r][:C].float()
        print(f"row {r}: cos {float((a * b).sum() / (a.norm() * b.norm())):.9f}", flush=True)
    print("run %.1fs" % (time.time() - t0), flush=True)
    paths = sorted(glob.glob(os.path.join(BUILD, "cache", "mul_mat_mx4.*.compiled.axmodel")), key=os.path.getmtime)
    dst = board_io.kernel_file(OUT, f"mul_mat_mx4_n{C}_k{K}", CORES)
    shutil.copy(paths[-1], dst)
    print("fixed:", board_io.fix(dst), flush=True)
    print("wrote", dst, os.path.getsize(dst), flush=True)


if __name__ == "__main__":
    main()
