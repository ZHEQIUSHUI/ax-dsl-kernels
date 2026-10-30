"""Board check of NVFP4 matmuls: projection kernels x [1, K] . W[N, K]^T -> out [1, N] f32, the rows split over CORES
(MXFP8 / MXFP4 / NVFP4 at N = 8192, K = 2048: the GDN qkv weight; NVFP4 also on one core at N = 2048), written with
board_io.fix to <dir>/<variant>/ with their inputs (<IO name>.bin) and the torch result on the dequantized weights
(ref_out.bin); after `axrun <dir>/<variant>/model.axmodel <dir>/<variant>` on the board, `python nv4_board.py --cmp
<dir>` prints cos / max error per variant. Usage: MODEL=<dir> WFMT=mx8 python nv4_board.py <dir>"""
import glob
import os
import shutil
import sys

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import board_io
import mb_nv4 as nv
import q35_moe as mo
import q35_mx_build as mb

K = 2048
VARIANTS = [("mx8_n8192_c4", 8192, 4), ("mx4_n8192_c4", 8192, 4), ("nv4_n8192_c4", 8192, 4), ("nv4_n2048_c1", 2048, 1),
            ("nvt_n8192_c4", 8192, 4), ("nvt_n2048_c1", 2048, 1)]  # nvt: scale2 a runtime input t (fp32 [1])
if os.environ.get("ONLY_NVT"):
    VARIANTS = [v for v in VARIANTS if v[0].startswith("nvt")]
CUR = {}


def kern_t(x, w, s, t, out):
    c = ax.program_id(0)
    n, cores = CUR["n"], CUR["cores"]
    rows = slice(c * (n // cores), (c + 1) * (n // cores))
    ax.dsl.store(linear(ax.dsl.load(x), w[rows], s[rows], "p", weight_scale2=t, output_dtype=ax.fp32), r=out[:, rows])
    return out


def kern(x, w, s, out):
    c = ax.program_id(0)
    n, cores = CUR["n"], CUR["cores"]
    rows = slice(c * (n // cores), (c + 1) * (n // cores))
    kw = dict(weight_scale2=CUR["s2"]) if CUR["s2"] is not None else {}
    ax.dsl.store(linear(ax.dsl.load(x), w[rows], s[rows], "p", output_dtype=ax.fp32, **kw), r=out[:, rows])
    return out


def raw(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.numpy().tobytes()


def build(d):
    torch.manual_seed(0)
    x = torch.randn((1, K)).bfloat16()
    for tag, n, cores in VARIANTS:
        w = torch.randn((n, K)) * 0.02
        fmt = tag[:3]
        if fmt == "mx8":
            q, s = mb.quant_mxfp8(w)
            ref, s2 = mb.dequant(q, s), None
        elif fmt == "mx4":
            q, s = mo.quant_mxfp4(w)
            ref, s2 = mo.dequant_mxfp4(q, s), None
            q, s = q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e8m0fnu)
        else:  # nv4 / nvt
            q, s, s2, ref = nv.quant_nvfp4(w)
            q = q.view(torch.float4_e2m1fn_x2)
        CUR.update(n=n, cores=cores, s2=s2)
        out = torch.zeros((1, n))
        dump = f"{d}/build_{tag}"
        os.makedirs(dump, exist_ok=True)
        os.chdir(dump)
        ax.init(chip_num=1, core_num=cores)
        if fmt == "nvt":
            t = torch.from_numpy(s2.copy())
            b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(kern_t)
            b[cores](x, q, s, t, out)
        else:
            b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(kern)
            b[cores](x, q, s, out)
        refo = x.float() @ ref.float().T
        sim_cos = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), refo.reshape(-1).double(), dim=0).item()
        vd = f"{d}/{tag}"
        os.makedirs(vd, exist_ok=True)
        (p,) = glob.glob(f"{dump}/cache/kern*.compiled.axmodel")
        shutil.copy(p, f"{vd}/model.axmodel")
        board_io.fix(f"{vd}/model.axmodel")
        for name, tt in (("x", x), ("w", q), ("s", s)) + ((("t", torch.from_numpy(s2.copy())),) if fmt == "nvt" else ()):
            open(f"{vd}/{name}.bin", "wb").write(raw(tt))
        open(f"{vd}/ref_out.bin", "wb").write(refo.numpy().astype(np.float32).tobytes())
        print(f"{tag}: simulator cos vs torch {sim_cos:.6f}, weight {q.numel() * q.element_size() / 2**20:.2f} MiB")


def cmp(d):
    for tag, n, cores in VARIANTS:
        vd = f"{d}/{tag}"
        if not os.path.exists(f"{vd}/out_out.bin"):
            print(f"{tag}: no board output")
            continue
        got = np.fromfile(f"{vd}/out_out.bin", dtype=np.float32)
        ref = np.fromfile(f"{vd}/ref_out.bin", dtype=np.float32)
        c = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
        print(f"{tag}: board cos vs torch {c:.6f}, max abs err {np.abs(got - ref).max():.4g} (ref max {np.abs(ref).max():.4g})")


if __name__ == "__main__":
    if sys.argv[1] == "--cmp":
        cmp(os.path.abspath(sys.argv[2]))
    else:
        build(os.path.abspath(sys.argv[1]))
