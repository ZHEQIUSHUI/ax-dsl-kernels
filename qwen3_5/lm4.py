"""the output projection with MXFP4 weights on CORES cores: mul_mat_mx4_n<C>_k<K> (x [1, K] bf16, w [C, K] E2M1 pairs,
s [C, K / 32] E8M0 -> out [1, C] f32), C / CORES rows per core. Checked on the simulator against torch on the
dequantized weights. Usage: [LM_QUANT=moe] [LM_TILE=<rows>] LM_C=248320 LM_K=4096 CORES=4 python lm4.py <dir>"""
import glob, os, shutil, sys
import torch
from loguru import logger
import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import board_io
# the MXFP4 quantizer (E2M1 codes + E8M0 scales, as ggml's): q35_layer_pk's for the dense models, q35_moe's for the MoE
# ones (LM_QUANT=moe: q35_layer_pk does not import with an MoE config)
K = __import__("q35_moe" if os.environ.get("LM_QUANT", "pk") == "moe" else "q35_layer_pk")

C, KK, CORES = int(os.environ.get("LM_C", 248320)), int(os.environ.get("LM_K", 4096)), int(os.environ.get("CORES", 4))
TILE = int(os.environ.get("LM_TILE", 0)) or max(16, min(1024, (256 * 1024) // (KK // 2)))  # rows per tile


def mul_mat_mx4(x, w, s, out):
    c = ax.program_id(0)
    rows = slice(c * (C // CORES), (c + 1) * (C // CORES))
    o = linear(ax.dsl.load(x), w[rows], s[rows], "lm4", output_dtype=ax.fp32, oc_tile_size=TILE)
    ax.dsl.store(o, r=out[:, rows])
    return out


def main():
    out_dir = os.path.abspath(sys.argv[1]); os.makedirs(out_dir, exist_ok=True)
    dump = os.path.abspath(os.environ.get("DUMP", "out_lm4")); os.makedirs(dump, exist_ok=True); os.chdir(dump)
    torch.manual_seed(0)
    x = torch.randn((1, KK)).bfloat16()
    w = torch.randn((C, KK)) * KK**-0.5
    pk, sc = K.quant_mxfp4(w)
    out = torch.zeros((1, C))
    ax.init(chip_num=1, core_num=CORES)
    ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(mul_mat_mx4)[CORES](
        x, pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu), out)
    code = torch.stack([pk & 15, pk >> 4], -1).reshape(C, -1).long()
    dq = K.E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0) * torch.exp2(sc.float() - 127).repeat_interleave(32, -1)
    ref = x.float() @ dq.T
    cos = torch.nn.functional.cosine_similarity(out.reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
    (p,) = glob.glob(f"{dump}/cache/mul_mat_mx4.*.compiled.axmodel")
    dst = board_io.kernel_file(out_dir, f"mul_mat_mx4_n{C}_k{KK}", CORES)
    shutil.copy(p, dst)
    logger.info(f"wrote {dst}: cos vs torch {cos:.6f}")


if __name__ == "__main__":
    main()
