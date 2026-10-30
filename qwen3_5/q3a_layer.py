"""A Qwen3-ASR audio encoder layer on four cores in one call: each core its N / CORES rows through LayerNorm1 and the
q / k / v projections, its K / V rows to all cores through shared OCM, bidirectional attention of its rows over all N
(key mask for the padding), out_proj, LayerNorm2 and the MLP; its rows of the output (only the rows it read).

  qwen3a_layer<N>_h<H>   x [N, H] bf16 -> out [N, H] bf16 (q3a_audio.py's pre / attn / post in one; weights bf16 as
                         theirs: w_qkv [3 H, H], b_qkv [3 H], w_o [H, H], b_o, ln1 / ln2 w / b, w_up [F, H], b_up,
                         w_down [H, F], b_down; kmask [1, N] bf16: 0 for a real row, -3e38 for the padding)

Every product over the whole K (no accumulation across calls or cores); the softmax over all N keys in fp32 in one
pass. Each core reads back from DDR only what it wrote itself; the shared K / V are read once more before the output
is stored (KEEP_SHM: their space is not handed to a later tensor while a slower core still reads them).

Written as qwen3a_layer<N>_h<H>_npu<CORES>.axmodel. Checked on the simulator: layer LAYER of the checkpoint on random
rows (NREAL real) against torch, twice with the shared OCM poisoned between the runs.
Usage: MODEL=<Qwen3-ASR -hf dir> N=128 CORES=4 python q3a_layer.py [kernel_dir]
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger

import q3a_audio as A

import axnn.dsl as ax
from axnn.dsl import parallel as axd
import board_io

v = A.v
H, NH, HD, F, N = A.H, A.NH, A.HD, A.F, A.N
CORES = int(os.environ.get("CORES", 4))
NQ = N // CORES  # rows per core
KEEP_SHM = os.environ.get("KEEP_SHM", "1") == "1"
assert N % (CORES * 16) == 0


def keep_shm(y, live):
    """y itself, read after every shared OCM tensor in live (selected away by a false condition: no arithmetic)"""
    if not KEEP_SHM:
        return y
    one = (1,) * len(y.shape)
    off = ax.dsl.full(shape=one, dtype=ax.uint8, fill_value=0)
    for t in live:
        e = t[tuple(slice(0, 1) for _ in t.shape)].reshape(*one)
        y = ax.dsl.where(off, ax.dsl.cast(e, dtype=y.dtype), y, output_dtype=y.dtype)
    return y


def layer(x, ln1_w, ln1_b, w_qkv, b_qkv, w_o, b_o, ln2_w, ln2_b, w_up, b_up, w_down, b_down, kmask, out):
    c = ax.program_id(0)
    rows = slice(c * NQ, (c + 1) * NQ)
    xt = ax.dsl.load(x[rows]).reshape(1, NQ, H)
    h = v.layernorm(xt, ln1_w, ln1_b)
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=HD**-0.5)
    part = lambda i, nm: v.proj(h, w_qkv[i * H : (i + 1) * H], b_qkv[i * H : (i + 1) * H], nm, output_dtype=ax.fp32)  # noqa: E731
    q = ax.dsl.mul(part(0, "q"), scale, output_dtype=ax.bf16).reshape(NQ, H)
    ksh = ax.dsl.create_tensor((N, H), ax.bf16, ax.MemoryType.SHARED_OCM, name="k_all")
    vsh = ax.dsl.create_tensor((N, H), ax.bf16, ax.MemoryType.SHARED_OCM, name="v_all")
    ax.dsl.move(r=ksh[rows], x=ax.dsl.cast(part(1, "k"), dtype=ax.bf16).reshape(NQ, H))
    ax.dsl.move(r=vsh[rows], x=ax.dsl.cast(part(2, "v"), dtype=ax.bf16).reshape(NQ, H))
    axd.pub(ksh)
    axd.pub(vsh)
    axd.sub(ksh)
    axd.sub(vsh)
    kall = ax.dsl.create_tensor((N, H), ax.bf16, ax.MemoryType.OCM, name="k_loc")
    vall = ax.dsl.create_tensor((N, H), ax.bf16, ax.MemoryType.OCM, name="v_loc")
    ax.dsl.move(r=kall, x=ksh)
    ax.dsl.move(r=vall, x=vsh)
    km = ax.dsl.load(kmask)  # [1, N]
    o = ax.dsl.create_tensor((NQ, H), ax.bf16, ax.MemoryType.OCM, name="attn")
    for hh in range(NH):
        cs = slice(hh * HD, (hh + 1) * HD)
        s = ax.dsl.matmul(q[:, cs], kall[:, cs], transposed=True, output_dtype=ax.fp32)  # [NQ, N]
        p = ax.dsl.softmax(ax.dsl.add(s, km, output_dtype=ax.fp32), dim=-1, output_dtype=ax.bf16)
        ax.dsl.move(r=o[:, cs], x=ax.dsl.matmul(p, vall[:, cs], output_dtype=ax.bf16))
    a = v.proj(o.reshape(1, NQ, H), w_o, b_o, "o", output_dtype=ax.fp32)
    x1 = ax.dsl.add(xt, a, output_dtype=ax.bf16)
    u = ax.dsl.gelu(v.proj(v.layernorm(x1, ln2_w, ln2_b), w_up, b_up, "up", output_dtype=ax.fp32), output_dtype=ax.bf16)
    d = v.proj(u, w_down, b_down, "down", output_dtype=ax.fp32)
    y = keep_shm(ax.dsl.add(x1, d, output_dtype=ax.bf16).reshape(NQ, H), [ksh, vsh])
    ax.dsl.store(y, r=out[rows])
    return out


ORDER = ["x", "ln1_w", "ln1_b", "w_qkv", "b_qkv", "w_o", "b_o", "ln2_w", "ln2_b", "w_up", "b_up", "w_down", "b_down",
         "kmask", "out"]


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    name = f"qwen3a_layer{N}_h{H}"
    dump = f"{os.path.abspath(os.environ.get('DUMP', 'out_q3a_layer'))}/{name}"
    os.makedirs(dump, exist_ok=True)
    os.chdir(dump)
    ax.init(chip_num=1, core_num=CORES)
    torch.manual_seed(0)
    n = int(os.environ.get("NREAL", N - 24))
    w = A.layer_weights(int(os.environ.get("LAYER", 0)))
    x = torch.randn((n, H)) * 0.5
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    kmask = torch.zeros((1, N))
    kmask[0, n:] = -3e38
    args = dict(x=bf(torch.cat([x, torch.zeros((N - n, H))])), kmask=bf(kmask), out=torch.zeros((N, H), dtype=torch.bfloat16),
                **{k: bf(w[k]) for k in ("ln1_w", "ln1_b", "w_qkv", "b_qkv", "w_o", "b_o", "ln2_w", "ln2_b", "w_up", "b_up",
                                         "w_down", "b_down")})
    kk = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(layer)
    kk[CORES](*[args[a] for a in ORDER])
    first = args["out"].clone()
    args["out"] = torch.full((N, H), float("nan"), dtype=torch.bfloat16)  # the second run's output poisoned before
    kk[CORES](*[args[a] for a in ORDER])
    same = bool(torch.equal(first[:n].view(torch.int16), args["out"][:n].view(torch.int16)))
    ref = A.reference(x, w)
    logger.info(f"{name} (n={n}): out vs torch cos {v.cos(first[:n].float(), ref):.6f}, second run "
                f"{'byte-identical' if same else 'DIFFERS'}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        (path,) = glob.glob(f"{dump}/cache/layer.*.compiled.axmodel")
        dst = board_io.kernel_file(out_dir, name, CORES)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
