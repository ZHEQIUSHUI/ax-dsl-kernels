"""The Qwen3-ASR audio encoder's tail on four cores in one call: each core its N / CORES rows through the post
LayerNorm and the projector (linear_1, GELU erf, linear_2), its rows of the output.

  qwen3a_proj<N>_h<H>   x [N, H] bf16 (the last layer's output) -> out [N, O] fp32 (ln_w / ln_b [H], w1 [H, H], b1 [H],
                        w2 [O, H], b2 [O] bf16; O the language model's hidden size)

No shared OCM: the cores share nothing. Every product over the whole K.
Written as qwen3a_proj<N>_h<H>_npu<CORES>.axmodel. Checked on the simulator against torch on random rows, twice with
the output poisoned between the runs.
Usage: MODEL=<Qwen3-ASR -hf dir> N=128 CORES=4 python q3a_proj.py [kernel_dir]
"""
import glob
import os
import shutil
import sys

import torch
from loguru import logger
from safetensors import safe_open

import q3a_audio as A

import axnn.dsl as ax
import board_io

v = A.v
H, N = A.H, A.N
O = A.AC["output_dim"]
CORES = int(os.environ.get("CORES", 4))
NQ = N // CORES  # rows per core
assert N % (CORES * 16) == 0


def proj(x, ln_w, ln_b, w1, b1, w2, b2, out):
    c = ax.program_id(0)
    rows = slice(c * NQ, (c + 1) * NQ)
    xt = ax.dsl.load(x[rows]).reshape(1, NQ, H)
    u = ax.dsl.gelu(v.proj(v.layernorm(xt, ln_w, ln_b), w1, b1, "p1", output_dtype=ax.fp32), output_dtype=ax.bf16)
    y = v.proj(u, w2, b2, "p2", output_dtype=ax.fp32).reshape(NQ, O)
    ax.dsl.store(y, r=out[rows])
    return out


ORDER = ["x", "ln_w", "ln_b", "w1", "b1", "w2", "b2", "out"]


def weights():
    with safe_open(f"{A.MODEL}/model.safetensors", "pt") as f:
        g = lambda n: f.get_tensor(n).float()  # noqa: E731
        return dict(ln_w=g("model.audio_tower.ln_post.weight"), ln_b=g("model.audio_tower.ln_post.bias"),
                    w1=g("model.multi_modal_projector.linear_1.weight"), b1=g("model.multi_modal_projector.linear_1.bias"),
                    w2=g("model.multi_modal_projector.linear_2.weight"), b2=g("model.multi_modal_projector.linear_2.bias"))


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    name = f"qwen3a_proj{N}_h{H}"
    dump = f"{os.path.abspath(os.environ.get('DUMP', 'out_q3a_proj'))}/{name}"
    os.makedirs(dump, exist_ok=True)
    os.chdir(dump)
    ax.init(chip_num=1, core_num=CORES)
    torch.manual_seed(0)
    n = int(os.environ.get("NREAL", N - 24))
    w = weights()
    x = torch.randn((n, H)) * 2.0
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    args = dict(x=bf(torch.cat([x, torch.zeros((N - n, H))])), out=torch.zeros((N, O), dtype=torch.float32),
                **{k: bf(t) for k, t in w.items()})
    kk = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=dump)(proj)
    kk[CORES](*[args[a] for a in ORDER])
    first = args["out"].clone()
    args["out"] = torch.full((N, O), float("nan"), dtype=torch.float32)  # the second run's output poisoned before
    kk[CORES](*[args[a] for a in ORDER])
    same = bool(torch.equal(first[:n].view(torch.int32), args["out"][:n].view(torch.int32)))
    xr = args["x"][:n].float()
    h = torch.nn.functional.layer_norm(xr, (H,), w["ln_w"], w["ln_b"], 1e-5)
    ref = torch.nn.functional.gelu(h @ w["w1"].T + w["b1"]) @ w["w2"].T + w["b2"]
    logger.info(f"{name} (n={n}, O={O}): out vs torch cos {v.cos(first[:n], ref):.6f}, max |d| "
                f"{(first[:n] - ref).abs().max().item():.4f} of max {ref.abs().max().item():.3f}, second run "
                f"{'byte-identical' if same else 'DIFFERS'}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        (path,) = glob.glob(f"{dump}/cache/proj.*.compiled.axmodel")
        dst = board_io.kernel_file(out_dir, name, CORES)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
