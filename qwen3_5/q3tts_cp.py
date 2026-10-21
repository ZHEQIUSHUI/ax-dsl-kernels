"""The Qwen3-TTS code predictor's layers (Qwen3-0.6B's shapes) as one kernel per position, for ggml-axera's generic
executor (ggml-axera-gen.cpp): the NL layers of q35_step.attn_block in a row, the hidden state on chip between them,
instead of NL calls of the single-layer kernel (each call costs its own start and its input / output round trip):

  qwen3tts_cp_stack<NL>[_mx8]_c<LMAX>
      x [1, 1, H] bf16, cos / sin [1, ROT/2] bf16, pos [1] i32 (the cell written), nblk [1] i32, mask [LMAX/BC, BC] bf16,
      per layer i: ln1_i, w_q_i, w_k_i, w_v_i, q_norm_i, k_norm_i, w_o_i, ln2_i, w_gate_i, w_up_i, w_down_i (MXFP8:
      w_*_i fp8 with their scales s_*_i)
      -> out [1, 1, H] bf16, kc_i / vc_i [LMAX, HKV * D] f16 (llama.cpp's cell layout, resident: row pos written)

The kernel function is written into the build directory (the builder reads its source) and imported. Checked on the
simulator against NL calls of the single-layer kernel (q35_step.attn_layer, or q35_mx_build.attn_layer_mx) for
positions 0 .. STEPS-1 with the model's layers 0 .. NL-1: out and K / V.

Usage: MODEL=<Qwen3-0.6B dir> KVL=cell LMAX=256 BC=256 NL=5 [WFMT=mx8] [STEPS=2] python q3tts_cp.py <kernel_dir>
"""
import glob
import importlib
import os
import shutil
import sys

os.environ.setdefault("KVL", "cell")
os.environ.setdefault("LMAX", "256")
os.environ.setdefault("BC", "256")

import torch
from loguru import logger

import axnn.dsl as ax
import board_io
import q35_step as q

NL = int(os.environ.get("NL", 5))
MX = os.environ.get("WFMT", "bf16") == "mx8"
STEPS = int(os.environ.get("STEPS", 2))
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q3tts_cp"))
if MX:
    import q35_mx_build as mb  # patches q.proj / q.slice_w to the MX linear
LAYER_W = ["ln1", "w_q", "w_k", "w_v", "q_norm", "k_norm", "w_o", "ln2", "w_gate", "w_up", "w_down"]
NAME = f"qwen3tts_cp_stack{NL}{'_mx8' if MX else ''}_c{q.LMAX}"


def layer_inputs():
    """the per-layer input names of the kernel, in its argument order"""
    names = []
    for w in LAYER_W:
        names.append(w)
        if MX and w in mb.ATTN_W:
            names.append(mb.scale_name(w))
    return names


def body(a):
    """the kernel: a = its arguments by name"""
    if MX:
        mb.SCALES.clear()
        for i in range(NL):
            for w in mb.ATTN_W:
                mb.SCALES[id(a[f"{w}_{i}"])] = a[f"{mb.scale_name(w)}_{i}"]
    xt = ax.dsl.load(a["x"])
    cos_t, sin_t = ax.dsl.load(a["cos"]), ax.dsl.load(a["sin"])
    for i in range(NL):
        l = {w: a[f"{w}_{i}"] for w in LAYER_W}
        xt = q.attn_block(xt, l["ln1"], l["w_q"], l["w_k"], l["w_v"], l["q_norm"], l["k_norm"], cos_t, sin_t,
                          a[f"kc_{i}"], a[f"vc_{i}"], a["pos"], a["nblk"], a["mask"], l["w_o"], l["ln2"], l["w_gate"],
                          l["w_up"], l["w_down"], tag=f"l{i}_")
    ax.dsl.store(xt, r=a["out"])


def kernel_function():
    """the kernel as a function of a module in BUILD (the builder reads its source and its return statement)"""
    ins = ["x", "cos", "sin", "pos", "nblk", "mask"] + [f"{n}_{i}" for i in range(NL) for n in layer_inputs()]
    outs = ["out"] + [f"{kv}_{i}" for i in range(NL) for kv in ("kc", "vc")]
    src = (f"import q3tts_cp as S\n\n\ndef cp_stack({', '.join(ins + outs)}):\n    S.body(locals())\n"
           f"    return {', '.join(outs)}\n")
    os.makedirs(BUILD, exist_ok=True)
    open(f"{BUILD}/cp_stack_kernel.py", "w").write(src)
    sys.path.insert(0, BUILD)
    return importlib.import_module("cp_stack_kernel").cp_stack, ins, outs


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    fn, ins, outs = kernel_function()

    args = []  # per layer: the kernel's weight inputs (MXFP8: quantized), and the single-layer kernel's
    for i in range(NL):
        a = q.attn_args(W, i)
        args.append(mb.mx_args(a, mb.ATTN_W)[0] if MX else a)
    single = mb.attn_layer_mx if MX else q.attn_layer
    single_order = mb.order(mb.attn_layer_mx) if MX else q.ATTN_ORDER
    os.makedirs(f"{BUILD}/stack", exist_ok=True)
    os.makedirs(f"{BUILD}/single", exist_ok=True)
    os.chdir(f"{BUILD}/stack")  # (a builder keeps its compiled kernels in cache/ of the directory it is made in)
    k_stack = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/stack")(fn)
    os.chdir(f"{BUILD}/single")
    k_single = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/single")(single)

    kv = lambda: torch.zeros((q.LMAX, q.HKV * q.D), dtype=torch.half)  # noqa: E731
    kc_s, vc_s = [kv() for _ in range(NL)], [kv() for _ in range(NL)]  # the stack's caches
    kc_r, vc_r = [kv() for _ in range(NL)], [kv() for _ in range(NL)]  # the single-layer calls'
    emb = W("embed_tokens.weight")
    worst = 1.0
    for pos in range(STEPS):
        x = emb[1000 + 7 * pos].reshape(1, 1, q.H).contiguous()
        c, s = q.rope_cos_sin(pos)
        m = torch.full((q.LMAX,), -3e38)
        m[: pos + 1] = 0
        common = dict(cos=c, sin=s, pos=torch.tensor([pos], dtype=torch.int32),
                      nblk=torch.tensor([pos // q.BC + 1], dtype=torch.int32), mask=m.reshape(q.LMAX // q.BC, q.BC).bfloat16())

        # reference: NL calls of the single-layer kernel
        h = x
        for i in range(NL):
            out = torch.zeros((1, 1, q.H), dtype=torch.bfloat16)
            case = dict(args[i], x=h, kc=kc_r[i], vc=vc_r[i], out=out, **common)
            q.RESIDENT.update(kc=kc_r[i].clone(), vc=vc_r[i].clone())
            if MX:
                mb.SCALES.clear()
            os.chdir(f"{BUILD}/single")
            k_single[1](*[case[n] for n in single_order])
            q.RESIDENT.clear()
            h = out

        # the stack
        out = torch.zeros((1, 1, q.H), dtype=torch.bfloat16)
        case = dict(x=x, out=out, **common)
        for i in range(NL):
            case.update({f"{n}_{i}": args[i][n] for n in layer_inputs()})
            case[f"kc_{i}"], case[f"vc_{i}"] = kc_s[i], vc_s[i]
            q.RESIDENT.update({f"kc_{i}": kc_s[i].clone(), f"vc_{i}": vc_s[i].clone()})
        os.chdir(f"{BUILD}/stack")
        k_stack[1](*[case[n] for n in ins + outs])
        q.RESIDENT.clear()

        co = cos(out, h)
        ck = min(cos(kc_s[i][pos], kc_r[i][pos]) for i in range(NL))
        cv = min(cos(vc_s[i][pos], vc_r[i][pos]) for i in range(NL))
        same = bool(torch.equal(out, h))
        logger.info(f"{NAME} pos {pos}: out cos {co:.6f} (bit-identical {same}), K cos min {ck:.6f}, V cos min {cv:.6f}")
        worst = min(worst, co, ck, cv)

    path = max(glob.glob(f"{BUILD}/stack/cache/cp_stack.*.compiled.axmodel"), key=os.path.getmtime)
    dst = board_io.kernel_file(out_dir, NAME)
    shutil.copy(path, dst)
    logger.info(f"wrote {dst} (worst cos vs the single-layer calls {worst:.6f})")


if __name__ == "__main__":
    main()
