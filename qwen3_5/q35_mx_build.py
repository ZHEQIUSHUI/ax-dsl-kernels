"""MXFP8 (E4M3 + E8M0 scale per 32 input channels) version of the on-board Qwen3.5-0.8B generator data.

Reuses q35_step's layer kernels unchanged: proj() is swapped for an MX linear whose scale is looked up from the
wrapper's extra *_s inputs, so the kernel bodies stay the bf16 ones. Every projection in the 24 layers and the
lm_head is MXFP8; norms, conv, GDN parameters stay as before; the embedding table for the lookup stays bf16.

Writes <out>/L<i>/<input>.bin (fp8 weights + u8 scales), layers.txt, lm_w.bin / lm_s.bin, the three axmodels
({gdn,attn,lm_head}_npu1.axmodel), and copies embed.bin, final_norm.bin, vocab.bin, eos.txt, meta.txt from <bf16_dir>.
Checks on the simulator: each MX kernel against the bf16 kernel on the same input (layer 0, layer 3, lm_head chunk 0).

Usage: LMAX=1024 BC=64 python q35_mx_build.py board_chat board_mx
"""
import glob
import os
import shutil
import sys

os.environ.setdefault("LMAX", "1024")
os.environ.setdefault("BC", "64")

import torch
from loguru import logger
from onnx import TensorProto

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm
from axnn.dsl.kernels.utils.linear import linear
from axnn.frontend import graph_utils
import board_io
import q35_step as q
import q35_chat_build as cb

# 导出器缺 MX 运行时输入的 dtype 映射：按 uint8 字节流导出（仅本脚本内生效，不改仓库）
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXE8M0", TensorProto.UINT8)
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXFP8_E4M3", TensorProto.UINT8)

# compiled axmodel 的 MX scale 输入：新版 builder 写成 FLOAT8E8M0（ONNX 24），板上运行时（2026-10）不认，加载报
# 0x80060084；按 uint8 字节写入，与旧版导出的 decode kernel 一致，字节内容不变
from axnn.dsl import builder_ as _builder
from axnn.opset import datatype as _datatype

_io_dtype = _builder.KernelBuilder._compiled_axmodel_io_dtype
_builder.KernelBuilder._compiled_axmodel_io_dtype = staticmethod(
    lambda dt: _datatype.DType("U8") if dt.typename == "MXE8M0" else _io_dtype(dt))

BF16_DIR = OUT = None  # set from the command line in main()
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_mx"))
GROUP = 32
# WFMT=nv4: the projections in NVFP4 instead (E2M1 pairs, an E4M3 scale per NVG input channels, an FP32 tensor scale2:
# t_<name> beside s_<name>, a runtime input so one kernel serves every layer)
NV4 = os.environ.get("WFMT") == "nv4"
NVG = 16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
VC = cb.VC

GDN_W = ["w_qkv", "w_z", "w_b", "w_a", "w_out", "w_gate", "w_up", "w_down"]
ATTN_W = ["w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down"]


def quant_mxfp8(w):
    """[OC, IC] -> (fp8 e4m3 [OC, IC], e8m0 scale [OC, IC/32]); exponent chosen so nothing saturates."""
    g = w.float().reshape(w.shape[0], -1, GROUP)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(2.0**-126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q8 = (g / torch.exp2(e)).to(torch.float8_e4m3fn).reshape(w.shape)
    scale = (e.squeeze(-1) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q8.contiguous(), scale.contiguous()


def dequant(q8, scale):
    s = torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(GROUP, dim=-1)
    return q8.float() * s


SCALES = {}


def oc_tile_fp8(cols):
    return max(16, min(256, (512 * 1024) // cols))  # fp8 一行 1 字节，tile 仍约 512 KB 封顶 256 行


# MX projection tiles. Default: OC tiles of about 512 KB of weights over the whole K, which leaves wide-K projections
# with few output rows per tile (9B MLP down: 58 rows over K 12288, 27B: 50 over 17408) and prefill far from the matmul
# peak. PRJ_ROWS=<rows>: every MX projection in tiles of that many output rows over the whole K (the backend splits K
# inside a tile and accumulates there). 9B prefill 128 with PRJ_ROWS=256: gdn 1.74 -> 1.20 ms, attn 1.90 -> 1.42 ms per
# call on the board. Input-channel tiles accumulated through MatMul NMC (linear's ic_tile_size) are not used: on the board
# their results change from run to run
PRJ_ROWS = int(os.environ.get("PRJ_ROWS", 0))


def oc_tile_fp4(cols):
    return max(16, min(1024, (256 * 1024) // (cols // 2)))  # fp4 tiles around 256 KB, as the experts'


def proj_mx(x, w, name, output_dtype=ax.bf16):
    s, s2 = SCALES.get(id(w)), SCALES2.get(id(w))
    if s2 is not None:  # NVFP4
        return linear(x, w, s, name, weight_scale2=s2, output_dtype=output_dtype, oc_tile_size=oc_tile_fp4(w.shape[1]))
    tile = oc_tile_fp8(w.shape[1]) if s is not None else q.oc_tile(w.shape[1])
    if PRJ_ROWS and s is not None:
        tile = min(PRJ_ROWS, w.shape[0])
    return linear(x, w, s, name, output_dtype=output_dtype, oc_tile_size=tile)


def slice_w_mx(w, rows=slice(None), cols=slice(None)):
    """a block of an MX / NVFP4 weight with its scales (one per GROUP / NVG input channels): cols in whole groups"""
    ws = w[rows, cols]
    s, s2 = SCALES.get(id(w)), SCALES2.get(id(w))
    if s is not None:
        gs = NVG if s2 is not None else GROUP
        g = slice(None) if cols == slice(None) else slice(cols.start // gs, cols.stop // gs)
        SCALES[id(ws)] = s[rows, g]
    if s2 is not None:
        SCALES2[id(ws)] = s2
    return ws


q.proj = proj_mx  # gdn_layer / attn_layer / mlp_residual 里的 proj 都换成 MX 版
q.slice_w = slice_w_mx


def gdn_layer_mx(x, ln1, w_qkv, s_qkv, w_z, s_z, w_b, s_b, w_a, s_a, conv_w, a_log, dt_bias, norm_w, w_out, s_out,
                 ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, state_in, conv_in, out, state_out, conv_out):
    SCALES.clear()
    for w, s in [(w_qkv, s_qkv), (w_z, s_z), (w_b, s_b), (w_a, s_a), (w_out, s_out), (w_gate, s_gate), (w_up, s_up),
                 (w_down, s_down)]:
        SCALES[id(w)] = s
    # builder 按 return 里的变量名确定输出，所以先调用再返回
    q.gdn_layer(x, ln1, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, norm_w, w_out, ln2, w_gate, w_up, w_down,
                state_in, conv_in, out, state_out, conv_out)
    return out, state_out, conv_out


def attn_layer_mx(x, ln1, w_q, s_q, w_k, s_k, w_v, s_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, s_o,
                  ln2, w_gate, s_gate, w_up, s_up, w_down, s_down, out):
    SCALES.clear()
    for w, s in [(w_q, s_q), (w_k, s_k), (w_v, s_v), (w_o, s_o), (w_gate, s_gate), (w_up, s_up), (w_down, s_down)]:
        SCALES[id(w)] = s
    q.attn_layer(x, ln1, w_q, w_k, w_v, q_norm, k_norm, cos, sin, kc, vc, pos, nblk, mask, w_o, ln2, w_gate, w_up,
                 w_down, out)
    return out, kc, vc


def lm_head_mx(x, norm, w, s, logits):
    h = rmsnorm(ax.dsl.load(x), ax.dsl.load(norm).reshape(1, 1, q.H), eps=q.EPS)
    lg = linear(h, w, s, "lm_head", output_dtype=ax.bf16, oc_tile_size=256)
    ax.dsl.store(lg, r=logits)
    return logits


def scale_name(w):
    return "s_" + w[2:]  # w_qkv -> s_qkv


def quant_nvfp4(w):
    """[N, K] -> (E2M1 pairs [N, K / 2] as float4_e2m1fn_x2, low nibble first; E4M3 scales [N, K / NVG]; FP32 scale2
    [1]): scale2 = amax / (6 * 448), each block's scale its amax / 6 / scale2 in E4M3, elements to the nearest E2M1"""
    n, k = w.shape
    s2 = (w.float().abs().max() / (6.0 * 448.0)).clamp_min(2.0**-126)
    g = w.float().reshape(n, k // NVG, NVG)
    sc = (g.abs().amax(-1, keepdim=True) / 6.0 / s2).to(torch.float8_e4m3fn)
    d = sc.float() * s2
    x = torch.where(d > 0, g / d, torch.zeros_like(g)).reshape(n, k)
    code = ((x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1) | ((x < 0).long() << 3)).to(torch.uint8)
    pk = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return pk.view(torch.float4_e2m1fn_x2), sc.reshape(n, k // NVG).contiguous(), s2.reshape(1).float()


def dequant_nvfp4(pk, sc, s2):
    code = pk.view(torch.uint8)
    code = torch.stack([code & 15, code >> 4], -1).reshape(code.shape[0], -1).long()
    v = E2M1[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    return (v.reshape(v.shape[0], -1, NVG) * (sc.float() * s2).unsqueeze(-1)).reshape(v.shape[0], -1)


def mx_args(a, names):
    """bf16 layer args -> MX args (adds s_*; NVFP4 also t_*: scale2), plus the dequantized weights for the bf16
    reference run."""
    m, deq = dict(a), dict(a)
    for n in names:
        if NV4:
            w4, s, s2 = quant_nvfp4(a[n])
            m[n], m[scale_name(n)], m["t_" + n[2:]] = w4, s, s2
            deq[n] = dequant_nvfp4(w4, s, s2).bfloat16()
            continue
        w8, s = quant_mxfp8(a[n])
        m[n], m[scale_name(n)] = w8, s
        deq[n] = dequant(w8, s).bfloat16()
    return m, deq


SCALES2 = {}  # id(NVFP4 weight) -> its scale2 (q35 kernels' t_<name> inputs)


def nv_twin(fn):
    """NVFP4 version of an MX kernel wrapper fn(..., w_<n>, s_<n>, ...): the same inputs plus t_<n> after each s_<n>,
    their weights' scale2 registered, then fn itself (its SCALES registration and body); same name and outputs"""
    import importlib
    import inspect
    names = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    ins = []
    for n in names:
        ins.append(n)
        if n.startswith("s_") and "w_" + n[2:] in names:
            ins.append("t_" + n[2:])
    ret = [ln.strip() for ln in inspect.getsource(fn).splitlines() if ln.strip().startswith("return ")][-1]
    key = f"{fn.__module__}.{fn.__name__}"  # the same wrapper name in several modules (single / four cores)
    TWINS[key] = fn
    src = (f"import q35_mx_build as M\n\n\ndef {fn.__name__}({', '.join(ins)}):\n    M.reg_scale2(locals())\n"
           f"    M.TWINS[{key!r}]({', '.join(names)})\n    {ret}\n")
    mod = f"nvtwin_{fn.__module__}_{fn.__name__}"
    os.makedirs(BUILD, exist_ok=True)
    open(f"{BUILD}/{mod}.py", "w").write(src)
    if BUILD not in sys.path:
        sys.path.insert(0, BUILD)
    importlib.invalidate_caches()  # the module was written after the path's directory listing was cached
    return getattr(importlib.import_module(mod), fn.__name__)


TWINS = {}


def reg_scale2(a):
    SCALES2.clear()
    for n, v in a.items():
        if n.startswith("t_") and "w_" + n[2:] in a:
            SCALES2[id(a["w_" + n[2:]])] = v


def order(fn):
    return list(fn.__code__.co_varnames[: fn.__code__.co_argcount])


def cos(a, b):
    return cb.cos(a, b)


def axmodel(name):
    (path,) = glob.glob(f"{BUILD}/cache/{name}.*.compiled.axmodel")
    return path


def dump(t, path):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        t = t.view(torch.uint8)
    t.numpy().tofile(path)


def main():
    global BF16_DIR, OUT
    BF16_DIR, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    emb = W("embed_tokens.weight")
    norm = q.one_plus(W("norm.weight"))
    x = emb[1000:1001].reshape(1, 1, q.H).contiguous()

    k_gdn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/gdn")(gdn_layer_mx)
    k_attn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/attn")(attn_layer_mx)
    k_head = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/lm_head")(lm_head_mx)
    k_gdn_bf = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/gdn_bf16")(q.gdn_layer)
    k_attn_bf = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/attn_bf16")(q.attn_layer)

    # 第 0 层（GDN）：MX kernel vs bf16 kernel（原权重 / 反量化权重）
    m, deq = mx_args(q.gdn_args(W, 0), GDN_W)
    io = dict(x=x, state_in=torch.zeros((q.NV, q.DK, q.DV)), conv_in=torch.zeros((1, q.CONV - 1, 2 * q.KD + q.VD),
              dtype=torch.bfloat16))
    outs = {}
    for tag, kern, args, names in [("mx", k_gdn, m, order(gdn_layer_mx)), ("bf16", k_gdn_bf, q.gdn_args(W, 0), q.GDN_ORDER),
                                   ("deq", k_gdn_bf, deq, q.GDN_ORDER)]:
        case = dict(args, **io, out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16),
                    state_out=torch.zeros((q.NV, q.DK, q.DV)),
                    conv_out=torch.zeros((1, q.CONV - 1, 2 * q.KD + q.VD), dtype=torch.bfloat16))
        SCALES.clear()
        kern[1](*[case[n] for n in names])
        outs[tag] = case["out"].clone()
    logger.info(f"gdn L0: mx~deq {cos(outs['mx'], outs['deq']):.6f}  mx~bf16 {cos(outs['mx'], outs['bf16']):.6f}")

    # 第 3 层（注意力），pos=0
    m, deq = mx_args(q.attn_args(W, 3), ATTN_W)
    c, s = q.rope_cos_sin(0)
    mask = torch.full((q.LMAX,), -3e38, dtype=torch.bfloat16)
    mask[0] = 0
    outs = {}
    for tag, kern, args, names in [("mx", k_attn, m, order(attn_layer_mx)),
                                   ("bf16", k_attn_bf, q.attn_args(W, 3), q.ATTN_ORDER),
                                   ("deq", k_attn_bf, deq, q.ATTN_ORDER)]:
        case = dict(args, x=x, cos=c, sin=s, kc=torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16),
                    vc=torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16),
                    pos=torch.tensor([0], dtype=torch.int32), nblk=torch.tensor([1], dtype=torch.int32),
                    mask=mask.reshape(q.LMAX // q.BC, q.BC), out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16))
        SCALES.clear()
        kern[1](*[case[n] for n in names])
        outs[tag] = case["out"].clone()
    logger.info(f"attn L3: mx~deq {cos(outs['mx'], outs['deq']):.6f}  mx~bf16 {cos(outs['mx'], outs['bf16']):.6f}")

    # lm_head 第 0 块
    lw, ls = quant_mxfp8(emb)
    logits = torch.zeros((1, 1, VC), dtype=torch.bfloat16)
    k_head[1](x, norm, lw[:VC].contiguous(), ls[:VC].contiguous(), logits)
    xf = x.float().reshape(1, q.H)
    hn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + q.EPS) * norm
    ref_deq = hn @ dequant(lw[:VC], ls[:VC]).T
    ref_bf = hn @ emb[:VC].float().T
    logger.info(f"lm_head chunk0: mx~deq {cos(logits.float(), ref_deq):.6f}  mx~bf16 {cos(logits.float(), ref_bf):.6f}")

    # 导出
    types = []
    for i, t in enumerate(q.LAYERS):
        gdn = t == "linear_attention"
        a = q.gdn_args(W, i) if gdn else q.attn_args(W, i)
        m, _ = mx_args(a, GDN_W if gdn else ATTN_W)
        types.append("gdn" if gdn else "attn")
        d = f"{OUT}/L{i:02d}"
        os.makedirs(d, exist_ok=True)
        for k, v in m.items():
            dump(v, f"{d}/{k}.bin")
    open(f"{OUT}/layers.txt", "w").write("\n".join(types) + "\n")
    dump(lw, f"{OUT}/lm_w.bin")
    dump(ls, f"{OUT}/lm_s.bin")
    shutil.copy(axmodel("gdn_layer_mx"), board_io.kernel_file(OUT, "gdn"))
    shutil.copy(axmodel("attn_layer_mx"), board_io.kernel_file(OUT, "attn"))
    shutil.copy(axmodel("lm_head_mx"), board_io.kernel_file(OUT, "lm_head"))
    for f in ["embed.bin", "final_norm.bin", "vocab.bin", "eos.txt", "meta.txt"]:
        shutil.copy(f"{BF16_DIR}/{f}", f"{OUT}/{f}")
    logger.info(f"wrote {OUT}")


if __name__ == "__main__":
    main()
