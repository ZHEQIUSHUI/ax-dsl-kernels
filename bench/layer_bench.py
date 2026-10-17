"""Phase-0 bench: one full decode layer in ONE call, single core, all weights runtime MXFP4 inputs.

QKV projection -> write new K/V into the resident cache at runtime `pos` -> GQA attention over the cache
(including the new token; runtime additive mask like llama.cpp's KQ_mask, runtime block count) ->
O projection + residual -> MLP (gate/up/silu*mul/down) + residual.
RMSNorm / RoPE are left out on purpose: they are element-wise and do not change what this checks.

kc/vc are read AND written by the kernel; on the board the host binds the input and output of the same
name to one buffer.

Env: H (1024), HQ (8), HKV (2), D (256), F (3584), LMAX (4096), BC (256), POS (1000), TN (256)
"""

import math
import os
import sys

import ml_dtypes
import numpy as np
import torch
from loguru import logger
from onnx import TensorProto

import axnn.dsl as ax
from axnn.frontend import graph_utils

# 导出器缺 MX 运行时输入的 dtype 映射：按 uint8 字节流导出（仅本脚本内生效，不改仓库）
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXE8M0", TensorProto.UINT8)
graph_utils.AX_TYPE_TO_TENSOR_TYPE.setdefault("MXFP4_X2", TensorProto.UINT8)

H = int(os.environ.get("H", 1024))
HQ = int(os.environ.get("HQ", 8))
HKV = int(os.environ.get("HKV", 2))
D = int(os.environ.get("D", 256))
F = int(os.environ.get("F", 3584))
LMAX = int(os.environ.get("LMAX", 4096))
BC = int(os.environ.get("BC", 256))
POS = int(os.environ.get("POS", 1000))
TN = int(os.environ.get("TN", 256))
G = HQ // HKV
GROUP = 32
DUMP = os.path.abspath(f"out_layer_h{H}_q{HQ}_kv{HKV}_d{D}_f{F}_l{LMAX}")


def gemv_into(xt, w, s, dst, n):
    """dst[:, 0:n] = xt @ dequant(w)^T, tiled over output channels."""
    for n0 in range(0, n, TN):
        ax.dsl.matmul(xt, ax.dsl.load(w[n0 : n0 + TN, :]), w_scale=ax.dsl.load(s[n0 : n0 + TN, :]),
                      transposed=True, output_dtype=ax.bf16, r=dst[:, n0 : n0 + TN])


def residual_gemv(xt, w, s, res, dst, n):
    """dst = res + xt @ dequant(w)^T, tiled over output channels."""
    for n0 in range(0, n, TN):
        d = ax.dsl.matmul(xt, ax.dsl.load(w[n0 : n0 + TN, :]), w_scale=ax.dsl.load(s[n0 : n0 + TN, :]),
                          transposed=True, output_dtype=ax.bf16)
        ax.dsl.add(res[:, n0 : n0 + TN], d, output_dtype=ax.bf16, r=dst[:, n0 : n0 + TN])


def decode_layer(x, wq, sq, wk, sk, wv, sv, wo, so, wg, sg, wu, su, wd, sd, kc, vc, pos, nblk, mask, out):
    xt = ax.dsl.load(x)  # [1, H]

    # 1. QKV 投影
    q = ax.dsl.create_tensor((1, HQ * D), ax.bf16, ax.MemoryType.OCM, name="q")
    k = ax.dsl.create_tensor((1, HKV * D), ax.bf16, ax.MemoryType.OCM, name="k")
    v = ax.dsl.create_tensor((1, HKV * D), ax.bf16, ax.MemoryType.OCM, name="v")
    gemv_into(xt, wq, sq, q, HQ * D)
    gemv_into(xt, wk, sk, k, HKV * D)
    gemv_into(xt, wv, sv, v, HKV * D)

    # 2. 新 token 的 K/V 写入常驻 cache 的 pos 行（逐头写）
    one_row = ax.dsl.from_numpy(np.array([1, 1], dtype=np.uint32))
    src0 = ax.dsl.from_numpy(np.zeros(shape=(2,), dtype=np.uint32))
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    for h in range(HKV):
        dst = ax.dsl.concat(ax.dsl.from_numpy(np.array([h], dtype=np.uint32)), p, dim=0)
        ax.dsl.dynamic_st(k[:, h * D : (h + 1) * D].reshape(1, 1, D), src0, dst, one_row, dim=(0, 1),
                          output_shape=kc.shape, r=kc)
        ax.dsl.dynamic_st(v[:, h * D : (h + 1) * D].reshape(1, 1, D), src0, dst, one_row, dim=(0, 1),
                          output_shape=vc.shape, r=vc)

    # 3. attention：按运行时块数循环读 cache（含新 token），加运行时 mask
    attn = ax.dsl.create_tensor((1, HQ * D), ax.bf16, ax.MemoryType.OCM, name="attn")
    kv_block_shape = ax.dsl.from_numpy(np.array([1, BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1.0 / math.sqrt(D))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_stop = ax.dsl.load(nblk)
    for h in range(HKV):
        head = ax.dsl.from_numpy(np.array([h], dtype=np.uint32))
        qh = ax.dsl.mul(q[:, h * G * D : (h + 1) * G * D].reshape(G, D), scale, output_dtype=ax.bf16)
        oi = ax.dsl.zeros(shape=(G, D), dtype=ax.bf16)
        li = ax.dsl.zeros(shape=(G,), dtype=ax.fp32)
        mi = ax.dsl.full(shape=(G, 1), dtype=ax.bf16, fill_value=-3e38)
        zero_col = ax.dsl.zeros(shape=(G, 1), dtype=ax.bf16)
        index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

        def body(index, kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape,
                 zero_col, minus_1):
            loop_mi, loop_li, loop_oi = mi, li, oi
            t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
            off = ax.dsl.concat(head, t0, dim=0)
            kj = ax.dsl.dynamic_ld(kc, off, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
            vj = ax.dsl.dynamic_ld(vc, off, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
            # mask 形状 [LMAX/BC, BC]，按整块读（dynamic_ld 的读取单元要 32B 对齐）
            mj = ax.dsl.dynamic_ld(mask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                                   output_shape=(1, BC))

            m_s = ax.dsl.create_tensor(shape=(G, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            ax.dsl.add(sj, mj, output_dtype=ax.bf16, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(G)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
            oij = pj @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(G, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
            return (kc, vc, mask, qh, loop_mi, loop_li, loop_oi, head, block_stride, kv_block_shape,
                    mask_block_shape, zero_col, minus_1)

        (kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape, zero_col,
         minus_1) = ax.dsl.range(
            index, range_start, range_stop, range_step, body,
            kc, vc, mask, qh, mi, li, oi, head, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1,
        )
        ax.dsl.div(oi, li.reshape(G, 1), output_dtype=ax.bf16,
                   r=attn[:, h * G * D : (h + 1) * G * D].reshape(G, D))

    # 4. O 投影 + 残差
    x1 = ax.dsl.create_tensor((1, H), ax.bf16, ax.MemoryType.OCM, name="x1")
    residual_gemv(attn, wo, so, xt, x1, H)

    # 5. MLP + 残差
    gt = ax.dsl.create_tensor((1, F), ax.bf16, ax.MemoryType.OCM, name="g")
    ut = ax.dsl.create_tensor((1, F), ax.bf16, ax.MemoryType.OCM, name="u")
    gemv_into(x1, wg, sg, gt, F)
    gemv_into(x1, wu, su, ut, F)
    hh = ax.dsl.mul(ax.dsl.silu(gt, output_dtype=ax.fp32), ut, output_dtype=ax.bf16)
    x2 = ax.dsl.create_tensor((1, H), ax.bf16, ax.MemoryType.OCM, name="x2")
    residual_gemv(hh, wd, sd, x1, x2, H)
    ax.dsl.store(x2, r=out)
    return out, kc, vc


LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def make_weight(rows, cols, g, lo=116, hi=121):
    codes = torch.randint(0, 16, (rows, cols), generator=g, dtype=torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    enc = torch.randint(lo, hi, (rows, cols // GROUP), generator=g, dtype=torch.uint8)
    deq = LUT[codes.long()] * (2.0 ** (enc.float() - 127)).repeat_interleave(GROUP, dim=-1)
    return packed, enc.view(torch.float8_e8m0fnu), deq


def make_case(seed=7):
    g = torch.Generator().manual_seed(seed)
    case = {}
    case["x"] = torch.randn((1, H), generator=g).bfloat16()
    deq = {}
    for name, rows, cols in (("q", HQ * D, H), ("k", HKV * D, H), ("v", HKV * D, H), ("o", H, HQ * D),
                             ("g", F, H), ("u", F, H), ("d", H, F)):
        w, s, dq = make_weight(rows, cols, g)
        case["w" + name], case["s" + name], deq[name] = w, s, dq
    case["kc"] = torch.randn((HKV, LMAX, D), generator=g).bfloat16()
    case["vc"] = torch.randn((HKV, LMAX, D), generator=g).bfloat16()
    case["pos"] = torch.tensor([POS], dtype=torch.int32)
    case["nblk"] = torch.tensor([POS // BC + 1], dtype=torch.int32)
    m = torch.full((LMAX,), -3e38)
    m[: POS + 1] = 0
    case["mask"] = m.reshape(LMAX // BC, BC).bfloat16()
    return case, deq


def reference(case, deq):
    bf = lambda t: t.bfloat16().float()  # noqa: E731
    x = case["x"].float()
    q, k, v = bf(x @ deq["q"].T), bf(x @ deq["k"].T), bf(x @ deq["v"].T)
    kc, vc = case["kc"].float().clone(), case["vc"].float().clone()
    for h in range(HKV):
        kc[h, POS] = k[0, h * D : (h + 1) * D]
        vc[h, POS] = v[0, h * D : (h + 1) * D]
    attn = torch.empty((1, HQ * D))
    for h in range(HKV):
        qh = q[0, h * G * D : (h + 1) * G * D].reshape(G, D) / math.sqrt(D)
        sc = qh @ kc[h, : POS + 1].T
        attn[0, h * G * D : (h + 1) * G * D] = (torch.softmax(sc, -1) @ vc[h, : POS + 1]).reshape(-1)
    x1 = bf(x + bf(attn) @ deq["o"].T)
    hh = torch.nn.functional.silu(x1 @ deq["g"].T) * (x1 @ deq["u"].T)
    x2 = x1 + bf(hh) @ deq["d"].T
    return x2, kc, vc


ORDER = ["x", "wq", "sq", "wk", "sk", "wv", "sv", "wo", "so", "wg", "sg", "wu", "su", "wd", "sd",
         "kc", "vc", "pos", "nblk", "mask"]


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(DUMP)  # 编译缓存（含 axmodel）写到当前目录的 cache/
    ax.init(chip_num=1, core_num=1)
    case, deq = make_case()
    ref_out, ref_kc, ref_vc = reference(case, deq)

    # 板端输入：<名字>.bin 原始字节；参考输出 ref_*.f32
    os.makedirs("board_in", exist_ok=True)
    for name in ORDER:
        t = case[name]
        t.view(torch.uint8).numpy().tofile(f"board_in/{name}.bin") if t.dtype in (
            torch.float4_e2m1fn_x2, torch.float8_e8m0fnu) else t.contiguous().view(
            torch.int16 if t.dtype == torch.bfloat16 else t.dtype).numpy().tofile(f"board_in/{name}.bin")
    ref_out.numpy().tofile("ref_out.f32")
    torch.stack([ref_kc[:, POS], ref_vc[:, POS]]).numpy().tofile("ref_kv_pos.f32")

    out = torch.empty((1, H), dtype=torch.bfloat16)
    kernel = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=DUMP)(decode_layer)
    kernel[1](*[case[n] for n in ORDER], out)

    c = torch.nn.functional.cosine_similarity(ref_out.reshape(-1), out.float().reshape(-1), dim=0).item()
    kv_ok = torch.allclose(case["kc"][:, POS].float(), ref_kc[:, POS], atol=1e-2)
    weight_mb = sum(case[n].numel() for n in ORDER if n[0] in "ws" and n not in ("x",)) / 1e6
    logger.info(f"RESULT layer H={H} HQ={HQ} HKV={HKV} D={D} F={F} LMAX={LMAX} POS={POS} weights={weight_mb:.1f}MB "
                f"cosine={c:.6f} kv_pos_written={kv_ok} dump={DUMP}")
    if c < 0.98:
        sys.exit(3)


if __name__ == "__main__":
    main()
