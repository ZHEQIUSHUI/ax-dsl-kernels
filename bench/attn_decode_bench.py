"""Phase-0 bench: single-core GQA decode attention over a runtime-length KV cache.

One new token, KV cache resident as runtime input [Hkv, LMAX, D]; the number of valid KV blocks is a
runtime input, so one axmodel serves every context length up to LMAX (online softmax, flash-style).

Env: HKV (2), G (q heads per kv head, 4), D (256), LMAX (4096), BC (256), NBLK (valid blocks for the check)
"""

import math
import os
import sys

import ml_dtypes
import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax

HKV = int(os.environ.get("HKV", 2))
G = int(os.environ.get("G", 4))
D = int(os.environ.get("D", 256))
LMAX = int(os.environ.get("LMAX", 4096))
BC = int(os.environ.get("BC", 256))
NBLK = int(os.environ.get("NBLK", LMAX // BC))
DUMP = os.path.abspath(f"out_attn_hkv{HKV}_g{G}_d{D}_l{LMAX}_bc{BC}")


def attn_decode(q, kc, vc, nblk, out):
    zero_u32 = ax.dsl.from_numpy(np.zeros(shape=(1,), dtype=np.uint32))
    kv_block_shape = ax.dsl.from_numpy(np.array([1, BC], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_stop = ax.dsl.load(nblk)  # 运行时有效块数

    for h in range(HKV):
        head = ax.dsl.from_numpy(np.array([h], dtype=np.uint32))
        qh = ax.dsl.load(q[h : h + 1]).reshape(G, D)
        oi = ax.dsl.zeros(shape=(G, D), dtype=ax.bf16)
        li = ax.dsl.zeros(shape=(G,), dtype=ax.fp32)
        mi = ax.dsl.full(shape=(G, 1), dtype=ax.bf16, fill_value=-3e38)
        zero_col = ax.dsl.zeros(shape=(G, 1), dtype=ax.bf16)
        index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)

        def body(index, kc, vc, qh, mi, li, oi, head, block_stride, kv_block_shape, zero_col, minus_1):
            loop_mi, loop_li, loop_oi = mi, li, oi
            token_offset = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
            offset = ax.dsl.concat(head, token_offset, dim=0)
            kj = ax.dsl.dynamic_ld(kc, offset, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)
            vj = ax.dsl.dynamic_ld(vc, offset, kv_block_shape, dim=(0, 1), output_shape=(1, BC, D)).reshape(BC, D)

            m_s = ax.dsl.create_tensor(shape=(G, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            ax.dsl.matmul(qh, kj, transposed=True, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            p = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(G)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=loop_li, output_dtype=ax.fp32)
            oij = p @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(G, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=loop_oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=loop_mi)
            return kc, vc, qh, loop_mi, loop_li, loop_oi, head, block_stride, kv_block_shape, zero_col, minus_1

        (kc, vc, qh, mi, li, oi, head, block_stride, kv_block_shape, zero_col, minus_1) = ax.dsl.range(
            index, range_start, range_stop, range_step, body,
            kc, vc, qh, mi, li, oi, head, block_stride, kv_block_shape, zero_col, minus_1,
        )
        res = ax.dsl.div(oi, li.reshape(G, 1), output_dtype=ax.bf16)
        ax.dsl.store(res.reshape(1, G, D), r=out[h : h + 1])
    return out


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(DUMP)  # 编译缓存（含 axmodel）写到当前目录的 cache/
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(3)
    q = (torch.randn((HKV, G, D)) / math.sqrt(D)).bfloat16()  # 已乘 1/sqrt(D)
    kc = torch.randn((HKV, LMAX, D)).bfloat16()
    vc = torch.randn((HKV, LMAX, D)).bfloat16()
    nblk = torch.tensor([NBLK], dtype=torch.int32)
    out = torch.empty((HKV, G, D), dtype=torch.bfloat16)

    kernel = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=DUMP)(attn_decode)
    kernel[1](q, kc, vc, nblk, out)

    n = NBLK * BC
    s = q.float() @ kc[:, :n].float().transpose(1, 2)
    ref = torch.softmax(s, dim=-1) @ vc[:, :n].float()
    c = torch.nn.functional.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0).item()
    logger.info(f"RESULT HKV={HKV} G={G} D={D} LMAX={LMAX} BC={BC} NBLK={NBLK} cosine={c:.6f} dump={DUMP}")
    if c < 0.99:
        sys.exit(3)


if __name__ == "__main__":
    main()
