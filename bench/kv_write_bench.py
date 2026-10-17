"""Phase-0 bench: write one token's K/V into a resident KV cache at a runtime position (in place).

Inputs: kc/vc [HKV, LMAX, D] bf16 (cache), kn/vn [HKV, 1, D] bf16 (new token), pos int32 [1].
The kernel returns kc/vc themselves, so the cache is an in/out buffer; only one slot is written.
"""

import os
import sys

import numpy as np
import torch
from loguru import logger

import axnn.dsl as ax

HKV = int(os.environ.get("HKV", 2))
D = int(os.environ.get("D", 256))
LMAX = int(os.environ.get("LMAX", 8192))
POS = int(os.environ.get("POS", 1234))
DUMP = os.path.abspath(f"out_kvw_hkv{HKV}_d{D}_l{LMAX}")


def kv_write(kc, vc, kn, vn, pos):
    # dynamic_st 一次只写首维为 1 的区域，按 KV 头逐个写
    shape = ax.dsl.from_numpy(np.array([1, 1], dtype=np.uint32))
    src = ax.dsl.from_numpy(np.zeros(shape=(2,), dtype=np.uint32))
    p = ax.dsl.cast(ax.dsl.load(pos), dtype=ax.uint32)
    for h in range(HKV):
        head = ax.dsl.from_numpy(np.array([h], dtype=np.uint32))
        dst = ax.dsl.concat(head, p, dim=0)  # [h, pos]
        k = ax.dsl.load(kn[h : h + 1])
        v = ax.dsl.load(vn[h : h + 1])
        ax.dsl.dynamic_st(k, src, dst, shape, dim=(0, 1), output_shape=kc.shape, r=kc)
        ax.dsl.dynamic_st(v, src, dst, shape, dim=(0, 1), output_shape=vc.shape, r=vc)
    return kc, vc


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(DUMP)  # 编译缓存（含 axmodel）写到当前目录的 cache/
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(5)
    kc = torch.randn((HKV, LMAX, D)).bfloat16()
    vc = torch.randn((HKV, LMAX, D)).bfloat16()
    kc0, vc0 = kc.clone(), vc.clone()
    kn = torch.randn((HKV, 1, D)).bfloat16()
    vn = torch.randn((HKV, 1, D)).bfloat16()
    pos = torch.tensor([POS], dtype=torch.int32)

    kernel = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=DUMP)(kv_write)
    kernel[1](kc, vc, kn, vn, pos)

    # DSL 把返回的 kc/vc 当作独立输出 buffer：只有写入的那一行有定义，其余行靠宿主把输入输出绑到同一块内存
    ok = torch.equal(kc[:, POS : POS + 1], kn) and torch.equal(vc[:, POS : POS + 1], vn)
    logger.info(f"RESULT HKV={HKV} D={D} LMAX={LMAX} POS={POS} slot_exact={ok} dump={DUMP}")
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
