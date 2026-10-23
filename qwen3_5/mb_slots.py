"""Simulator cycles of the px slots kernel (q35_moe_prefill.moe_slots_px, SLOTS x SLOT_G, CORES) and of its
prefetching variant moe_slots_px2, on random MXFP4 experts and distinct experts per slot.
Usage: MODEL=<35B> KVL=cell LMAX=4096 PCHUNK=128 WFMT=mx8 CORES=4 SLOTS=64 SLOT_G=64 SLOT_OUT=bf16 python mb_slots.py <dump>"""
import os
import sys

import torch
from loguru import logger

import axnn.dsl as ax
import q35_moe as mo
import q35_moe_prefill as mp

DUMP = os.path.abspath(sys.argv[1])


def main():
    torch.manual_seed(0)
    S, G, H, FE, NE, CORES = mp.S, mp.G, mp.H, mo.FE, mo.NE, mp.CORES
    gw, gs, dw, ds = [], [], [], []
    for _ in range(NE):
        pk, sc = mo.quant_mxfp4(torch.randn((2 * FE, H)) * 0.02)
        gw.append(pk), gs.append(sc)
        pk, sc = mo.quant_mxfp4(torch.randn((H, FE)) * 0.02)
        dw.append(pk), ds.append(sc)
    ex = (torch.stack(gw).view(torch.float4_e2m1fn_x2), torch.stack(gs).view(torch.float8_e8m0fnu),
          torch.stack(dw).view(torch.float4_e2m1fn_x2), torch.stack(ds).view(torch.float8_e8m0fnu))
    T = 2048
    h2 = (torch.randn((T, H)) * 0.5).bfloat16()
    tok = torch.zeros((S, 1 + G), dtype=torch.int32)
    tok[:, 0] = torch.arange(S) % NE
    tok[:, 1:] = torch.randint(0, T, (S, G), dtype=torch.int32)
    rw = torch.rand((S, G))
    res = {}
    for fn in (mp.moe_slots_px, mp.moe_slots_px2):
        out = torch.zeros((S, G, H), dtype=mp.OUT_DT.torch_dtype if hasattr(mp.OUT_DT, "torch_dtype") else torch.bfloat16)
        os.makedirs(f"{DUMP}/{fn.__name__}", exist_ok=True)
        os.chdir(f"{DUMP}/{fn.__name__}")
        ax.init(chip_num=1, core_num=CORES)
        b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/{fn.__name__}")(fn)
        b[CORES](h2, tok, rw, *ex, out)
        b[CORES](h2, tok, rw, *ex, out)
        res[fn.__name__] = (b.simulator_cycles, out.clone())
        logger.info(f"MB {fn.__name__} S={S} G={G} npu{CORES}: {b.simulator_cycles:,} cycles "
                    f"({b.simulator_cycles / (S // CORES):,.0f} per slot per core)")
    a, b_ = res["moe_slots_px"][1], res["moe_slots_px2"][1]
    logger.info(f"MB px2 vs px bitwise {bool(torch.equal(a.view(torch.int16), b_.view(torch.int16)))}")


if __name__ == "__main__":
    main()
