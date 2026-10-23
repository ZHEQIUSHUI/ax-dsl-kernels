"""Simulator cycles of the four-core attention prefill kernel (q35_layer_mc.mine()[1], this PCHUNK, the first full
attention layer's weights): run once to build, again from the cache for the cycle simulation.
Usage: MODEL=<35B> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=mx8 PCHUNK=<P> python mb_attnpre.py"""
import torch
from loguru import logger

import q35_layer_mc as L
import q35_step as q

mb = L.mb


def main():
    W = q.Weights(q.MODEL)
    mc = L.mine()[1]
    li = q.LAYERS.index("full_attention")
    a = q.attn_args(W, li)
    a = mb.mx_args(a, mb.ATTN_W)[0] if L.MX and not (L.MOE and L.P > 1) else a
    cs, sn = L.rope_rows(0)
    kc, vc = (torch.zeros((L.LMAX, L.HKV * L.D), dtype=torch.half) for _ in range(2))
    case = dict(a, x=L.tokens(W, 0), cos=cs, sin=sn, kc=kc, vc=vc, pos=torch.tensor([0], dtype=torch.int32),
                nblk=torch.tensor([(L.P - 1) // L.BC + 1], dtype=torch.int32), mask=L.mask_rows(0),
                out=torch.zeros((1, L.P, L.H), dtype=torch.bfloat16), **L.moe_outs())
    k = L.kern(mc, f"mbap{L.P}")
    for _ in range(2):  # build, then from the cache (the cycle simulation)
        q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
        L.run(k, L.CORES, case, L.order(mc))
        q.RESIDENT.clear()
    cyc = k.simulator_cycles
    logger.info(f"MB attn prefill P={L.P} npu{L.CORES}: {cyc:,} cycles = {cyc / 1e3:.1f} us at 1 GHz, "
                f"{cyc / L.P:,.0f} cycles per token")


if __name__ == "__main__":
    main()
