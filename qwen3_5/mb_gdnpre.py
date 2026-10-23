"""Simulator cycles of the four-core GDN prefill kernel (q35_layer_mc.mine()[0], this PCHUNK, MXFP8, layer 0's
weights) and per-token figures: run once to build, again from the cache for the cycle simulation.
Usage: MODEL=<35B> KVL=cell LMAX=4096 BC=256 CORES=4 WFMT=mx8 PCHUNK=<P> python mb_gdnpre.py"""
import os

import torch
from loguru import logger

import q35_layer_mc as L
import q35_step as q

mb = L.mb


def main():
    W = q.Weights(q.MODEL)
    mc = L.mine()[0]
    a = mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0]
    torch.manual_seed(0)
    case = dict(a, x=L.tokens(W, 0), state_in=torch.zeros((L.NV, L.DK, L.DV)),
                conv_in=torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16),
                out=torch.zeros((1, L.P, L.H), dtype=torch.bfloat16), state_out=torch.zeros((L.NV, L.DK, L.DV)),
                conv_out=torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16), **L.moe_outs())
    k = L.kern(mc, f"mbgp{L.P}")
    for _ in range(2):  # build, then from the cache (the cycle simulation)
        L.run(k, L.CORES, case, L.order(mc))
    cyc = k.simulator_cycles
    logger.info(f"MB gdn prefill P={L.P} npu{L.CORES}: {cyc:,} cycles = {cyc / 1e3:.1f} us at 1 GHz, "
                f"{cyc / L.P:,.0f} cycles per token")


if __name__ == "__main__":
    main()
