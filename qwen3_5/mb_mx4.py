"""Simulator micro-benchmark of the routed experts' MXFP4 matmuls on one core (as q35_layer_mc.moe_full): two experts
as now (gate-up and down per expert: four calls) against concatenated (one gate-up over both experts' rows, one down
over both experts' inputs), and one gate-up call alone (the per-call fixed cost). Prints simulator cycles per variant.
Usage: MODEL=<35B dir> WFMT=mx8 python mb_mx4.py <dump dir>"""
import os
import re
import sys

import torch
from loguru import logger

import axnn.dsl as ax
from axnn.dsl.kernels.utils.linear import linear
import q35_moe as mo

H, FE = 2048, 512
DUMP = os.path.abspath(sys.argv[1])
cycles = {}
logger.add(lambda m: cycles.__setitem__("last", int(re.search(r"simulator cycles: ([\d,]+)", m).group(1).replace(",", "")))
           if "simulator cycles" in m else None)


def q4(rows, cols):
    pk, sc = mo.quant_mxfp4(torch.randn((rows, cols)) * 0.02)
    return pk.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu)


def silu_mul(g, u):
    return ax.dsl.mul(ax.dsl.silu(g, output_dtype=ax.fp32), u, output_dtype=ax.bf16)


def sep(x, wc0, sc0, wd0, sd0, wc1, sc1, wd1, sd1, out):  # now: per expert gate-up and down
    h = ax.dsl.load(x)
    acc = None
    for i, (wc, sc, wd, sd) in enumerate(((wc0, sc0, wd0, sd0), (wc1, sc1, wd1, sd1))):
        gu = linear(h, wc, sc, f"gu{i}", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))
        a = silu_mul(gu[:, :, :FE], gu[:, :, FE:])
        y = linear(a, wd, sd, f"dn{i}", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(FE))
        acc = y if acc is None else ax.dsl.add(acc, y, output_dtype=ax.fp32)
    ax.dsl.store(acc, r=out)
    return out


def cat(x, wc, sc, wd, sd, out):  # concatenated: gate-up rows of both experts, down over both experts' inputs (K)
    h = ax.dsl.load(x)
    gu = linear(h, wc, sc, "gu", output_dtype=ax.bf16, oc_tile_size=mo.oc_tile_fp4(H))  # [1, 1, 2 * 2FE]
    a = ax.dsl.concat(silu_mul(gu[:, :, :FE], gu[:, :, FE : 2 * FE]),
                      silu_mul(gu[:, :, 2 * FE : 3 * FE], gu[:, :, 3 * FE :]), dim=-1)  # [1, 1, 2FE]
    y = linear(a, wd, sd, "dn", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(2 * FE))
    ax.dsl.store(y, r=out)
    return out


def one(x, wc, sc, out):  # a single gate-up call
    gu = linear(ax.dsl.load(x), wc, sc, "gu", output_dtype=ax.fp32, oc_tile_size=mo.oc_tile_fp4(H))
    ax.dsl.store(gu, r=out)
    return out


def run(fn, args):  # built and run once, then again from the cache: the cycle simulation (simulate_for_1_chip)
    os.makedirs(f"{DUMP}/{fn.__name__}", exist_ok=True)
    os.chdir(f"{DUMP}/{fn.__name__}")
    b = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{DUMP}/{fn.__name__}")(fn)
    b[1](*args)
    b[1](*args)
    return getattr(b, "simulator_cycles", cycles.get("last", -1))


def main():
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    x = torch.randn((1, 1, H)).bfloat16()
    e = [dict(zip(("wc", "sc"), q4(2 * FE, H))) | dict(zip(("wd", "sd"), q4(H, FE))) for _ in range(2)]
    out = torch.zeros((1, 1, H))
    res = {"sep (4 calls)": run(sep, (x, *[e[i][k] for i in range(2) for k in ("wc", "sc", "wd", "sd")], out))}
    wc = torch.cat([e[0]["wc"].view(torch.uint8), e[1]["wc"].view(torch.uint8)]).view(torch.float4_e2m1fn_x2)
    sc = torch.cat([e[0]["sc"].view(torch.uint8), e[1]["sc"].view(torch.uint8)]).view(torch.float8_e8m0fnu)
    wd = torch.cat([e[0]["wd"].view(torch.uint8), e[1]["wd"].view(torch.uint8)], dim=1).view(torch.float4_e2m1fn_x2)
    sd = torch.cat([e[0]["sd"].view(torch.uint8), e[1]["sd"].view(torch.uint8)], dim=1).view(torch.float8_e8m0fnu)
    res["cat (2 calls)"] = run(cat, (x, wc, sc, wd, sd, torch.zeros((1, 1, H))))
    res["one gate-up (1 call, 1 expert)"] = run(one, (x, e[0]["wc"], e[0]["sc"], torch.zeros((1, 1, 2 * FE))))
    res["one gate-up (1 call, 2 experts)"] = run(one, (x, wc, sc, torch.zeros((1, 1, 4 * FE))))
    for k, v in res.items():
        logger.info(f"MB {k}: {v:,} cycles")


if __name__ == "__main__":
    main()
