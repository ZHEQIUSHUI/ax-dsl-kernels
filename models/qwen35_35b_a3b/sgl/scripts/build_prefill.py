"""Compile one of the npu-sgl36 Qwen3.5-35B-A3B prefill kernels at real shape.

usage: python build_prefill_sgl.py <gdn|flash|moe> <out_dir>
Compiles only (raises at the simulator entry once the axmodel exists).
"""
import glob
import os
import shutil
import sys
from pathlib import Path

import torch
import axnn.dsl as ax
from axnn.dsl import builder_ as B

from sgl_kernel_axera.attention.flash_attn import flash_attention, flash_attn_with_kvcache
from sgl_kernel_axera.attention.gdn import GDNKernel, runtime_weights, split_conv_state
from sgl_kernel_axera.models.qwen36_35b_a3b.forward import clear_prefill_padding
from sgl_kernel_axera.moe.m_qwen3 import moe_kernel_expert_parallel_v2_scf
from sgl_kernel_axera.quantization import quantize_nvfp4
from test_gdn_attn import random_case, random_nvfp4_weights
from test_qwen35_runtime import moe_weights

H, HEADS, KVH, HD = 2048, 16, 2, 256
KH, VH, KD, VD = 16, 32, 128, 128
EXPERTS, INTER = 256, 512
CTX, ROWS = int(os.getenv("CTX", "16384")), 4096


class Compiled(Exception):
    pass


def no_sim(self, *args, **kwargs):
    raise Compiled()


if not os.getenv("SIM"):
    for cls in [c for c in vars(B).values() if isinstance(c, type) and hasattr(c, "simulate_for_1_chip")]:
        setattr(cls, "simulate_for_1_chip", no_sim)


def gdn():
    x, _, w, state, conv = random_case(256, hidden=H, key_heads=KH, value_heads=VH, key_dim=KD, value_dim=VD)
    w, _ = random_nvfp4_weights(w, seed=11)
    w = runtime_weights(w)
    w["w_pre"] = None
    for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"),
                           (state, *split_conv_state(conv, KH * KD, VH * VD)), strict=True):
        w[name + "_out"] = w[name + "_in"] = value.contiguous()
    out = torch.empty_like(x)
    valid = torch.full((1,), 256, dtype=torch.int32)
    config(GDNKernel, "gdn")[4, 4](x, None, output=None, residual_out=out, valid_tokens=valid, **w)


def flash():
    w = {}
    for name, shape in (("q", (HEADS * HD * 2, H)), ("k", (KVH * HD, H)), ("v", (KVH * HD, H)), ("o", (H, HEADS * HD))):
        for prefix, value in zip(("w_", "w_scale_", "w_scale2_"), quantize_nvfp4((torch.randn(shape) * 0.02).bfloat16())):
            w[prefix + name] = value
        w["w_zeropoint_" + name] = torch.zeros(1)
    rot = int(HD * 0.25)
    ang = torch.arange(CTX)[:, None] / (1e7 ** (torch.arange(0, rot, 2).float() / rot))[None, :]
    w.update(w_norm_attn=torch.ones(H, dtype=torch.bfloat16), w_norm_q=torch.ones(HD) / HD**0.5,
             w_norm_k=torch.ones(HD, dtype=torch.bfloat16), cos=None, sin=None,
             cos_sin_cache=torch.cat((ang.cos(), ang.sin()), dim=-1), partial_rotary_factor=0.25)
    x = (torch.randn(1, 256, H) * 0.03).bfloat16()
    out = torch.empty_like(x)
    cache = torch.zeros(CTX // 256, 2, 256, KVH, HD, dtype=torch.bfloat16)
    positions = torch.arange(256, dtype=torch.int32)[None, :].repeat(3, 1)
    indptr = torch.tensor([0, 0], dtype=torch.int32)
    indices = torch.arange(CTX // 256, dtype=torch.int32)
    last_len = torch.tensor([0], dtype=torch.int32)
    loc = torch.tensor([0, 0], dtype=torch.int32)
    config(flash_attn_with_kvcache, "flash")
    flash_attention(x, w, positions, out, cache, indptr, indices, last_len,
                    cache_location=loc, core_num=4, attn_output_gate=True,
                    partial_rotary_factor=0.25, has_mrope=True, mrope_section=(11, 11, 10),
                    mrope_interleaved=True)


def moe():
    x = (torch.randn(ROWS, H) * 0.03).bfloat16()
    out = torch.empty_like(x)
    config(moe_kernel_expert_parallel_v2_scf, "moe")[4, 4](x, output_ax=out, **moe_weights(H, EXPERTS, INTER))


def padding():
    x = torch.zeros(ROWS, H, dtype=torch.bfloat16)
    n = torch.tensor([ROWS], dtype=torch.int32)
    config(clear_prefill_padding, "padding")[4, 4](x, n)


VARIANTS = {"gdn": gdn, "flash": flash, "moe": moe, "padding": padding}

out_path = Path(sys.argv[2]).resolve()
out_path.mkdir(parents=True, exist_ok=True)


def config(kernel, name):
    d = out_path / name
    d.mkdir(parents=True, exist_ok=True)
    kernel.cache_path = d
    kernel.dump_path = d / "artifacts"
    kernel.enable_cache = True
    kernel.unique_axmodel = True
    return kernel
torch.manual_seed(35)
torch.set_num_threads(16)
ax.init(chip_num=1)
try:
    VARIANTS[sys.argv[1]]()
except Compiled:
    pass
name = {"gdn": "sgl35_gdn.axmodel", "flash": "sgl35_flash.axmodel",
        "moe": "sgl35_moe4096.axmodel", "padding": "sgl35_padding.axmodel"}[sys.argv[1]]
f = glob.glob(str(out_path / sys.argv[1] / "**" / "*.compiled.axmodel"), recursive=True)
assert len(f) == 1, f
shutil.copy(f[0], out_path / name)
print("done", sys.argv[1], out_path / name, (out_path / name).stat().st_size, flush=True)
