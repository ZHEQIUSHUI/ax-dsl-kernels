"""Rewrite a bf16 GGUF in place with the sgl-kernel-axera NVFP4 quantization applied.

usage: sim_nvfp4_gguf.py <bf16.gguf> <out.gguf>

Copies the file, then for every tensor the pack quantizes (the projections and
experts) replaces the bf16 values with dequantize(quantize_nvfp4(w)), so a CPU
llama.cpp runs the weights the NPU kernels hold.  The splits follow
pack_sgl35_current.py: attn_qkv is sliced q|k|v, experts are quantized one
expert at a time (gate and up independently).

GGUF stores ne0 fastest, so a tensor of shape (ne0=K, ne1=N, ne2=E) is in
memory (E, N, K) -- the [N, K] matrices quantize_nvfp4 takes.
"""
import shutil, sys, time
import numpy as np
import torch, gguf
from sgl_kernel_axera.quantization import quantize_nvfp4

BLK = ("attn_qkv", "attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "ssm_alpha", "ssm_beta",
       "ssm_out", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp",
       "ffn_down_shexp")


def dequant(weight, scales, global_scale):
    packed = weight.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    return values[codes] * scales.float().repeat_interleave(16, -1) * global_scale


def nv(rows):  # [N, K] float -> [N, K] bf16 dequantized
    return dequant(*quantize_nvfp4(rows.contiguous().to(torch.bfloat16))).to(torch.bfloat16)


def transform(name, t):
    """t: (E, N, K) or (N, K) float32. Returns the same shape (bf16 values in f32)."""
    if t.ndim == 3:
        out = torch.empty_like(t)
        for i in range(t.shape[0]):
            out[i] = nv(t[i])
        return out
    if name.endswith("attn_qkv.weight"):
        return torch.cat([nv(t[:2048]), nv(t[2048:4096]), nv(t[4096:])], 0)
    return nv(t)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copyfile(src, dst)
    r = gguf.GGUFReader(dst)
    f = open(dst, "r+b")
    t0 = time.time()
    todo = [x for x in r.tensors if x.name.split(".")[-2] in BLK and x.name.endswith(".weight")]
    print("tensors to transform:", len(todo), flush=True)
    for i, t in enumerate(todo):
        assert t.tensor_type == gguf.GGMLQuantizationType.BF16, (t.name, t.tensor_type)
        shape = tuple(reversed(t.shape))
        f.seek(t.data_offset)
        a = np.frombuffer(f.read(t.n_bytes), dtype="<u2").astype(np.uint32)
        x = ((a << 16) | (a >> 16)).view(np.float32).reshape(shape)
        y = transform(t.name, torch.from_numpy(x)).float().numpy()
        b = y.view(np.uint32)
        f.seek(t.data_offset)
        f.write(((b >> 16) | (b << 16)).astype("<u2").tobytes())
        if i % 20 == 0 or i == len(todo) - 1:
            print(f"{i+1}/{len(todo)} {t.name} {time.time()-t0:.0f}s", flush=True)
    f.close()
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
