"""CPU preparation of packed runtime quantized weights."""

import numpy as np
import torch
from axnn.opset import datatype


def quantize_nvfp4(weight):
    """Quantize a floating [N, K] matrix on CPU, once during model loading.

    Return packed E2M1 [N, K/2], E4M3 block scales [N, K/16] and a
    FP32 global decode scale [1]. Both casts round to nearest, ties to even.
    Only 1024 rows are expanded to FP32 at a time, including for large heads.
    """
    if weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError("NVFP4 weights require a matrix with K divisible by 16")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("NVFP4 quantization requires floating-point weights")
    weight = weight.detach().cpu()
    low, high = torch.aminmax(weight)
    amax = max(abs(low.item()), abs(high.item()))
    global_scale = torch.tensor([amax / (448 * 6) if amax else 1.0], dtype=torch.float32)
    n, k = weight.shape
    packed = np.empty((n, k // 2), dtype=np.uint8)
    scales = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn)
    for start in range(0, n, 1024):
        blocks = weight[start : start + 1024].float().reshape(-1, k // 16, 16)
        scale = (blocks.abs().amax(-1) / (6 * global_scale)).clamp(max=448).to(torch.float8_e4m3fn)
        scales[start : start + 1024] = scale
        decode_scale = scale.float() * global_scale
        # Zero/underflowed block scales decode to zero; avoid a zero divisor.
        scaled = blocks / torch.where(decode_scale > 0, decode_scale, 1).unsqueeze(-1)
        codes = datatype.fp32_to_mxfp4(scaled.numpy(), datatype.DType(datatype.nvfp4)).view(np.uint8)
        codes = codes.reshape(-1, k)
        # Consecutive input channels occupy low (even K) / high (odd K) nibbles.
        packed[start : start + 1024] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.from_numpy(packed).view(torch.float4_e2m1fn_x2), scales, global_scale
