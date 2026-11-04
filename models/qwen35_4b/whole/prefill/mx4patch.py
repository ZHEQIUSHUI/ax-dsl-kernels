"""Run the other team's Qwen3.8 fused-forward tests with MXFP4 (E8M0 per 32) weights instead of NVFP4."""
import numpy as np
import torch
from axnn.opset import datatype
import sgl_kernel_axera.quantization as q


def quantize_mxfp4(weight):
    weight = weight.detach().cpu().float()
    n, k = weight.shape
    assert k % 32 == 0
    blocks = weight.reshape(n, k // 32, 32)
    amax = blocks.abs().amax(-1)
    # OCP MX: scale = 2^(floor(log2(amax)) - 2) (E2M1 emax = 2)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2, torch.full_like(amax, -127))
    e = e.clamp(-127, 127)
    scaled = blocks / torch.exp2(e).unsqueeze(-1)
    codes = datatype.fp32_to_mxfp4(scaled.numpy(), datatype.DType(datatype.mxfp4)).view(np.uint8).reshape(n, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.from_numpy((e + 127).to(torch.uint8).numpy()).view(torch.float8_e8m0fnu)
    return torch.from_numpy(np.ascontiguousarray(packed)).view(torch.float4_e2m1fn_x2), scales, torch.ones(1)


def install():
    q.quantize_nvfp4 = quantize_mxfp4
    import test_gdn_attn, test_qwen38_fused_forward
    test_gdn_attn.quantize_nvfp4 = quantize_mxfp4
    test_qwen38_fused_forward.quantize_nvfp4 = quantize_mxfp4
