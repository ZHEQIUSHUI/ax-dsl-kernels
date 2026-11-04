"""NVFP4 LM-head packing, runtime scale inputs and fused FP32 projection."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_axera.quantization import quantize_nvfp4

from sgl_kernel_axera.gemm import rmsnorm_fp16_gemm


def dequantize(weight, scales, global_scale):
    """Independent E2M1 decode of low-even / high-odd packed columns."""
    packed = weight.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    return values[codes] * scales.float().repeat_interleave(16, -1) * global_scale


def test_quantize_nvfp4_layout_rounding_and_zero_blocks():
    source = torch.zeros(3, 32, dtype=torch.bfloat16)
    source[0, 0] = 448 * 6  # Make the global decode scale exactly one.
    source[1, :16] = torch.tensor(
        [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5]
    )
    weight, scales, scale2 = quantize_nvfp4(source)
    assert weight.dtype == torch.float4_e2m1fn_x2 and weight.shape == (3, 16)
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (3, 2)
    assert scale2.dtype == torch.float32 and scale2.tolist() == [1]
    expected_codes = torch.tensor([0, 0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14], dtype=torch.uint8)
    torch.testing.assert_close(weight.view(torch.uint8)[1, :8], expected_codes[::2] | (expected_codes[1::2] << 4))
    assert not weight.view(torch.uint8)[2].any()
    assert scales.float()[1, 0] == 1 and not scales.float()[2].any()
    assert torch.isfinite(dequantize(weight, scales, scale2)).all()
    zeros = quantize_nvfp4(torch.zeros(2, 16))
    assert zeros[2].item() == 1
    assert not dequantize(*zeros).any()


@pytest.mark.parametrize(
    "dtype,cores,m,n",
    [(torch.bfloat16, 4, 8, 1281), (torch.float16, 4, 17, 513), (torch.bfloat16, 4, 1, 64)],
)
def test_nvfp4_fused_norm_runtime_weights_fp32(dtype, cores, m, n):
    torch.manual_seed(93)
    a = torch.randn(m, 128, dtype=dtype)
    norm_weight = torch.randn(128, dtype=dtype)
    weight, scales, scale2 = quantize_nvfp4(torch.randn(n, 128, dtype=dtype) * 0.1)
    bias = torch.randn(n) / 16
    output = torch.empty(m, n, dtype=torch.float32)
    normalized = (a.float() * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6) * norm_weight.float()).to(
        torch.bfloat16
    )
    for multiplier in (1, 2):
        scale2.mul_(multiplier)
        actual = rmsnorm_fp16_gemm(
            a, norm_weight, weight, bias=bias, output=output, core_num=cores, b_scale=scales, b_scale2=scale2
        )
        expected = F.linear(normalized.float(), dequantize(weight, scales, scale2), bias)
        assert actual is output and actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        # Update packed weights and group scales without changing the kernel ABI.
        weight.view(torch.uint8).bitwise_xor_(0x88)
        scales.copy_((scales.float() * 0.5).to(scales.dtype))


@pytest.mark.parametrize("packed", (False, True))
def test_lm_head_preserves_fp32_norm_coefficients(packed):
    torch.manual_seed(20261030)
    a = torch.randn(8, 128).bfloat16()
    norm = 1 + (torch.randn(128) * 0.125).bfloat16().float()
    weight = torch.randn(64, 128).bfloat16() * 0.1
    if packed:
        weight, scales, scale2 = quantize_nvfp4(weight)
        decoded = dequantize(weight, scales, scale2)
    else:
        scales = scale2 = None
        decoded = weight.float()
    normalized = (a.float() * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6) * norm).bfloat16()
    rounded_norm = (
        a.float() * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6) * norm.bfloat16().float()
    ).bfloat16()
    assert not torch.equal(normalized, rounded_norm)
    output = rmsnorm_fp16_gemm(a, norm, weight, core_num=4, b_scale=scales, b_scale2=scale2)
    expected = F.linear(normalized.float(), decoded)
    torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
