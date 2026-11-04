"""Numerical tests for the Qwen GDN AXDSL implementation."""

from __future__ import annotations

import importlib
import inspect
import os

import ml_dtypes
import numpy as np
import axnn.dsl as ax
from axnn.dsl.ctx import ctx
import pytest
import torch

from sgl_kernel_axera.attention.gdn import GDNKernel, gdn


gdn_module = importlib.import_module("sgl_kernel_axera.attention.gdn")


@ax.builder(run_mode=ax.RunMode(os.environ.get("RUN_MODE", "PYRUN")), enable_cache=False)
def small_product_kernel(x, y, output):
    product = gdn_module.small_matmul_fp32(ax.dsl.load(x), ax.dsl.load(y))
    ax.dsl.store(product, r=output)
    return output


@ax.builder(run_mode=ax.RunMode(os.environ.get("RUN_MODE", "PYRUN")), enable_cache=False)
def inverse_kernel(negative_lower, output):
    inverse = gdn_module.inverse_unit_lower_recursive(ax.dsl.load(negative_lower), "test_gdn_inverse")
    ax.dsl.store(inverse, r=output)
    return output


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def wide_projection_kernel(
    x,
    q_weight,
    q_scale,
    q_scale2,
    k_weight,
    k_scale,
    k_scale2,
    v_weight,
    v_scale,
    v_scale2,
    z_weight,
    z_scale,
    z_scale2,
    q_output,
    k_output,
    v_output,
    z_output,
):
    x_scaled = ax.dsl.load(x)
    x_rstd = ax.dsl.full(shape=(x.shape[0], 1), dtype=ax.fp32, fill_value=1.0)
    q_projected, k_projected = gdn_module.project_normed_inputs_scaled_waves(
        x_scaled,
        x_rstd,
        (
            ("q_weight_pair_raw", q_weight, q_scale, q_scale2),
            ("k_weight_pair_raw", k_weight, k_scale, k_scale2),
        ),
        32,
        16,
    )
    ax.dsl.store(q_projected, r=q_output)
    ax.dsl.store(k_projected, r=k_output)
    v_projected, z_projected = gdn_module.project_normed_inputs_scaled_waves(
        x_scaled,
        x_rstd,
        (
            ("v_weight_nvfp4_raw", v_weight, v_scale, v_scale2),
            ("z_weight_nvfp4_raw", z_weight, z_scale, z_scale2),
        ),
        32,
        16,
    )
    ax.dsl.store(v_projected, r=v_output)
    ax.dsl.store(z_projected, r=z_output)
    return q_output, k_output, v_output, z_output


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def bf16_preprojection_wave_kernel(x, x_rstd, weight, output):
    (projected,) = gdn_module.project_normed_inputs_unscaled(
        ax.dsl.load(x),
        ax.dsl.load(x_rstd),
        weight,
        (512,),
        ic_tile_width=32,
    )
    ax.dsl.store(projected, r=output)
    return output


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def mxfp8_projection_wave_kernel(x, x_rstd, weight, scale, output):
    (projected,) = gdn_module.project_normed_inputs_scaled_waves(
        ax.dsl.load(x),
        ax.dsl.load(x_rstd),
        (("mxfp8_weight_raw", weight, scale, None),),
        ic_tile_width=32,
        scale_group_size=32,
    )
    ax.dsl.store(projected, r=output)
    return output


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def mxfp4_projection_wave_kernel(x, x_rstd, weight, scale, output):
    (projected,) = gdn_module.project_normed_inputs_scaled_waves(
        ax.dsl.load(x),
        ax.dsl.load(x_rstd),
        (("mxfp4_weight_raw", weight, scale, None),),
        ic_tile_width=32,
        scale_group_size=32,
    )
    ax.dsl.store(projected, r=output)
    return output


@ax.builder(run_mode=ax.RunMode(os.environ.get("RUN_MODE", "PYRUN")), enable_cache=False)
def nvfp4_ordered_projection_kernel(
    x, weight, scale, scale2, first_output, second_output, aliased, ic_tile_width, oc_tile_width
):
    core = ax.program_id(0)
    width = weight.shape[0] // ax.num_programs(0)
    owned = slice(core * width, (core + 1) * width)
    raw_buffers = None
    if aliased:
        raw = ax.dsl.create_tensor((x.shape[0], 2 * width + width // 2), x.dtype)
        raw_buffers = (raw[:, :width], raw[:, width : 2 * width])
    first, second = gdn_module.project_normed_inputs_scaled_waves(
        ax.dsl.load(x),
        ax.dsl.full((x.shape[0], 1), ax.fp32, 1),
        (("first", weight[owned], scale[owned], scale2), ("second", weight[owned], scale[owned], scale2)),
        ic_tile_width=ic_tile_width,
        scale_group_size=16,
        oc_tile_width=oc_tile_width,
        raw_buffers=raw_buffers,
    )
    ax.dsl.store(first, r=first_output[:, owned])
    ax.dsl.store(second, r=second_output[:, owned])
    return first_output, second_output


def bf16_randn(shape, scale=0.1):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(torch.bfloat16)


def linear_bf16(x, weight):
    return torch.matmul(x.float(), weight.float().transpose(-1, -2)).to(torch.bfloat16)


FP4_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def pack_fp4_codes(logical):
    """Pack FP4 codes with even columns in low nibbles, as runtime MatMul expects."""
    raw = np.ascontiguousarray(logical, dtype=np.uint8)
    bits = np.unpackbits(raw.reshape(-1), bitorder="little").reshape(-1, 8)[:, :4]
    packed = np.packbits(bits, bitorder="little")
    packed_shape = raw.shape[:-1] + (raw.shape[-1] // 2,)
    return torch.from_numpy(packed).view(torch.float4_e2m1fn_x2).reshape(packed_shape)


def random_nvfp4_weights(weights, seed):
    """Create exact packed NVFP4 payloads and their dequantized reference weights."""
    rng = np.random.default_rng(seed)
    packed_weights = dict(weights)
    reference_weights = dict(weights)
    for name in ("w_q", "w_k", "w_v", "w_z", "w_a", "w_b", "w_o"):
        shape = weights[name].shape
        codes = rng.integers(0, 16, size=shape, dtype=np.uint8)
        packed_weights[name] = pack_fp4_codes(codes)
        scale_key = f"{name}_scale"

        logical_weight = FP4_LUT[torch.from_numpy(codes).long()]
        group_size = 16
        scale_shape = (shape[0], shape[1] // group_size)
        encoded_scale = rng.uniform(0.008, 0.032, size=scale_shape)
        group_scale = torch.from_numpy(encoded_scale.astype(np.float32)).to(torch.float8_e4m3fn)
        scale2 = float(rng.uniform(0.25, 0.75))
        packed_weights[scale_key] = group_scale
        packed_weights[f"{scale_key}2"] = torch.tensor([scale2], dtype=torch.float32)
        expanded_scale = group_scale.float().repeat_interleave(group_size, dim=-1)
        reference_weights[name] = logical_weight * expanded_scale * scale2
    return packed_weights, reference_weights


MXFP8_BLOCK_SIZE = 32


def random_mxfp8_weights(weights, seed):
    """Create E4M3 MXFP8 weights with per-32-channel E8M0 scales.

    The hardware dequantizes each block as ``fp8 * 2 ** (scale - 127)``. The
    returned reference weights apply the same formula in FP32 so the PyTorch
    reference and the DSL projection use the same numerical contract.
    """
    generator = torch.Generator().manual_seed(seed)
    packed_weights = dict(weights)
    reference_weights = dict(weights)
    for name in ("w_q", "w_k", "w_v", "w_z", "w_a", "w_b", "w_o"):
        shape = weights[name].shape
        assert shape[-1] % MXFP8_BLOCK_SIZE == 0
        fp8_weight = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
        scale_shape = (shape[0], shape[-1] // MXFP8_BLOCK_SIZE)
        # Keep scales finite and close to 1.0 so the recurrence remains stable.
        scale_u8 = torch.randint(124, 131, scale_shape, generator=generator, dtype=torch.int64).to(torch.uint8)
        packed_weights[name] = fp8_weight
        packed_weights[f"{name}_scale"] = scale_u8.view(torch.float8_e8m0fnu)
        scale_value = torch.pow(2.0, scale_u8.float() - 127.0)
        reference_weights[name] = fp8_weight.float() * scale_value.repeat_interleave(MXFP8_BLOCK_SIZE, dim=-1)
    return packed_weights, reference_weights


def random_mxfp4_weights(weights, seed):
    """Create packed E2M1 MXFP4 weights with per-32-channel E8M0 scales."""
    rng = np.random.default_rng(seed)
    packed_weights = dict(weights)
    reference_weights = dict(weights)
    for name in ("w_q", "w_k", "w_v", "w_z", "w_a", "w_b", "w_o"):
        shape = weights[name].shape
        assert shape[-1] % 32 == 0
        codes = rng.integers(0, 16, size=shape, dtype=np.uint8)
        packed_weights[name] = pack_fp4_codes(codes)
        scale_shape = (shape[0], shape[-1] // 32)
        # Keep MXFP4 weights in the same small-value regime as the NVFP4
        # reference; large E8M0 exponents amplify recurrent gate error.
        scale_u8 = rng.integers(120, 125, size=scale_shape, dtype=np.uint8)
        packed_weights[f"{name}_scale"] = torch.from_numpy(scale_u8).view(torch.float8_e8m0fnu)
        logical_weight = FP4_LUT[torch.from_numpy(codes).long()]
        scale_value = torch.pow(2.0, torch.from_numpy(scale_u8).float() - 127.0)
        reference_weights[name] = logical_weight * scale_value.repeat_interleave(32, dim=-1)
    return packed_weights, reference_weights


def causal_conv1d_silu_reference(x, weight, state_in):
    state_len = state_in.shape[1]
    conv_size = weight.shape[1]
    assert state_len == conv_size - 1
    conv_input = torch.cat((state_in, x), dim=1)
    output = torch.zeros_like(x)
    for offset in range(conv_size):
        source_start = state_len - offset
        source_stop = source_start + x.shape[1]
        # PyTorch Conv1d stores oldest-to-newest taps; offset zero reads now.
        term = conv_input[:, source_start:source_stop, :].float() * weight[:, conv_size - 1 - offset].float().reshape(
            1, 1, -1
        )
        output = (output.float() + term).to(x.dtype)
    state_out = conv_input[:, -state_len:, :] if state_len else conv_input[:, :0, :]
    return torch.nn.functional.silu(output.float()).to(x.dtype), state_out


def gdn_reference(q, k, v, a, b, a_log, dt_bias, state_in):
    key_dim = q.shape[-1]
    value_heads = v.shape[2]
    value_heads_per_key = value_heads // q.shape[2]
    q_norm = (q.float() / torch.sqrt(q.float().square().sum(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    k_norm = (k.float() / torch.sqrt(k.float().square().sum(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    q_norm = (q_norm.float() * (key_dim**-0.5)).to(torch.bfloat16)
    g = -torch.exp(a_log.float()).reshape(1, 1, value_heads) * torch.nn.functional.softplus(
        a.float() + dt_bias.float().reshape(1, 1, value_heads)
    )
    beta = torch.sigmoid(b.float()).to(torch.bfloat16)

    state = state_in.clone().float()
    output = torch.empty(v.shape, dtype=torch.bfloat16)
    for token in range(q.shape[1]):
        for value_head in range(value_heads):
            key_head = value_head // value_heads_per_key
            state[:, value_head] *= torch.exp(g[:, token, value_head]).reshape(-1, 1, 1)
            prediction = torch.einsum(
                "bk,bkv->bv",
                k_norm[:, token, key_head].float(),
                state[:, value_head],
            ).to(torch.bfloat16)
            error = (
                (v[:, token, value_head].float() - prediction.float()).to(torch.bfloat16).float()
                * beta[:, token, value_head].float().reshape(-1, 1)
            ).to(torch.bfloat16)
            state[:, value_head] += torch.einsum(
                "bk,bv->bkv",
                k_norm[:, token, key_head].float(),
                error.float(),
            )
            output[:, token, value_head] = torch.einsum(
                "bk,bkv->bv",
                q_norm[:, token, key_head].float(),
                state[:, value_head],
            ).to(torch.bfloat16)
    return output, state


def random_case(
    seq_len,
    *,
    hidden=64,
    key_heads=4,
    value_heads=8,
    key_dim=16,
    value_dim=16,
    conv_size=4,
    seed=20260730,
):
    torch.manual_seed(seed + seq_len)
    batch = 1
    key_width = key_heads * key_dim
    value_width = value_heads * value_dim
    hidden_states = bf16_randn((batch, seq_len, hidden), 0.08)
    residual = bf16_randn((batch, seq_len, hidden), 0.04)
    weights = {
        "input_norm_w": (torch.rand((hidden,), dtype=torch.float32) * 0.5 + 0.75).to(torch.bfloat16),
        "w_q": bf16_randn((key_width, hidden), 0.08),
        "w_k": bf16_randn((key_width, hidden), 0.08),
        "w_v": bf16_randn((value_width, hidden), 0.08),
        "w_z": bf16_randn((value_width, hidden), 0.08),
        "w_a": bf16_randn((value_heads, hidden), 0.08),
        "w_b": bf16_randn((value_heads, hidden), 0.08),
        "w_q_conv": bf16_randn((key_width, conv_size), 0.2),
        "w_k_conv": bf16_randn((key_width, conv_size), 0.2),
        "w_v_conv": bf16_randn((value_width, conv_size), 0.2),
        "a_log": torch.randn((value_heads,), dtype=torch.float32) * 0.02,
        "dt_bias": bf16_randn((value_heads,), 0.02),
        "o_norm_w": (torch.rand((value_dim,), dtype=torch.float32) * 0.5 + 0.75).to(torch.bfloat16),
        "w_o": bf16_randn((hidden, value_width), 0.08),
    }
    temporal_state = torch.randn((batch, value_heads, key_dim, value_dim), dtype=torch.float32) * 0.03
    conv_state = torch.cat(
        (
            bf16_randn((batch, conv_size - 1, key_width), 0.03),
            bf16_randn((batch, conv_size - 1, key_width), 0.03),
            bf16_randn((batch, conv_size - 1, value_width), 0.03),
        ),
        dim=-1,
    )
    return hidden_states, residual, weights, temporal_state, conv_state


FIXTURE_GDN_CONFIG = {
    "hidden": 64,
    "key_heads": 4,
    "value_heads": 8,
    "key_dim": 16,
    "value_dim": 16,
    "conv_size": 4,
    "input_norm_eps": 2e-6,
    "o_norm_eps": 3e-6,
    # Long-recurrence outputs cross the output BF16 rounding midpoint.
    "fused_atol": 0.0625,
}

FIXTURE_NVFP4_GDN_CONFIG = {
    "hidden": 64,
    "key_heads": 4,
    "value_heads": 8,
    "key_dim": 16,
    "value_dim": 16,
    "conv_size": 4,
    "input_norm_eps": 2e-6,
    "o_norm_eps": 3e-6,
    "weight_format": "nvfp4",
    "fused_atol": 0.0625,
}

# GDN fields from Qwen3.5-9B config.json. rms_norm_eps applies to both the input
# RMSNorm and the gated output RMSNorm.
QWEN3_5_9B_GDN_CONFIG = {
    "hidden": 4096,
    "key_heads": 16,
    "value_heads": 32,
    "key_dim": 128,
    "value_dim": 128,
    "conv_size": 4,
    "input_norm_eps": 1e-6,
    "o_norm_eps": 1e-6,
    # A 4096-wide BF16 output projection has BF16 ULP up to 0.5 at the
    # magnitudes produced by the production-shape random recurrence.
    "fused_atol": 0.375,
    # FP32 recurrent-state MAC accumulation differs by at most one low-order
    # quantum at key_dim=128 in the observed production-shape matrix.
    "temporal_atol": 4e-3,
}

QWEN3_5_9B_NVFP4_GDN_CONFIG = {
    **QWEN3_5_9B_GDN_CONFIG,
    "weight_format": "nvfp4",
    "fused_atol": 0.75,
    "temporal_atol": 8e-3,
}

QWEN3_5_9B_MXFP8_GDN_CONFIG = {
    **QWEN3_5_9B_GDN_CONFIG,
    "weight_format": "mxfp8",
    "fused_atol": 0.75,
    "temporal_atol": 8e-3,
}

QWEN3_5_9B_MXFP4_GDN_CONFIG = {
    **QWEN3_5_9B_GDN_CONFIG,
    "weight_format": "mxfp4",
    "fused_atol": 0.75,
    "temporal_atol": 8e-3,
}

GDN_TEST_CONFIGS = {
    "fixture": FIXTURE_GDN_CONFIG,
    "fixture_nvfp4": FIXTURE_NVFP4_GDN_CONFIG,
    "fixture_nvfp4_head_tail": {**FIXTURE_NVFP4_GDN_CONFIG, "key_heads": 8, "value_heads": 24},
    "fixture_mxfp8": {**FIXTURE_GDN_CONFIG, "weight_format": "mxfp8"},
    "fixture_mxfp4": {**FIXTURE_GDN_CONFIG, "weight_format": "mxfp4"},
    # Qwen3.8 uses three value heads per key head rather than two.
    "fixture_nvfp4_ratio3": {**FIXTURE_NVFP4_GDN_CONFIG, "value_heads": 12},
    "qwen3_5_9b": QWEN3_5_9B_GDN_CONFIG,
    "qwen3_5_9b_nvfp4": QWEN3_5_9B_NVFP4_GDN_CONFIG,
    "qwen3_5_9b_mxfp8": QWEN3_5_9B_MXFP8_GDN_CONFIG,
    "qwen3_5_9b_mxfp4": QWEN3_5_9B_MXFP4_GDN_CONFIG,
    "qwen36_35b_nvfp4": {**QWEN3_5_9B_NVFP4_GDN_CONFIG, "hidden": 2048},
    "qwen38_27b_nvfp4": {**QWEN3_5_9B_NVFP4_GDN_CONFIG, "hidden": 5120, "value_heads": 48},
}


def reference(case, input_norm_eps=2e-6, o_norm_eps=3e-6, weights_override=None):
    hidden_states, residual, case_weights, temporal_state, conv_state = case
    weights = case_weights if weights_override is None else weights_override
    batch, seq_len, hidden = hidden_states.shape
    key_width = weights["w_q"].shape[0]
    value_width = weights["w_v"].shape[0]
    value_heads, key_dim, value_dim = temporal_state.shape[1:]
    key_heads = key_width // key_dim
    q_state, k_state, v_state = torch.split(conv_state, (key_width, key_width, value_width), dim=-1)

    residual_base = (hidden_states.float() + residual.float()).to(torch.bfloat16)
    x_scaled = (residual_base.float() * weights["input_norm_w"].float().reshape(1, 1, -1)).to(torch.bfloat16)
    x_sqrt = torch.sqrt(residual_base.float().square().mean(dim=-1, keepdim=True) + input_norm_eps)

    def project(name):
        return (linear_bf16(x_scaled, weights[name]).float() / x_sqrt).to(torch.bfloat16)

    q, q_state_out = causal_conv1d_silu_reference(project("w_q"), weights["w_q_conv"], q_state)
    k, k_state_out = causal_conv1d_silu_reference(project("w_k"), weights["w_k_conv"], k_state)
    v, v_state_out = causal_conv1d_silu_reference(project("w_v"), weights["w_v_conv"], v_state)
    q = q.reshape(batch, seq_len, key_heads, key_dim)
    k = k.reshape(batch, seq_len, key_heads, key_dim)
    v = v.reshape(batch, seq_len, value_heads, value_dim)
    z = project("w_z").reshape(batch, seq_len, value_heads, value_dim)
    a = project("w_a")
    b = project("w_b")

    core_output, temporal_out = gdn_reference(
        q,
        k,
        v,
        a,
        b,
        weights["a_log"],
        weights["dt_bias"],
        temporal_state,
    )
    variance = core_output.float().square().mean(dim=-1, keepdim=True)
    normed = core_output.float() * torch.rsqrt(variance + o_norm_eps)
    normed = (normed * weights["o_norm_w"].float().reshape(1, 1, 1, value_dim)).to(torch.bfloat16)
    normed = (normed.float() * torch.nn.functional.silu(z.float()).to(torch.bfloat16).float()).to(torch.bfloat16)
    attention_output = linear_bf16(normed.reshape(batch, seq_len, value_heads * value_dim), weights["w_o"])
    fused = (residual_base.float() + attention_output.float()).to(torch.bfloat16)
    return (
        fused,
        temporal_out,
        torch.cat((q_state_out, k_state_out, v_state_out), dim=-1),
    )


def run(case, *, core_num=4, chunk_size=2, input_norm_eps=2e-6, o_norm_eps=3e-6):
    return gdn(
        *case,
        core_num=core_num,
        chunk_size=chunk_size,
        input_norm_eps=input_norm_eps,
        o_norm_eps=o_norm_eps,
    )


@pytest.mark.parametrize("core_num", [4])
@pytest.mark.parametrize(
    "valid,conv_size,weight_format",
    [(6, 4, "bf16"), (1, 4, "nvfp4"), (255, 4, "nvfp4"), (256, 4, "nvfp4"), (33, 4, "nvfp4_ratio3"), (1, 258, "nvfp4")],
)
def test_runtime_tail_preserves_state_for_decode(valid, conv_size, weight_format, core_num):
    # NVFP4 packs scales in pairs of 16 channels; each core's ratio-three
    # output projection needs a value dimension of 32 to retain complete pairs.
    case = random_case(
        256,
        seed=20261028,
        value_heads=12 if weight_format == "nvfp4_ratio3" else 8,
        value_dim=32 if weight_format == "nvfp4_ratio3" else 16,
        conv_size=conv_size,
    )
    reference_weights = None
    if weight_format.startswith("nvfp4"):
        weights, reference_weights = random_nvfp4_weights(case[2], seed=20261029)
        case = (*case[:2], weights, *case[3:])
    # Nonzero padding and initial histories expose accidental state updates.
    real = (case[0][:, :valid], case[1][:, :valid], *case[2:])
    expected = reference(real, weights_override=reference_weights)
    baseline = run(real, core_num=core_num, chunk_size=32)
    actual = gdn(
        *case,
        core_num=core_num,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
        valid_tokens=torch.tensor([valid], dtype=torch.int32),
    )
    torch.testing.assert_close(actual[1][:, :valid].float(), expected[0].float(), rtol=5e-2, atol=0.0625)
    torch.testing.assert_close(actual[2], expected[1], rtol=5e-2, atol=3e-3)
    # State selection is a bit copy. Compare it exactly with the existing
    # unpadded path; NVFP4 projection rounding differs from dequantized Torch.
    torch.testing.assert_close(actual[3], baseline[3], rtol=0, atol=0)

    continuation = random_case(1, seed=20261030, value_heads=case[3].shape[1], value_dim=case[3].shape[-1])
    expected_decode = reference(
        (*continuation[:2], case[2], expected[1], expected[2]), weights_override=reference_weights
    )
    decoded = gdn(
        *continuation[:2],
        case[2],
        actual[2],
        actual[3],
        core_num=core_num,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
    )
    baseline_decode = run((*continuation[:2], case[2], baseline[2], baseline[3]), core_num=core_num, chunk_size=32)
    torch.testing.assert_close(decoded[1].float(), expected_decode[0].float(), rtol=5e-2, atol=0.0625)
    torch.testing.assert_close(decoded[2], expected_decode[1], rtol=5e-2, atol=3e-3)
    torch.testing.assert_close(decoded[3], baseline_decode[3], rtol=0, atol=0)


def test_runtime_tail_lengths_share_one_kernel_signature():
    case = random_case(256, seed=20261028)
    valid = torch.tensor([6], dtype=torch.int32)
    keys = set()
    for count in (6, 256, 7, 33, 255):
        valid.fill_(count)
        gdn(*case, core_num=4, valid_tokens=valid)
        keys.add(GDNKernel.gen_cache_key())
    gdn(*case, core_num=4)
    keys.add(GDNKernel.gen_cache_key())
    assert len(keys) == 1


@pytest.mark.parametrize(
    "gdn_config,seq_len,core_num,chunk_size,seed",
    [
        pytest.param("fixture_nvfp4_ratio3", 1, 4, 32, 20261022, id="nvfp4-ratio3-decode"),
        pytest.param("fixture_nvfp4_ratio3", 256, 4, 32, 20261022, id="nvfp4-ratio3-prefill256"),
        pytest.param("fixture_nvfp4_head_tail", 64, 4, 32, 20261022, id="nvfp4-head-tail-prefill"),
        pytest.param("fixture", 5, 4, 2, 20260730, id="5-4-2-20260730"),
        pytest.param("fixture", 33, 4, 32, 20260731, id="33-4-32-20260731"),
        pytest.param("fixture_mxfp8", 33, 4, 32, 20261022, id="mxfp8-prefill-tail"),
        pytest.param("fixture_mxfp4", 33, 4, 32, 20261022, id="mxfp4-prefill-tail"),
        pytest.param("fixture", 1, 4, 32, 20260731, id="1-4-32-20260731"),
        pytest.param("fixture", 256, 4, 32, 20260731, id="256-4-32-20260731"),
        *[
            pytest.param(model, rows, 4, 32, 20261029, id=f"{model}-{rows}")
            for model in ("qwen36_35b_nvfp4", "qwen38_27b_nvfp4")
            for rows in (1, 256)
        ],
        pytest.param("qwen3_5_9b", 1, 4, 32, 20260733, id="qwen3_5_9b-1-4-32-20260733"),
        pytest.param("qwen3_5_9b", 256, 4, 32, 20260733, id="qwen3_5_9b-256-4-32-20260733"),
        pytest.param(
            "qwen3_5_9b_nvfp4",
            256,
            4,
            32,
            20260735,
            id="qwen3_5_9b-nvfp4-256-4-32-20260735",
        ),
        pytest.param(
            "qwen3_5_9b_nvfp4",
            1,
            4,
            32,
            20260735,
            id="qwen3_5_9b-nvfp4-1-4-32-20260735",
        ),
        pytest.param(
            "qwen3_5_9b_mxfp8",
            1,
            4,
            32,
            20260737,
            id="qwen3_5_9b-mxfp8-1-4-32-20260737",
        ),
        pytest.param(
            "qwen3_5_9b_mxfp8",
            256,
            4,
            32,
            20260737,
            id="qwen3_5_9b-mxfp8-256-4-32-20260737",
        ),
        pytest.param(
            "qwen3_5_9b_mxfp4",
            1,
            4,
            32,
            20260738,
            id="qwen3_5_9b-mxfp4-1-4-32-20260738",
        ),
        pytest.param(
            "qwen3_5_9b_mxfp4",
            256,
            4,
            32,
            20260738,
            id="qwen3_5_9b-mxfp4-256-4-32-20260738",
        ),
    ],
)
def test_random_gdn_matches_full_pytorch_reference(gdn_config, seq_len, core_num, chunk_size, seed):
    config = GDN_TEST_CONFIGS[gdn_config]
    dimensions = {
        name: config[name] for name in ("hidden", "key_heads", "value_heads", "key_dim", "value_dim", "conv_size")
    }
    case = random_case(seq_len, seed=seed, **dimensions)
    reference_weights = None
    if config.get("weight_format") == "nvfp4":
        packed_weights, reference_weights = random_nvfp4_weights(
            case[2],
            seed=seed + 17,
        )
        case = (*case[:2], packed_weights, *case[3:])
    elif config.get("weight_format") == "mxfp8":
        packed_weights, reference_weights = random_mxfp8_weights(
            case[2],
            seed=seed + 17,
        )
        case = (*case[:2], packed_weights, *case[3:])
    elif config.get("weight_format") == "mxfp4":
        packed_weights, reference_weights = random_mxfp4_weights(
            case[2],
            seed=seed + 17,
        )
        case = (*case[:2], packed_weights, *case[3:])
    fused_ref, temporal_ref, conv_ref = reference(
        case,
        input_norm_eps=config["input_norm_eps"],
        o_norm_eps=config["o_norm_eps"],
        weights_override=reference_weights,
    )
    output, residual_out, temporal_out, conv_out = run(
        case,
        core_num=core_num,
        chunk_size=chunk_size,
        input_norm_eps=config["input_norm_eps"],
        o_norm_eps=config["o_norm_eps"],
    )

    assert torch.equal(output, torch.zeros_like(output))

    for name, actual, expected in (
        ("residual", residual_out, fused_ref),
        ("temporal state", temporal_out, temporal_ref),
        ("conv state", conv_out, conv_ref),
    ):
        similarity = torch.nn.functional.cosine_similarity(
            actual.float().reshape(-1), expected.float().reshape(-1), dim=-1
        )
        assert similarity.item() > 0.99, f"{name}: cosine={similarity.item():.6f}"


def test_random_gdn_supports_unequal_key_value_dimensions():
    case = random_case(
        3,
        key_heads=4,
        value_heads=8,
        key_dim=16,
        value_dim=8,
        seed=20260732,
    )
    fused_ref, temporal_ref, conv_ref = reference(case)
    output, residual_out, temporal_out, conv_out = run(case, core_num=4, chunk_size=2)

    assert torch.equal(output, torch.zeros_like(output))
    torch.testing.assert_close(residual_out.float(), fused_ref.float(), rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(temporal_out, temporal_ref, rtol=5e-2, atol=3e-3)
    torch.testing.assert_close(conv_out.float(), conv_ref.float(), rtol=0, atol=0)


@pytest.mark.parametrize("weight_format,core_num", [("nvfp4", 4)])
def test_decode_state_continuation_matches_prefill(weight_format, core_num):
    case = random_case(2, key_heads=8, value_heads=24)
    if weight_format == "nvfp4":
        packed_weights, _ = random_nvfp4_weights(case[2], seed=20261022)
        case = (*case[:2], packed_weights, *case[3:])
    full = run(case, core_num=core_num, chunk_size=2)
    hidden_states, residual, weights, temporal, conv = case
    residual_parts = []
    for token in range(2):
        token_case = (
            hidden_states[:, token : token + 1],
            residual[:, token : token + 1],
            weights,
            temporal,
            conv,
        )
        _, residual_token, temporal, conv = run(token_case, core_num=core_num)
        residual_parts.append(residual_token)

    torch.testing.assert_close(
        torch.cat(residual_parts, dim=1).float(),
        full[1].float(),
        rtol=5e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(temporal, full[2], rtol=5e-2, atol=3e-3)
    torch.testing.assert_close(conv, full[3], rtol=0, atol=0)


@pytest.mark.parametrize("reduction", [13, 16, 19])
def test_small_matmul_fp32_matches_reference(reduction):
    generator = torch.Generator().manual_seed(20261008)
    x = torch.randn((2, 7, reduction), generator=generator)
    y = torch.randn((2, reduction, 16), generator=generator)
    expected = x.double() @ y.double()
    actual = torch.zeros((2, 7, 16), dtype=torch.float32)

    ax.init(chip_num=1, core_num=1)
    small_product_kernel[1](x, y, actual)

    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize(
    "size,batch_count",
    [
        pytest.param(1, 1, id="single-token"),
        pytest.param(2, 2, id="batched-base-case"),
        pytest.param(17, 1, id="padding-and-recursion"),
        pytest.param(32, 2, id="batched-c32"),
        pytest.param(256, 2, id="maximum-chunk"),
    ],
)
def test_fp32_inverse_matches_correlated_triangular_reference(size, batch_count):
    positions = torch.arange(size)
    distance = positions[:, None] - positions[None, :]
    negative_lower = torch.tril(-0.96 * torch.exp(-0.012 * distance.float()), diagonal=-1)
    if batch_count > 1:
        negative_lower = torch.stack((negative_lower, negative_lower * 0.7))

    identity = torch.eye(size, dtype=torch.float64).expand(negative_lower.shape)
    expected = torch.linalg.solve_triangular(identity - negative_lower.double(), identity, upper=False)
    actual = torch.zeros_like(negative_lower)

    ax.init(chip_num=1, core_num=1)
    inverse_kernel[1](negative_lower, actual)

    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        (identity - negative_lower.double()) @ actual.double(),
        identity,
        rtol=0,
        atol=3e-6,
    )


@pytest.mark.parametrize("core_num", [4])
def test_gdn_default_core_selection_matches_explicit_core(monkeypatch, core_num):
    monkeypatch.setenv("SGLANG_AXERA_KERNEL_CORE_NUM", str(core_num))
    case = random_case(1, seed=20261020 + core_num)

    default_result = gdn(*case, chunk_size=2, input_norm_eps=2e-6, o_norm_eps=3e-6)
    explicit_result = gdn(
        *case,
        core_num=core_num,
        chunk_size=2,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
    )

    for default_value, explicit_value in zip(default_result, explicit_result):
        torch.testing.assert_close(default_value.float(), explicit_value.float(), rtol=0, atol=0)


@pytest.mark.parametrize("seq_len,core_num,chunk_size", [(512, 4, 32)])
def test_gdn_missing_residual_matches_explicit_zero_residual(seq_len, core_num, chunk_size):
    case = random_case(seq_len, seed=20261024)
    zero_residual_case = (case[0], torch.zeros_like(case[1]), *case[2:])

    missing_residual = gdn(
        case[0],
        None,
        *case[2:],
        core_num=core_num,
        chunk_size=chunk_size,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
    )
    explicit_zero_residual = gdn(
        *zero_residual_case,
        core_num=core_num,
        chunk_size=chunk_size,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
    )

    for missing_value, explicit_value in zip(missing_residual, explicit_zero_residual):
        torch.testing.assert_close(missing_value.float(), explicit_value.float(), rtol=0, atol=0)


def test_gdn_kernel_is_a_launchable_builder_function():
    assert not inspect.isclass(GDNKernel)
    assert GDNKernel.func.__name__ == "GDNKernel"
    assert GDNKernel.func.__module__ == gdn_module.__name__
    assert GDNKernel.const_inputs == []
    case = random_case(1)
    with pytest.raises(ValueError, match="one or four cores"):
        gdn(*case, core_num=2)


def test_gdn_learned_weights_have_no_constant_graph_ownership():
    module_source = inspect.getsource(gdn_module)
    assert 'const_inputs=["y"]' not in module_source
    assert "constant_tensor(" not in module_source
    assert "ax.dsl.from_numpy(w_" not in module_source

    assert GDNKernel.const_inputs == []
    conv_source = inspect.getsource(gdn_module.stateful_causal_depthwise_conv1d_silu)
    assert "ax.dsl.conv(" in conv_source
    assert "for window_offset" not in conv_source


@pytest.mark.parametrize("aliased", [False, True])
def test_nvfp4_decode_full_rows_preserve_ic_wave_rounding(aliased):
    generator = torch.Generator().manual_seed(20261103)
    hidden, outputs, ic_tile = 2048, 2048, 1536
    codes = torch.randint(0, 16, (outputs, hidden), dtype=torch.uint8, generator=generator)
    weight = pack_fp4_codes(codes)
    scales = torch.full((outputs, hidden // 16), 1 / 64, dtype=torch.float8_e4m3fn)
    x = (torch.randn(1, hidden, generator=generator) * 0.125).bfloat16()
    first = torch.empty(1, outputs, dtype=torch.bfloat16)
    second = torch.empty_like(first)
    decoded = FP4_LUT[codes.long()] / 64
    expected = torch.zeros_like(first)
    for start in range(0, hidden, ic_tile):
        stop = min(start + ic_tile, hidden)
        partial = torch.nn.functional.linear(x[:, start:stop].float(), decoded[:, start:stop])
        expected = (expected.float() + partial).bfloat16()

    ax.init(chip_num=1, core_num=4)
    nvfp4_ordered_projection_kernel[4, 4](x, weight, scales, torch.ones(1), first, second, aliased, ic_tile, 512)
    for result in (first, second):
        torch.testing.assert_close(result, expected, rtol=0.02, atol=0.003)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("hidden,rows,inputs", [(512, 1, 4096), (2048, 1, 4096), (5120, 16, 6144)])
def test_nvfp4_decode_output_projection_preserves_scale_waves(aliased, hidden, rows, inputs, tmp_path):
    generator = torch.Generator().manual_seed(20261104)
    layers, cores = 2, 4
    codes = torch.randint(0, 16, (layers, hidden, inputs), dtype=torch.uint8, generator=generator)
    weights = pack_fp4_codes(codes)
    scales = (torch.randint(1, 4, (layers, hidden, inputs // 16), generator=generator).float() / 128).to(
        torch.float8_e4m3fn
    )
    decoded = FP4_LUT[codes.long()] * scales.float().repeat_interleave(16, dim=-1)
    x = (torch.randn(rows, inputs, generator=generator) * 0.125).bfloat16()
    residual = (torch.randn(1, rows, hidden, generator=generator) * 0.1).bfloat16()
    output = torch.empty_like(residual)
    mode = ax.RunMode(os.getenv("RUN_MODE", "PYRUN"))

    @ax.builder(run_mode=mode, enable_cache=True, cache_path=tmp_path / "cache", dump_path=tmp_path / "artifacts")
    def kernel(x, residual, weight, scale, scale2, layer, output):
        core = ax.program_id(0)
        owned = slice(core * (hidden // cores), (core + 1) * (hidden // cores))
        input_slice = slice(core * (inputs // cores), (core + 1) * (inputs // cores))
        index = ax.dsl.load(layer)
        banks = {weight.name: (weight, (index, hidden, 0)), scale.name: (scale, (index, hidden, 0))}
        shared = ax.dsl.create_tensor(residual.shape, residual.dtype, ax.MemoryType.SHARED_OCM)
        ax.dsl.load(residual[:, :, owned], r=shared[:, :, owned])
        ax.parallel.pub(shared)
        ax.parallel.sub(shared)
        destination = shared if aliased else ax.dsl.create_tensor(shared.shape, shared.dtype, ax.MemoryType.SHARED_OCM)
        gdn_module.store_row_parallel_o_proj_residual_nmc(
            ax.dsl.load(x[:, input_slice]),
            weight[:1].reshape(hidden, inputs),
            scale[:1].reshape(hidden, inputs // 16),
            scale2,
            shared,
            output,
            destination,
            input_slice,
            core,
            cores,
            name_prefix="test_decode_o",
            ic_tile=512,
            banks=banks,
        )
        return output

    ax.init(chip_num=1, core_num=cores)
    for layer in (0, 1, 0):
        expected = residual.clone()
        for owner in range(cores):
            oc = slice(owner * (hidden // cores), (owner + 1) * (hidden // cores))
            for round_ in range(cores):
                contributor = (owner - round_) % cores
                for wave in range(inputs // cores // 512):
                    start = contributor * (inputs // cores) + wave * 512
                    partial = (
                        torch.nn.functional.linear(
                            x[:, start : start + 512].float(), decoded[layer, oc, start : start + 512]
                        )
                        .mul(0.5)
                        .bfloat16()
                    )
                    expected[:, :, oc] = (expected[:, :, oc].float() + partial).bfloat16()
        kernel[cores, cores](
            x, residual, weights, scales, torch.full((1,), 0.5), torch.tensor([layer], dtype=torch.int32), output
        )
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.004)


def test_wide_nvfp4_projection_records_balanced_ic_waves(monkeypatch):
    generator = torch.Generator().manual_seed(771)
    q_codes = torch.randint(1, 16, (32, 64), dtype=torch.uint8, generator=generator)
    k_codes = torch.randint(1, 16, q_codes.shape, dtype=torch.uint8, generator=generator)
    x = torch.randn((2, 64), dtype=torch.bfloat16, generator=generator)
    q_weight = pack_fp4_codes(q_codes)
    k_weight = pack_fp4_codes(k_codes)
    z_codes = torch.randint(1, 16, q_codes.shape, dtype=torch.uint8, generator=generator)
    z_weight = pack_fp4_codes(z_codes)
    v_weight = pack_fp4_codes(torch.randint(1, 16, q_codes.shape, dtype=torch.uint8, generator=generator))
    q_scale, k_scale, v_scale, z_scale = (
        (torch.rand((32, 4), generator=generator) * 0.02 + 0.01).to(torch.float8_e4m3fn) for _ in range(4)
    )
    q_scale2, k_scale2, v_scale2, z_scale2 = (torch.rand((1,), generator=generator) * 0.5 + 0.25 for _ in range(4))
    q_output = torch.empty((2, 32), dtype=torch.bfloat16)
    k_output = torch.empty_like(q_output)
    z_output = torch.empty_like(q_output)
    v_output = torch.empty_like(q_output)

    monkeypatch.setattr(gdn_module, "GDN_NVFP4_PROJECTION_TILE_WIDTH", 16)
    ax.init(chip_num=1, core_num=1)
    wide_projection_kernel[1](
        x,
        q_weight,
        q_scale,
        q_scale2,
        k_weight,
        k_scale,
        k_scale2,
        v_weight,
        v_scale,
        v_scale2,
        z_weight,
        z_scale,
        z_scale2,
        q_output,
        k_output,
        v_output,
        z_output,
    )

    core_ctx = ctx.sub_ctxs[0][0]
    matmuls = [(position, op) for position, op in enumerate(core_ctx.ops) if op.type == "AxMatMul"]
    assert len(matmuls) == 16

    # Verify source ranges and producer ordering independently of temporary names.
    tensors = {**ctx.sub_ctxs[0][-1].tensors, **core_ctx.tensors}
    loads = {
        op.outputs_spec["r"]: (position, tensors[op.inputs_spec["x"]])
        for position, op in enumerate(core_ctx.ops)
        if op.type == "AxSlice" and set(op.inputs_spec) == {"x"}
    }
    for start, names in ((0, ("q_weight", "k_weight")), (8, ("v_weight", "z_weight"))):
        first_matmul_position = matmuls[start][0]
        for wave in range(2):
            for tile in range(2):
                for projection, name in enumerate(names):
                    _, op = matmuls[start + wave * 4 + tile * 2 + projection]
                    load_position, source = loads[op.inputs_spec["y"]]
                    expected = tensors[name][tile * 16 : (tile + 1) * 16, wave * 32 : (wave + 1) * 32]
                    assert source.name == expected.name
                    assert source.shape == expected.shape
                    assert source.offset == expected.offset
                    assert source.bit_strides == expected.bit_strides
                    assert load_position < first_matmul_position
    for output in (q_output, k_output, v_output, z_output):
        assert torch.isfinite(output).all() and torch.count_nonzero(output)


def test_bf16_preprojection_prefetches_ic_waves():
    x = torch.randn((1, 2, 64), dtype=torch.bfloat16)
    x_rstd = torch.ones((1, 2, 1), dtype=torch.float32)
    weight = torch.randn((512, 64), dtype=torch.bfloat16)
    output = torch.empty((1, 2, 512), dtype=torch.bfloat16)

    ax.init(chip_num=1, core_num=1)
    bf16_preprojection_wave_kernel[1](x, x_rstd, weight, output)

    core_ctx = ctx.sub_ctxs[0][0]
    matmuls = [(position, op) for position, op in enumerate(core_ctx.ops) if op.type == "AxMatMul"]
    assert len(matmuls) == 4
    assert [op.attrs["output_add_assign"] for _, op in matmuls] == [False, True, False, True]

    load_positions = [
        position
        for position, op in enumerate(core_ctx.ops)
        if op.type == "AxSlice" and op.name.startswith("ld:") and op.name not in ("ld:x_ocm_0", "ld:x_rstd_ocm_0")
    ]
    assert len(load_positions) == 4
    matmul_positions = [position for position, _ in matmuls]
    # Each tile's second wave is staged before the first wave is consumed, and
    # the next tile's first wave is staged before the current tile accumulates.
    assert (
        load_positions[0]
        < load_positions[1]
        < matmul_positions[0]
        < load_positions[2]
        < matmul_positions[1]
        < load_positions[3]
        < matmul_positions[2]
        < matmul_positions[3]
    )


def test_mxfp8_projection_records_32_channel_scale_waves():
    x = torch.randn((1, 2, 64), dtype=torch.bfloat16)
    x_rstd = torch.ones((1, 2, 1), dtype=torch.float32)
    weight = torch.randn((32, 64), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.full((32, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    output = torch.empty((1, 2, 32), dtype=torch.bfloat16)

    ax.init(chip_num=1, core_num=1)
    mxfp8_projection_wave_kernel[1](x, x_rstd, weight, scale, output)

    core_ctx = ctx.sub_ctxs[0][0]
    matmuls = [op for op in core_ctx.ops if op.type == "AxMatMul"]
    assert len(matmuls) == 2
    assert [op.attrs["output_add_assign"] for op in matmuls] == [False, True]
    assert [core_ctx.tensors[op.inputs_spec["y_scale"]].shape for op in matmuls] == [
        (32, 1),
        (32, 1),
    ]


def test_mxfp4_projection_records_32_channel_scale_waves():
    x = torch.randn((1, 2, 64), dtype=torch.bfloat16)
    x_rstd = torch.ones((1, 2, 1), dtype=torch.float32)
    codes = np.random.default_rng(20260739).integers(0, 16, size=(32, 64), dtype=np.uint8)
    weight = pack_fp4_codes(codes)
    scale = torch.full((32, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    output = torch.empty((1, 2, 32), dtype=torch.bfloat16)

    ax.init(chip_num=1, core_num=1)
    mxfp4_projection_wave_kernel[1](x, x_rstd, weight, scale, output)

    core_ctx = ctx.sub_ctxs[0][0]
    matmuls = [op for op in core_ctx.ops if op.type == "AxMatMul"]
    assert len(matmuls) == 2
    assert [op.attrs["output_add_assign"] for op in matmuls] == [False, True]
    assert [core_ctx.tensors[op.inputs_spec["y_scale"]].shape for op in matmuls] == [
        (32, 1),
        (32, 1),
    ]


def test_gdn_runtime_conv_weight_uses_dsl_layout_without_embedding_values():
    checkpoint = torch.arange(2 * 4, dtype=torch.bfloat16).reshape(2, 4)

    runtime = gdn_module.runtime_conv_weight(checkpoint)

    assert isinstance(runtime, torch.Tensor)
    assert runtime.shape == (2, 1, 4)
    assert runtime.dtype == torch.bfloat16
    torch.testing.assert_close(runtime.float(), checkpoint.unsqueeze(1).float())
    assert runtime.data_ptr() == checkpoint.data_ptr()


def test_gdn_runtime_pre_projection_weight_is_core_contiguous():
    weights = random_case(1)[2]
    weights = gdn_module.runtime_weights(weights)

    runtime = gdn_module.runtime_pre_projection_weight(weights, core_num=4)

    assert runtime.shape == (2 * 64 + 2 * 128 + 2 * 8, 64)
    core_width = 2 * 16 + 2 * 32 + 2 * 2
    torch.testing.assert_close(
        runtime[:core_width].float(),
        torch.cat(
            (
                weights["w_q"][:16],
                weights["w_k"][:16],
                weights["w_v"][:32],
                weights["w_z"][:32],
                weights["w_a"][:2],
                weights["w_b"][:2],
            )
        ).float(),
    )
    torch.testing.assert_close(
        runtime[core_width : 2 * core_width].float(),
        torch.cat(
            (
                weights["w_q"][16:32],
                weights["w_k"][16:32],
                weights["w_v"][32:64],
                weights["w_z"][32:64],
                weights["w_a"][2:4],
                weights["w_b"][2:4],
            )
        ).float(),
    )


def test_gdn_runtime_weight_conversion_preserves_signed_int4_storage():
    storage = np.arange(-8, 8, dtype=np.int8).astype(ml_dtypes.int4).reshape(2, 8)

    runtime = gdn_module.as_runtime_tensor(storage)

    assert runtime.dtype == torch.int8
    assert runtime.shape == storage.shape
    assert torch.equal(runtime, torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8))


def test_gdn_runtime_fc_scale_normalization_supports_scalar_channel_and_group_layouts():
    weight = torch.ones((64, 64), dtype=torch.bfloat16)

    scalar = gdn_module.normalized_fc_scale(0.25, weight, "w_q_scale")
    per_channel = gdn_module.normalized_fc_scale(np.arange(64, dtype=np.float32), weight, "w_q_scale")
    group16 = gdn_module.normalized_fc_scale(np.ones((64, 4), dtype=np.float32), weight, "w_q_scale")
    already_batched = gdn_module.normalized_fc_scale(
        torch.ones((1, 64, 4), dtype=torch.float32),
        weight,
        "w_q_scale",
    )

    assert scalar.shape == per_channel.shape == (1, 64, 1)
    assert group16.shape == already_batched.shape == (64, 4)
    assert torch.equal(scalar, torch.full_like(scalar, 0.25))
    assert torch.equal(per_channel.flatten(), torch.arange(64, dtype=torch.float32))
    assert torch.equal(gdn_module.normalized_scale2(2.0, "w_q_scale2"), torch.tensor([2.0]))
    with pytest.raises(ValueError, match="w_q_scale shape"):
        gdn_module.normalized_fc_scale(np.ones((65, 1), dtype=np.float32), weight, "w_q_scale")
    with pytest.raises(ValueError, match="w_q_scale2 must contain"):
        gdn_module.normalized_scale2(np.ones((2,), dtype=np.float32), "w_q_scale2")


@pytest.mark.parametrize("with_residual", [True, False])
def test_gdn_launch_passes_all_weights_and_scales_as_runtime_tensors(monkeypatch, with_residual):
    captured = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["args"] = args

            return launch

    monkeypatch.setattr(gdn_module.ax, "init", lambda **kwargs: captured.setdefault("init", kwargs))
    monkeypatch.setattr(gdn_module, "GDNKernel", FakeKernel())

    case = random_case(1, seed=20261015)
    hidden_states, residual, weights, temporal, conv = case
    weights = dict(weights)
    weights["w_a"] = weights["w_a"].float().numpy()
    weights["w_q_scale"] = np.full((weights["w_q"].shape[0], 1), 0.5, dtype=np.float32)
    weights["w_q_scale2"] = np.array([2.0], dtype=np.float32)

    gdn(
        hidden_states,
        residual if with_residual else None,
        weights,
        temporal,
        conv,
        core_num=4,
        chunk_size=1,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
    )

    args = captured["args"]
    assert captured["init"] == {"chip_num": 1}
    assert captured["grid"] == (4, 4)
    assert len(args) == 41
    assert args[1] is (residual if with_residual else None)
    # Scaled projections consume separate runtime weights, not a packed BF16 matrix.
    assert args[9] is None
    expected_weights = gdn_module.runtime_weights(weights)
    expected_weights["w_pre"] = gdn_module.runtime_pre_projection_weight(expected_weights, core_num=4)
    for index, name in enumerate(
        (
            "input_norm_w",
            "w_q",
            "w_k",
            "w_v",
            "w_z",
            "w_a",
            "w_b",
            "w_pre",
            "w_q_conv",
            "w_k_conv",
            "w_v_conv",
            "a_log",
            "dt_bias",
            "o_norm_w",
            "w_o",
        ),
        start=2,
    ):
        expected = expected_weights[name]
        if expected is None:
            assert args[index] is None, (index, name)
        else:
            assert isinstance(args[index], torch.Tensor), (index, name)
            assert torch.equal(args[index], expected), (index, name)
    assert args[13].dtype == torch.float32
    assert args[27].dtype == torch.float32
    assert torch.equal(args[27], torch.full_like(args[27], 0.5))
    assert args[27].shape == (1, weights["w_q"].shape[0], 1)
    assert torch.equal(args[28], torch.full((1,), 2.0, dtype=torch.float32))
    for index in range(29, 41):
        assert args[index] is None, index


def test_gdn_runtime_weight_change_is_visible_in_output():
    case = random_case(1, seed=20261016)
    baseline = run(case, core_num=4, chunk_size=1)
    hidden_states, residual, weights, temporal, conv = case
    changed_weights = dict(weights)
    changed_weights["w_q"] = weights["w_q"] * 2.0 + 0.125

    changed = gdn(
        hidden_states,
        residual,
        changed_weights,
        temporal,
        conv,
        core_num=4,
        chunk_size=1,
        input_norm_eps=2e-6,
        o_norm_eps=3e-6,
    )

    assert not torch.equal(baseline[1], changed[1])


def test_public_wrapper_rejects_invalid_contracts():
    case = random_case(1)
    hidden_states, residual, weights, temporal, conv = case
    with pytest.raises(ValueError, match="one dense request"):
        gdn(hidden_states.expand(2, -1, -1), residual, weights, temporal, conv)
    with pytest.raises(ValueError, match="temporal state"):
        gdn(hidden_states, residual, weights, temporal.to(torch.bfloat16), conv)
    bad_head_dim = torch.zeros((*temporal.shape[:-1], 12), dtype=torch.float32)
    with pytest.raises(ValueError, match="power-of-two"):
        gdn(hidden_states, residual, weights, bad_head_dim, conv)
    with pytest.raises(ValueError, match="one or four cores"):
        gdn(hidden_states, residual, weights, temporal, conv, core_num=2)
    with pytest.raises(ValueError, match="epsilons"):
        gdn(
            hidden_states,
            residual,
            weights,
            temporal,
            conv,
            input_norm_eps=0,
        )
