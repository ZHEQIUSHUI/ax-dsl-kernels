"""Sequential per-layer DeepSeek-V4 first-four-block DSL port.

This module mirrors the block order in the checkpoint ``inference/model.py``:

``HCA -> RMSNorm -> Attention -> HCA -> HCA -> RMSNorm -> MoE -> HCA``.

The four attention policies are fixed to ``(SWA, SWA, CSA-4+Indexer,
CSA-128)``.  Persistent decode continuation and quantized checkpoint storage
remain outside this compact prefill correctness boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import axnn.dsl as ax
from axnn.dsl import parallel as axd

from sgl_kernel_axera.attention.csa import (
    csa_compress_value,
    csa_indexer_value,
)
from sgl_kernel_axera.attention.hca import hca_compress_value
from sgl_kernel_axera.attention.mhc import post_hc_value, pre_hc_value
from sgl_kernel_axera.moe.m_dsv4 import (
    hash_moe_partial_value,
    hash_moe_value,
    score_moe_partial_value,
    selected_moe_multicore_value,
    score_moe_value,
    selected_moe_value,
)
from sgl_kernel_axera.attention.parallel_utils import validate_core_num
from sgl_kernel_axera.attention.attn_utils.common import rmsnorm as common_rmsnorm
from sgl_kernel_axera.attention.attn_utils.common import tail_rope as common_tail_rope
from sgl_kernel_axera.attention.swa import (
    swa_attention_value,
    window_indices,
)

MAX_LAYER_COUNT = 4


def run_mode() -> ax.RunMode:
    """Return the selected DSL backend; CRUN is the model-port default."""
    return ax.RunMode(os.environ.get("RUN_MODE", "CRUN"))


def kernel_weight(value: torch.Tensor) -> torch.Tensor:
    """Convert FP32 MatMul weights to the AX8860 CRUN representation."""
    if run_mode() == ax.RunMode.CRUN and value.dtype == torch.float32:
        return value.to(torch.bfloat16)
    return value


def constant(value: float):
    """Create a scalar FP32 DSL constant."""
    return ax.dsl.from_numpy(np.asarray([value], dtype=np.float32))


def tiled_linear_value(x, weight, *, output_dtype, output_tile: int = 256):
    """Record a linear projection while loading bounded DDR weight slices."""
    input_width = x.shape[-1]
    output_width = weight.shape[0]
    rows = 1
    for size in x.shape[:-1]:
        rows *= size
    flat = x.reshape(rows, input_width)
    parts = []
    for start in range(0, output_width, output_tile):
        stop = min(start + output_tile, output_width)
        weight_part = weight[start:stop, :]
        if weight_part.memory_type == ax.MemoryType.DDR:
            weight_part = ax.dsl.load(weight_part)
        parts.append(
            ax.dsl.matmul(
                flat,
                weight_part,
                transposed=True,
                output_dtype=output_dtype,
            )
        )
    output = parts[0] if len(parts) == 1 else ax.dsl.concat(*parts, dim=-1)
    return output.reshape(*x.shape[:-1], output_width)


def rmsnorm_value(x, *, eps: float, output_dtype, weight=None):
    """Record the shared FP32 RMSNorm boundary used by attention and FFN."""
    return common_rmsnorm(x, weight=weight, eps=eps, output_dtype=output_dtype)


def rope_value(x, cos, sin, *, rope_dim: int, inverse: bool = False):
    """Record compact-table interleaved RoPE through the shared tail helper."""
    cos_table = cos.reshape(1, cos.shape[0], cos.shape[1])
    sin_table = sin.reshape(1, sin.shape[0], sin.shape[1])
    if inverse:
        sin_table = ax.dsl.mul(sin_table, constant(-1.0), output_dtype=ax.fp32)
    return common_tail_rope(x, cos_table, sin_table, rope_dim)


def rope_token_value(x, cos, sin, *, rope_dim: int, inverse: bool = False):
    """Record token-major interleaved RoPE through the shared compact helper."""
    return rope_value(x, cos, sin, rope_dim=rope_dim, inverse=inverse)


def rope_head_value(x, cos, sin, *, rope_dim: int, inverse: bool = False):
    """Record head-major interleaved RoPE through the shared compact helper."""
    return rope_value(x, cos, sin, rope_dim=rope_dim, inverse=inverse)


def hadamard_value(x):
    """Record a normalized Walsh-Hadamard transform on the last axis."""
    width = x.shape[-1]
    prefix = x.shape[:-1]
    value = x
    size = 1
    while size < width:
        value = value.reshape(*prefix, width // (2 * size), 2, size)
        left = value[..., :1, :]
        right = value[..., 1:2, :]
        value = ax.dsl.concat(
            ax.dsl.add(left, right, output_dtype=ax.fp32),
            ax.dsl.sub(left, right, output_dtype=ax.fp32),
            dim=-2,
        ).reshape(*prefix, width)
        size *= 2
    return ax.dsl.mul(value, constant(width**-0.5), output_dtype=x.dtype)


def attention_input_value(x, wq_a, wq_b, wkv, q_norm_weight, kv_norm_weight, cos, sin, config):
    """Record Q LoRA, Q/KV normalization, projection, and RoPE."""
    batch, seq_len, _ = x.shape
    qr = tiled_linear_value(x, wq_a, output_dtype=x.dtype)
    qr = rmsnorm_value(
        qr,
        eps=config.norm_eps,
        output_dtype=x.dtype,
        weight=q_norm_weight,
    )
    query = tiled_linear_value(qr, wq_b, output_dtype=x.dtype)
    query = query.reshape(batch, seq_len, config.head_num, config.head_dim)
    query = rmsnorm_value(query, eps=config.norm_eps, output_dtype=x.dtype)
    query = rope_head_value(query, cos, sin, rope_dim=config.rope_head_dim)
    kv = tiled_linear_value(x, wkv, output_dtype=x.dtype)
    kv = rmsnorm_value(
        kv,
        eps=config.norm_eps,
        output_dtype=x.dtype,
        weight=kv_norm_weight,
    )
    kv = rope_token_value(kv, cos, sin, rope_dim=config.rope_head_dim)
    return qr, query, kv


def attention_output_value(attention, cos, sin, wo_a, wo_b, config):
    """Record inverse RoPE and grouped low-rank output projection."""
    batch, seq_len, heads, head_dim = attention.shape
    group_width = heads // config.output_groups * head_dim
    value = rope_head_value(
        attention,
        cos,
        sin,
        rope_dim=config.rope_head_dim,
        inverse=True,
    ).reshape(batch, seq_len, config.output_groups, group_width)
    projected_groups = []
    for group in range(config.output_groups):
        projected_groups.append(
            tiled_linear_value(
                value[:, :, group : group + 1, :].reshape(batch * seq_len, group_width),
                wo_a[group : group + 1, :, :].reshape(config.output_lora_rank, group_width),
                output_dtype=attention.dtype,
            ).reshape(batch, seq_len, config.output_lora_rank)
        )
    projected = projected_groups[0]
    for group_value in projected_groups[1:]:
        projected = ax.dsl.concat(projected, group_value, dim=-1)
    return tiled_linear_value(projected, wo_b, output_dtype=attention.dtype)


def query_projection_value(x, weight, *, output_dtype):
    """Record one projection used by the CSA Indexer."""
    return tiled_linear_value(x, weight, output_dtype=output_dtype)


def gather_candidates_value(cache, indices):
    """Gather `[B,S,K,D]` candidates from `[B,T,D]` using safe indices."""
    batch, seq_len, candidate_count, head_dim = indices.shape
    cache_value = ax.dsl.expand(
        cache.reshape(batch, 1, cache.shape[1], cache.shape[2]),
        shape=(batch, seq_len, cache.shape[1], cache.shape[2]),
    )
    return ax.dsl.gather(cache_value, indices, dim=2).reshape(batch, seq_len, candidate_count, head_dim)


def gather_compressed(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Eager gather used by the mathematical reference."""
    safe = indices.clamp_min(0).long()
    expanded = cache[:, None, :, :].expand(-1, indices.shape[1], -1, -1)
    gathered = torch.gather(
        expanded,
        2,
        safe[..., None].expand(*safe.shape, cache.shape[-1]),
    )
    return torch.where((indices >= 0)[..., None], gathered, torch.zeros_like(gathered))


def rope_frequencies(
    seq_len: int,
    rope_dim: int,
    theta: float = 10000.0,
    *,
    positions=None,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
):
    """Create model.py-compatible cosine/sine values consumed by DSL/eager RoPE.

    Compressed attention layers use the checkpoint's YaRN interpolation.  The
    uncompressed SWA path deliberately keeps ``original_seq_len=0`` to match
    model.py's base-frequency branch.
    """
    if seq_len <= 0 or rope_dim <= 0 or rope_dim % 2:
        raise ValueError("seq_len and rope_dim must be positive even dimensions")
    position_values = torch.arange(seq_len, dtype=torch.float32) if positions is None else positions.float()
    inverse = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    if original_seq_len > 0:

        def correction_dim(num_rotations: float) -> float:
            return rope_dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(rope_dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        smooth = 1.0 - ramp
        inverse = inverse / factor * (1.0 - smooth) + inverse * smooth
    angles = torch.outer(position_values, inverse)
    return angles.cos(), angles.sin()


@dataclass(frozen=True)
class DeepSeekV4FourLayerConfig:
    hidden_dim: int
    hc_mult: int
    head_num: int
    head_dim: int
    q_lora_rank: int
    output_groups: int
    output_lora_rank: int
    rope_head_dim: int
    window_size: int
    moe_inter_dim: int
    routed_experts: int
    activated_experts: int
    index_head_num: int
    index_head_dim: int
    index_topk: int
    compress_ratios: tuple[int, ...] = (0, 0, 4, 128)
    hash_layer_count: int = 3
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-6
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    preselected_moe: bool = False
    fp8_moe: bool = False
    # YaRN parameters are needed for real checkpoint compressed-layer RoPE.
    # Defaults preserve the compact mathematical fixtures' base frequencies.
    original_seq_len: int = 0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1

    def __post_init__(self):
        expected_ratios = (0, 0, 4, 128)
        if (
            not 1 <= len(self.compress_ratios) <= MAX_LAYER_COUNT
            or self.compress_ratios != expected_ratios[: len(self.compress_ratios)]
        ):
            raise ValueError("layers 1..4 require the matching prefix of compress_ratios=(0,0,4,128)")
        if self.hidden_dim <= 0 or self.head_num <= 0 or self.head_dim <= 0:
            raise ValueError("hidden, head count, and head dimensions must be positive")
        if self.head_num % self.output_groups:
            raise ValueError("head_num must be divisible by output_groups")
        if self.rope_head_dim <= 0 or self.rope_head_dim > self.head_dim or self.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be positive, even, and no larger than head_dim")
        if self.index_head_dim & (self.index_head_dim - 1):
            raise ValueError("index_head_dim must be a power of two for the Hadamard transform")
        if self.hash_layer_count != 3:
            raise ValueError("the checkpoint routes exactly the first three layers by token hash")
        if self.fp8_moe and not self.preselected_moe:
            raise ValueError("FP8 MoE weights currently require preselected_moe")

    @property
    def layer_count(self) -> int:
        return len(self.compress_ratios)


@dataclass
class DeepSeekV4AttentionWeights:
    attn_sink: torch.Tensor
    wq_a: torch.Tensor
    wq_b: torch.Tensor
    wkv: torch.Tensor
    wo_a: torch.Tensor
    wo_b: torch.Tensor
    q_norm_weight: Optional[torch.Tensor] = None
    kv_norm_weight: Optional[torch.Tensor] = None
    compressor_wkv: Optional[torch.Tensor] = None
    compressor_wgate: Optional[torch.Tensor] = None
    compressor_ape: Optional[torch.Tensor] = None
    compressor_norm_weight: Optional[torch.Tensor] = None
    index_wq_b: Optional[torch.Tensor] = None
    index_gate_weight: Optional[torch.Tensor] = None
    index_compressor_wkv: Optional[torch.Tensor] = None
    index_compressor_wgate: Optional[torch.Tensor] = None
    index_compressor_ape: Optional[torch.Tensor] = None
    index_compressor_norm_weight: Optional[torch.Tensor] = None


@dataclass
class DeepSeekV4MoEWeights:
    gate_weight: torch.Tensor
    expert_w1: torch.Tensor
    expert_w2: torch.Tensor
    expert_w3: torch.Tensor
    shared_w1: torch.Tensor
    shared_w2: torch.Tensor
    shared_w3: torch.Tensor
    hash_table: Optional[torch.Tensor] = None
    gate_bias: Optional[torch.Tensor] = None
    selected_indices: Optional[torch.Tensor] = None


@dataclass
class DeepSeekV4BlockWeights:
    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    attn_norm_weight: Optional[torch.Tensor]
    ffn_norm_weight: Optional[torch.Tensor]
    attention: DeepSeekV4AttentionWeights
    moe: DeepSeekV4MoEWeights


@dataclass
class DeepSeekV4PackedWeights:
    """Layer-stacked tensors sliced before each independent DSL builder call."""

    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    attn_sink: torch.Tensor
    wq_a: torch.Tensor
    wq_b: torch.Tensor
    wkv: torch.Tensor
    wo_a: torch.Tensor
    wo_b: torch.Tensor
    q_norm_weight: torch.Tensor
    kv_norm_weight: torch.Tensor
    attn_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor
    moe_gate: torch.Tensor
    expert_w1: torch.Tensor
    expert_w2: torch.Tensor
    expert_w3: torch.Tensor
    shared_w1: torch.Tensor
    shared_w2: torch.Tensor
    shared_w3: torch.Tensor
    hash_table: torch.Tensor
    gate_bias: torch.Tensor
    csa4_wkv: torch.Tensor
    csa4_wgate: torch.Tensor
    csa4_ape: torch.Tensor
    csa4_norm: torch.Tensor
    index_wq_b: torch.Tensor
    index_gate: torch.Tensor
    index_wkv: torch.Tensor
    index_wgate: torch.Tensor
    index_ape: torch.Tensor
    index_norm: torch.Tensor
    csa128_wkv: torch.Tensor
    csa128_wgate: torch.Tensor
    csa128_ape: torch.Tensor
    csa128_norm: torch.Tensor


@dataclass
class DeepSeekV4PortOutput:
    """Visible Block boundaries from the selected DSL schedule."""

    output: torch.Tensor
    layer_outputs: list[torch.Tensor]


def stack_weights(values: Sequence[torch.Tensor], *, matmul: bool) -> torch.Tensor:
    """Stack one tensor per layer, converting MatMul weights when required."""
    converted = [kernel_weight(value) if matmul else value for value in values]
    return torch.stack(converted, dim=0).contiguous()


def norm_weight(value: Optional[torch.Tensor], width: int) -> torch.Tensor:
    """Return a learned RMSNorm weight or the compact-case identity weight."""
    if value is None:
        return torch.ones((width,), dtype=torch.float32)
    return value


def pack_four_layer_weights(
    config: DeepSeekV4FourLayerConfig,
    weights: Sequence[DeepSeekV4BlockWeights],
) -> DeepSeekV4PackedWeights:
    """Validate and pack the selected heterogeneous Blocks for sequential launches."""
    if len(weights) != config.layer_count:
        raise ValueError(f"expected {config.layer_count} Block weights, got {len(weights)}")
    for layer_id, block in enumerate(weights):
        validate_block_weights(config, layer_id, block)
    attention = [block.attention for block in weights]
    moe = [block.moe for block in weights]
    # Attention topology is independent of the compact MoE expert-selection
    # mode.  ``preselected_moe`` only limits routed expert weights; disabling
    # CSA/HCA packing here silently turned layers 2/3 into invalid dummy
    # compressors whenever the real-checkpoint memory-saving mode was used.
    csa4 = attention[2] if config.layer_count > 2 else None
    csa128 = attention[3] if config.layer_count > 3 else None

    def optional_tensor(value, *, matmul=False):
        if value is None:
            return torch.ones((1,), dtype=torch.bfloat16 if matmul else torch.float32)
        return kernel_weight(value) if matmul else value

    def moe_tensor(name):
        values = [getattr(value, name) for value in moe]
        if config.fp8_moe:
            values = [value.to(torch.float8_e4m3fn) for value in values]
        return stack_weights(values, matmul=True)

    return DeepSeekV4PackedWeights(
        hc_attn_fn=stack_weights([block.hc_attn_fn for block in weights], matmul=True),
        hc_attn_scale=stack_weights([block.hc_attn_scale for block in weights], matmul=False),
        hc_attn_base=stack_weights([block.hc_attn_base for block in weights], matmul=False),
        hc_ffn_fn=stack_weights([block.hc_ffn_fn for block in weights], matmul=True),
        hc_ffn_scale=stack_weights([block.hc_ffn_scale for block in weights], matmul=False),
        hc_ffn_base=stack_weights([block.hc_ffn_base for block in weights], matmul=False),
        attn_sink=stack_weights([value.attn_sink for value in attention], matmul=False),
        wq_a=stack_weights([value.wq_a for value in attention], matmul=True),
        wq_b=stack_weights([value.wq_b for value in attention], matmul=True),
        wkv=stack_weights([value.wkv for value in attention], matmul=True),
        wo_a=stack_weights([value.wo_a for value in attention], matmul=True),
        wo_b=stack_weights([value.wo_b for value in attention], matmul=True),
        q_norm_weight=stack_weights(
            [norm_weight(value.q_norm_weight, config.q_lora_rank) for value in attention],
            matmul=False,
        ),
        kv_norm_weight=stack_weights(
            [norm_weight(value.kv_norm_weight, config.head_dim) for value in attention],
            matmul=False,
        ),
        attn_norm_weight=stack_weights(
            [norm_weight(block.attn_norm_weight, config.hidden_dim) for block in weights],
            matmul=False,
        ),
        ffn_norm_weight=stack_weights(
            [norm_weight(block.ffn_norm_weight, config.hidden_dim) for block in weights],
            matmul=False,
        ),
        moe_gate=stack_weights(
            [
                value.gate_weight.index_select(0, value.selected_indices.reshape(-1).long())
                if config.preselected_moe
                else value.gate_weight
                for value in moe
            ],
            matmul=True,
        ),
        expert_w1=moe_tensor("expert_w1"),
        expert_w2=moe_tensor("expert_w2"),
        expert_w3=moe_tensor("expert_w3"),
        shared_w1=moe_tensor("shared_w1"),
        shared_w2=moe_tensor("shared_w2"),
        shared_w3=moe_tensor("shared_w3"),
        hash_table=(
            torch.zeros((1, 1, config.activated_experts), dtype=torch.int32)
            if config.preselected_moe
            else torch.stack(
                [value.hash_table.to(torch.int32) for value in moe[: min(config.layer_count, 3)]], dim=0
            ).contiguous()
        ),
        gate_bias=(
            torch.zeros((config.routed_experts,), dtype=torch.float32)
            if config.preselected_moe or config.layer_count <= config.hash_layer_count
            else moe[config.hash_layer_count].gate_bias
        ),
        csa4_wkv=optional_tensor(csa4.compressor_wkv if csa4 else None, matmul=True),
        csa4_wgate=optional_tensor(csa4.compressor_wgate if csa4 else None, matmul=True),
        csa4_ape=optional_tensor(csa4.compressor_ape if csa4 else None),
        csa4_norm=(norm_weight(csa4.compressor_norm_weight, config.head_dim) if csa4 else optional_tensor(None)),
        index_wq_b=optional_tensor(csa4.index_wq_b if csa4 else None, matmul=True),
        index_gate=optional_tensor(csa4.index_gate_weight if csa4 else None, matmul=True),
        index_wkv=optional_tensor(csa4.index_compressor_wkv if csa4 else None, matmul=True),
        index_wgate=optional_tensor(csa4.index_compressor_wgate if csa4 else None, matmul=True),
        index_ape=optional_tensor(csa4.index_compressor_ape if csa4 else None),
        index_norm=(
            norm_weight(csa4.index_compressor_norm_weight, config.index_head_dim) if csa4 else optional_tensor(None)
        ),
        csa128_wkv=optional_tensor(csa128.compressor_wkv if csa128 else None, matmul=True),
        csa128_wgate=optional_tensor(csa128.compressor_wgate if csa128 else None, matmul=True),
        csa128_ape=optional_tensor(csa128.compressor_ape if csa128 else None),
        csa128_norm=(norm_weight(csa128.compressor_norm_weight, config.head_dim) if csa128 else optional_tensor(None)),
    )


@dataclass
class DeepSeekV4AttentionTrace:
    output: torch.Tensor
    query_rank: torch.Tensor
    query: torch.Tensor
    window_kv: torch.Tensor
    candidate_indices: torch.Tensor
    compressed: Optional[torch.Tensor]
    index_scores: Optional[torch.Tensor]
    compressed_indices: Optional[torch.Tensor]
    attention_output: torch.Tensor


@dataclass
class DeepSeekV4BlockTrace:
    output: torch.Tensor
    attention_input: torch.Tensor
    attention: DeepSeekV4AttentionTrace
    after_attention: torch.Tensor
    ffn_input: torch.Tensor
    routed_experts: torch.Tensor
    ffn_output: torch.Tensor


@dataclass
class DeepSeekV4FourLayerTrace:
    output: torch.Tensor
    layers: list[DeepSeekV4BlockTrace]


def causal_compressed_indices(ratio: int, batch_size: int, seq_len: int, *, device=None) -> torch.Tensor:
    """Match model.py's prefill ``get_compress_topk_idxs`` before cache offset."""
    groups = seq_len // ratio
    positions = torch.arange(groups, dtype=torch.int32, device=device).repeat(seq_len, 1)
    valid_count = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device).reshape(-1, 1) // ratio
    indices = torch.where(positions < valid_count, positions, torch.full_like(positions, -1))
    return indices.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def index_score_mask(ratio: int, batch_size: int, seq_len: int, *, device=None):
    valid = causal_compressed_indices(ratio, batch_size, seq_len, device=device) >= 0
    return torch.where(
        valid,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), -float("inf"), dtype=torch.float32, device=device),
    )


def validate_attention_weights(
    config: DeepSeekV4FourLayerConfig,
    layer_id: int,
    weights: DeepSeekV4AttentionWeights,
):
    ratio = config.compress_ratios[layer_id]
    group_width = config.head_num // config.output_groups * config.head_dim
    expected = {
        "attn_sink": (config.head_num,),
        "wq_a": (config.q_lora_rank, config.hidden_dim),
        "wq_b": (config.head_num * config.head_dim, config.q_lora_rank),
        "wkv": (config.head_dim, config.hidden_dim),
        "wo_a": (config.output_groups, config.output_lora_rank, group_width),
        "wo_b": (config.hidden_dim, config.output_groups * config.output_lora_rank),
    }
    for name, shape in expected.items():
        if tuple(getattr(weights, name).shape) != shape:
            raise ValueError(f"layer {layer_id} attention {name} must have shape {shape}")
    for name, shape in (
        ("q_norm_weight", (config.q_lora_rank,)),
        ("kv_norm_weight", (config.head_dim,)),
    ):
        value = getattr(weights, name)
        if value is not None and tuple(value.shape) != shape:
            raise ValueError(f"layer {layer_id} {name} must have shape {shape}")
    if ratio:
        channels = (2 if ratio == 4 else 1) * config.head_dim
        if weights.compressor_wkv is None or weights.compressor_wgate is None or weights.compressor_ape is None:
            raise ValueError(f"layer {layer_id} requires main CSA compressor weights")
        if tuple(weights.compressor_wkv.shape) != (channels, config.hidden_dim):
            raise ValueError("main CSA compressor projection shape is incompatible")
        if weights.compressor_wgate.shape != weights.compressor_wkv.shape:
            raise ValueError("main CSA compressor projections must have identical shapes")
        if tuple(weights.compressor_ape.shape) != (ratio, channels):
            raise ValueError("main CSA APE shape is incompatible")
        if weights.compressor_norm_weight is not None and tuple(weights.compressor_norm_weight.shape) != (
            config.head_dim,
        ):
            raise ValueError("main CSA norm shape is incompatible")
    if ratio == 4:
        index_channels = 2 * config.index_head_dim
        index_tensors = (
            weights.index_wq_b,
            weights.index_gate_weight,
            weights.index_compressor_wkv,
            weights.index_compressor_wgate,
            weights.index_compressor_ape,
        )
        if any(value is None for value in index_tensors):
            raise ValueError("ratio-4 layer requires Indexer projections and compressor weights")
        if tuple(weights.index_wq_b.shape) != (
            config.index_head_num * config.index_head_dim,
            config.q_lora_rank,
        ):
            raise ValueError("Indexer query projection shape is incompatible")
        if tuple(weights.index_gate_weight.shape) != (
            config.index_head_num,
            config.hidden_dim,
        ):
            raise ValueError("Indexer gate projection shape is incompatible")
        if tuple(weights.index_compressor_wkv.shape) != (
            index_channels,
            config.hidden_dim,
        ):
            raise ValueError("Indexer compressor projection shape is incompatible")
        if weights.index_compressor_wgate.shape != weights.index_compressor_wkv.shape:
            raise ValueError("Indexer compressor projections must have identical shapes")
        if tuple(weights.index_compressor_ape.shape) != (ratio, index_channels):
            raise ValueError("Indexer compressor APE shape is incompatible")
        if weights.index_compressor_norm_weight is not None and tuple(weights.index_compressor_norm_weight.shape) != (
            config.index_head_dim,
        ):
            raise ValueError("Indexer compressor norm shape is incompatible")


def validate_block_weights(
    config: DeepSeekV4FourLayerConfig,
    layer_id: int,
    weights: DeepSeekV4BlockWeights,
):
    mix = (2 + config.hc_mult) * config.hc_mult
    hc_shape = (mix, config.hc_mult * config.hidden_dim)
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        if tuple(getattr(weights, name).shape) != hc_shape:
            raise ValueError(f"layer {layer_id} {name} has an incompatible shape")
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        if tuple(getattr(weights, name).shape) != (3,):
            raise ValueError(f"layer {layer_id} {name} must have shape (3,)")
    for name in ("hc_attn_base", "hc_ffn_base"):
        if tuple(getattr(weights, name).shape) != (mix,):
            raise ValueError(f"layer {layer_id} {name} must have shape ({mix},)")
    for name in ("attn_norm_weight", "ffn_norm_weight"):
        value = getattr(weights, name)
        if value is not None and tuple(value.shape) != (config.hidden_dim,):
            raise ValueError(f"layer {layer_id} {name} must have shape ({config.hidden_dim},)")
    validate_attention_weights(config, layer_id, weights.attention)
    moe = weights.moe
    expert_count = config.activated_experts if config.preselected_moe else config.routed_experts
    expert_shape = (expert_count, config.moe_inter_dim, config.hidden_dim)
    if tuple(moe.gate_weight.shape) != (config.routed_experts, config.hidden_dim):
        raise ValueError("MoE gate weight shape is incompatible")
    if tuple(moe.expert_w1.shape) != expert_shape or tuple(moe.expert_w3.shape) != expert_shape:
        raise ValueError("MoE expert w1/w3 shape is incompatible")
    if tuple(moe.expert_w2.shape) != (
        expert_count,
        config.hidden_dim,
        config.moe_inter_dim,
    ):
        raise ValueError("MoE expert w2 shape is incompatible")
    if config.preselected_moe:
        if moe.selected_indices is None or tuple(moe.selected_indices.shape) != (
            1,
            1,
            config.activated_experts,
        ):
            raise ValueError("preselected MoE requires indices with shape [1,1,activated_experts]")
    elif layer_id < config.hash_layer_count and moe.hash_table is None:
        raise ValueError("hash-routed layers require a token-to-expert table")
    if not config.preselected_moe and layer_id >= config.hash_layer_count and moe.gate_bias is None:
        raise ValueError("score-routed layers require a gate bias")


def model_constants(config: DeepSeekV4FourLayerConfig, layer_id: int, batch: int, seq_len: int):
    """Create only the constants consumed by one statically expanded layer."""
    ratio = config.compress_ratios[layer_id]
    yarn_kwargs = {
        "original_seq_len": config.original_seq_len if ratio else 0,
        "factor": config.rope_factor,
        "beta_fast": config.beta_fast,
        "beta_slow": config.beta_slow,
    }
    cos, sin = rope_frequencies(
        seq_len,
        config.rope_head_dim,
        config.compress_rope_theta if ratio else config.rope_theta,
        **yarn_kwargs,
    )
    logical_window = window_indices(config.window_size, batch, seq_len)
    safe_window = logical_window.clamp_min(0)[..., None].expand(batch, seq_len, config.window_size, config.head_dim)
    window_mask = torch.where(
        logical_window >= 0,
        torch.zeros((), dtype=torch.float32),
        torch.full((), -float("inf"), dtype=torch.float32),
    )
    values = {
        "cos": cos,
        "sin": sin,
        "window_gather": safe_window.to(torch.int32),
        "window_mask": window_mask,
    }
    if ratio == 4 and seq_len >= ratio:
        csa4_cos, csa4_sin = rope_frequencies(
            seq_len // 4,
            config.rope_head_dim,
            config.compress_rope_theta,
            positions=torch.arange(0, seq_len, 4),
            **yarn_kwargs,
        )
        values.update(
            csa4_cos=csa4_cos,
            csa4_sin=csa4_sin,
            csa4_score_mask=index_score_mask(4, batch, seq_len),
        )
    elif ratio == 128 and seq_len >= ratio:
        csa128_cos, csa128_sin = rope_frequencies(
            seq_len // 128,
            config.rope_head_dim,
            config.compress_rope_theta,
            positions=torch.arange(0, seq_len, 128),
            **yarn_kwargs,
        )
        values.update(
            csa128_cos=csa128_cos,
            csa128_sin=csa128_sin,
            csa128_mask=index_score_mask(128, batch, seq_len),
        )
    return {name: ax.dsl.from_numpy(value.cpu().numpy(), prefix=f"ds_v4_{name}") for name, value in values.items()}


def deepseek_attention_value(
    x,
    layer_id: int,
    weights: DeepSeekV4AttentionWeights,
    constants,
    config: DeepSeekV4FourLayerConfig,
):
    """Record one complete attention sublayer in the four-layer graph."""
    batch, seq_len, _ = x.shape
    ratio = config.compress_ratios[layer_id]
    cos = constants["cos"]
    sin = constants["sin"]
    qr, query, kv = attention_input_value(
        x,
        weights.wq_a,
        weights.wq_b,
        weights.wkv,
        weights.q_norm_weight,
        weights.kv_norm_weight,
        cos,
        sin,
        config,
    )
    candidates = gather_candidates_value(kv, constants["window_gather"])
    additive_mask = constants["window_mask"]

    if ratio == 4 and seq_len >= ratio:
        compressed = csa_compress_value(
            x,
            weights.compressor_wkv,
            weights.compressor_wgate,
            weights.compressor_ape,
            ratio=4,
            eps=config.norm_eps,
            output_dtype=x.dtype,
            norm_weight=weights.compressor_norm_weight,
        )
        compressed = rope_token_value(
            compressed,
            constants["csa4_cos"],
            constants["csa4_sin"],
            rope_dim=config.rope_head_dim,
        )
        index_query = query_projection_value(qr, weights.index_wq_b, output_dtype=x.dtype).reshape(
            batch,
            seq_len,
            config.index_head_num,
            config.index_head_dim,
        )
        index_query = hadamard_value(rope_head_value(index_query, cos, sin, rope_dim=config.rope_head_dim))
        index_cache = csa_compress_value(
            x,
            weights.index_compressor_wkv,
            weights.index_compressor_wgate,
            weights.index_compressor_ape,
            ratio=4,
            eps=config.norm_eps,
            output_dtype=x.dtype,
            norm_weight=weights.index_compressor_norm_weight,
        )
        index_cache = hadamard_value(
            rope_token_value(
                index_cache,
                constants["csa4_cos"],
                constants["csa4_sin"],
                rope_dim=config.rope_head_dim,
            )
        )
        index_gate = query_projection_value(x, weights.index_gate_weight, output_dtype=ax.fp32)
        _, positions = csa_indexer_value(
            index_query,
            index_cache,
            index_gate,
            constants["csa4_score_mask"],
            index_topk=config.index_topk,
            softmax_scale=(config.index_head_dim**-0.5 * config.index_head_num**-0.5),
        )
        gather_indices = ax.dsl.expand(
            positions.reshape(batch, seq_len, config.index_topk, 1),
            shape=(
                batch,
                seq_len,
                config.index_topk,
                config.head_dim,
            ),
        )
        candidates = ax.dsl.concat(
            candidates,
            gather_candidates_value(compressed, gather_indices),
            dim=2,
        )
        additive_mask = ax.dsl.concat(
            additive_mask,
            ax.dsl.gather(constants["csa4_score_mask"], positions, dim=2),
            dim=2,
        )
    elif ratio == 128 and seq_len >= ratio:
        compressed = hca_compress_value(
            x,
            weights.compressor_wkv,
            weights.compressor_wgate,
            weights.compressor_ape,
            weights.compressor_norm_weight,
            constants["csa128_cos"],
            constants["csa128_sin"],
            eps=config.norm_eps,
            rotary_dim=config.rope_head_dim,
            output_dtype=x.dtype,
        )
        # ``hca_compress_value`` already applies the block-position RoPE.
        # Unlike ratio-4 CSA, C128 must not take a second dense compressor or
        # a second RoPE pass here.
        candidates = ax.dsl.concat(
            candidates,
            ax.dsl.expand(
                compressed.reshape(batch, 1, 1, config.head_dim),
                shape=(batch, seq_len, 1, config.head_dim),
            ),
            dim=2,
        )
        additive_mask = ax.dsl.concat(additive_mask, constants["csa128_mask"], dim=2)

    attention = swa_attention_value(
        query,
        candidates,
        additive_mask,
        weights.attn_sink,
        softmax_scale=config.head_dim**-0.5,
        output_dtype=x.dtype,
    )
    return attention_output_value(attention, cos, sin, weights.wo_a, weights.wo_b, config)


def reduce_moe_partials(partial, layer_id: int, output_dtype):
    """Collect one FP32 MoE partial per core and return the common reduced value."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    partials = ax.dsl.create_tensor(
        (core_num, *partial.shape),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"ds_v4_layer_{layer_id}_moe_partials",
    )
    ax.dsl.move(partial.reshape(1, *partial.shape), r=partials[core_id : core_id + 1])
    axd.sync_all()
    combined = partials[0:1].reshape(*partial.shape)
    for reduce_core in range(1, core_num):
        combined = ax.dsl.add(
            combined,
            partials[reduce_core : reduce_core + 1].reshape(*partial.shape),
            output_dtype=ax.fp32,
        )
    return ax.dsl.cast(combined, dtype=output_dtype)


def deepseek_block_value(
    hidden,
    input_ids,
    layer_id: int,
    weights: DeepSeekV4BlockWeights,
    constants,
    config: DeepSeekV4FourLayerConfig,
):
    """Record one `model.py::Block` without launching an inner kernel."""
    residual = hidden
    reduced, post, comb = pre_hc_value(
        hidden,
        weights.hc_attn_fn,
        weights.hc_attn_scale,
        weights.hc_attn_base,
        eps=config.hc_eps,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        output_dtype=hidden.dtype,
    )
    attention_input = rmsnorm_value(
        reduced,
        eps=config.norm_eps,
        output_dtype=hidden.dtype,
        weight=weights.attn_norm_weight,
    )
    attention_output = deepseek_attention_value(attention_input, layer_id, weights.attention, constants, config)
    hidden = post_hc_value(
        attention_output,
        residual,
        post,
        comb,
        output_dtype=hidden.dtype,
    )

    residual = hidden
    reduced, post, comb = pre_hc_value(
        hidden,
        weights.hc_ffn_fn,
        weights.hc_ffn_scale,
        weights.hc_ffn_base,
        eps=config.hc_eps,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        output_dtype=hidden.dtype,
    )
    ffn_input = rmsnorm_value(
        reduced,
        eps=config.norm_eps,
        output_dtype=hidden.dtype,
        weight=weights.ffn_norm_weight,
    )
    moe = weights.moe
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    if config.preselected_moe:
        if core_num == 1:
            ffn_output = selected_moe_value(
                ffn_input,
                moe.gate_weight,
                moe.expert_w1,
                moe.expert_w2,
                moe.expert_w3,
                moe.shared_w1,
                moe.shared_w2,
                moe.shared_w3,
                layer_id=layer_id,
                route_scale=config.route_scale,
                swiglu_limit=config.swiglu_limit,
                output_dtype=hidden.dtype,
            )
        else:
            ffn_output = selected_moe_multicore_value(
                ffn_input,
                moe.gate_weight,
                moe.expert_w1,
                moe.expert_w2,
                moe.expert_w3,
                moe.shared_w1,
                moe.shared_w2,
                moe.shared_w3,
                layer_id=layer_id,
                route_scale=config.route_scale,
                swiglu_limit=config.swiglu_limit,
                output_dtype=hidden.dtype,
            )
    elif layer_id < config.hash_layer_count:
        # The checkpoint HASH path applies its selected weight before Down in
        # ``model.py``.  Keep that explicit compatibility boundary for the
        # first three layers; ROUTED layers below use the canonical
        # post-Down weighted-sum ABI.
        selected = ax.dsl.index_select(moe.hash_table, input_ids, dim=0)
        if core_num == 1:
            ffn_output, _ = hash_moe_value(
                ffn_input,
                moe.gate_weight,
                selected,
                moe.expert_w1,
                moe.expert_w2,
                moe.expert_w3,
                moe.shared_w1,
                moe.shared_w2,
                moe.shared_w3,
                route_scale=config.route_scale,
                swiglu_limit=config.swiglu_limit,
                output_dtype=hidden.dtype,
                route_after_down=False,
            )
        else:
            expert_num = moe.expert_w1.shape[0]
            expert_start = core_id * expert_num // core_num
            expert_stop = (core_id + 1) * expert_num // core_num
            partial, _ = hash_moe_partial_value(
                ffn_input,
                moe.gate_weight,
                selected,
                moe.expert_w1,
                moe.expert_w2,
                moe.expert_w3,
                moe.shared_w1,
                moe.shared_w2,
                moe.shared_w3,
                expert_start=expert_start,
                expert_stop=expert_stop,
                include_shared=core_id == 0,
                route_scale=config.route_scale,
                swiglu_limit=config.swiglu_limit,
                output_dtype=hidden.dtype,
                route_after_down=False,
            )
            ffn_output = reduce_moe_partials(partial, layer_id, hidden.dtype)
    else:
        if core_num == 1:
            ffn_output, _ = score_moe_value(
                ffn_input,
                moe.gate_weight,
                moe.gate_bias,
                moe.expert_w1,
                moe.expert_w2,
                moe.expert_w3,
                moe.shared_w1,
                moe.shared_w2,
                moe.shared_w3,
                topk=config.activated_experts,
                route_scale=config.route_scale,
                swiglu_limit=config.swiglu_limit,
                output_dtype=hidden.dtype,
            )
        else:
            expert_num = moe.expert_w1.shape[0]
            expert_start = core_id * expert_num // core_num
            expert_stop = (core_id + 1) * expert_num // core_num
            partial, _ = score_moe_partial_value(
                ffn_input,
                moe.gate_weight,
                moe.gate_bias,
                moe.expert_w1,
                moe.expert_w2,
                moe.expert_w3,
                moe.shared_w1,
                moe.shared_w2,
                moe.shared_w3,
                topk=config.activated_experts,
                expert_start=expert_start,
                expert_stop=expert_stop,
                include_shared=core_id == 0,
                route_scale=config.route_scale,
                swiglu_limit=config.swiglu_limit,
                output_dtype=hidden.dtype,
            )
            ffn_output = reduce_moe_partials(partial, layer_id, hidden.dtype)
    return post_hc_value(ffn_output, residual, post, comb, output_dtype=hidden.dtype)


@ax.builder(
    run_mode=run_mode(),
    enable_cache=False,
    dump_dsl_perf=(os.environ.get("DUMP_DSL_PERF") == "1" or os.environ.get("RUN_DEEPSEEK_V4_FLASH") == "1"),
)
def deepseek_v4_flash_layer_kernel(
    hidden,
    input_ids,
    hc_attn_fn,
    hc_attn_scale,
    hc_attn_base,
    hc_ffn_fn,
    hc_ffn_scale,
    hc_ffn_base,
    attn_sink,
    wq_a,
    wq_b,
    wkv,
    wo_a,
    wo_b,
    q_norm_weight,
    kv_norm_weight,
    attn_norm_weight,
    ffn_norm_weight,
    moe_gate,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    hash_table,
    gate_bias,
    csa4_wkv,
    csa4_wgate,
    csa4_ape,
    csa4_norm,
    index_wq_b,
    index_gate,
    index_wkv,
    index_wgate,
    index_ape,
    index_norm,
    csa128_wkv,
    csa128_wgate,
    csa128_ape,
    csa128_norm,
    output,
    layer_id,
    config,
):
    """Record and execute one Block; the host chains its output to the next layer."""
    batch, seq_len = hidden.shape[:2]

    def load_small(value, *, element_limit: int = 1 << 20):
        """Keep compact test weights in OCM while large checkpoint tensors stay in DDR."""
        element_count = 1
        for size in value.shape:
            element_count *= size
        return ax.dsl.load(value) if element_count < element_limit else value

    attention = DeepSeekV4AttentionWeights(
        attn_sink=ax.dsl.load(attn_sink),
        # Keep large projection matrices as DDR views.  ``tiled_linear_value``
        # loads bounded row tiles; loading the complete tensor here would
        # consume the per-core OCM before the multi-core graph is scheduled.
        wq_a=load_small(wq_a),
        wq_b=load_small(wq_b),
        wkv=load_small(wkv),
        wo_a=load_small(wo_a),
        wo_b=load_small(wo_b),
        q_norm_weight=ax.dsl.load(q_norm_weight),
        kv_norm_weight=ax.dsl.load(kv_norm_weight),
    )
    ratio = config.compress_ratios[layer_id]
    if ratio == 4:
        attention.compressor_wkv = ax.dsl.load(csa4_wkv)
        attention.compressor_wgate = ax.dsl.load(csa4_wgate)
        attention.compressor_ape = ax.dsl.load(csa4_ape)
        attention.compressor_norm_weight = ax.dsl.load(csa4_norm)
        attention.index_wq_b = ax.dsl.load(index_wq_b)
        attention.index_gate_weight = ax.dsl.load(index_gate)
        attention.index_compressor_wkv = ax.dsl.load(index_wkv)
        attention.index_compressor_wgate = ax.dsl.load(index_wgate)
        attention.index_compressor_ape = ax.dsl.load(index_ape)
        attention.index_compressor_norm_weight = ax.dsl.load(index_norm)
    elif ratio == 128:
        attention.compressor_wkv = ax.dsl.load(csa128_wkv)
        attention.compressor_wgate = ax.dsl.load(csa128_wgate)
        attention.compressor_ape = ax.dsl.load(csa128_ape)
        attention.compressor_norm_weight = ax.dsl.load(csa128_norm)

    moe = DeepSeekV4MoEWeights(
        gate_weight=ax.dsl.load(moe_gate),
        # Expert and shared projections are also consumed through tiled
        # MatMul.  Preserve their DDR views so each tile can be swapped in
        # independently on every core.
        expert_w1=load_small(expert_w1),
        expert_w2=load_small(expert_w2),
        expert_w3=load_small(expert_w3),
        shared_w1=load_small(shared_w1),
        shared_w2=load_small(shared_w2),
        shared_w3=load_small(shared_w3),
    )
    if not config.preselected_moe and layer_id < config.hash_layer_count:
        moe.hash_table = ax.dsl.load(hash_table)
    elif not config.preselected_moe:
        moe.gate_bias = ax.dsl.load(gate_bias)

    weights = DeepSeekV4BlockWeights(
        hc_attn_fn=ax.dsl.load(hc_attn_fn),
        hc_attn_scale=ax.dsl.load(hc_attn_scale),
        hc_attn_base=ax.dsl.load(hc_attn_base),
        hc_ffn_fn=ax.dsl.load(hc_ffn_fn),
        hc_ffn_scale=ax.dsl.load(hc_ffn_scale),
        hc_ffn_base=ax.dsl.load(hc_ffn_base),
        attn_norm_weight=ax.dsl.load(attn_norm_weight),
        ffn_norm_weight=ax.dsl.load(ffn_norm_weight),
        attention=attention,
        moe=moe,
    )
    value = deepseek_block_value(
        ax.dsl.load(hidden),
        ax.dsl.load(input_ids),
        layer_id,
        weights,
        model_constants(config, layer_id, batch, seq_len),
        config,
    )
    if ax.program_id(0) == 0:
        ax.dsl.store(value, r=output)
    return output


def packed_layer_arguments(packed: DeepSeekV4PackedWeights, layer_id: int):
    """Return only one layer's tensors for one independent Block launch."""

    def layer_value(stacked):
        return stacked[layer_id]

    hash_layer = min(layer_id, packed.hash_table.shape[0] - 1)
    return (
        layer_value(packed.hc_attn_fn),
        layer_value(packed.hc_attn_scale),
        layer_value(packed.hc_attn_base),
        layer_value(packed.hc_ffn_fn),
        layer_value(packed.hc_ffn_scale),
        layer_value(packed.hc_ffn_base),
        layer_value(packed.attn_sink),
        layer_value(packed.wq_a),
        layer_value(packed.wq_b),
        layer_value(packed.wkv),
        layer_value(packed.wo_a),
        layer_value(packed.wo_b),
        layer_value(packed.q_norm_weight),
        layer_value(packed.kv_norm_weight),
        layer_value(packed.attn_norm_weight),
        layer_value(packed.ffn_norm_weight),
        layer_value(packed.moe_gate),
        layer_value(packed.expert_w1),
        layer_value(packed.expert_w2),
        layer_value(packed.expert_w3),
        layer_value(packed.shared_w1),
        layer_value(packed.shared_w2),
        layer_value(packed.shared_w3),
        packed.hash_table[hash_layer],
        packed.gate_bias,
        packed.csa4_wkv,
        packed.csa4_wgate,
        packed.csa4_ape,
        packed.csa4_norm,
        packed.index_wq_b,
        packed.index_gate,
        packed.index_wkv,
        packed.index_wgate,
        packed.index_ape,
        packed.index_norm,
        packed.csa128_wkv,
        packed.csa128_wgate,
        packed.csa128_ape,
        packed.csa128_norm,
    )


class DeepSeekV4FlashPort:
    """Host wrapper that launches each Block independently and chains outputs."""

    def __init__(
        self,
        config: DeepSeekV4FourLayerConfig,
        weights: Sequence[DeepSeekV4BlockWeights],
        core_num: int = 1,
    ):
        validate_core_num(core_num, core_num, "configured model core slots")
        self.config = config
        self.core_num = core_num
        self.weights = list(weights)
        self.packed = pack_four_layer_weights(config, weights)

    def __call__(self, hidden: torch.Tensor, input_ids: torch.Tensor):
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_dim:
            raise ValueError("hidden must have shape [B,S,D]")
        if tuple(input_ids.shape) != tuple(hidden.shape[:2]):
            raise ValueError("input_ids must match hidden batch and sequence dimensions")
        if self.config.preselected_moe and tuple(hidden.shape[:2]) != (1, 1):
            raise ValueError("checkpoint preselected-expert validation requires shape [1,1,D]")
        if not self.config.preselected_moe and hidden.shape[1] % 128:
            raise ValueError("the prefill sequence must be divisible by 128")
        current = hidden.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        layer_outputs = []
        token_ids = input_ids.to(torch.int32)
        ax.init(chip_num=1, core_num=self.core_num)
        for layer_id in range(self.config.layer_count):
            output = torch.empty_like(current)
            deepseek_v4_flash_layer_kernel[self.core_num](
                current,
                token_ids,
                *packed_layer_arguments(self.packed, layer_id),
                output,
                layer_id,
                self.config,
            )
            current = output
            layer_outputs.append(output)
        return DeepSeekV4PortOutput(current, layer_outputs)

    def reference(self, hidden: torch.Tensor, input_ids: torch.Tensor):
        """Evaluate the selected Blocks with eager mathematical references."""
        value = hidden.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        traces = []
        for layer_id, weights in enumerate(self.weights):
            trace = reference_block(self.config, layer_id, weights, value, input_ids)
            value = trace.output
            traces.append(trace)
        return DeepSeekV4FourLayerTrace(value, traces)

    def compare(self, hidden: torch.Tensor, input_ids: torch.Tensor):
        """Run the selected per-layer DSL schedule and its eager reference."""
        return self(hidden, input_ids), self.reference(hidden, input_ids)


DeepSeekV4FourLayerModel = DeepSeekV4FlashPort


def eager_rmsnorm(x: torch.Tensor, eps: float, weight: Optional[torch.Tensor] = None):
    value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        value = value * weight.float()
    return value.to(x.dtype)


def eager_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int, inverse=False):
    prefix = x[..., : x.shape[-1] - rope_dim]
    pairs = x[..., x.shape[-1] - rope_dim :].float().reshape(*x.shape[:-1], rope_dim // 2, 2)
    shape = [1, x.shape[1]] + [1] * (x.ndim - 3) + [rope_dim // 2]
    cos_value = cos.reshape(*shape)
    sin_value = sin.reshape(*shape) * (-1.0 if inverse else 1.0)
    real = pairs[..., 0] * cos_value - pairs[..., 1] * sin_value
    imag = pairs[..., 0] * sin_value + pairs[..., 1] * cos_value
    rotated = torch.stack((real, imag), dim=-1).flatten(-2)
    return torch.cat((prefix.float(), rotated), dim=-1).to(x.dtype)


def eager_hadamard(x: torch.Tensor):
    value = x.float()
    width = value.shape[-1]
    size = 1
    while size < width:
        value = value.reshape(*value.shape[:-1], width // (2 * size), 2, size)
        left, right = value[..., 0, :], value[..., 1, :]
        value = torch.stack((left + right, left - right), dim=-2).reshape(*x.shape)
        size *= 2
    return (value * width**-0.5).to(x.dtype)


def reference_pre_hc(config, hidden, fn, scale, base):
    batch, seq_len = hidden.shape[:2]
    flat = hidden.float().reshape(batch * seq_len, -1)
    mixes = F.linear(flat, kernel_weight(fn).float())
    mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.hc_eps)
    hc = config.hc_mult
    pre = torch.sigmoid(mixes[:, :hc] * scale.float()[0] + base.float()[:hc]) + config.hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * scale.float()[1] + base.float()[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].reshape(batch * seq_len, hc, hc)
    comb = comb * scale.float()[2] + base.float()[2 * hc :].reshape(1, hc, hc)
    comb = comb.softmax(-1) + config.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + config.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    reduced = (pre.reshape(batch, seq_len, hc, 1) * hidden.float()).sum(2).to(hidden.dtype)
    return (
        reduced,
        post.reshape(batch, seq_len, hc),
        comb.reshape(batch, seq_len, hc, hc),
    )


def reference_post_hc(value, residual, post, comb):
    expanded = post.unsqueeze(-1) * value.float().unsqueeze(2)
    expanded += torch.einsum("bsij,bsid->bsjd", comb.float(), residual.float())
    return expanded.to(residual.dtype)


# Keep the old reference names available to downstream diagnostics while the
# canonical names follow the mHC module terminology.
reference_hca_pre = reference_pre_hc
reference_hca_post = reference_post_hc


def reference_compress(config, x, ratio, wkv, wgate, ape, head_dim, norm_weight=None):
    batch, seq_len, _ = x.shape
    groups = seq_len // ratio
    coff = 2 if ratio == 4 else 1
    kv = F.linear(x.float(), kernel_weight(wkv).float()).reshape(batch, groups, ratio, coff, head_dim)
    score = F.linear(x.float(), kernel_weight(wgate).float()).reshape(batch, groups, ratio, coff, head_dim)
    score = score + ape.float().reshape(1, 1, ratio, coff, head_dim)
    if coff == 2:
        previous_kv = torch.zeros((batch, groups, ratio, head_dim), dtype=torch.float32)
        previous_score = torch.full_like(previous_kv, -float("inf"))
        if groups > 1:
            previous_kv[:, 1:] = kv[:, :-1, :, 0, :]
            previous_score[:, 1:] = score[:, :-1, :, 0, :]
        kv = torch.cat((previous_kv, kv[:, :, :, 1, :]), dim=2)
        score = torch.cat((previous_score, score[:, :, :, 1, :]), dim=2)
    else:
        kv = kv.reshape(batch, groups, ratio, head_dim)
        score = score.reshape(batch, groups, ratio, head_dim)
    compressed = (kv * score.softmax(dim=2)).sum(2)
    return eager_rmsnorm(compressed.to(x.dtype), config.norm_eps, norm_weight)


def reference_attention(config, layer_id, weights, x):
    batch, seq_len, _ = x.shape
    ratio = config.compress_ratios[layer_id]
    theta = config.compress_rope_theta if ratio else config.rope_theta
    cos, sin = rope_frequencies(
        seq_len,
        config.rope_head_dim,
        theta,
        original_seq_len=config.original_seq_len if ratio else 0,
        factor=config.rope_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
    )
    cos, sin = cos.to(x.device), sin.to(x.device)
    qr = F.linear(x.float(), kernel_weight(weights.wq_a).float()).to(x.dtype)
    qr = eager_rmsnorm(qr, config.norm_eps, weights.q_norm_weight)
    query = F.linear(qr.float(), kernel_weight(weights.wq_b).float()).to(x.dtype)
    query = eager_rmsnorm(query.reshape(batch, seq_len, config.head_num, config.head_dim), config.norm_eps)
    query = eager_rope(query, cos, sin, config.rope_head_dim)
    kv = F.linear(x.float(), kernel_weight(weights.wkv).float()).to(x.dtype)
    kv = eager_rope(
        eager_rmsnorm(kv, config.norm_eps, weights.kv_norm_weight),
        cos,
        sin,
        config.rope_head_dim,
    )
    window = window_indices(config.window_size, batch, seq_len).to(x.device)
    candidates = gather_compressed(kv, window)
    candidate_indices = window
    compressed = index_scores = compressed_indices = None
    if ratio and seq_len >= ratio:
        compressed = reference_compress(
            config,
            x,
            ratio,
            weights.compressor_wkv,
            weights.compressor_wgate,
            weights.compressor_ape,
            config.head_dim,
            weights.compressor_norm_weight,
        )
        compressed_positions = torch.arange(0, seq_len, ratio, device=x.device)
        compressed_cos, compressed_sin = rope_frequencies(
            compressed.shape[1],
            config.rope_head_dim,
            config.compress_rope_theta,
            positions=compressed_positions,
            original_seq_len=config.original_seq_len,
            factor=config.rope_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
        )
        compressed_cos, compressed_sin = compressed_cos.to(x.device), compressed_sin.to(x.device)
        compressed = eager_rope(compressed, compressed_cos, compressed_sin, config.rope_head_dim)
        if ratio == 4:
            index_query = F.linear(qr.float(), kernel_weight(weights.index_wq_b).float()).to(x.dtype)
            index_query = index_query.reshape(batch, seq_len, config.index_head_num, config.index_head_dim)
            index_query = eager_hadamard(eager_rope(index_query, cos, sin, config.rope_head_dim))
            index_cache = reference_compress(
                config,
                x,
                ratio,
                weights.index_compressor_wkv,
                weights.index_compressor_wgate,
                weights.index_compressor_ape,
                config.index_head_dim,
                weights.index_compressor_norm_weight,
            )
            index_cache = eager_hadamard(eager_rope(index_cache, compressed_cos, compressed_sin, config.rope_head_dim))
            gate = F.linear(x.float(), kernel_weight(weights.index_gate_weight).float())
            index_scores = torch.einsum("bshd,btd->bsht", index_query.float(), index_cache.float()).relu()
            index_scores = (index_scores * gate.unsqueeze(-1)).sum(2)
            index_scores *= config.index_head_dim**-0.5 * config.index_head_num**-0.5
            index_scores += index_score_mask(ratio, batch, seq_len, device=x.device)
            topk = min(config.index_topk, index_cache.shape[1])
            # AX TopK keeps the lower source index when ReLU creates equal
            # zero scores.  Stable argsort gives the eager reference the same
            # deterministic tie rule as the recorded kernel.
            compressed_indices = torch.argsort(index_scores, dim=-1, descending=True, stable=True)[..., :topk].to(
                torch.int32
            )
            valid_count = torch.arange(1, seq_len + 1, device=x.device).reshape(1, seq_len, 1) // ratio
            compressed_indices = torch.where(
                compressed_indices < valid_count,
                compressed_indices,
                torch.full_like(compressed_indices, -1),
            )
        else:
            compressed_indices = causal_compressed_indices(ratio, batch, seq_len, device=x.device)
        candidates = torch.cat((candidates, gather_compressed(compressed, compressed_indices)), dim=2)
        candidate_indices = torch.cat((candidate_indices, compressed_indices), dim=2)

    valid = candidate_indices >= 0
    scores = torch.einsum("bshd,bskd->bshk", query.float(), candidates.float())
    scores = scores * config.head_dim**-0.5
    scores = scores.masked_fill(~valid[:, :, None, :], -float("inf"))
    sink_scores = weights.attn_sink.float()[None, None, :, None].expand(batch, seq_len, -1, -1)
    shifted_scores = torch.cat((scores, sink_scores), dim=-1)
    shifted_scores -= shifted_scores.max(-1, keepdim=True).values
    numerator_weights = torch.exp(shifted_scores[..., :-1]).to(x.dtype).float()
    denominator = torch.exp(shifted_scores).sum(-1, keepdim=True)
    attention_value = torch.einsum("bshk,bskd->bshd", numerator_weights, candidates.float()) / denominator
    attention_value = attention_value.to(x.dtype)
    inverse = eager_rope(attention_value, cos, sin, config.rope_head_dim, inverse=True)
    group_width = config.head_num // config.output_groups * config.head_dim
    grouped = inverse.reshape(batch, seq_len, config.output_groups, group_width)
    projected = []
    for group in range(config.output_groups):
        projected.append(F.linear(grouped[:, :, group].float(), kernel_weight(weights.wo_a[group]).float()).to(x.dtype))
    output = F.linear(torch.cat(projected, dim=-1).float(), kernel_weight(weights.wo_b).float()).to(x.dtype)
    return DeepSeekV4AttentionTrace(
        output,
        qr,
        query,
        kv,
        candidate_indices,
        compressed,
        index_scores,
        compressed_indices,
        attention_value,
    )


def reference_expert(config, x, w1, w2, w3, route_weight=None):
    gate = F.linear(x.float(), kernel_weight(w1).float()).to(x.dtype)
    up = F.linear(x.float(), kernel_weight(w3).float()).to(x.dtype)
    gate = gate.float().clamp(max=config.swiglu_limit)
    up = up.float().clamp(-config.swiglu_limit, config.swiglu_limit)
    hidden = F.silu(gate) * up
    if route_weight is not None:
        hidden = hidden * route_weight
    hidden = hidden.to(x.dtype)
    return F.linear(hidden.float(), kernel_weight(w2).float()).to(x.dtype)


def reference_moe(config, layer_id, weights, x, input_ids):
    flat = x.reshape(-1, config.hidden_dim)
    raw = F.softplus(F.linear(flat.float(), kernel_weight(weights.gate_weight).float())).sqrt()
    if config.preselected_moe:
        indices = weights.selected_indices.reshape(-1, config.activated_experts).long()
    elif layer_id < config.hash_layer_count:
        indices = weights.hash_table[input_ids.long()].reshape(-1, config.activated_experts).long()
    else:
        indices = (raw + weights.gate_bias.float()).topk(config.activated_experts, dim=-1).indices
    selected = raw.gather(1, indices)
    selected = selected / selected.sum(-1, keepdim=True) * config.route_scale
    combined = torch.zeros_like(flat, dtype=torch.float32)
    if config.preselected_moe:
        for slot in range(config.activated_experts):
            value = reference_expert(
                config,
                flat,
                weights.expert_w1[slot],
                weights.expert_w2[slot],
                weights.expert_w3[slot],
                selected[:, slot : slot + 1],
            )
            combined += value.float()
    else:
        sparse = torch.zeros((flat.shape[0], config.routed_experts), dtype=torch.float32)
        sparse.scatter_(1, indices, selected)
        for expert in range(config.routed_experts):
            value = reference_expert(
                config,
                flat,
                weights.expert_w1[expert],
                weights.expert_w2[expert],
                weights.expert_w3[expert],
                sparse[:, expert : expert + 1],
            )
            combined += value.float()
    shared = reference_expert(config, flat, weights.shared_w1, weights.shared_w2, weights.shared_w3)
    return (combined + shared.float()).to(x.dtype).reshape_as(x), indices.to(torch.int32).reshape(
        *x.shape[:2], config.activated_experts
    )


def reference_block(config, layer_id, weights, hidden, input_ids):
    residual = hidden
    reduced, post, comb = reference_pre_hc(
        config,
        hidden,
        weights.hc_attn_fn,
        weights.hc_attn_scale,
        weights.hc_attn_base,
    )
    attention_input_value = eager_rmsnorm(reduced, config.norm_eps, weights.attn_norm_weight)
    attention_trace = reference_attention(config, layer_id, weights.attention, attention_input_value)
    after_attention = reference_post_hc(attention_trace.output, residual, post, comb)
    residual = after_attention
    reduced, post, comb = reference_pre_hc(
        config,
        after_attention,
        weights.hc_ffn_fn,
        weights.hc_ffn_scale,
        weights.hc_ffn_base,
    )
    ffn_input = eager_rmsnorm(reduced, config.norm_eps, weights.ffn_norm_weight)
    ffn_output, selected = reference_moe(config, layer_id, weights.moe, ffn_input, input_ids)
    output = reference_post_hc(ffn_output, residual, post, comb)
    return DeepSeekV4BlockTrace(
        output,
        attention_input_value,
        attention_trace,
        after_attention,
        ffn_input,
        selected,
        ffn_output,
    )


__all__ = [
    "DeepSeekV4AttentionWeights",
    "DeepSeekV4BlockWeights",
    "DeepSeekV4FlashPort",
    "DeepSeekV4FourLayerConfig",
    "DeepSeekV4FourLayerModel",
    "DeepSeekV4FourLayerTrace",
    "DeepSeekV4MoEWeights",
    "DeepSeekV4PackedWeights",
    "DeepSeekV4PortOutput",
    "causal_compressed_indices",
    "deepseek_v4_flash_layer_kernel",
    "pack_four_layer_weights",
    "packed_layer_arguments",
]
