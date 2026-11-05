"""DeepSeek Sparse Attention prefill kernel for Axera DSL.

The module owns the DSA Indexer, sparse MLA attention, builder, validation, and
public wrapper. Shared pre-attention math and host-side scale normalization are
reused from :mod:`sgl_kernel_axera.attention.attn_utils.common`. It deliberately
does not import reference kernels or helpers from ``axnn.dsl.kernels``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.opset import datatype
from sgl_kernel_axera.attention.attn_utils import (
    DSAIndexNorm,
    DSASparseAttentionImpl,
    MLA_CONST_INPUTS,
    as_dsa_index_norm,
    as_dsa_sparse_impl,
    normalize_fc_scale,
    normalize_head_fc_scale,
    normalize_scale2,
    record_mla_absorbed_query,
    record_mla_shared_projections,
    rmsnorm,
    rope_impl,
)


@dataclass(frozen=True)
class DSAConfig:
    """Describe the MLA and Indexer dimensions used by one DSA layer.

    The field names follow ``DeepSeek-v3.2/inference/model.py::ModelArgs``.
    Tensor dimensions are validated against this structure before the DSL graph
    is recorded.

    Args:
        dim: Hidden-state width ``D``.
        n_heads: MLA attention-head count ``H``.
        q_lora_rank: Query low-rank width ``Rq``.
        kv_lora_rank: Compressed KV width ``Rkv``.
        qk_nope_head_dim: Non-RoPE query/key width ``Dn`` per MLA head.
        qk_rope_head_dim: RoPE width ``Dr`` per MLA head and in the Indexer.
        v_head_dim: Value width ``Dv`` per MLA head.
        index_n_heads: Indexer head count ``Hi``.
        index_head_dim: Indexer head width ``Di``.
        index_topk: Maximum selected token count ``K``.
        softmax_scale: MLA attention score scale. When ``None``, use
            ``1 / sqrt(Dn + Dr)``.
        scale_fmt: Index activation scale format. A non-``None`` value enables
            UE8M0-compatible power-of-two scale rounding.
        index_k_norm_type: Index-key normalization, ``"layer"`` or ``"rms"``.
        index_use_hadamard: Apply the normalized Hadamard transform to Indexer
            Q/K before FP8 quantization.
        index_init_tokens: Initial logical tokens forced into every Top-K row.
        index_local_tokens: Most recent logical tokens forced into Top-K.
        index_token_block_size: Indexer cache tile width.
        index_query_block_size: Indexer query group width.
        sparse_attention_impl: Sparse MLA implementation, ``"token"`` or
            ``"block_dense"``.
        sparse_query_block_size: Sparse MLA query tile width.
        sparse_cache_block_size: Sparse MLA cache tile width.
        core_num: Active NPU core count; the contiguous-prefill implementation
            supports 1 or 4.
    """

    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int = 2048
    softmax_scale: Optional[float] = None
    scale_fmt: Optional[str] = "ue8m0"
    index_k_norm_type: DSAIndexNorm = DSAIndexNorm.LAYER
    index_use_hadamard: bool = True
    index_init_tokens: int = 0
    index_local_tokens: int = 0
    index_token_block_size: int = 128
    index_query_block_size: int = 16
    sparse_attention_impl: DSASparseAttentionImpl = DSASparseAttentionImpl.BLOCK_DENSE
    sparse_query_block_size: int = 32
    sparse_cache_block_size: int = 128
    core_num: int = 4

    @property
    def attention_softmax_scale(self) -> float:
        """Return the configured or dimension-derived MLA score scale."""
        if self.softmax_scale is not None:
            return self.softmax_scale
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


@dataclass(frozen=True)
class DSAProjection:
    """Bundle one constant projection weight with its quantization scales.

    Args:
        weight: Constant projection weight in the layout documented by
            :class:`DSAWeights`.
        scale: Primary quantization scale, or ``None`` for an identity scale.
        scale2: Scalar secondary scale, or ``None`` for an identity scale.
    """

    weight: Any
    scale: Any = None
    scale2: Any = None


@dataclass(frozen=True)
class DSAWeights:
    """Group the MLA and Indexer parameters of one DeepSeek DSA layer.

    Names match the reference MLA/Indexer structure. ``wkv_b_key`` is the
    transposed key-absorption portion of reference ``wkv_b`` and
    ``wkv_b_value`` is its value-expansion portion.

    Args:
        attn_norm: Input RMSNorm weight ``[D]`` for the fused residual layer.
        wq_a: MLA query down projection ``[Rq, D]``.
        q_norm: Query low-rank RMSNorm weight ``[Rq]``.
        wq_b: MLA query up projection ``[H * (Dn + Dr), Rq]``.
        wkv_a: Compressed KV and positional-key projection ``[Rkv + Dr, D]``.
        kv_norm: Compressed KV RMSNorm weight ``[Rkv]``.
        wkv_b_key: Per-head key-absorption weights ``[H, Rkv, Dn]``.
        wkv_b_value: Per-head value-expansion weights ``[H, Dv, Rkv]``.
        wo: MLA output projection ``[D, H * Dv]``.
        index_wq_b: Index-query projection ``[Hi * Di, Rq]``.
        index_wk: Index-key projection ``[Di, D]``.
        index_k_norm_weight: Index-key normalization weight ``[Di]``.
        index_k_norm_bias: LayerNorm bias ``[Di]`` or ``None`` for RMSNorm.
        index_weights_proj: Per-head Indexer gate projection ``[Hi, D]``.
    """

    attn_norm: torch.Tensor
    wq_a: DSAProjection
    q_norm: torch.Tensor
    wq_b: DSAProjection
    wkv_a: DSAProjection
    kv_norm: torch.Tensor
    wkv_b_key: DSAProjection
    wkv_b_value: DSAProjection
    wo: DSAProjection
    index_wq_b: DSAProjection
    index_wk: DSAProjection
    index_k_norm_weight: torch.Tensor
    index_k_norm_bias: Optional[torch.Tensor]
    index_weights_proj: DSAProjection


@dataclass
class DSACache:
    """Own the persistent MLA and Indexer caches updated by :func:`dsa`.

    Args:
        kv_cache: BF16 compressed MLA cache ``[B, L, Rkv]``.
        pe_cache: BF16 rotary-key cache ``[B, L, Dr]``.
        index_k_cache: FP8 E4M3 Indexer key cache ``[B, L, Di]``.
        index_k_scale_cache: FP32 Indexer key scales ``[B, L, 1]``.
    """

    kv_cache: torch.Tensor
    pe_cache: torch.Tensor
    index_k_cache: torch.Tensor
    index_k_scale_cache: torch.Tensor


FreqsCis = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


FP8_E4M3_MAX = 448.0


def index_layer_norm(x, weight, bias, eps=1e-6):
    """Record the SGLang index-key LayerNorm policy with FP32 statistics."""
    width = ax.dsl.full((1,), ax.fp32, fill_value=x.shape[-1])
    mean = ax.dsl.div(
        ax.dsl.sum(x, dim=-1, keepdim=True, output_dtype=ax.fp32),
        width,
        output_dtype=ax.fp32,
    )
    centered = ax.dsl.sub(x, mean, output_dtype=ax.fp32)
    variance = ax.dsl.div(
        ax.dsl.sum(
            ax.dsl.mul(centered, centered, output_dtype=ax.fp32),
            dim=-1,
            keepdim=True,
            output_dtype=ax.fp32,
        ),
        width,
        output_dtype=ax.fp32,
    )
    epsilon = ax.dsl.full((1,), ax.fp32, fill_value=eps)
    normalized = ax.dsl.div(
        centered,
        ax.dsl.sqrt(ax.dsl.add(variance, epsilon, output_dtype=ax.fp32)),
        output_dtype=ax.fp32,
    )
    scaled = ax.dsl.mul(normalized, weight, output_dtype=x.dtype)
    return ax.dsl.add(scaled, bias, output_dtype=x.dtype)


def index_norm(x, weight, bias, norm_type):
    """Apply the model-selected index-key normalization."""
    norm_type = as_dsa_index_norm(norm_type)
    if norm_type is DSAIndexNorm.RMS:
        return rmsnorm(x, weight)
    assert bias is not None, "LayerNorm requires index_k_norm_bias"
    return index_layer_norm(x, weight, bias)


def partial_rope(x, cos, sin, position_ids, rope_dim, name):
    """Apply RoPE to the leading index dimension and preserve the suffix."""
    assert 0 < rope_dim <= x.shape[-1]
    assert rope_dim % 2 == 0
    result = ax.dsl.create_tensor(x.shape, x.dtype, name=name)
    rope_impl(
        x[..., :rope_dim],
        cos,
        sin,
        position_ids,
        result[..., :rope_dim],
        head_less=len(x.shape) == 3,
    )
    if rope_dim < x.shape[-1]:
        ax.dsl.move(x[..., rope_dim:], r=result[..., rope_dim:])
    return result


def normalized_hadamard(x, name):
    """Record an orthonormal Hadamard transform along the last dimension."""
    dim = x.shape[-1]
    assert dim > 0 and dim & (dim - 1) == 0, "Hadamard width must be a power of two"
    rows = int(np.prod(x.shape[:-1]))
    current = x.reshape(rows, dim)
    half_width = 1
    stage = 0
    while half_width < dim:
        group_count = dim // (2 * half_width)
        grouped = current.reshape(rows, group_count, 2, half_width)
        left = grouped[:, :, 0:1, :].reshape(rows, group_count, half_width)
        right = grouped[:, :, 1:2, :].reshape(rows, group_count, half_width)
        next_value = ax.dsl.create_tensor(
            (rows, group_count, 2, half_width),
            x.dtype,
            name=f"{name}_stage{stage}",
        )
        ax.dsl.add(
            left,
            right,
            output_dtype=x.dtype,
            r=next_value[:, :, 0:1, :].reshape(left.shape),
        )
        ax.dsl.sub(
            left,
            right,
            output_dtype=x.dtype,
            r=next_value[:, :, 1:2, :].reshape(left.shape),
        )
        current = next_value.reshape(rows, dim)
        half_width *= 2
        stage += 1

    scale = ax.dsl.from_numpy(np.array([dim**-0.5], dtype=np.float32))
    return ax.dsl.mul(current, scale, output_dtype=x.dtype).reshape(x.shape)


def quantize_index_rows(x, use_ue8m0, name):
    """Quantize last-dimension groups to FP8 E4M3 and return FP32 inverse scales."""
    rows = int(np.prod(x.shape[:-1]))
    width = x.shape[-1]
    x_2d = x.reshape(rows, width)
    x_fp32 = ax.dsl.cast(x_2d, dtype=ax.fp32)
    neg_one = ax.dsl.from_numpy(np.array([-1.0], dtype=np.float32))
    abs_x = ax.dsl.maximum(
        x_fp32,
        ax.dsl.mul(x_fp32, neg_one, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    max_abs = ax.dsl.max(abs_x, dim=-1, keepdim=True)
    epsilon = ax.dsl.full(max_abs.shape, ax.fp32, fill_value=1e-12)
    max_abs = ax.dsl.maximum(max_abs, epsilon, output_dtype=ax.fp32)
    fp8_max = ax.dsl.full(max_abs.shape, ax.fp32, fill_value=FP8_E4M3_MAX)
    inv_scale = ax.dsl.div(max_abs, fp8_max, output_dtype=ax.fp32)
    if use_ue8m0:
        # SGLang's UE8M0 mode rounds each inverse scale upward to a power of two.
        exponent = ax.dsl.ceil(ax.dsl.log2(inv_scale, output_dtype=ax.fp32))
        ln_two = ax.dsl.from_numpy(np.array([np.log(2.0)], dtype=np.float32))
        inv_scale = ax.dsl.exp(
            ax.dsl.mul(exponent, ln_two, output_dtype=ax.fp32),
            output_dtype=ax.fp32,
        )

    scaled = ax.dsl.div(x_fp32, inv_scale, output_dtype=ax.fp32)
    lower = ax.dsl.full(scaled.shape, ax.fp32, fill_value=-FP8_E4M3_MAX)
    upper = ax.dsl.full(scaled.shape, ax.fp32, fill_value=FP8_E4M3_MAX)
    scaled = ax.dsl.minimum(
        ax.dsl.maximum(scaled, lower, output_dtype=ax.fp32),
        upper,
        output_dtype=ax.fp32,
    )
    quantized = ax.dsl.create_tensor((rows, width), ax.fp8_e4m3, name=f"{name}_fp8")
    ax.dsl.cast(scaled, dtype=ax.fp8_e4m3, r=quantized)
    return quantized.reshape(*x.shape), inv_scale.reshape(*(x.shape[:-1] + (1,)))


def index_tile_logits(q_fp8, q_scale, gate, k_fp8, k_scale, softmax_scale):
    """Compute one gated MQA index-logit tile without materializing dequantized caches."""
    q_bf16 = ax.dsl.cast(q_fp8, dtype=ax.bf16)
    k_bf16 = ax.dsl.cast(k_fp8, dtype=ax.bf16)
    per_head = ax.dsl.matmul(q_bf16, k_bf16, transposed=True, output_dtype=ax.fp32)
    per_head = ax.dsl.maximum(
        per_head,
        ax.dsl.zeros(per_head.shape, ax.fp32),
        output_dtype=ax.fp32,
    )
    head_weight = ax.dsl.mul(gate, q_scale, output_dtype=ax.fp32)
    scale = ax.dsl.from_numpy(np.array([softmax_scale], dtype=np.float32))
    head_weight = ax.dsl.mul(head_weight, scale, output_dtype=ax.fp32)
    weighted = ax.dsl.mul(per_head, head_weight.reshape(head_weight.shape[0], 1), output_dtype=ax.fp32)
    logits = ax.dsl.sum(weighted, dim=0, output_dtype=ax.fp32)
    return ax.dsl.mul(logits, k_scale.reshape(k_scale.shape[0]), output_dtype=ax.fp32)


def index_tile_logits_batched(q_bf16, head_weight, k_fp8, k_scale):
    """Compute ``[Bq, Ti]`` gated MQA logits with one Query-batched MatMul.

    Args:
        q_bf16: BF16 ``[Bq, Hi, Di]`` dequantized index queries.
        head_weight: FP32 ``[Bq, Hi]`` containing gate, query scale, and
            ``1 / sqrt(Di)``.
        k_fp8: FP8 E4M3 ``[Ti, Di]`` index-key cache tile.
        k_scale: FP32 ``[Ti]`` inverse scale for ``k_fp8``.
    Returns:
        FP32 ``[Bq, Ti]`` index logits.
    """
    query_count, index_head_num, index_dim = q_bf16.shape
    token_count = k_fp8.shape[0]
    k_bf16 = ax.dsl.cast(k_fp8, dtype=ax.bf16)
    per_head = ax.dsl.matmul(
        q_bf16.reshape(query_count * index_head_num, index_dim),
        k_bf16,
        transposed=True,
        output_dtype=ax.fp32,
    ).reshape(query_count, index_head_num, token_count)
    per_head = ax.dsl.maximum(
        per_head,
        ax.dsl.zeros(per_head.shape, ax.fp32),
        output_dtype=ax.fp32,
    )
    weighted = ax.dsl.mul(
        per_head,
        head_weight.reshape(query_count, index_head_num, 1),
        output_dtype=ax.fp32,
    )
    logits = ax.dsl.sum(weighted, dim=1, output_dtype=ax.fp32)
    return ax.dsl.mul(logits, k_scale.reshape(1, token_count), output_dtype=ax.fp32)


def force_index_tokens(logits, token_start, valid_count, init_tokens, local_tokens):
    """Force configured initial and local logical positions into Top-K."""
    boost = np.zeros(logits.shape, dtype=np.float32)
    token_stop = token_start + logits.shape[-1]
    if init_tokens > 0:
        start = max(token_start, 0)
        stop = min(token_stop, init_tokens)
        if stop > start:
            boost[start - token_start : stop - token_start] = 1.0e30
    if local_tokens > 0:
        local_start = max(0, valid_count - local_tokens)
        start = max(token_start, local_start)
        stop = min(token_stop, valid_count)
        if stop > start:
            boost[start - token_start : stop - token_start] = 1.0e30
    if np.any(boost):
        return ax.dsl.add(logits, ax.dsl.from_numpy(boost), output_dtype=ax.fp32)
    return logits


def force_index_tokens_batched(logits, token_start, valid_counts, init_tokens, local_tokens):
    """Apply the forced-token policy to a static ``[Bq, Ti]`` history tile."""
    boost = np.zeros(logits.shape, dtype=np.float32)
    token_stop = token_start + logits.shape[-1]
    for row, valid_count in enumerate(valid_counts):
        if init_tokens > 0:
            start = max(token_start, 0)
            stop = min(token_stop, init_tokens)
            if stop > start:
                boost[row, start - token_start : stop - token_start] = 1.0e30
        if local_tokens > 0:
            local_start = max(0, valid_count - local_tokens)
            start = max(token_start, local_start)
            stop = min(token_stop, valid_count)
            if stop > start:
                boost[row, start - token_start : stop - token_start] = 1.0e30
    if np.any(boost):
        return ax.dsl.add(logits, ax.dsl.from_numpy(boost), output_dtype=ax.fp32)
    return logits


def select_index_topk_batched(
    query_fp8,
    query_scale,
    gate,
    current_k_fp8,
    current_k_scale,
    index_k_cache_in,
    index_k_scale_in,
    selected_indices,
    *,
    seq_slice,
    cached_seq_len,
    index_topk,
    token_block_size,
    query_block_size,
    init_tokens,
    local_tokens,
    softmax_scale,
):
    """Select exact token-level Top-K from one private full-logit row per Query.

    Historical key tiles are loaded once per Query group. Invalid future positions
    remain ``-inf`` in the private OCM tensor, so no global dense-logit tensor is
    published to SHOCM or DDR.
    """
    local_seq_len = seq_slice.stop - seq_slice.start
    index_head_num = query_fp8.shape[-2]
    index_dim = query_fp8.shape[-1]
    total_seq_len = cached_seq_len + current_k_fp8.shape[1]
    query_bf16 = ax.dsl.cast(query_fp8, dtype=ax.bf16).reshape(
        local_seq_len,
        index_head_num,
        index_dim,
    )
    head_weight = ax.dsl.mul(
        gate,
        query_scale.reshape(local_seq_len, index_head_num),
        output_dtype=ax.fp32,
    )
    index_scale = ax.dsl.from_numpy(np.array([softmax_scale], dtype=np.float32))
    head_weight = ax.dsl.mul(head_weight, index_scale, output_dtype=ax.fp32)

    for query_start in range(0, local_seq_len, query_block_size):
        query_stop = min(query_start + query_block_size, local_seq_len)
        query_count = query_stop - query_start
        global_query_start = seq_slice.start + query_start
        valid_counts = [cached_seq_len + global_query_start + row + 1 for row in range(query_count)]
        q_block = query_bf16[query_start:query_stop, :, :]
        weight_block = head_weight[query_start:query_stop, :]
        history_logit_tiles = []

        # Reuse each immutable history tile across all Queries in the local group.
        for token_start in range(0, cached_seq_len, token_block_size):
            token_stop = min(token_start + token_block_size, cached_seq_len)
            key_tile = ax.dsl.load(index_k_cache_in[0:1, token_start:token_stop, :]).reshape(
                token_stop - token_start,
                index_dim,
            )
            key_scale_tile = ax.dsl.load(index_k_scale_in[0:1, token_start:token_stop, :]).reshape(
                token_stop - token_start
            )
            logits = index_tile_logits_batched(
                q_block,
                weight_block,
                key_tile,
                key_scale_tile,
            )
            logits = force_index_tokens_batched(
                logits,
                token_start,
                valid_counts,
                init_tokens,
                local_tokens,
            )
            history_logit_tiles.append(logits)

        if len(history_logit_tiles) == 1:
            history_logits = history_logit_tiles[0]
        elif history_logit_tiles:
            history_logits = ax.dsl.concat(*history_logit_tiles, dim=-1)
        else:
            history_logits = None

        # Current-token validity differs by Query. Keep this small causal tail
        # row-wise while the dominant history scan remains Query-batched.
        full_logit_rows = []
        for query_offset in range(query_count):
            global_query_index = global_query_start + query_offset
            local_query_index = query_start + query_offset
            current_valid = global_query_index + 1
            q_row = query_fp8[
                0:1,
                local_query_index : local_query_index + 1,
                :,
                :,
            ].reshape(index_head_num, index_dim)
            q_scale_row = query_scale[
                0:1,
                local_query_index : local_query_index + 1,
                :,
                :,
            ].reshape(index_head_num)
            gate_row = gate[local_query_index : local_query_index + 1, :].reshape(index_head_num)
            current_logit_tiles = []
            for current_start in range(0, current_valid, token_block_size):
                current_stop = min(current_start + token_block_size, current_valid)
                key_tile = current_k_fp8[0:1, current_start:current_stop, :].reshape(
                    current_stop - current_start,
                    index_dim,
                )
                key_scale_tile = current_k_scale[0:1, current_start:current_stop, :].reshape(
                    current_stop - current_start
                )
                logits = index_tile_logits(
                    q_row,
                    q_scale_row,
                    gate_row,
                    key_tile,
                    key_scale_tile,
                    softmax_scale,
                )
                token_start = cached_seq_len + current_start
                logits = force_index_tokens(
                    logits,
                    token_start,
                    cached_seq_len + current_valid,
                    init_tokens,
                    local_tokens,
                )
                current_logit_tiles.append(logits.reshape(1, current_stop - current_start))

            row_parts = []
            if history_logits is not None:
                row_parts.append(history_logits[query_offset : query_offset + 1, :])
            if len(current_logit_tiles) == 1:
                row_parts.append(current_logit_tiles[0])
            else:
                row_parts.append(ax.dsl.concat(*current_logit_tiles, dim=-1))
            invalid_count = current_k_fp8.shape[1] - current_valid
            if invalid_count > 0:
                row_parts.append(
                    ax.dsl.full(
                        (1, invalid_count),
                        ax.fp32,
                        fill_value=-float("inf"),
                        prefix=f"dsa_index_invalid_q{global_query_index}",
                    )
                )
            full_logit_rows.append(row_parts[0] if len(row_parts) == 1 else ax.dsl.concat(*row_parts, dim=-1))

        full_logits = full_logit_rows[0] if len(full_logit_rows) == 1 else ax.dsl.concat(*full_logit_rows, dim=0)
        topk_values = ax.dsl.create_tensor((query_count, index_topk), ax.fp32)
        topk_positions = ax.dsl.create_tensor((query_count, index_topk), ax.int32)
        ax.dsl.topk(
            full_logits,
            k=index_topk,
            dim=-1,
            r_values=topk_values,
            r_indices=topk_positions,
        )

        if valid_counts[0] >= index_topk:
            selected_block = topk_positions
        else:
            # Preserve the existing ``-1`` padding contract for early Queries
            # whose causal cache contains fewer than K valid positions.
            logical_positions = np.full((query_count, total_seq_len), -1, dtype=np.int32)
            for row, valid_count in enumerate(valid_counts):
                logical_positions[row, :valid_count] = np.arange(valid_count, dtype=np.int32)
            selected_block = ax.dsl.gather(
                ax.dsl.from_numpy(logical_positions),
                topk_positions,
                dim=-1,
            )

        ax.dsl.move(
            selected_block,
            r=selected_indices[
                0:1,
                global_query_start : global_query_start + query_count,
                0:1,
                :,
            ].reshape(query_count, index_topk),
        )


def lightning_indexer_prefill(
    x_norm,
    q_lora,
    position_ids,
    cos,
    sin,
    w_iq,
    w_scale_iq,
    w_scale2_iq,
    w_ik,
    w_scale_ik,
    w_scale2_ik,
    w_ig,
    w_scale_ig,
    w_scale2_ig,
    index_k_norm_weight,
    index_k_norm_bias,
    index_k_cache_out,
    index_k_scale_out,
    *,
    core_id,
    core_num,
    seq_slice,
    cached_seq_len,
    index_topk,
    rope_dim,
    index_k_cache_in=None,
    index_k_scale_in=None,
    index_k_norm_type=DSAIndexNorm.LAYER,
    use_hadamard=False,
    use_ue8m0=True,
    init_tokens=0,
    local_tokens=0,
    token_block_size=128,
    query_block_size=16,
    topk_indices_override=None,
):
    """Record the full SGLang-aligned contiguous-prefill lightning indexer.

    The current index-K tile and finalized indices are positional SHOCM tensors. Historical
    index-K remains an immutable FP8 DDR source. Query rows are sequence-sharded; each owner
    computes all index heads and writes disjoint Top-K rows.
    """
    index_k_norm_type = as_dsa_index_norm(index_k_norm_type)
    batch, seq_len, hidden_dim = x_norm.shape
    assert batch == 1, "the first contiguous DSA prefill implementation supports B=1"
    assert seq_len % core_num == 0, "sequence length must divide the active core count"
    assert seq_slice.stop - seq_slice.start == seq_len // core_num
    assert index_topk > 0
    assert token_block_size > 0
    assert query_block_size > 0
    assert w_ik.shape[1] == hidden_dim
    index_dim = w_ik.shape[0]
    index_head_num = w_ig.shape[0]
    assert w_iq.shape == (index_head_num * index_dim, q_lora.shape[-1])
    assert w_ig.shape[1] == hidden_dim
    assert index_k_norm_weight.shape[-1] == index_dim
    assert index_k_cache_out.shape[-1] == index_dim
    assert index_k_scale_out.shape[-1] == 1
    assert cached_seq_len + seq_len <= index_k_cache_out.shape[1]
    assert (index_k_cache_in is None) == (index_k_scale_in is None)
    if cached_seq_len > 0:
        assert index_k_cache_in.shape == (batch, cached_seq_len, index_dim)
        assert index_k_scale_in.shape == (batch, cached_seq_len, 1)

    shared_scope = ax.MemoryType.SHARED_OCM if core_num > 1 else ax.MemoryType.OCM
    current_k_fp8 = ax.dsl.create_tensor(
        (batch, seq_len, index_dim),
        ax.fp8_e4m3,
        shared_scope,
        name="dsa_current_index_k_fp8",
    )
    current_k_scale = ax.dsl.create_tensor(
        (batch, seq_len, 1),
        ax.fp32,
        shared_scope,
        name="dsa_current_index_k_scale",
    )
    selected_indices = ax.dsl.create_tensor(
        (batch, seq_len, 1, index_topk),
        ax.int32,
        shared_scope,
        name="dsa_internal_topk_indices",
    )

    # Publish the current index-K rows before any query scans the current tile.
    key_raw = ax.dsl.matmul(
        x_norm[:, seq_slice, :],
        w_ik,
        w_scale=w_scale_ik,
        w_scale2=w_scale2_ik,
        transposed=True,
        output_dtype=ax.bf16,
        const_inputs=["y"],
    )
    norm_weight = ax.dsl.load(index_k_norm_weight)
    norm_bias = ax.dsl.load(index_k_norm_bias) if index_k_norm_bias is not None else None
    key = index_norm(key_raw, norm_weight, norm_bias, index_k_norm_type)
    key = partial_rope(
        key,
        cos,
        sin,
        position_ids[seq_slice],
        rope_dim,
        f"dsa_index_k_rope_core{core_id}",
    )
    if use_hadamard:
        key = normalized_hadamard(key, f"dsa_index_k_hadamard_core{core_id}")
    key_fp8, key_scale = quantize_index_rows(key, use_ue8m0, f"dsa_index_k_core{core_id}")
    ax.dsl.move(key_fp8, r=current_k_fp8[:, seq_slice, :])
    ax.dsl.move(key_scale, r=current_k_scale[:, seq_slice, :])

    output_slice = slice(cached_seq_len + seq_slice.start, cached_seq_len + seq_slice.stop)
    ax.dsl.store(current_k_fp8[:, seq_slice, :], r=index_k_cache_out[:, output_slice, :])
    ax.dsl.store(current_k_scale[:, seq_slice, :], r=index_k_scale_out[:, output_slice, :])
    if core_id == 0 and cached_seq_len > 0:
        for start in range(0, cached_seq_len, token_block_size):
            stop = min(start + token_block_size, cached_seq_len)
            history_k = ax.dsl.load(index_k_cache_in[:, start:stop, :])
            history_scale = ax.dsl.load(index_k_scale_in[:, start:stop, :])
            ax.dsl.store(history_k, r=index_k_cache_out[:, start:stop, :])
            ax.dsl.store(history_scale, r=index_k_scale_out[:, start:stop, :])
    if core_num > 1:
        ax.parallel.sync_all()

    if topk_indices_override is not None:
        assert topk_indices_override.shape == selected_indices.shape
        override_slice = ax.dsl.load(topk_indices_override[:, seq_slice, :, :])
        ax.dsl.move(override_slice, r=selected_indices[:, seq_slice, :, :])
        if core_num > 1:
            ax.parallel.sync_all()
        return selected_indices

    total_seq_len = cached_seq_len + seq_len
    if total_seq_len <= index_topk:
        # The short path still updates the index-K cache but avoids Q/gate/logit work.
        for query_index in range(seq_slice.start, seq_slice.stop):
            valid_count = cached_seq_len + query_index + 1
            row = np.full((index_topk,), -1, dtype=np.int32)
            row[:valid_count] = np.arange(valid_count, dtype=np.int32)
            ax.dsl.move(
                ax.dsl.from_numpy(row),
                r=selected_indices[0:1, query_index : query_index + 1, 0:1, :].reshape(index_topk),
            )
        if core_num > 1:
            ax.parallel.sync_all()
        return selected_indices

    local_seq_len = seq_slice.stop - seq_slice.start
    query = ax.dsl.matmul(
        q_lora[:, seq_slice, :],
        w_iq,
        w_scale=w_scale_iq,
        w_scale2=w_scale2_iq,
        transposed=True,
        output_dtype=ax.bf16,
        const_inputs=["y"],
    ).reshape(batch, local_seq_len, index_head_num, index_dim)
    query = partial_rope(
        query,
        cos,
        sin,
        position_ids[seq_slice],
        rope_dim,
        f"dsa_index_q_rope_core{core_id}",
    )
    if use_hadamard:
        query = normalized_hadamard(query, f"dsa_index_q_hadamard_core{core_id}")
    query_fp8, query_scale = quantize_index_rows(query, use_ue8m0, f"dsa_index_q_core{core_id}")
    gate = ax.dsl.matmul(
        x_norm[:, seq_slice, :],
        w_ig,
        w_scale=w_scale_ig,
        w_scale2=w_scale2_ig,
        transposed=True,
        output_dtype=ax.bf16,
        const_inputs=["y"],
    ).reshape(local_seq_len, index_head_num)
    gate_scale = ax.dsl.from_numpy(np.array([index_head_num**-0.5], dtype=np.float32))
    gate = ax.dsl.mul(gate, gate_scale, output_dtype=ax.fp32)
    index_softmax_scale = index_dim**-0.5

    select_index_topk_batched(
        query_fp8,
        query_scale,
        gate,
        current_k_fp8,
        current_k_scale,
        index_k_cache_in,
        index_k_scale_in,
        selected_indices,
        seq_slice=seq_slice,
        cached_seq_len=cached_seq_len,
        index_topk=index_topk,
        token_block_size=token_block_size,
        query_block_size=min(query_block_size, local_seq_len),
        init_tokens=init_tokens,
        local_tokens=local_tokens,
        softmax_scale=index_softmax_scale,
    )
    if core_num > 1:
        ax.parallel.sync_all()
    return selected_indices


def block_dense_mla_attention(
    q,
    history_latent_cache,
    history_rope_cache,
    current_latent_cache,
    current_rope_cache,
    topk_indices,
    output,
    *,
    cached_seq_len,
    softmax_scale,
    query_block_size=32,
    cache_block_size=128,
    query_position_offset=0,
    materialize_query_group=False,
    name_prefix="",
):
    """按连续 KV block 记录数学等价的 token-sparse MLA attention。

    Indexer 的输出仍是逐 Query 的 token Top-K。本实现将一个 Query Group 的 Top-K
    scatter 成 ``[Bq, C + S]`` 选择 mask，再流式扫描连续历史 DDR block 和当前
    SHOCM block。QK 将连续的 ``[Bq,H,Rkv+Dr]`` Query 展平到 ``Bq*H`` 行，
    softmax/PV 仍按四头分组。未选择、越界和违反因果关系的 token 概率严格置零，
    因此生产 Indexer 输出唯一索引时与 token gather 路径数学等价。

    该验证版本不执行运行时 block sort/unique，因此不会跳过 Query Group 完全未
    命中的 cache block。它用于隔离连续 block IO、Query Group MatMul 和 online
    softmax 的精度与性能；真正的稀疏 block planner 需要后端动态 primitive。

    Args:
        q: BF16 private OCM，shape ``[S, H, Rkv + Dr]``。
        history_latent_cache: BF16 immutable DDR，shape ``[C, Rkv]``；``C=0`` 时为 None。
        history_rope_cache: BF16 immutable DDR，shape ``[C, Dr]``；与 latent history 同时存在。
        current_latent_cache: BF16 SHOCM，shape ``[S, Rkv]``。
        current_rope_cache: BF16 SHOCM，shape ``[S, Dr]``。
        topk_indices: INT32，shape ``[S, K]``；生产路径要求每行有效索引唯一。
        output: BF16 private OCM，shape ``[S, H, Rkv]``。
        cached_seq_len: Python int，历史 token 数 ``C``。
        softmax_scale: Python float，attention score scale。
        query_block_size: Python int，同时计算的 Query 数 ``Bq``；默认 32 是
            ``S=128,C=4096,K=2048`` 四核 case 的已验证策略。
        cache_block_size: Python int，每次连续加载的 cache token 数 ``Bc``。
        query_position_offset: Python int，``q`` 在当前 prefill tile 中的起始
            token 位置；分块生产 Q 时用于恢复全局因果上界。
        materialize_query_group: 是否将当前 ``Bq`` 的 Q 和组合 K tile 搬到
            带 ``core_num`` 前导维的 SHOCM workspace；每个 core 只访问自己的
            slice，用于隔离上游布局并避免 private OCM 碎片影响 QK 调度。
        name_prefix: 临时 tensor 的稳定命名后缀。

    Returns:
        None；结果写入 ``output``。
    """
    seq_len, head_num, q_width = q.shape
    latent_rank = current_latent_cache.shape[-1]
    rope_dim = current_rope_cache.shape[-1]
    current_seq_len = current_latent_cache.shape[0]
    total_seq_len = cached_seq_len + current_seq_len
    assert q_width == latent_rank + rope_dim
    assert output.shape == (seq_len, head_num, latent_rank)
    assert topk_indices.shape[0] == seq_len
    assert topk_indices.shape[-1] > 0
    assert query_block_size > 0
    assert cache_block_size > 0
    assert query_position_offset >= 0
    assert query_position_offset + seq_len <= current_seq_len
    assert current_latent_cache.shape == (current_seq_len, latent_rank)
    assert current_rope_cache.shape == (current_seq_len, rope_dim)
    if cached_seq_len > 0:
        assert history_latent_cache.shape == (cached_seq_len, latent_rank)
        assert history_rope_cache.shape == (cached_seq_len, rope_dim)
    else:
        assert history_latent_cache is None
        assert history_rope_cache is None

    zero_i32 = ax.dsl.from_numpy(np.array([0], dtype=np.int32))
    one_i32 = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    total_i32 = ax.dsl.from_numpy(np.array([total_seq_len], dtype=np.int32))
    neg_mask = ax.dsl.from_numpy(np.array([-3.0e4], dtype=np.float32))
    scale = ax.dsl.from_numpy(np.array([softmax_scale], dtype=np.float32))
    matmul_head_block = 4
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    multi_core = core_num > 1

    cache_sources = []
    if cached_seq_len > 0:
        cache_sources.append((history_latent_cache, history_rope_cache, 0, cached_seq_len, True))
    cache_sources.append(
        (
            current_latent_cache,
            current_rope_cache,
            cached_seq_len,
            current_seq_len,
            False,
        )
    )

    for query_start in range(0, seq_len, query_block_size):
        query_stop = min(query_start + query_block_size, seq_len)
        query_count = query_stop - query_start
        query_indices = topk_indices[query_start:query_stop, :]
        if topk_indices.memory_type == ax.MemoryType.DDR:
            query_indices = ax.dsl.load(query_indices)

        # Scatter 只接受安全位置。无效位置统一写入末尾哨兵列，避免 clamp 到
        # 0/T-1 后覆盖同一行中已经写入的有效 token。
        causal_limit = ax.dsl.from_numpy(
            np.arange(
                cached_seq_len + query_position_offset + query_start + 1,
                cached_seq_len + query_position_offset + query_stop + 1,
                dtype=np.int32,
            ).reshape(query_count, 1)
        )
        lower_valid = ax.dsl.minimum(ax.dsl.maximum(query_indices + one_i32, zero_i32), one_i32)
        upper_valid = ax.dsl.minimum(ax.dsl.maximum(total_i32 - query_indices, zero_i32), one_i32)
        causal_valid = ax.dsl.minimum(ax.dsl.maximum(causal_limit - query_indices, zero_i32), one_i32)
        valid = lower_valid * upper_valid * causal_valid
        clamped_indices = ax.dsl.minimum(ax.dsl.maximum(query_indices, zero_i32), total_i32 - one_i32)
        safe_indices = valid * clamped_indices + (one_i32 - valid) * total_i32
        selected_mask_with_sentinel = ax.dsl.full(
            (query_count, total_seq_len + 1),
            ax.bf16,
            fill_value=0,
        )
        ax.dsl.scatter(
            ax.dsl.full(query_indices.shape, ax.bf16, fill_value=1),
            safe_indices,
            dim=1,
            r=selected_mask_with_sentinel,
        )
        selected_mask = selected_mask_with_sentinel[:, :total_seq_len]

        head_ranges = [
            (head_start, min(head_start + matmul_head_block, head_num))
            for head_start in range(0, head_num, matmul_head_block)
        ]
        # 完整 head 维连续，reshape 只是 OCM alias。将 Query 与 head 合并到
        # MatMul 的 M 维，避免后端把 Bq 记录成大量 width=4 的小 QK。
        q_group_2d = q[query_start:query_stop, :, :].reshape(query_count * head_num, q_width)
        if materialize_query_group:
            if multi_core:
                q_group_workspace = ax.dsl.create_tensor(
                    (core_num, query_count * head_num, q_width),
                    q.dtype,
                    ax.MemoryType.SHARED_OCM,
                    name=f"dsa_q_group_workspace{name_prefix}_q{query_start}",
                )
                q_group_compact = q_group_workspace[core_id : core_id + 1, :, :].reshape(*q_group_2d.shape)
            else:
                q_group_compact = ax.dsl.create_tensor(
                    q_group_2d.shape,
                    q.dtype,
                    name=f"dsa_q_group_compact{name_prefix}_q{query_start}",
                )
            ax.dsl.move(q_group_2d, r=q_group_compact)
            q_group_2d = q_group_compact
        max_running = []
        sum_running = []
        output_running = []
        for head_start, head_stop in head_ranges:
            head_count = head_stop - head_start
            max_running.append(
                ax.dsl.full(
                    (query_count, head_count),
                    ax.fp32,
                    fill_value=-3.0e4,
                )
            )
            sum_running.append(ax.dsl.full((query_count, head_count), ax.fp32, fill_value=0))
            output_running.append(
                ax.dsl.full(
                    (query_count, head_count, latent_rank),
                    q.dtype,
                    fill_value=0,
                )
            )

        for (
            latent_cache,
            rope_cache,
            global_offset,
            source_len,
            source_is_ddr,
        ) in cache_sources:
            for cache_start in range(0, source_len, cache_block_size):
                cache_stop = min(cache_start + cache_block_size, source_len)
                block_count = cache_stop - cache_start
                latent_block = latent_cache[cache_start:cache_stop, :]
                rope_block = rope_cache[cache_start:cache_stop, :]
                if source_is_ddr:
                    latent_block = ax.dsl.load(latent_block)
                    rope_block = ax.dsl.load(rope_block)

                if materialize_query_group and multi_core:
                    key_block_workspace = ax.dsl.create_tensor(
                        (core_num, block_count, latent_rank + rope_dim),
                        q.dtype,
                        ax.MemoryType.SHARED_OCM,
                        name=f"dsa_block_k_workspace{name_prefix}_q{query_start}_t{global_offset + cache_start}",
                    )
                    key_block = key_block_workspace[core_id : core_id + 1, :, :].reshape(
                        block_count, latent_rank + rope_dim
                    )
                else:
                    key_block = ax.dsl.create_tensor(
                        (block_count, latent_rank + rope_dim),
                        q.dtype,
                        name=f"dsa_block_k{name_prefix}_q{query_start}_t{global_offset + cache_start}",
                    )
                ax.dsl.move(latent_block, r=key_block[:, :latent_rank])
                ax.dsl.move(rope_block, r=key_block[:, latent_rank:])
                latent_transposed = ax.dsl.create_tensor(
                    (latent_rank, block_count),
                    q.dtype,
                    name=f"dsa_block_vt{name_prefix}_q{query_start}_t{global_offset + cache_start}",
                )
                ax.dsl.transpose(latent_block, perm=(1, 0), r=latent_transposed)

                token_start = global_offset + cache_start
                token_stop = global_offset + cache_stop
                selected_block = selected_mask[:, token_start:token_stop]
                invalid_fp32 = ax.dsl.cast(one_i32 - ax.dsl.cast(selected_block, dtype=ax.int32), dtype=ax.fp32)
                mask = ax.dsl.mul(invalid_fp32, neg_mask, output_dtype=ax.fp32)
                valid_fp32 = ax.dsl.cast(selected_block, dtype=ax.fp32).reshape(query_count, 1, block_count)

                score_all = ax.dsl.matmul(
                    q_group_2d,
                    key_block,
                    transposed=True,
                    output_dtype=ax.fp32,
                ).reshape(query_count, head_num, block_count)
                score_all = ax.dsl.mul(score_all, scale, output_dtype=ax.fp32)
                score_all = ax.dsl.add(
                    score_all,
                    mask.reshape(query_count, 1, block_count),
                    output_dtype=ax.fp32,
                )

                for head_block_index, (head_start, head_stop) in enumerate(head_ranges):
                    head_count = head_stop - head_start
                    # Softmax/PV 继续按四头分组；只合并共享 K 的 QK 阶段。
                    score = score_all[:, head_start:head_stop, :]
                    block_max = ax.dsl.max(score, dim=-1)
                    previous_max = max_running[head_block_index]
                    new_max = ax.dsl.maximum(previous_max, block_max)
                    previous_coefficient = ax.dsl.exp(previous_max - new_max, output_dtype=ax.fp32)
                    probability_fp32 = ax.dsl.exp(
                        score - new_max.reshape(query_count, head_count, 1),
                        output_dtype=ax.fp32,
                    )
                    probability_fp32 = ax.dsl.mul(
                        probability_fp32,
                        valid_fp32,
                        output_dtype=ax.fp32,
                    )
                    block_sum = ax.dsl.sum(probability_fp32, dim=-1, output_dtype=ax.fp32)

                    ax.dsl.mul(
                        sum_running[head_block_index],
                        previous_coefficient,
                        r=sum_running[head_block_index],
                    )
                    ax.dsl.add(
                        sum_running[head_block_index],
                        block_sum,
                        r=sum_running[head_block_index],
                    )
                    ax.dsl.mul(
                        output_running[head_block_index],
                        previous_coefficient.reshape(query_count, head_count, 1),
                        r=output_running[head_block_index],
                    )
                    block_output = ax.dsl.matmul(
                        ax.dsl.cast(probability_fp32, dtype=q.dtype),
                        latent_transposed,
                        transposed=True,
                        output_dtype=q.dtype,
                    )
                    ax.dsl.add(
                        output_running[head_block_index],
                        block_output,
                        r=output_running[head_block_index],
                    )
                    max_running[head_block_index] = new_max

        for head_block_index, (head_start, head_stop) in enumerate(head_ranges):
            head_count = head_stop - head_start
            ax.dsl.div(
                output_running[head_block_index],
                sum_running[head_block_index].reshape(query_count, head_count, 1),
                r=output[query_start:query_stop, head_start:head_stop, :],
            )


def sparse_mla_attention(
    q,
    history_latent_cache,
    history_rope_cache,
    current_latent_cache,
    current_rope_cache,
    topk_indices,
    output,
    *,
    cached_seq_len,
    softmax_scale,
    selected_block_size=128,
    name_prefix="",
):
    """记录单 batch、本地 head 分片的 token-level sparse MLA attention 图。

    Shape 记号：``S`` 为当前 query 长度，``H`` 为当前 core 的本地 head 数，
    ``C`` 为历史 cache 长度，``K`` 为每个 query 的候选数，``Rkv`` 为 latent
    rank，``Dr`` 为 RoPE 维度。当前实现要求 activation/cache/output 为 BF16，
    score、softmax max/sum 使用 FP32。同一 Query 的全部本地 head 共享 selected
    KV，并合并为一个 2D QK/softmax/PV 计算组；不同 Query 的索引不同，不跨 Query
    合并。

    Args:
        q: BF16、private OCM，shape 为 ``[S, H, Rkv + Dr]``；已经完成
            query absorption 和 RoPE。
        history_latent_cache: BF16、immutable DDR，shape 为 ``[C, Rkv]``；
            ``C == 0`` 时必须为 ``None``。
        history_rope_cache: BF16、immutable DDR，shape 为 ``[C, Dr]``；
            必须与 ``history_latent_cache`` 同时存在或同时为 ``None``。
        current_latent_cache: BF16、SHOCM，shape 为 ``[S, Rkv]``；保存当前
            prefill tile 的 latent cache。
        current_rope_cache: BF16、SHOCM，shape 为 ``[S, Dr]``；保存当前
            prefill tile 的 RoPE cache。
        topk_indices: INT32，shape 为 ``[S, K]``。生产路径使用 Indexer 写入的
            SHOCM；独立 sparse-attention 调试允许使用 DDR 输入。元素是 logical
            cache ``[0, C + S)`` 的全局 token index，负数和越界值作为 padding。
        output: BF16、private OCM 输出，shape 为 ``[S, H, Rkv]``；由本函数
            原地写入。
        cached_seq_len: Python ``int``，历史 token 数 ``C``，参与静态图记录。
        softmax_scale: Python ``float``，记录为 FP32 attention score scale。
        selected_block_size: Python ``int``，每次物化的候选 token 上限 ``Kb``，
            默认 128。
        name_prefix: Python ``str``，循环中临时 tensor 的稳定命名后缀。

    Returns:
        ``None``；结果写入 ``output``。
    """
    seq_len, head_num, q_width = q.shape
    assert topk_indices.shape[0] == seq_len
    latent_rank = current_latent_cache.shape[-1]
    rope_dim = current_rope_cache.shape[-1]
    assert output.shape == (seq_len, head_num, latent_rank)
    assert q_width == latent_rank + rope_dim
    assert current_latent_cache.shape[0] == seq_len
    assert current_rope_cache.shape == (seq_len, rope_dim)
    if cached_seq_len > 0:
        assert history_latent_cache is not None
        assert history_rope_cache is not None
        assert history_latent_cache.shape == (cached_seq_len, latent_rank)
        assert history_rope_cache.shape == (cached_seq_len, rope_dim)
    else:
        assert history_latent_cache is None
        assert history_rope_cache is None

    selected_count = topk_indices.shape[-1]
    assert selected_count > 0
    assert selected_block_size > 0
    single_selected_block = selected_count <= selected_block_size
    total_seq_len = cached_seq_len + seq_len

    zero_i32 = ax.dsl.from_numpy(np.array([0], dtype=np.int32))
    one_i32 = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    cached_i32 = ax.dsl.from_numpy(np.array([cached_seq_len], dtype=np.int32))
    total_i32 = ax.dsl.from_numpy(np.array([total_seq_len], dtype=np.int32))
    current_last_i32 = ax.dsl.from_numpy(np.array([seq_len - 1], dtype=np.int32))
    neg_mask = ax.dsl.from_numpy(np.array([-3.0e4], dtype=np.float32))
    scale = ax.dsl.from_numpy(np.array([softmax_scale], dtype=np.float32))
    for q_index in range(seq_len):
        causal_limit = ax.dsl.from_numpy(np.array([cached_seq_len + q_index + 1], dtype=np.int32))
        q_row = q[q_index : q_index + 1, :, :]
        # Token gather 的 selected K 每个 Query 不同，不能跨 Query 合并；但同一
        # Query 的所有本地 head 共享 K，可零复制展开成一个 2D QK。
        q_row_2d = q_row.reshape(head_num, q_width)
        max_running = None
        sum_running = None
        output_running = None

        for selected_start in range(0, selected_count, selected_block_size):
            selected_stop = min(selected_start + selected_block_size, selected_count)
            block_count = selected_stop - selected_start
            index_block = topk_indices[q_index : q_index + 1, selected_start:selected_stop]
            if topk_indices.memory_type == ax.MemoryType.DDR:
                index_block = ax.dsl.load(index_block)
            indices = index_block.reshape(block_count)

            # 原始 index 负责有效性和因果判断；clamp 后的副本只用于安全 gather。
            lower_valid = ax.dsl.minimum(ax.dsl.maximum(indices + one_i32, zero_i32), one_i32)
            upper_valid = ax.dsl.minimum(ax.dsl.maximum(total_i32 - indices, zero_i32), one_i32)
            causal_valid = ax.dsl.minimum(ax.dsl.maximum(causal_limit - indices, zero_i32), one_i32)
            valid = lower_valid * upper_valid * causal_valid

            # history DDR 与 current SHOCM 分别 gather，再在 private OCM 合并。
            current_indices = indices - cached_i32
            safe_current_indices = ax.dsl.minimum(
                ax.dsl.maximum(current_indices, zero_i32),
                current_last_i32,
            )
            is_current = ax.dsl.minimum(
                ax.dsl.maximum(current_indices + one_i32, zero_i32),
                one_i32,
            )
            selected_latent = ax.dsl.create_tensor(
                (block_count, latent_rank),
                current_latent_cache.dtype,
                name=f"dsa_latent{name_prefix}_q{q_index}_k{selected_start}",
            )
            selected_rope = ax.dsl.create_tensor(
                (block_count, rope_dim),
                current_rope_cache.dtype,
                name=f"dsa_rope{name_prefix}_q{q_index}_k{selected_start}",
            )
            ax.dsl.index_select(
                current_latent_cache,
                safe_current_indices,
                dim=0,
                r=selected_latent,
            )
            ax.dsl.index_select(
                current_rope_cache,
                safe_current_indices,
                dim=0,
                r=selected_rope,
            )

            if cached_seq_len > 0:
                history_last_i32 = ax.dsl.from_numpy(np.array([cached_seq_len - 1], dtype=np.int32))
                safe_history_indices = ax.dsl.minimum(
                    ax.dsl.maximum(indices, zero_i32),
                    history_last_i32,
                )
                selected_history_latent = ax.dsl.create_tensor(
                    (block_count, latent_rank),
                    current_latent_cache.dtype,
                    name=f"dsa_history_latent{name_prefix}_q{q_index}_k{selected_start}",
                )
                selected_history_rope = ax.dsl.create_tensor(
                    (block_count, rope_dim),
                    current_rope_cache.dtype,
                    name=f"dsa_history_rope{name_prefix}_q{q_index}_k{selected_start}",
                )
                ax.dsl.index_select(
                    history_latent_cache,
                    safe_history_indices,
                    dim=0,
                    r=selected_history_latent,
                )
                ax.dsl.index_select(
                    history_rope_cache,
                    safe_history_indices,
                    dim=0,
                    r=selected_history_rope,
                )
                current_weight = ax.dsl.cast(is_current, dtype=selected_latent.dtype).reshape(block_count, 1)
                history_weight = ax.dsl.cast(
                    one_i32 - is_current,
                    dtype=selected_latent.dtype,
                ).reshape(block_count, 1)
                ax.dsl.mul(selected_latent, current_weight, r=selected_latent)
                ax.dsl.mul(selected_rope, current_weight, r=selected_rope)
                # 非 alias Mul 输出可触发现有 Mul-Add -> FMA 图融合。
                weighted_history_latent = ax.dsl.mul(selected_history_latent, history_weight)
                weighted_history_rope = ax.dsl.mul(selected_history_rope, history_weight)
                ax.dsl.add(selected_latent, weighted_history_latent, r=selected_latent)
                ax.dsl.add(selected_rope, weighted_history_rope, r=selected_rope)

            # QK 使用拼接后的 K；PV 复用一次转置后的 contiguous latent。
            selected_k = ax.dsl.create_tensor(
                (block_count, latent_rank + rope_dim),
                current_latent_cache.dtype,
                name=f"dsa_k{name_prefix}_q{q_index}_k{selected_start}",
            )
            ax.dsl.move(selected_latent, r=selected_k[:, :latent_rank])
            ax.dsl.move(selected_rope, r=selected_k[:, latent_rank:])
            selected_latent_transposed = ax.dsl.create_tensor(
                (latent_rank, block_count),
                selected_latent.dtype,
                name=f"dsa_latent_t{name_prefix}_q{q_index}_k{selected_start}",
            )
            ax.dsl.transpose(selected_latent, perm=(1, 0), r=selected_latent_transposed)

            invalid = one_i32 - valid
            invalid_fp32 = ax.dsl.cast(invalid, dtype=ax.fp32)
            mask = ax.dsl.mul(invalid_fp32, neg_mask, output_dtype=ax.fp32)

            score_all_2d = ax.dsl.matmul(
                q_row_2d,
                selected_k,
                transposed=True,
                output_dtype=ax.fp32,
            )
            score_all_2d = ax.dsl.mul(score_all_2d, scale, output_dtype=ax.fp32)
            score_all_2d = ax.dsl.add(
                score_all_2d,
                mask.reshape(1, block_count),
                output_dtype=ax.fp32,
            )
            if selected_start == 0 and single_selected_block:
                # 单 block 不保留 max 状态，在完整 all-head tensor 上融合
                # reduce_max_sub，避免对子 view 写回产生重复 spill/free。
                block_max_all_2d = ax.dsl.max(score_all_2d, dim=-1, keepdim=True)
                ax.dsl.sub(score_all_2d, block_max_all_2d, r=score_all_2d)
                shifted_score_2d = score_all_2d
            else:
                block_max_2d = ax.dsl.max(score_all_2d, dim=-1)
                block_max = block_max_2d.reshape(1, head_num)
                shifted_score_2d = score_all_2d - block_max_2d.reshape(head_num, 1)

            if block_count * 16 >= 256:
                # exp_sum 是 2D 算子；完整 [H, Kb] view 同时覆盖所有本地 head。
                probability_bf16_2d = ax.dsl.exp(shifted_score_2d, output_dtype=ax.bf16)
                probability_fp32_2d = ax.dsl.exp(shifted_score_2d, output_dtype=ax.fp32)
                block_sum_1d = ax.dsl.sum(probability_fp32_2d, dim=-1, output_dtype=ax.fp32)
            else:
                # exp_sum 的 BF16 Exp 流至少需要一个 256-bit 归约块。
                probability_fp32_2d = ax.dsl.exp(shifted_score_2d, output_dtype=ax.fp32)
                block_sum_1d = ax.dsl.sum(probability_fp32_2d, dim=-1, output_dtype=ax.fp32)
                probability_bf16_2d = ax.dsl.cast(probability_fp32_2d, dtype=ax.bf16)

            probability_bf16 = probability_bf16_2d.reshape(1, head_num, block_count)
            block_sum = block_sum_1d.reshape(1, head_num)

            if selected_start == 0:
                # 首 block 无历史贡献，直接初始化 all-head online-softmax 状态。
                if not single_selected_block:
                    max_running = block_max
                sum_running = block_sum
                output_running = ax.dsl.matmul(
                    probability_bf16,
                    selected_latent_transposed,
                    transposed=True,
                    output_dtype=ax.bf16,
                )
                continue

            previous_max = ax.dsl.create_tensor(max_running.shape, dtype=max_running.dtype)
            ax.dsl.move(max_running, r=previous_max)
            max_running = ax.dsl.maximum(max_running, block_max, r=max_running)

            previous_coefficient = ax.dsl.exp(previous_max - max_running, output_dtype=ax.fp32)
            block_coefficient = ax.dsl.exp(block_max - max_running, output_dtype=ax.fp32)

            sum_running = ax.dsl.mul(sum_running, previous_coefficient, r=sum_running)
            scaled_block_sum = ax.dsl.mul(block_sum, block_coefficient, output_dtype=ax.fp32)
            sum_running = ax.dsl.add(sum_running, scaled_block_sum, r=sum_running)

            output_running = ax.dsl.mul(
                output_running,
                previous_coefficient.reshape(1, head_num, 1),
                r=output_running,
            )
            block_output = ax.dsl.matmul(
                probability_bf16,
                selected_latent_transposed,
                transposed=True,
                output_dtype=ax.bf16,
            )
            block_output = ax.dsl.mul(
                block_output,
                block_coefficient.reshape(1, head_num, 1),
                output_dtype=ax.bf16,
            )
            output_running = ax.dsl.add(output_running, block_output, r=output_running)

        ax.dsl.div(
            output_running,
            sum_running.reshape(1, head_num, 1),
            r=output[q_index : q_index + 1, :, :],
        )


def dsa_prefill_impl(
    x,  # [batch_size, max_seq_len, hidden_dim]
    seq_lens,  # [batch_size] int32, per-batch runtime query lengths
    w_attn_norm,  # [1, hidden_dim]
    w_dq,  # [q_lora_rank, hidden_dim]
    w_scale_dq,
    w_scale2_dq,
    w_uq,  # [q_head_num * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank]
    w_scale_uq,
    w_scale2_uq,
    w_dkv,  # [kv_lora_rank + qk_rope_head_dim, hidden_dim]
    w_scale_dkv,
    w_scale2_dkv,
    w_kv_b_0,  # [q_head_num, kv_lora_rank, qk_nope_head_dim]
    w_scale_kv_b_0,
    w_scale2_kv_b_0,
    w_kv_b_1,  # [q_head_num, v_head_dim, kv_lora_rank]
    w_scale_kv_b_1,
    w_scale2_kv_b_1,
    w_q_norm,  # [1, q_lora_rank]
    w_kv_norm,  # [1, kv_lora_rank]
    w_o,  # [hidden_dim, q_head_num * v_head_dim]
    w_scale_o,
    w_scale2_o,
    cos,  # [max_seq_len, qk_rope_head_dim//2]
    sin,  # [max_seq_len, qk_rope_head_dim//2]
    position_ids,  # [batch_size, seq_len]
    kv_cache_out,  # [batch_size, seq_len, kv_lora_rank]
    pe_cache_out,  # [batch_size, seq_len, qk_rope_head_dim]
    # attn_out,  # [B, seq_len, q_head_num, v_head_dim] debug
    output,
    softmax_scale=1.0,
    kv_offset=0,
    kv_cache_in=None,  # [B, CACHED_SEQ_LEN, kv_lora_rank]
    pe_cache_in=None,  # [B, CACHED_SEQ_LEN, qk_rope_head_dim]
    topk_indices=None,  # [B, seq_len, 1, selected_count]
    w_iq=None,  # [index_head_num * index_dim, q_lora_rank]
    w_scale_iq=None,
    w_scale2_iq=None,
    w_ik=None,  # [index_dim, hidden_dim]
    w_scale_ik=None,
    w_scale2_ik=None,
    w_ig=None,  # [index_head_num, hidden_dim]
    w_scale_ig=None,
    w_scale2_ig=None,
    index_k_norm_weight=None,  # [index_dim]
    index_k_norm_bias=None,  # [index_dim], required by LayerNorm
    index_k_cache_out=None,  # FP8 [B, max_seq_len, index_dim]
    index_k_scale_out=None,  # FP32 [B, max_seq_len, 1]
    index_k_cache_in=None,  # FP8 [B, cached_seq_len, index_dim]
    index_k_scale_in=None,  # FP32 [B, cached_seq_len, 1]
    index_topk=2048,
    index_rope_dim=64,
    index_k_norm_type=DSAIndexNorm.LAYER,
    index_use_hadamard=False,
    index_use_ue8m0=True,
    index_init_tokens=0,
    index_local_tokens=0,
    index_token_block_size=128,
    index_query_block_size=16,
    sparse_attention_impl=DSASparseAttentionImpl.TOKEN,
    sparse_query_block_size=32,
    sparse_cache_block_size=128,
):
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    multi_core = core_num > 1
    index_k_norm_type = as_dsa_index_norm(index_k_norm_type)
    sparse_attention_impl = as_dsa_sparse_impl(sparse_attention_impl)
    w_dtype_type = datatype.DType(w_dq.dtype).type
    MAX_SEQ_LEN = kv_cache_out.shape[1]
    B, seq_len, hidden_dim = x.shape
    v_head_dim = w_kv_b_1.shape[1]
    # seq_len multiple core slicing
    seq_len_factor = (seq_len + core_num - 1) // core_num
    slice_seq_len = slice(
        min(core_id * seq_len_factor, seq_len - 1),
        min((core_id + 1) * seq_len_factor, seq_len),
    )
    # q_lora_rank multiple core slicing
    q_lora_rank = w_dq.shape[0]
    q_lora_rank_factor = (q_lora_rank + core_num - 1) // core_num
    slice_q_lora_rank = slice(
        min(core_id * q_lora_rank_factor, q_lora_rank - 1),
        min((core_id + 1) * q_lora_rank_factor, q_lora_rank),
    )
    # q_head_num multiple core slicing
    q_head_num, kv_lora_rank, qk_nope_head_dim = w_kv_b_0.shape

    kv_pe_dim = w_dkv.shape[0]
    qk_rope_head_dim = kv_pe_dim - kv_lora_rank
    q_head_num_factor = (q_head_num + core_num - 1) // core_num
    slice_q_head_num = slice(
        min(core_id * q_head_num_factor, q_head_num - 1),
        min((core_id + 1) * q_head_num_factor, q_head_num),
    )
    q_head_num_cur_core = slice_q_head_num.stop - slice_q_head_num.start
    assert q_head_num_cur_core > 0, "q_head_num_cur_core must be positive"
    # kv_pe_dim multiple core slicing
    kv_pe_dim_factor = (kv_pe_dim + core_num - 1) // core_num
    slice_kv_pe_dim = slice(
        min(core_id * kv_pe_dim_factor, kv_pe_dim - 1),
        min((core_id + 1) * kv_pe_dim_factor, kv_pe_dim),
    )

    # Slice the kv_cache and pe_cache for current output seq_len
    assert (
        kv_offset + seq_len <= MAX_SEQ_LEN
    ), f"kv_offset + seq_len must <= MAX_SEQ_LEN, but {kv_offset=}, {seq_len=}, {MAX_SEQ_LEN=}"
    _kv_cache_out = kv_cache_out[:, kv_offset : kv_offset + seq_len, :]
    _pe_cache_out = pe_cache_out[:, kv_offset : kv_offset + seq_len, :]
    internal_indexer = w_iq is not None
    sparse_attention = topk_indices is not None or internal_indexer
    cached_seq_len = kv_cache_in.shape[1] if kv_cache_in is not None else 0
    if sparse_attention:
        if topk_indices is not None:
            assert len(topk_indices.shape) == 4
            assert topk_indices.shape[:3] == (B, seq_len, 1)
            assert topk_indices.shape[-1] > 0
            assert topk_indices.dtype == ax.int32
        assert (kv_cache_in is None) == (pe_cache_in is None)
        if pe_cache_in is not None:
            assert pe_cache_in.shape[:2] == (B, cached_seq_len)
        assert kv_offset == cached_seq_len
    if internal_indexer:
        assert all(
            value is not None
            for value in (
                w_scale_iq,
                w_scale2_iq,
                w_ik,
                w_scale_ik,
                w_scale2_ik,
                w_ig,
                w_scale_ig,
                w_scale2_ig,
                index_k_norm_weight,
                index_k_cache_out,
                index_k_scale_out,
            )
        )
        assert (index_k_cache_in is None) == (index_k_scale_in is None)
        if cached_seq_len > 0:
            assert index_k_cache_in is not None
            assert index_k_cache_in.shape[1] == cached_seq_len

    x_mem_scope = ax.MemoryType.SHARED_OCM if multi_core else ax.MemoryType.OCM
    (
        x_shocm,
        x_norm_shocm,
        r_norm_dq,
        r_kv_norm_pe_shocm,
        position_ids,
        seq_len_mask_i32,
    ) = record_mla_shared_projections(
        x,
        w_attn_norm,
        w_dq,
        w_scale_dq,
        w_scale2_dq,
        w_dkv,
        w_scale_dkv,
        w_scale2_dkv,
        w_q_norm,
        w_kv_norm,
        cos,
        sin,
        position_ids,
        core_id=core_id,
        core_num=core_num,
        seq_slice=slice_seq_len,
        q_lora_slice=slice_q_lora_rank,
        kv_pe_slice=slice_kv_pe_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        seq_lens=seq_lens,
        name_prefix=f"dsa_core{core_id}",
    )
    r_kv_norm_shocm = r_kv_norm_pe_shocm[:, :, :kv_lora_rank]
    r_pe_rope_shocm = r_kv_norm_pe_shocm[:, :, kv_lora_rank:]

    if internal_indexer:
        internal_topk_indices = lightning_indexer_prefill(
            x_norm_shocm,
            r_norm_dq,
            position_ids,
            cos,
            sin,
            w_iq,
            w_scale_iq,
            w_scale2_iq,
            w_ik,
            w_scale_ik,
            w_scale2_ik,
            w_ig,
            w_scale_ig,
            w_scale2_ig,
            index_k_norm_weight,
            index_k_norm_bias,
            index_k_cache_out,
            index_k_scale_out,
            core_id=core_id,
            core_num=core_num,
            seq_slice=slice_seq_len,
            cached_seq_len=cached_seq_len,
            index_topk=index_topk,
            rope_dim=index_rope_dim,
            index_k_cache_in=index_k_cache_in,
            index_k_scale_in=index_k_scale_in,
            index_k_norm_type=index_k_norm_type,
            use_hadamard=index_use_hadamard,
            use_ue8m0=index_use_ue8m0,
            init_tokens=index_init_tokens,
            local_tokens=index_local_tokens,
            token_block_size=index_token_block_size,
            query_block_size=index_query_block_size,
            topk_indices_override=topk_indices,
        )
        # The Indexer always publishes one on-chip handoff. A debug override
        # bypasses logits/Top-K but retains current index-K cache production.
        topk_indices = internal_topk_indices

    if core_id == 0 and sparse_attention and cached_seq_len > 0:
        # Preserve the public combined-cache output contract. Sparse attention
        # reads immutable history and current SHOCM directly; it never reads
        # this launch's DDR output.
        cache_copy_block = 64
        for b in range(B):
            for cache_start in range(0, cached_seq_len, cache_copy_block):
                cache_stop = min(cache_start + cache_copy_block, cached_seq_len)
                kv_cache_block = ax.dsl.load(kv_cache_in[b : b + 1, cache_start:cache_stop, :])
                pe_cache_block = ax.dsl.load(pe_cache_in[b : b + 1, cache_start:cache_stop, :])
                ax.dsl.store(
                    x=kv_cache_block,
                    r=kv_cache_out[b : b + 1, cache_start:cache_stop, :],
                )
                ax.dsl.store(
                    x=pe_cache_block,
                    r=pe_cache_out[b : b + 1, cache_start:cache_stop, :],
                )

    if core_id == 0:
        # 将 r_kv_norm_shocm 存入 kv_cache_out
        ax.dsl.store(r=_kv_cache_out[:, :seq_len, :], x=r_kv_norm_shocm)
        # 将 r_pe_rope_shocm 存入 pe_cache_out
        ax.dsl.store(r=_pe_cache_out[:, :seq_len, :], x=r_pe_rope_shocm)
    # DSA shares one selected-KV producer across the complete local head
    # shard, while its arithmetic remains in validated four-head shapes. Keep
    # dense MLA on its original four-head streaming path and OCM footprint.
    head_block = q_head_num_cur_core if sparse_attention else min(4, q_head_num_cur_core)

    x_running = x_shocm
    hidden_dims_per_core = hidden_dim // core_num
    q_head_core_start = core_id * q_head_num_factor

    for h_start in range(0, q_head_num_cur_core, head_block):
        h_stop = min(h_start + head_block, q_head_num_cur_core)
        head_block_cur = h_stop - h_start

        global_h_start = q_head_core_start + h_start
        global_h_stop = q_head_core_start + h_stop
        q_nope_wkv_b0_with_pe = record_mla_absorbed_query(
            r_norm_dq,
            w_uq,
            w_scale_uq,
            w_scale2_uq,
            w_kv_b_0,
            w_scale_kv_b_0,
            w_scale2_kv_b_0,
            cos,
            sin,
            position_ids,
            global_head_start=global_h_start,
            global_head_stop=global_h_stop,
            name_prefix=f"dsa_core{core_id}",
        )

        r_attn_compress_block = ax.dsl.create_tensor(
            (B, seq_len, head_block_cur, v_head_dim),
            dtype=x.dtype,
            name=f"r_attn_compress_core{core_id}_h{h_start}",
        )

        if sparse_attention and sparse_attention_impl is DSASparseAttentionImpl.BLOCK_DENSE:
            # Keep attention tiles consecutive, then run B1. Four independent
            # private tiles fit fragmented OCM, unlike one contiguous full-S
            # result, and avoid B1 CONV jobs contending with the next tile.
            r_attn_tiles = []
            for b in range(B):
                q_head = q_nope_wkv_b0_with_pe[b : b + 1, :, :, :].reshape(
                    seq_len, head_block_cur, kv_lora_rank + qk_rope_head_dim
                )
                for query_start in range(0, seq_len, sparse_query_block_size):
                    query_stop = min(query_start + sparse_query_block_size, seq_len)
                    query_count = query_stop - query_start
                    r_attn_tile = ax.dsl.create_tensor(
                        (query_count, head_block_cur, kv_lora_rank),
                        dtype=x.dtype,
                        name=f"r_attn_tile_core{core_id}_b{b}_h{h_start}_q{query_start}",
                    )
                    block_dense_mla_attention(
                        q_head[query_start:query_stop, :, :],
                        (
                            kv_cache_in[b : b + 1, :, :].reshape(cached_seq_len, kv_lora_rank)
                            if kv_cache_in is not None
                            else None
                        ),
                        (
                            pe_cache_in[b : b + 1, :, :].reshape(cached_seq_len, qk_rope_head_dim)
                            if pe_cache_in is not None
                            else None
                        ),
                        r_kv_norm_shocm[b : b + 1, :, :].reshape(seq_len, kv_lora_rank),
                        r_pe_rope_shocm[b : b + 1, :, :].reshape(seq_len, qk_rope_head_dim),
                        topk_indices[b : b + 1, query_start:query_stop, :, :].reshape(query_count, -1),
                        r_attn_tile,
                        cached_seq_len=cached_seq_len,
                        softmax_scale=softmax_scale,
                        query_block_size=query_count,
                        cache_block_size=sparse_cache_block_size,
                        query_position_offset=query_start,
                        materialize_query_group=True,
                        name_prefix=f"_prefill_core{core_id}_b{b}_h{h_start}_q{query_start}",
                    )
                    r_attn_tiles.append((b, query_start, query_stop, r_attn_tile))

            for b, query_start, query_stop, r_attn_tile in r_attn_tiles:
                query_count = query_stop - query_start
                for head_i in range(h_start, h_stop):
                    head_local_i = head_i - h_start
                    ax.dsl.matmul(
                        r_attn_tile[:, head_local_i : head_local_i + 1, :].reshape(1, query_count, kv_lora_rank),
                        w_kv_b_1[slice_q_head_num, :, :][head_i : head_i + 1, :, :].reshape(v_head_dim, kv_lora_rank),
                        w_scale=w_scale_kv_b_1[slice_q_head_num, :, :][head_i : head_i + 1, :, :].reshape(
                            v_head_dim, -1
                        ),
                        w_scale2=w_scale2_kv_b_1,
                        transposed=False,
                        output_dtype=ax.bf16,
                        const_inputs=["y"],
                        r=r_attn_compress_block[
                            b : b + 1,
                            query_start:query_stop,
                            head_local_i : head_local_i + 1,
                            :,
                        ].reshape(1, query_count, v_head_dim),
                    )
        else:
            r_attn = ax.dsl.create_tensor(
                (B, seq_len, head_block_cur, kv_lora_rank),
                dtype=x.dtype,
                name=f"r_attn_core{core_id}_h{h_start}",
            )
            for b in range(B):
                q_head = q_nope_wkv_b0_with_pe[b : b + 1, :, :, :].reshape(
                    seq_len, head_block_cur, kv_lora_rank + qk_rope_head_dim
                )
                output_head = r_attn[b : b + 1, :, :, :].reshape(seq_len, head_block_cur, kv_lora_rank)
                name_prefix = f"_prefill_core{core_id}_b{b}_h{h_start}"
                sparse_mla_attention(
                    q_head,
                    (
                        kv_cache_in[b : b + 1, :, :].reshape(cached_seq_len, kv_lora_rank)
                        if kv_cache_in is not None
                        else None
                    ),
                    (
                        pe_cache_in[b : b + 1, :, :].reshape(cached_seq_len, qk_rope_head_dim)
                        if pe_cache_in is not None
                        else None
                    ),
                    r_kv_norm_shocm[b : b + 1, :, :].reshape(seq_len, kv_lora_rank),
                    r_pe_rope_shocm[b : b + 1, :, :].reshape(seq_len, qk_rope_head_dim),
                    topk_indices[b : b + 1, :, :, :].reshape(seq_len, -1),
                    output_head,
                    cached_seq_len=cached_seq_len,
                    softmax_scale=softmax_scale,
                    name_prefix=name_prefix,
                )

            # Apply B1 immediately, then release the large latent attention block.
            for head_i in range(h_start, h_stop):
                head_local_i = head_i - h_start
                for b in range(B):
                    ax.dsl.matmul(
                        r_attn[b : b + 1, :, head_local_i : head_local_i + 1, :].reshape(1, seq_len, kv_lora_rank),
                        w_kv_b_1[slice_q_head_num, :, :][head_i : head_i + 1, :, :].reshape(v_head_dim, kv_lora_rank),
                        w_scale=w_scale_kv_b_1[slice_q_head_num, :, :][head_i : head_i + 1, :, :].reshape(
                            v_head_dim, -1
                        ),
                        w_scale2=w_scale2_kv_b_1,
                        transposed=False,
                        output_dtype=ax.bf16,
                        const_inputs=["y"],
                        r=r_attn_compress_block[b : b + 1, :, head_local_i : head_local_i + 1, :].reshape(
                            1, seq_len, v_head_dim
                        ),
                    )
        if multi_core:
            axd.sync_all()

        projection_head_block = 4
        for projection_h_start in range(0, head_block_cur, projection_head_block):
            projection_h_stop = min(projection_h_start + projection_head_block, head_block_cur)
            r_attn_compress_sub = r_attn_compress_block[:, :, projection_h_start:projection_h_stop, :].reshape(
                B * seq_len, -1
            )
            global_h_start = q_head_core_start + h_start + projection_h_start
            global_h_stop = q_head_core_start + h_start + projection_h_stop
            slice_q_head_group_dims = slice(global_h_start * v_head_dim, global_h_stop * v_head_dim)

            # 10. Keep the validated four-head output-FC accumulation order
            # while sparse attention shares one selected KV block across the
            # complete local head shard.
            for j in range(core_num):
                # 核交错，解x_shocm bank冲突，控制不同核的起始block不同，core0 [0,1,2,3] core1 [1,2,3,0] ...
                hidden_dim_offset = ((j + core_id) % core_num) * hidden_dims_per_core
                slice_hidden_dim = slice(hidden_dim_offset, hidden_dim_offset + hidden_dims_per_core)
                w_o_sub = w_o[slice_hidden_dim, slice_q_head_group_dims]

                w_scale_ic_slice = {
                    datatype.DataType.FP8_E4M3: slice(0, None),
                    datatype.DataType.FP8_E5M2: slice(0, None),
                    datatype.DataType.MXFP4: slice(
                        global_h_start * v_head_dim // 32,
                        global_h_stop * v_head_dim // 32,
                    ),
                    datatype.DataType.NVFP4: slice(
                        global_h_start * v_head_dim // 16,
                        global_h_stop * v_head_dim // 16,
                    ),
                }[w_dtype_type]
                w_scale_o_sub = w_scale_o[slice_hidden_dim, w_scale_ic_slice] if w_scale_o.size > 1 else w_scale_o
                r_fc_o = ax.dsl.matmul(
                    r_attn_compress_sub,
                    w_o_sub,
                    w_scale=w_scale_o_sub,
                    w_scale2=w_scale2_o,
                    transposed=True,
                    output_dtype=ax.bf16,
                    const_inputs=["y"],
                ).reshape(B, seq_len, hidden_dims_per_core)
                r_sum_shocm = ax.dsl.create_tensor(
                    x_shocm.shape,
                    x_shocm.dtype,
                    ax.MemoryType.SHARED_OCM,
                    name=f"r_sum_shocm_{core_id}_{h_start}_{projection_h_start}_{j}",
                )
                ax.dsl.add(
                    x=r_fc_o,
                    y=x_running[:, :, slice_hidden_dim],
                    r=r_sum_shocm[:, :, slice_hidden_dim],
                )
                if multi_core:
                    axd.sync_all()
                x_running = r_sum_shocm

    output_mask = ax.dsl.cast(seq_len_mask_i32, dtype=x_running.dtype)
    output_masked = ax.dsl.create_tensor(
        x_running.shape,
        x_running.dtype,
        x_mem_scope,
        name=f"mla_output_masked_core{core_id}",
    )
    ax.dsl.mul(x_running, output_mask, output_dtype=x_running.dtype, r=output_masked)
    ax.dsl.store(r=output[:, slice_seq_len, :], x=output_masked[:, slice_seq_len, :])

    if internal_indexer:
        return output, kv_cache_out, pe_cache_out, index_k_cache_out, index_k_scale_out
    return output, kv_cache_out, pe_cache_out


DSA_CONST_INPUTS = MLA_CONST_INPUTS + [
    "w_iq",  # [index_head_num * index_dim, q_lora_rank]
    "w_scale_iq",
    "w_scale2_iq",
    "w_ik",  # [index_dim, hidden_dim]
    "w_scale_ik",
    "w_scale2_ik",
    "w_ig",  # [index_head_num, hidden_dim]
    "w_scale_ig",
    "w_scale2_ig",
]


@ax.builder(
    const_inputs=DSA_CONST_INPUTS,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
)
def dsa_builder(
    x,  # BF16 [B, S, D]
    seq_lens,  # INT32 [B]
    w_attn_norm,  # FP32 [D]
    w_dq,  # FP8 [Rq, D]
    w_scale_dq,
    w_scale2_dq,
    w_uq,  # FP8 [H * (Dn + Dr), Rq]
    w_scale_uq,
    w_scale2_uq,
    w_dkv,  # FP8 [Rkv + Dr, D]
    w_scale_dkv,
    w_scale2_dkv,
    w_kv_b_0,  # FP8 [H, Rkv, Dn]
    w_scale_kv_b_0,
    w_scale2_kv_b_0,
    w_kv_b_1,  # FP8 [H, Dv, Rkv]
    w_scale_kv_b_1,
    w_scale2_kv_b_1,
    w_q_norm,  # FP32 [Rq]
    w_kv_norm,  # FP32 [Rkv]
    w_o,  # FP8 [D, H * Dv]
    w_scale_o,
    w_scale2_o,
    w_iq,  # FP8 [Hi * Di, Rq]
    w_scale_iq,
    w_scale2_iq,
    w_ik,  # FP8 [Di, D]
    w_scale_ik,
    w_scale2_ik,
    w_ig,  # FP8 [Hi, D]
    w_scale_ig,
    w_scale2_ig,
    index_k_norm_weight,  # FP32 [Di]
    index_k_norm_bias,  # FP32 [Di], required for LayerNorm
    cos,  # FP32 [max_seq_len, Dr / 2]
    sin,  # FP32 [max_seq_len, Dr / 2]
    position_ids,  # INT32 [S]
    kv_cache_out,  # BF16 [B, max_seq_len, Rkv]
    pe_cache_out,  # BF16 [B, max_seq_len, Dr]
    index_k_cache_out,  # FP8 [B, max_seq_len, Di]
    index_k_scale_out,  # FP32 [B, max_seq_len, 1]
    output,  # BF16 [B, S, D]
    softmax_scale=1.0,
    kv_offset=0,
    kv_cache_in=None,  # BF16 [B, C, Rkv]
    pe_cache_in=None,  # BF16 [B, C, Dr]
    index_k_cache_in=None,  # FP8 [B, C, Di]
    index_k_scale_in=None,  # FP32 [B, C, 1]
    topk_indices_override=None,  # INT32 [B, S, 1, K], debug only
    index_topk=2048,
    index_rope_dim=64,
    index_k_norm_type=DSAIndexNorm.LAYER,
    index_use_hadamard=False,
    index_use_ue8m0=True,
    index_init_tokens=0,
    index_local_tokens=0,
    index_token_block_size=128,
    index_query_block_size=16,
    sparse_attention_impl=DSASparseAttentionImpl.TOKEN,
    sparse_query_block_size=32,
    sparse_cache_block_size=128,
):
    """Record Indexer, sparse MLA, output projection, residual, and cache export.

    The external API starts from hidden states. ``topk_indices_override`` exists only for
    sparse-attention debugging; production indices are generated internally and handed to
    sparse MLA through OCM/SHOCM without a DDR round trip.
    """
    dsa_prefill_impl(
        x,
        seq_lens,
        w_attn_norm,
        w_dq,
        w_scale_dq,
        w_scale2_dq,
        w_uq,
        w_scale_uq,
        w_scale2_uq,
        w_dkv,
        w_scale_dkv,
        w_scale2_dkv,
        w_kv_b_0,
        w_scale_kv_b_0,
        w_scale2_kv_b_0,
        w_kv_b_1,
        w_scale_kv_b_1,
        w_scale2_kv_b_1,
        w_q_norm,
        w_kv_norm,
        w_o,
        w_scale_o,
        w_scale2_o,
        cos,
        sin,
        position_ids,
        kv_cache_out,
        pe_cache_out,
        output,
        softmax_scale=softmax_scale,
        kv_offset=kv_offset,
        kv_cache_in=kv_cache_in,
        pe_cache_in=pe_cache_in,
        topk_indices=topk_indices_override,
        w_iq=w_iq,
        w_scale_iq=w_scale_iq,
        w_scale2_iq=w_scale2_iq,
        w_ik=w_ik,
        w_scale_ik=w_scale_ik,
        w_scale2_ik=w_scale2_ik,
        w_ig=w_ig,
        w_scale_ig=w_scale_ig,
        w_scale2_ig=w_scale2_ig,
        index_k_norm_weight=index_k_norm_weight,
        index_k_norm_bias=index_k_norm_bias,
        index_k_cache_out=index_k_cache_out,
        index_k_scale_out=index_k_scale_out,
        index_k_cache_in=index_k_cache_in,
        index_k_scale_in=index_k_scale_in,
        index_topk=index_topk,
        index_rope_dim=index_rope_dim,
        index_k_norm_type=index_k_norm_type,
        index_use_hadamard=index_use_hadamard,
        index_use_ue8m0=index_use_ue8m0,
        index_init_tokens=index_init_tokens,
        index_local_tokens=index_local_tokens,
        index_token_block_size=index_token_block_size,
        index_query_block_size=index_query_block_size,
        sparse_attention_impl=sparse_attention_impl,
        sparse_query_block_size=sparse_query_block_size,
        sparse_cache_block_size=sparse_cache_block_size,
    )
    return output, kv_cache_out, pe_cache_out, index_k_cache_out, index_k_scale_out


def validate_dsa_builder_inputs(
    x,
    seq_lens,
    w_ik,
    index_k_norm_weight,
    index_k_norm_bias,
    kv_cache_out,
    pe_cache_out,
    index_k_cache_out,
    index_k_scale_out,
    output,
    kv_offset,
    kv_cache_in,
    pe_cache_in,
    index_k_cache_in,
    index_k_scale_in,
    index_topk,
    index_rope_dim,
    index_k_norm_type,
    core_num,
):
    """Validate the public contiguous-prefill contract before graph recording."""
    index_k_norm_type = as_dsa_index_norm(index_k_norm_type)
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"x must have shape [1, S, D], got {tuple(x.shape)}")
    batch, seq_len, hidden_dim = x.shape
    if seq_len <= 0:
        raise ValueError("x sequence length must be positive")
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be INT32 with shape {(batch,)}, got {tuple(seq_lens.shape)}")
    if core_num not in (1, 4):
        raise ValueError(f"DSA contiguous prefill supports core_num 1 or 4, got {core_num}")
    if seq_len % core_num != 0:
        raise ValueError(f"sequence length {seq_len} must divide core_num {core_num}")
    if w_ik.ndim != 2 or w_ik.shape[1] != hidden_dim:
        raise ValueError(f"w_ik must have shape [Di, {hidden_dim}], got {w_ik.shape}")
    index_dim = w_ik.shape[0]
    if tuple(index_k_norm_weight.shape) != (index_dim,):
        raise ValueError(f"index_k_norm_weight must have shape {(index_dim,)}")
    if index_k_norm_type is DSAIndexNorm.LAYER and (
        index_k_norm_bias is None or tuple(index_k_norm_bias.shape) != (index_dim,)
    ):
        raise ValueError(f"LayerNorm requires index_k_norm_bias with shape {(index_dim,)}")
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")
    if not 0 < index_rope_dim <= index_dim or index_rope_dim % 2 != 0:
        raise ValueError(f"index_rope_dim must be positive, even, and <= {index_dim}")

    cache_capacity = kv_cache_out.shape[1]
    if tuple(kv_cache_out.shape[:2]) != (batch, cache_capacity):
        raise ValueError("kv_cache_out must have shape [B, L, Rkv]")
    if pe_cache_out.shape[:2] != (batch, cache_capacity):
        raise ValueError("pe_cache_out must have shape [B, L, Dr]")
    if tuple(index_k_cache_out.shape) != (batch, cache_capacity, index_dim):
        raise ValueError(f"index_k_cache_out must have shape {(batch, cache_capacity, index_dim)}")
    if tuple(index_k_scale_out.shape) != (batch, cache_capacity, 1):
        raise ValueError(f"index_k_scale_out must have shape {(batch, cache_capacity, 1)}")
    if tuple(output.shape) != (batch, seq_len, hidden_dim):
        raise ValueError(f"output must have shape {(batch, seq_len, hidden_dim)}")
    if kv_offset + seq_len > cache_capacity:
        raise ValueError("kv_offset + sequence length exceeds output cache capacity")

    if (kv_cache_in is None) != (pe_cache_in is None):
        raise ValueError("kv_cache_in and pe_cache_in must be provided together")
    if (index_k_cache_in is None) != (index_k_scale_in is None):
        raise ValueError("index_k_cache_in and index_k_scale_in must be provided together")
    cached_seq_len = 0 if kv_cache_in is None else kv_cache_in.shape[1]
    if kv_offset != cached_seq_len:
        raise ValueError(f"kv_offset must equal cached sequence length {cached_seq_len}, got {kv_offset}")
    if cached_seq_len > 0:
        if pe_cache_in.shape[:2] != (batch, cached_seq_len):
            raise ValueError("pe_cache_in must match kv_cache_in batch and sequence dimensions")
        if tuple(index_k_cache_in.shape) != (batch, cached_seq_len, index_dim):
            raise ValueError(f"index_k_cache_in must have shape {(batch, cached_seq_len, index_dim)}")
        if tuple(index_k_scale_in.shape) != (batch, cached_seq_len, 1):
            raise ValueError(f"index_k_scale_in must have shape {(batch, cached_seq_len, 1)}")


def run_dsa_builder(
    x,
    seq_lens,
    w_attn_norm,
    w_dq,
    w_scale_dq,
    w_scale2_dq,
    w_uq,
    w_scale_uq,
    w_scale2_uq,
    w_dkv,
    w_scale_dkv,
    w_scale2_dkv,
    w_kv_b_0,
    w_scale_kv_b_0,
    w_scale2_kv_b_0,
    w_kv_b_1,
    w_scale_kv_b_1,
    w_scale2_kv_b_1,
    w_q_norm,
    w_kv_norm,
    w_o,
    w_scale_o,
    w_scale2_o,
    w_iq,
    w_scale_iq,
    w_scale2_iq,
    w_ik,
    w_scale_ik,
    w_scale2_ik,
    w_ig,
    w_scale_ig,
    w_scale2_ig,
    index_k_norm_weight,
    index_k_norm_bias,
    cos,
    sin,
    position_ids,
    kv_cache_out,
    pe_cache_out,
    index_k_cache_out,
    index_k_scale_out,
    output,
    *,
    softmax_scale=1.0,
    kv_offset=0,
    kv_cache_in=None,
    pe_cache_in=None,
    index_k_cache_in=None,
    index_k_scale_in=None,
    topk_indices_override=None,
    index_topk=2048,
    index_rope_dim=64,
    index_k_norm_type=DSAIndexNorm.LAYER,
    index_use_hadamard=False,
    index_use_ue8m0=True,
    index_init_tokens=0,
    index_local_tokens=0,
    index_token_block_size=128,
    index_query_block_size=16,
    sparse_attention_impl=DSASparseAttentionImpl.BLOCK_DENSE,
    sparse_query_block_size=32,
    sparse_cache_block_size=128,
    core_num=4,
):
    """Run the flattened builder-facing DSA contiguous-prefill interface.

    The kernel performs the shared MLA projections, lightning Indexer, sparse
    MLA attention, output projection, residual addition, and cache export.
    ``topk_indices_override`` is a debug-only input that bypasses index logits
    and Top-K while retaining current index-key cache generation.

    Args:
        x: Current hidden states, BF16 ``[1, S, D]``.
        seq_lens: Runtime valid query lengths, INT32 ``[1]``.
        w_attn_norm: Input RMSNorm weight ``[D]``.
        w_dq: MLA query down-projection weight ``[Rq, D]``.
        w_scale_dq: Scale for ``w_dq``.
        w_scale2_dq: Secondary scale for ``w_dq``.
        w_uq: MLA query up-projection weight ``[H*(Dn+Dr), Rq]``.
        w_scale_uq: Scale for ``w_uq``.
        w_scale2_uq: Secondary scale for ``w_uq``.
        w_dkv: Compressed KV/position projection weight ``[Rkv+Dr, D]``.
        w_scale_dkv: Scale for ``w_dkv``.
        w_scale2_dkv: Secondary scale for ``w_dkv``.
        w_kv_b_0: Per-head query absorption weight ``[H, Rkv, Dn]``.
        w_scale_kv_b_0: Scale for ``w_kv_b_0``.
        w_scale2_kv_b_0: Secondary scale for ``w_kv_b_0``.
        w_kv_b_1: Per-head value expansion weight ``[H, Dv, Rkv]``.
        w_scale_kv_b_1: Scale for ``w_kv_b_1``.
        w_scale2_kv_b_1: Secondary scale for ``w_kv_b_1``.
        w_q_norm: Query LoRA RMSNorm weight ``[Rq]``.
        w_kv_norm: Latent-KV RMSNorm weight ``[Rkv]``.
        w_o: Attention output projection weight ``[D, H*Dv]``.
        w_scale_o: Scale for ``w_o``.
        w_scale2_o: Secondary scale for ``w_o``.
        w_iq: Index-query projection weight ``[Hi*Di, Rq]``.
        w_scale_iq: Scale for ``w_iq``.
        w_scale2_iq: Secondary scale for ``w_iq``.
        w_ik: Index-key projection weight ``[Di, D]``.
        w_scale_ik: Scale for ``w_ik``.
        w_scale2_ik: Secondary scale for ``w_ik``.
        w_ig: Index gate projection weight ``[Hi, D]``.
        w_scale_ig: Scale for ``w_ig``.
        w_scale2_ig: Secondary scale for ``w_ig``.
        index_k_norm_weight: Index-key normalization weight ``[Di]``.
        index_k_norm_bias: LayerNorm bias ``[Di]`` or ``None`` for RMSNorm.
        cos: RoPE cosine table ``[L, Dr/2]``.
        sin: RoPE sine table ``[L, Dr/2]``.
        position_ids: Current logical positions, INT32 ``[S]``.
        kv_cache_out: Combined latent cache output ``[1, L, Rkv]``.
        pe_cache_out: Combined rotary cache output ``[1, L, Dr]``.
        index_k_cache_out: Combined FP8 index-key cache ``[1, L, Di]``.
        index_k_scale_out: Combined FP32 index-key scales ``[1, L, 1]``.
        output: Residual layer output ``[1, S, D]``.
        softmax_scale: Sparse MLA score scale.
        kv_offset: Current-token offset; must equal the historical cache length.
        kv_cache_in: Immutable historical latent cache or ``None``.
        pe_cache_in: Immutable historical rotary cache or ``None``.
        index_k_cache_in: Immutable historical FP8 index-key cache or ``None``.
        index_k_scale_in: Immutable historical index-key scales or ``None``.
        topk_indices_override: Optional debug indices ``[1, S, 1, K]``.
        index_topk: Selected token count ``K``.
        index_rope_dim: RoPE prefix width in each index vector.
        index_k_norm_type: ``"layer"`` or ``"rms"``.
        index_use_hadamard: Apply the common orthonormal Hadamard rotation.
        index_use_ue8m0: Round inverse FP8 scales upward to powers of two.
        index_init_tokens: Number of initial tokens forced into Top-K.
        index_local_tokens: Number of local tokens forced into Top-K.
        index_token_block_size: Indexer token tile size.
        index_query_block_size: Indexer query group size.
        sparse_attention_impl: ``"token"`` or ``"block_dense"``.
        sparse_query_block_size: Sparse attention query group size.
        sparse_cache_block_size: Sparse attention cache tile size.
        core_num: Active NPU core count, currently 1 or 4.

    Returns:
        ``(output, kv_cache_out, pe_cache_out, index_k_cache_out,
        index_k_scale_out)``.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    validate_dsa_builder_inputs(
        x,
        seq_lens,
        w_ik,
        index_k_norm_weight,
        index_k_norm_bias,
        kv_cache_out,
        pe_cache_out,
        index_k_cache_out,
        index_k_scale_out,
        output,
        kv_offset,
        kv_cache_in,
        pe_cache_in,
        index_k_cache_in,
        index_k_scale_in,
        index_topk,
        index_rope_dim,
        index_k_norm_type,
        core_num,
    )

    w_scale_dq = normalize_fc_scale(w_scale_dq, w_dq)
    w_scale2_dq = normalize_scale2(w_scale2_dq)
    w_scale_uq = normalize_fc_scale(w_scale_uq, w_uq)
    w_scale2_uq = normalize_scale2(w_scale2_uq)
    w_scale_dkv = normalize_fc_scale(w_scale_dkv, w_dkv)
    w_scale2_dkv = normalize_scale2(w_scale2_dkv)
    w_scale_kv_b_0 = normalize_head_fc_scale(w_scale_kv_b_0, w_kv_b_0)
    w_scale2_kv_b_0 = normalize_scale2(w_scale2_kv_b_0)
    w_scale_kv_b_1 = normalize_head_fc_scale(w_scale_kv_b_1, w_kv_b_1)
    w_scale2_kv_b_1 = normalize_scale2(w_scale2_kv_b_1)
    w_scale_o = normalize_fc_scale(w_scale_o, w_o)
    w_scale2_o = normalize_scale2(w_scale2_o)
    w_scale_iq = normalize_fc_scale(w_scale_iq, w_iq)
    w_scale2_iq = normalize_scale2(w_scale2_iq)
    w_scale_ik = normalize_fc_scale(w_scale_ik, w_ik)
    w_scale2_ik = normalize_scale2(w_scale2_ik)
    w_scale_ig = normalize_fc_scale(w_scale_ig, w_ig)
    w_scale2_ig = normalize_scale2(w_scale2_ig)

    ax.init(chip_num=1, core_num=core_num)
    dsa_builder[core_num](
        x,
        seq_lens,
        w_attn_norm,
        w_dq,
        w_scale_dq,
        w_scale2_dq,
        w_uq,
        w_scale_uq,
        w_scale2_uq,
        w_dkv,
        w_scale_dkv,
        w_scale2_dkv,
        w_kv_b_0,
        w_scale_kv_b_0,
        w_scale2_kv_b_0,
        w_kv_b_1,
        w_scale_kv_b_1,
        w_scale2_kv_b_1,
        w_q_norm,
        w_kv_norm,
        w_o,
        w_scale_o,
        w_scale2_o,
        w_iq,
        w_scale_iq,
        w_scale2_iq,
        w_ik,
        w_scale_ik,
        w_scale2_ik,
        w_ig,
        w_scale_ig,
        w_scale2_ig,
        index_k_norm_weight,
        index_k_norm_bias,
        cos,
        sin,
        position_ids,
        kv_cache_out,
        pe_cache_out,
        index_k_cache_out,
        index_k_scale_out,
        output,
        softmax_scale=softmax_scale,
        kv_offset=kv_offset,
        kv_cache_in=kv_cache_in,
        pe_cache_in=pe_cache_in,
        index_k_cache_in=index_k_cache_in,
        index_k_scale_in=index_k_scale_in,
        topk_indices_override=topk_indices_override,
        index_topk=index_topk,
        index_rope_dim=index_rope_dim,
        index_k_norm_type=index_k_norm_type,
        index_use_hadamard=index_use_hadamard,
        index_use_ue8m0=index_use_ue8m0,
        index_init_tokens=index_init_tokens,
        index_local_tokens=index_local_tokens,
        index_token_block_size=index_token_block_size,
        index_query_block_size=index_query_block_size,
        sparse_attention_impl=sparse_attention_impl,
        sparse_query_block_size=sparse_query_block_size,
        sparse_cache_block_size=sparse_cache_block_size,
    )
    return output, kv_cache_out, pe_cache_out, index_k_cache_out, index_k_scale_out


def split_freqs_cis(freqs_cis: FreqsCis, seq_len: int, rope_dim: int):
    """Convert the reference RoPE input into real cosine and sine tables.

    Args:
        freqs_cis: Complex RoPE values ``[S, Dr/2]`` as used by the DeepSeek
            reference, or a ``(cos, sin)`` pair with the same shape.
        seq_len: Current query length ``S``.
        rope_dim: RoPE vector width ``Dr``.

    Returns:
        A pair of contiguous FP32 cosine and sine tensors ``[S, Dr/2]``.

    Raises:
        TypeError: If ``freqs_cis`` is neither a complex tensor nor a pair.
        ValueError: If the RoPE table shape does not match the current input.
    """
    if isinstance(freqs_cis, torch.Tensor):
        if not torch.is_complex(freqs_cis):
            raise TypeError("freqs_cis must be a complex tensor or a (cos, sin) pair")
        cos, sin = freqs_cis.real, freqs_cis.imag
    elif isinstance(freqs_cis, tuple) and len(freqs_cis) == 2:
        cos, sin = freqs_cis
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            raise TypeError("cos and sin must be torch tensors")
    else:
        raise TypeError("freqs_cis must be a complex tensor or a (cos, sin) pair")

    expected_shape = (seq_len, rope_dim // 2)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise ValueError(
            f"freqs_cis must provide cos/sin shape {expected_shape}, " f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    return cos.float().contiguous(), sin.float().contiguous()


def validate_dsa_interface(
    x: torch.Tensor,
    start_pos: int,
    weights: DSAWeights,
    cache: DSACache,
    config: DSAConfig,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    topk_indices_override: Optional[torch.Tensor],
    effective_topk: int,
):
    """Validate the reference-structured public DSA interface."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"x must have shape [1, S, D], got {tuple(x.shape)}")
    batch, seq_len, hidden_dim = x.shape
    if hidden_dim != config.dim:
        raise ValueError(f"x hidden width must be {config.dim}, got {hidden_dim}")
    if not isinstance(start_pos, int) or isinstance(start_pos, bool) or start_pos < 0:
        raise ValueError(f"start_pos must be a non-negative integer, got {start_pos!r}")
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be INT32 with shape {(batch,)}")
    if torch.any(seq_lens < 0) or torch.any(seq_lens > seq_len):
        raise ValueError(f"seq_lens values must be in [0, {seq_len}]")
    if config.n_heads <= 0 or config.index_n_heads <= 0:
        raise ValueError("n_heads and index_n_heads must be positive")
    if config.core_num not in (1, 4):
        raise ValueError(f"core_num must be 1 or 4, got {config.core_num}")
    if seq_len % config.core_num != 0:
        raise ValueError(f"sequence length {seq_len} must be divisible by core_num {config.core_num}")
    if config.index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {config.index_topk}")
    if config.qk_rope_head_dim <= 0 or config.qk_rope_head_dim % 2 != 0:
        raise ValueError("qk_rope_head_dim must be positive and even")
    if config.qk_rope_head_dim > config.index_head_dim:
        raise ValueError("qk_rope_head_dim must not exceed index_head_dim")
    index_k_norm_type = as_dsa_index_norm(config.index_k_norm_type)
    as_dsa_sparse_impl(config.sparse_attention_impl)

    projection_shapes = {
        "wq_a": (config.q_lora_rank, config.dim),
        "wq_b": (
            config.n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            config.q_lora_rank,
        ),
        "wkv_a": (
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.dim,
        ),
        "wkv_b_key": (
            config.n_heads,
            config.kv_lora_rank,
            config.qk_nope_head_dim,
        ),
        "wkv_b_value": (
            config.n_heads,
            config.v_head_dim,
            config.kv_lora_rank,
        ),
        "wo": (config.dim, config.n_heads * config.v_head_dim),
        "index_wq_b": (
            config.index_n_heads * config.index_head_dim,
            config.q_lora_rank,
        ),
        "index_wk": (config.index_head_dim, config.dim),
        "index_weights_proj": (config.index_n_heads, config.dim),
    }
    projections = {
        "wq_a": weights.wq_a,
        "wq_b": weights.wq_b,
        "wkv_a": weights.wkv_a,
        "wkv_b_key": weights.wkv_b_key,
        "wkv_b_value": weights.wkv_b_value,
        "wo": weights.wo,
        "index_wq_b": weights.index_wq_b,
        "index_wk": weights.index_wk,
        "index_weights_proj": weights.index_weights_proj,
    }
    for name, expected_shape in projection_shapes.items():
        actual_shape = tuple(projections[name].weight.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{name}.weight must have shape {expected_shape}, got {actual_shape}")

    norm_values = (
        ("attn_norm", weights.attn_norm, (config.dim,)),
        ("q_norm", weights.q_norm, (config.q_lora_rank,)),
        ("kv_norm", weights.kv_norm, (config.kv_lora_rank,)),
        ("index_k_norm_weight", weights.index_k_norm_weight, (config.index_head_dim,)),
    )
    for name, value, expected_shape in norm_values:
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")
    if index_k_norm_type is DSAIndexNorm.LAYER:
        if weights.index_k_norm_bias is None or tuple(weights.index_k_norm_bias.shape) != (config.index_head_dim,):
            raise ValueError("LayerNorm requires index_k_norm_bias with shape " f"{(config.index_head_dim,)}")

    if cache.kv_cache.ndim != 3:
        raise ValueError("kv_cache must have rank 3")
    cache_capacity = cache.kv_cache.shape[1]
    cache_values = (
        ("kv_cache", cache.kv_cache, (batch, cache_capacity, config.kv_lora_rank), torch.bfloat16),
        ("pe_cache", cache.pe_cache, (batch, cache_capacity, config.qk_rope_head_dim), torch.bfloat16),
        (
            "index_k_cache",
            cache.index_k_cache,
            (batch, cache_capacity, config.index_head_dim),
            torch.float8_e4m3fn,
        ),
        ("index_k_scale_cache", cache.index_k_scale_cache, (batch, cache_capacity, 1), torch.float32),
    )
    for name, value, expected_shape, expected_dtype in cache_values:
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")
        actual_dtype = value.dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"{name} must have dtype {expected_dtype}, got {actual_dtype}")
    if start_pos + seq_len > cache_capacity:
        raise ValueError("start_pos + sequence length exceeds cache capacity")
    if tuple(output.shape) != tuple(x.shape) or output.dtype != torch.bfloat16:
        raise ValueError(
            f"output must be BF16 with shape {tuple(x.shape)}, " f"got {tuple(output.shape)} and {output.dtype}"
        )
    if topk_indices_override is not None:
        expected_shape = (batch, seq_len, 1, effective_topk)
        if tuple(topk_indices_override.shape) != expected_shape or topk_indices_override.dtype != torch.int32:
            raise ValueError(f"topk_indices_override must be INT32 with shape {expected_shape}")


def dsa(
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: FreqsCis,
    weights: DSAWeights,
    cache: DSACache,
    config: DSAConfig,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    topk_indices_override: Optional[torch.Tensor] = None,
):
    """Run one DeepSeek Sparse Attention MLA/Indexer step on Axera NPU.

    The interface follows the reference ``MLA.forward(x, start_pos,
    freqs_cis, mask)`` structure while making layer weights and persistent
    caches explicit for framework integration. The function supports
    contiguous prefill/extend calls. It records the MLA projections, Indexer
    FP8 key generation and Top-K selection, sparse attention, residual output,
    and all cache updates without importing a backend or reference DSA kernel.

    Args:
        x: Current hidden states ``[1, S, D]``. Values are converted to BF16.
        start_pos: Number of historical tokens already stored in ``cache``.
        freqs_cis: Current-position complex RoPE values ``[S, Dr/2]`` or a
            ``(cos, sin)`` pair with the same shape.
        weights: MLA and Indexer weights using reference structure names.
        cache: Caller-owned MLA and Indexer caches. History is read from
            ``[:start_pos]`` and current rows are written at
            ``[start_pos:start_pos + S]``.
        config: DSA dimensions and recording policy.
        seq_lens: Optional runtime valid lengths ``[1]`` in INT32. Defaults to
            the full current sequence length.
        output: Optional caller-owned BF16 output ``[1, S, D]``. When omitted,
            a new tensor is allocated.
        topk_indices_override: Debug-only causal token indices
            ``[1, S, 1, min(K, start_pos + S)]``. It bypasses score/Top-K work
            but still records and updates the current Indexer key cache.

    Returns:
        The DSA residual output ``[1, S, D]``. Every tensor owned by ``cache``
        is updated in place.

    Raises:
        TypeError: If ``freqs_cis`` does not use a supported representation.
        ValueError: If dimensions, cache capacity, runtime lengths, or the
            optional debug indices violate the contiguous-prefill contract.
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"x must have shape [1, S, D], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    batch, seq_len, _ = x.shape
    cos, sin = split_freqs_cis(freqs_cis, seq_len, config.qk_rope_head_dim)
    if seq_lens is None:
        seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=x.device)
    if output is None:
        output = torch.empty_like(x)
    effective_topk = min(config.index_topk, start_pos + seq_len)
    validate_dsa_interface(
        x,
        start_pos,
        weights,
        cache,
        config,
        seq_lens,
        output,
        topk_indices_override,
        effective_topk,
    )

    position_ids = torch.arange(seq_len, dtype=torch.int32, device=x.device)
    if start_pos:
        kv_cache_in = cache.kv_cache[:, :start_pos, :].clone()
        pe_cache_in = cache.pe_cache[:, :start_pos, :].clone()
        index_k_cache_in = cache.index_k_cache[:, :start_pos, :].clone()
        index_k_scale_in = cache.index_k_scale_cache[:, :start_pos, :].clone()
    else:
        kv_cache_in = None
        pe_cache_in = None
        index_k_cache_in = None
        index_k_scale_in = None

    run_dsa_builder(
        x,
        seq_lens,
        weights.attn_norm,
        weights.wq_a.weight,
        weights.wq_a.scale,
        weights.wq_a.scale2,
        weights.wq_b.weight,
        weights.wq_b.scale,
        weights.wq_b.scale2,
        weights.wkv_a.weight,
        weights.wkv_a.scale,
        weights.wkv_a.scale2,
        weights.wkv_b_key.weight,
        weights.wkv_b_key.scale,
        weights.wkv_b_key.scale2,
        weights.wkv_b_value.weight,
        weights.wkv_b_value.scale,
        weights.wkv_b_value.scale2,
        weights.q_norm,
        weights.kv_norm,
        weights.wo.weight,
        weights.wo.scale,
        weights.wo.scale2,
        weights.index_wq_b.weight,
        weights.index_wq_b.scale,
        weights.index_wq_b.scale2,
        weights.index_wk.weight,
        weights.index_wk.scale,
        weights.index_wk.scale2,
        weights.index_weights_proj.weight,
        weights.index_weights_proj.scale,
        weights.index_weights_proj.scale2,
        weights.index_k_norm_weight,
        weights.index_k_norm_bias,
        cos,
        sin,
        position_ids,
        cache.kv_cache,
        cache.pe_cache,
        cache.index_k_cache,
        cache.index_k_scale_cache,
        output,
        softmax_scale=config.attention_softmax_scale,
        kv_offset=start_pos,
        kv_cache_in=kv_cache_in,
        pe_cache_in=pe_cache_in,
        index_k_cache_in=index_k_cache_in,
        index_k_scale_in=index_k_scale_in,
        topk_indices_override=topk_indices_override,
        index_topk=effective_topk,
        index_rope_dim=config.qk_rope_head_dim,
        index_k_norm_type=config.index_k_norm_type,
        index_use_hadamard=config.index_use_hadamard,
        index_use_ue8m0=config.scale_fmt is not None,
        index_init_tokens=config.index_init_tokens,
        index_local_tokens=config.index_local_tokens,
        index_token_block_size=config.index_token_block_size,
        index_query_block_size=config.index_query_block_size,
        sparse_attention_impl=config.sparse_attention_impl,
        sparse_query_block_size=config.sparse_query_block_size,
        sparse_cache_block_size=config.sparse_cache_block_size,
        core_num=config.core_num,
    )
    return output


__all__ = ["DSACache", "DSAConfig", "DSAProjection", "DSAWeights", "dsa"]
