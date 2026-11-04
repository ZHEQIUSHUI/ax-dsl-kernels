"""Public Multi-head Latent Attention (MLA) kernel for sglang backend integration.

Port of the axnn DeepSeek-V3/R1 MLA prefill kernel (``mla_kernel``) into the
``sgl_kernel_axera.attention`` package, aligned to the same input/output
contract so prefill (empty KV cache) and decode/extend (cached latent KV) share
one kernel. Common pre-attention calculations are imported from
``attention.attn_utils.common``; this module owns dense FlashAttention and the MLA
public wrapper.

The fused DeepSeek-V3/R1-style latent-attention layer computes::

    x_norm    = rmsnorm(x)
    q_latent  = rmsnorm(x_norm @ w_dq)
    kv_latent = x_norm @ w_dkv                     # kv_lora_rank + rope dims
    kv_norm   = rmsnorm(kv_latent[:, :kv_lora_rank])
    q         = q_latent @ w_uq                    # nope + rope split
    pe        = rope(kv_latent[:, kv_lora_rank:])  # current-token PE
    attn      = causal_attn(q, [kv_cache_in | kv_norm], [pe_cache_in | pe])
    output    = (attn @ w_kv_b_1) @ w_o            # per-head B1 + output FC

The current tokens' ``kv_norm`` and roped ``pe`` are stored into
``kv_cache_out`` / ``pe_cache_out`` so the caller can maintain its own cache.
See :func:`mla` for the complete contract.
"""

from __future__ import annotations

import os

import ml_dtypes
import numpy as np
import torch

import axnn.dsl as ax
from axnn.opset import datatype
from axnn.dsl import parallel as axd
from sgl_kernel_axera.attention.attn_utils import (
    MLA_CONST_INPUTS,
    normalize_fc_scale,
    normalize_head_fc_scale,
    normalize_scale2,
    record_mla_absorbed_query,
    record_mla_shared_projections,
)


def flash_attn_causal_impl_with_kv_load(
    Q,
    K_CACHE,
    V_CACHE,
    k_cur,
    v_cur,
    O,
    prefill=True,
    name_prefix="",
    mla=False,
    softmax_scale=1.0,
):
    """Causal FlashAttention streaming-softmax loop over cached + current KV.

    Q: (Q_SEQ_LEN_M, Q_HEAD_NUM, head_dim). K_CACHE/V_CACHE are the cached
    prefix K/V, k_cur/v_cur the current tokens' K/V. ``mla=True`` stores the
    latent K/V into a single ``[kv_lora_rank + qk_rope_head_dim]`` K tile.
    """
    Q_SEQ_LEN_M, Q_HEAD_NUM, head_dim = Q.shape
    K_CACHE_SEQ_LEN = V_CACHE_SEQ_LEN = 0
    if K_CACHE is not None and V_CACHE is not None:
        K_CACHE_SEQ_LEN, _ = K_CACHE.shape
        V_CACHE_SEQ_LEN, _ = V_CACHE.shape
        assert K_CACHE_SEQ_LEN == V_CACHE_SEQ_LEN, "K_CACHE and V_CACHE must have the same sequence length"
        if mla:
            assert head_dim == K_CACHE.shape[-1] + V_CACHE.shape[-1]
    assert k_cur.shape[0] == v_cur.shape[0]
    assert Q_SEQ_LEN_M == v_cur.shape[0]
    K_TOTAL_SEQ_LEN = K_CACHE_SEQ_LEN + Q_SEQ_LEN_M

    KV_PAD_LEN = (K_TOTAL_SEQ_LEN * k_cur.dtype.num_bits + 255) // 256 * 256 // k_cur.dtype.num_bits - K_TOTAL_SEQ_LEN
    K_TOTAL_SEQ_LEN_PAD = K_TOTAL_SEQ_LEN + KV_PAD_LEN
    q_seq_len_m_factor = Q_SEQ_LEN_M
    k_factor = min(256, K_TOTAL_SEQ_LEN_PAD)  # K tile block size
    # mask row indices (q_seq_len_m_factor, 1) and col indices (1, k_factor)
    row_indices = np.arange(q_seq_len_m_factor).reshape(q_seq_len_m_factor, 1)  # (q_seq_len_m_factor, 1)

    zeros = ax.dsl.from_numpy(
        np.full(
            shape=(Q_HEAD_NUM, q_seq_len_m_factor, 1),
            dtype=ml_dtypes.bfloat16,
            fill_value=0,
        )
    )
    # kv_cache ocm block - use name_prefix to ensure unique names across loop iterations
    # Note: For multi-block sequences, we need unique buffers per j iteration to avoid duplicate labels
    num_q_blocks = (Q_SEQ_LEN_M + q_seq_len_m_factor - 1) // q_seq_len_m_factor
    num_k_blocks = (K_TOTAL_SEQ_LEN_PAD + k_factor - 1) // k_factor
    # Multi-block case: create unique buffer pairs for each j iteration
    k_ocm_block = [
        [
            ax.dsl.create_tensor(
                (k_factor, head_dim),
                dtype=k_cur.dtype,
                name=f"k_ocm_block{name_prefix}_i{i}_j{j}",
            )
            for j in range(num_k_blocks)
        ]
        for i in range(num_q_blocks)
    ]
    v_head_dim = head_dim if not mla else v_cur.shape[-1]
    v_ocm_block = [
        [
            ax.dsl.create_tensor(
                (k_factor, v_head_dim),
                dtype=v_cur.dtype,
                name=f"v_ocm_block{name_prefix}_i{i}_j{j}",
            )
            for j in range(num_k_blocks)
        ]
        for i in range(num_q_blocks)
    ]

    if KV_PAD_LEN > 0:
        kv_pad_name = f"kv_pad{name_prefix}" if name_prefix else None
        kv_pad = ax.dsl.zeros(shape=(KV_PAD_LEN, head_dim), dtype=k_cur.dtype, prefix=kv_pad_name)

    # iterate over Q blocks
    for i in range(num_q_blocks):
        q_m_start = i * q_seq_len_m_factor
        q_m_end = min((i + 1) * q_seq_len_m_factor, Q_SEQ_LEN_M)
        Qi = Q[q_m_start:q_m_end, :, :]  # (q_seq_len_m_factor, head_num, head_dim)

        # per-block max/sum/O accumulators
        max_old = ax.dsl.full(shape=(Q_HEAD_NUM, q_seq_len_m_factor, 1), dtype=ax.bf16, fill_value=-3e38)
        L_old = ax.dsl.zeros(
            shape=(
                Q_HEAD_NUM,
                q_seq_len_m_factor,
            ),
            dtype=ax.fp32,
        )  # fp32 sum for precision
        Oi_old_block = ax.dsl.zeros(shape=(Q_HEAD_NUM, q_seq_len_m_factor, v_head_dim), dtype=ax.bf16)

        # iterate over K blocks
        for j in range(num_k_blocks):
            k_start = j * k_factor
            k_end = min((j + 1) * k_factor, K_TOTAL_SEQ_LEN_PAD)
            k_seq_len_factor_cur = k_end - k_start
            assert k_seq_len_factor_cur <= k_factor
            col_indices = np.arange(k_seq_len_factor_cur).reshape(1, k_seq_len_factor_cur)  # include pad

            mid_block = j < num_k_blocks - 1
            if mid_block:  # middle block stores max_old at head and -max_old at tail
                q_kt = ax.dsl.create_tensor(
                    shape=(
                        Q_HEAD_NUM,
                        q_seq_len_m_factor,
                        1 + k_seq_len_factor_cur + 1,
                    ),
                    dtype=ax.bf16,
                )  # (q_seq_len_m_factor, 1 + k_seq_len_factor_cur + 1) bf16
            else:  # last block, only stores max_old at head
                q_kt = ax.dsl.create_tensor(
                    shape=(Q_HEAD_NUM, q_seq_len_m_factor, 1 + k_seq_len_factor_cur),
                    dtype=ax.bf16,
                )  # (q_seq_len_m_factor, 1 + k_seq_len_factor_cur) bf16
            # | max_old | q_kt | zero |
            ax.dsl.move(r=q_kt[:, :, :1], x=max_old)
            if mid_block:
                ax.dsl.move(r=q_kt[:, :, -1:], x=zeros)

            mask_bf16 = None
            # mask add
            if prefill:
                # global coords: col_indices + k_start > row_indices + q_m_start
                mask_bf16 = np.where(
                    col_indices + k_start - K_CACHE_SEQ_LEN > row_indices + q_m_start,
                    np.float32(-np.inf),
                    0.0,
                )
            elif KV_PAD_LEN > 0 and k_start < K_TOTAL_SEQ_LEN and k_end > K_TOTAL_SEQ_LEN:
                # decode mask only for kv_cache pad case
                mask_bf16 = np.where(
                    col_indices + k_start - K_CACHE_SEQ_LEN >= 1,
                    np.float32(-np.inf),
                    0.0,
                )

            # Get the appropriate buffer pair for this j iteration
            cur_k_ocm_block = k_ocm_block[i][j]
            cur_v_ocm_block = v_ocm_block[i][j]

            # | K_CACHE_SEQ_LEN   |  Q_SEQ_LEN_M     | KV_PAD_LEN        |
            # | K_CACHE(DDR) load |  k_cur(OCM) move | kv_pad(OCM)  move |

            # load kv_cache block
            if k_start < K_CACHE_SEQ_LEN:
                valid_kv_end = min((j + 1) * k_factor, K_CACHE_SEQ_LEN)
                valid_kv_cache_seq_len = valid_kv_end - k_start
                assert valid_kv_cache_seq_len <= k_factor
                if mla:
                    kv_lora_rank = V_CACHE.shape[-1]
                    ax.dsl.load(
                        r=cur_k_ocm_block[:valid_kv_cache_seq_len, :kv_lora_rank],
                        x=V_CACHE[k_start:valid_kv_end, :],
                    )
                    ax.dsl.load(
                        r=cur_k_ocm_block[:valid_kv_cache_seq_len, kv_lora_rank:],
                        x=K_CACHE[k_start:valid_kv_end, :],
                    )
                else:
                    ax.dsl.load(
                        r=cur_k_ocm_block[:valid_kv_cache_seq_len, :],
                        x=K_CACHE[k_start:valid_kv_end, :],
                    )
                ax.dsl.load(
                    r=cur_v_ocm_block[:valid_kv_cache_seq_len, :],
                    x=V_CACHE[k_start:valid_kv_end, :],
                )

            # move k_cur&v_cur to cur_k_ocm_block/cur_v_ocm_block if needed
            if k_start < K_CACHE_SEQ_LEN + Q_SEQ_LEN_M and k_end > K_CACHE_SEQ_LEN:
                kv_cur_start = max(k_start - K_CACHE_SEQ_LEN, 0)
                kv_cur_end = min(k_end - K_CACHE_SEQ_LEN, Q_SEQ_LEN_M)
                kv_cur_len = kv_cur_end - kv_cur_start
                k_ocm_block_offset = max(K_CACHE_SEQ_LEN - k_start, 0)
                ax.dsl.move(
                    r=cur_k_ocm_block[k_ocm_block_offset : k_ocm_block_offset + kv_cur_len, :],
                    x=k_cur[kv_cur_start:kv_cur_end, :],
                )
                ax.dsl.move(
                    r=cur_v_ocm_block[k_ocm_block_offset : k_ocm_block_offset + kv_cur_len, :],
                    x=v_cur[kv_cur_start:kv_cur_end, :],
                )

            # pad cur_k_ocm_block/cur_v_ocm_block if needed
            if KV_PAD_LEN > 0 and k_start < K_TOTAL_SEQ_LEN and k_end > K_TOTAL_SEQ_LEN:
                # k_start 到 K_TOTAL_SEQ_LEN 是有效数据，之后是 padding
                kv_pad_offset = K_TOTAL_SEQ_LEN - k_start
                assert kv_pad_offset + KV_PAD_LEN == k_seq_len_factor_cur
                ax.dsl.move(
                    r=cur_k_ocm_block[kv_pad_offset : kv_pad_offset + KV_PAD_LEN, :],
                    x=kv_pad,
                )
                if mla:
                    kv_lora_rank = v_cur.shape[-1]
                    ax.dsl.move(
                        r=cur_v_ocm_block[kv_pad_offset : kv_pad_offset + KV_PAD_LEN, :],
                        x=kv_pad[:, :kv_lora_rank],
                    )
                else:
                    ax.dsl.move(
                        r=cur_v_ocm_block[kv_pad_offset : kv_pad_offset + KV_PAD_LEN, :],
                        x=kv_pad,
                    )

            Kj = cur_k_ocm_block[:k_seq_len_factor_cur, :]  # (k_seq_len_factor_cur, head_dim)
            Vj = cur_v_ocm_block[:k_seq_len_factor_cur, :]  # (k_seq_len_factor_cur, head_dim)

            # q_kt = Qi @ Kj^T
            q_kt_tmp = ax.dsl.matmul(Qi, Kj, transposed=True)
            ax.dsl.transpose(q_kt_tmp, perm=(1, 0, 2), r=q_kt[:, :, 1 : 1 + k_seq_len_factor_cur])
            if softmax_scale != 1.0:
                scale_tensor = ax.dsl.from_numpy(
                    np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=softmax_scale)
                )
                ax.dsl.mul(
                    q_kt[:, :, 1 : 1 + k_seq_len_factor_cur],
                    scale_tensor,
                    r=q_kt[:, :, 1 : 1 + k_seq_len_factor_cur],
                )

            mask_bf16 = None
            # mask add
            if prefill:
                mask_bf16 = np.where(
                    col_indices + k_start - K_CACHE_SEQ_LEN > row_indices + q_m_start,
                    np.float32(-np.inf),
                    0.0,
                )
            elif KV_PAD_LEN > 0 and k_start < K_TOTAL_SEQ_LEN and k_end > K_TOTAL_SEQ_LEN:
                mask_bf16 = np.where(
                    col_indices + k_start - K_CACHE_SEQ_LEN >= 1,
                    np.float32(-np.inf),
                    0.0,
                )

            if mask_bf16 is not None and not np.all(mask_bf16 == 0):
                mask_tensor = ax.dsl.from_numpy(mask_bf16.astype(ml_dtypes.bfloat16)).reshape(*mask_bf16.shape)
                ax.dsl.add(
                    q_kt[:, :, 1 : 1 + k_seq_len_factor_cur],
                    mask_tensor,
                    r=q_kt[:, :, 1 : 1 + k_seq_len_factor_cur],
                )

            # step1: compare with previous block's max (max_old stored in q_kt first column)
            # | max_old | q_kt |
            max_update = ax.dsl.max(q_kt[:, :, : 1 + k_seq_len_factor_cur], dim=-1, keepdim=True)

            # | max_old - max_update | q_kt - max_update | -max_update |
            q_kt -= max_update
            # step2: P = exp(q_kt - max_update)
            P = ax.dsl.exp(q_kt[:, :, 1 : 1 + k_seq_len_factor_cur])
            P_fp32 = ax.dsl.exp(q_kt[:, :, 1 : 1 + k_seq_len_factor_cur], output_dtype=ax.fp32)  # fp32 P
            # step3: L_curr = sum(P)
            L_curr = ax.dsl.sum(P_fp32, dim=-1, output_dtype=ax.fp32)

            # L_update = L_old * exp(max_old - max_update) + L_curr
            ratio_m = ax.dsl.exp(q_kt[:, :, :1], output_dtype=ax.fp32).reshape(-1, q_seq_len_m_factor)
            L_update = ratio_m * L_old + L_curr

            # Oi_update_block = Oi_old_block * ratio_m[..., None] + P @ Vj
            PV = P @ Vj
            ratio_O = ax.dsl.mul(
                ratio_m.reshape(-1, q_seq_len_m_factor, 1),
                Oi_old_block,
                output_dtype=ax.fp32,
            )
            Oi_update_block = ax.dsl.add(ratio_O, PV, output_dtype=ax.bf16)

            # write back
            Oi_old_block = Oi_update_block
            L_old = L_update

            if mid_block:
                # q_kt last column holds -max_update; * -1 restores max_old
                minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
                max_old = q_kt[:, :, -1:] * minus_1

        # normalize O at block end
        L_old_bf16 = ax.dsl.cast(L_old, dtype=Oi_old_block.dtype)
        Oi_old_block = ax.dsl.div(Oi_old_block, L_old_bf16.reshape(-1, q_seq_len_m_factor, 1))
        Oi_old_block = ax.dsl.transpose(Oi_old_block, perm=(1, 0, 2))
        ax.dsl.move(Oi_old_block, r=O[q_m_start:q_m_end, :, :])


@ax.builder(
    const_inputs=MLA_CONST_INPUTS,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
)
def mla_kernel(
    x,  # [B, seq_len, hidden_dim]
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
    cos,  # [max_seq_len, qk_rope_head_dim // 2]
    sin,  # [max_seq_len, qk_rope_head_dim // 2]
    position_ids,  # [B, seq_len]
    kv_cache_out,  # [B, max_seq_len, kv_lora_rank]
    pe_cache_out,  # [B, max_seq_len, qk_rope_head_dim]
    output,  # [B, seq_len, hidden_dim]
    softmax_scale=1.0,
    kv_offset=0,
    kv_cache_in=None,  # [B, cached_seq_len, kv_lora_rank]
    pe_cache_in=None,  # [B, cached_seq_len, qk_rope_head_dim]
):
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    multi_core = core_num > 1
    w_dtype = datatype.DType(w_dq.dtype).type
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

    (
        x_shocm,
        _x_norm_shocm,
        r_norm_dq,
        r_kv_norm_pe_shocm,
        position_ids,
        _seq_len_mask,
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
        name_prefix=f"mla_core{core_id}",
    )
    r_kv_norm_shocm = r_kv_norm_pe_shocm[:, :, :kv_lora_rank]
    r_pe_rope_shocm = r_kv_norm_pe_shocm[:, :, kv_lora_rank:]

    if core_id == 0:
        # store r_kv_norm_shocm into kv_cache_out and r_pe_rope_shocm into pe_cache_out
        ax.dsl.store(r=_kv_cache_out[:, :seq_len, :], x=r_kv_norm_shocm)
        ax.dsl.store(r=_pe_cache_out[:, :seq_len, :], x=r_pe_rope_shocm)

    # Stream the attention path in head blocks. Keeping the full latent-Q and
    # attention outputs in OCM exceeds AX8860's per-core OCM for the 671B
    # configuration (32 heads per core). B1 reduces each head from
    # kv_lora_rank to v_head_dim, then project and accumulate immediately.
    # Keeping the pipeline strictly one-block-at-a-time avoids both the OCM
    # overflow and backend lifetime issues seen when several blocks are later
    # reconstructed through concat or DDR intermediates.
    head_block = min(4, q_head_num_cur_core)

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
            name_prefix=f"mla_core{core_id}",
        )

        r_attn = ax.dsl.create_tensor(
            (B, seq_len, head_block_cur, kv_lora_rank),
            dtype=x.dtype,
            name=f"r_attn_core{core_id}_h{h_start}",
        )
        for b in range(B):
            kv_head = None
            pe_head = None
            q_head = q_nope_wkv_b0_with_pe[b : b + 1, :, :, :].reshape(
                seq_len, head_block_cur, kv_lora_rank + qk_rope_head_dim
            )
            if kv_cache_in is not None:
                kv_head = kv_cache_in[b : b + 1, :, :].reshape(-1, kv_lora_rank)
            if pe_cache_in is not None:
                pe_head = pe_cache_in[b : b + 1, :, :].reshape(-1, qk_rope_head_dim)
            kv_pe_cur = r_kv_norm_pe_shocm[b : b + 1, :, :].reshape(seq_len, -1)
            kv_cur = r_kv_norm_shocm[b : b + 1, :, :].reshape(seq_len, -1)
            output_head = r_attn[b : b + 1, :, :, :].reshape(seq_len, head_block_cur, kv_lora_rank)
            name_prefix = f"_prefill_core{core_id}_b{b}_h{h_start}"
            flash_attn_causal_impl_with_kv_load(
                q_head,
                pe_head,
                kv_head,
                kv_pe_cur,
                kv_cur,
                output_head,
                prefill=True,
                name_prefix=name_prefix,
                mla=True,
                softmax_scale=softmax_scale,
            )

        # 9. Apply B1 immediately, then release the large latent attention
        # block.
        r_attn_compress_block = ax.dsl.create_tensor(
            (B, seq_len, head_block_cur, v_head_dim),
            dtype=x.dtype,
            name=f"r_attn_compress_core{core_id}_h{h_start}",
        )
        for head_i in range(h_start, h_stop):
            head_local_i = head_i - h_start
            for b in range(B):
                ax.dsl.matmul(
                    r_attn[b : b + 1, :, head_local_i : head_local_i + 1, :].reshape(1, seq_len, kv_lora_rank),
                    w_kv_b_1[slice_q_head_num, :, :][head_i : head_i + 1, :, :].reshape(v_head_dim, kv_lora_rank),
                    w_scale=w_scale_kv_b_1[slice_q_head_num, :, :][head_i : head_i + 1, :, :].reshape(v_head_dim, -1),
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

        r_attn_compress_block = r_attn_compress_block.reshape(B * seq_len, -1)
        global_h_start = q_head_core_start + h_start
        global_h_stop = q_head_core_start + h_stop
        slice_q_head_group_dims = slice(global_h_start * v_head_dim, global_h_stop * v_head_dim)

        # 10. Output FC for this block, followed immediately by the original
        # cross-core shared-OCM reduction.
        for j in range(core_num):
            # core interleave on hidden_dim avoids x_shocm bank conflicts; the
            # starting block differs per core (core0 [0,1,2,3], core1 [1,2,3,0], ...)
            hidden_dim_offset = ((j + core_id) % core_num) * hidden_dims_per_core
            slice_hidden_dim = slice(hidden_dim_offset, hidden_dim_offset + hidden_dims_per_core)
            w_o_sub = w_o[slice_hidden_dim, slice_q_head_group_dims]

            w_scale_ic_slice = {
                datatype.DataType.BF16: slice(0, None),
                datatype.DataType.FP8_E4M3: slice(0, None),
                datatype.DataType.FP8_E5M2: slice(0, None),
                datatype.DataType.MXFP4: slice(global_h_start * v_head_dim // 32, global_h_stop * v_head_dim // 32),
                datatype.DataType.NVFP4: slice(global_h_start * v_head_dim // 16, global_h_stop * v_head_dim // 16),
            }[w_dtype]
            w_scale_o_sub = w_scale_o[slice_hidden_dim, w_scale_ic_slice] if w_scale_o.size > 1 else w_scale_o
            r_fc_o = ax.dsl.matmul(
                r_attn_compress_block,
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
                name=f"r_sum_shocm_{core_id}_{h_start}_{j}",
            )
            ax.dsl.add(
                x=r_fc_o,
                y=x_running[:, :, slice_hidden_dim],
                r=r_sum_shocm[:, :, slice_hidden_dim],
            )
            if multi_core:
                axd.sync_all()
            x_running = r_sum_shocm

    ax.dsl.store(r=output[:, slice_seq_len, :], x=x_running[:, slice_seq_len, :])

    return output, kv_cache_out, pe_cache_out


def mla(
    x,  # [B, seq_len, hidden_dim] bf16
    w_attn_norm,  # [hidden_dim] fp32/bf16
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
    w_q_norm,  # [q_lora_rank] fp32/bf16
    w_kv_norm,  # [kv_lora_rank] fp32/bf16
    w_o,  # [hidden_dim, q_head_num * v_head_dim]
    w_scale_o,
    w_scale2_o,
    cos,  # [max_seq_len, qk_rope_head_dim // 2]
    sin,  # [max_seq_len, qk_rope_head_dim // 2]
    position_ids,  # [B, seq_len] or [seq_len] int32
    kv_cache_out,  # [B, max_seq_len, kv_lora_rank] bf16
    pe_cache_out,  # [B, max_seq_len, qk_rope_head_dim] bf16
    output,  # [B, seq_len, hidden_dim] bf16
    *,
    softmax_scale=1.0,
    kv_offset=0,
    kv_cache_in=None,  # [B, cached_seq_len, kv_lora_rank] bf16
    pe_cache_in=None,  # [B, cached_seq_len, qk_rope_head_dim] bf16
    core_num=1,
):
    """Run fused DeepSeek-V3/R1 MLA prefill/decode on the Axera NPU.

    Parameters
    ----------
    x : (B, seq_len, hidden_dim) bfloat16
    w_attn_norm : (hidden_dim,) — input RMSNorm weight.
    w_dq, w_uq, w_dkv, w_kv_b_0, w_kv_b_1, w_o : latent-attention weights, with
        the matching ``w_scale_*`` / ``w_scale2_*`` per-weight scale arrays.
        Scales may be ``None`` / scalar for plain bf16 weights; they are
        normalized to per-output-channel identity arrays internally.
    w_q_norm : (q_lora_rank,), w_kv_norm : (kv_lora_rank,) — norm weights.
    cos, sin : (max_seq_len, qk_rope_head_dim // 2) — RoPE tables.
    position_ids : (B, seq_len) or (seq_len,) int32.
    kv_cache_out : (B, max_seq_len, kv_lora_rank) bf16 — current tokens' latent
        K cache, written at ``[kv_offset:kv_offset+seq_len]``.
    pe_cache_out : (B, max_seq_len, qk_rope_head_dim) bf16 — current tokens'
        roped PE cache, written at ``[kv_offset:kv_offset+seq_len]``.
    output : (B, seq_len, hidden_dim) bf16 — attention-layer output.
    softmax_scale : softmax temperature on the QK scores.
    kv_offset : offset into the cache tensors for the current tokens.
    kv_cache_in : (B, cached_seq_len, kv_lora_rank) bf16 or None — prefix latent
        K (e.g. a populated KV cache) the query attends to.
    pe_cache_in : (B, cached_seq_len, qk_rope_head_dim) bf16 or None — prefix
        roped PE the query attends to.
    core_num : number of NPU cores used to run the kernel.

    Returns
    -------
    (output, kv_cache_out, pe_cache_out)
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

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

    ax.init(chip_num=1, core_num=core_num)
    mla_kernel[core_num](
        x,
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
    )
    return output, kv_cache_out, pe_cache_out
