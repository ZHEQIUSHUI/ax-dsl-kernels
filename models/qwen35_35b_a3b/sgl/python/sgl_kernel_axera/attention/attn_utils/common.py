"""Shared attention graph helpers used by MLA, DSA, FlashAttention, and GDN.

The module owns layout-independent RMSNorm/RoPE recording and the shared MLA
projection prefix. FlashAttention and GDN keep their cache, recurrence, and
parallel scheduling logic local while reusing these public AX DSL helpers.
"""

from __future__ import annotations


from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.dsl import util as ax_util
from axnn.opset import datatype

from .numeric import BF16_DTYPE, MXE8M0_DTYPE, MXFP4_DTYPE, MXFP8_E4M3_DTYPE
from sgl_kernel_axera.models.bank_access import gather_rows, load_tile, store_tile


class AttentionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class AttentionShardDim(str, Enum):
    SEQUENCE = "sequence"
    HIDDEN = "hidden"


class TensorParallelLayout(str, Enum):
    TP2_ONE_CHIP = "tp2_one_chip"
    TP1 = "tp1"


class DSAIndexNorm(str, Enum):
    LAYER = "layer"
    RMS = "rms"


class DSASparseAttentionImpl(str, Enum):
    TOKEN = "token"
    BLOCK_DENSE = "block_dense"


def as_phase(value: AttentionPhase) -> AttentionPhase:
    return value if isinstance(value, AttentionPhase) else AttentionPhase(value)


def as_shard_dim(value: AttentionShardDim) -> AttentionShardDim:
    return value if isinstance(value, AttentionShardDim) else AttentionShardDim(value)


def as_tp_layout(value: TensorParallelLayout) -> TensorParallelLayout:
    return value if isinstance(value, TensorParallelLayout) else TensorParallelLayout(value)


def as_dsa_index_norm(value: DSAIndexNorm) -> DSAIndexNorm:
    return value if isinstance(value, DSAIndexNorm) else DSAIndexNorm(value)


def as_dsa_sparse_impl(value: DSASparseAttentionImpl) -> DSASparseAttentionImpl:
    return value if isinstance(value, DSASparseAttentionImpl) else DSASparseAttentionImpl(value)


def configured_run_mode(default: ax.RunMode = ax.RunMode.CRUN) -> ax.RunMode:
    """Resolve the process run mode once at the configuration boundary."""
    return ax.RunMode(os.environ.get("RUN_MODE", default.value))


def env_flag(name: str, default: bool) -> bool:
    """Parse an integer environment switch before it reaches kernel logic."""
    return bool(int(os.environ.get(name, int(default))))


MX_GROUP_SIZE = 32
MX_WEIGHT_DTYPES = (MXFP4_DTYPE, MXFP8_E4M3_DTYPE)
ACTIVATION_QUANT_BLOCK_SIZE = 128
# AXERA's MX MatMul can consume one BF16 activation tile directly.  For the
# production DSV4 projections this avoids expanding one graph node per K=128
# activation group; keep the group-wise reference path for small unit-test
# shapes where it is useful for checking the FP8 ABI.
ACTIVATION_QUANT_FAST_PATH_MIN_WIDTH = 1024
CONV_EUS_PER_CORE = 4
MHC_STATUS_SEQ_TILE = 32
MHC_FFN_QUERY_TILE = 40
MHC_COLLAPSED_HIDDEN_TILE = 256
MHC_RESIDUAL_HIDDEN_TILE = 1024
ATTENTION_PREFIX_SEQ_TILE = 16
ATTENTION_PROJECTION_SEQ_TILE = 64
ATTENTION_QUERY_SEQ_TILE = 64
CSA_TOPK_SOURCE_CAPACITY = 512
CSA_INDEX_SOURCE_BLOCK = 4096
MOE_EXPERT_CAPACITY = 128
MOE_ROUTER_TOKEN_TILE = 128
MOE_DISPATCH_EXPERT_GROUP = 4
MOE_SLOT_LOOKUP_TILE = 64
MOE_WEIGHTED_SUM_TILE = 64
MOE_WEIGHTED_SUM_HIDDEN_PARTS = 4
SHARED_EXPERT_GATE_UP_IC_TILE = 448
SHARED_EXPERT_DOWN_OC_TILE = 960
SHARED_EXPERT_DECODE_PRELOAD_WAVES_MXFP8 = 6
SHARED_EXPERT_DECODE_PRELOAD_WAVES_MXFP4 = 8


def kernel_cache_enabled() -> bool:
    """Enable the CRUN cache only for the production DSV4 launch path."""
    return (
        configured_run_mode() is ax.RunMode.CRUN
        and env_flag("SGLANG_AXERA_DSV4_ENABLE_KERNEL_CACHE", True)
        and not env_flag("DUMP_DSL_PERF", False)
    )


def kernel_cache_path() -> str:
    """Return the versioned cache directory shared by AXERA builders.

    AXNN's cache key does not include helper source files imported from an
    external distribution.  A changed helper can therefore reuse an old
    ``.compiled.axmodel``.  Namespace the cache with a stable fingerprint of
    this package so a source change starts a fresh, safe cache without
    deleting the previous artifacts.
    """
    base = Path(
        os.environ.get(
            "SGLANG_AXERA_DSV4_KERNEL_CACHE",
            os.environ.get("SGLANG_CACHE_DIR", "cache"),
        )
    )
    if not env_flag("SGLANG_AXERA_DSV4_CACHE_SOURCE_FINGERPRINT", True):
        return str(base)
    return str(base / _kernel_cache_namespace())


@lru_cache(maxsize=1)
def _kernel_cache_namespace() -> str:
    """Build a stable namespace from the installed AXERA helper sources."""
    package_root = Path(__file__).resolve().parents[2]
    digest = hashlib.blake2b(digest_size=8)
    for source in sorted(package_root.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        digest.update(source.relative_to(package_root).as_posix().encode())
        digest.update(source.read_bytes())
    return f"axera-dsv4-{digest.hexdigest()}"


@dataclass
class MHCState:
    post: object
    comb: object
    collapsed: object
    residual_path: object


@dataclass(frozen=True)
class KernelMeta:
    core_id: int
    core_num: int
    batch: int
    seq_len: int
    hidden: int


@dataclass
class AttentionPrefix:
    state: MHCState
    normalized: object
    q_residual: object
    kv: object
    cos: object
    sin: object


MLA_CONST_INPUTS = [
    "w_dq",
    "w_scale_dq",
    "w_scale2_dq",
    "w_uq",
    "w_scale_uq",
    "w_scale2_uq",
    "w_dkv",
    "w_scale_dkv",
    "w_scale2_dkv",
    "w_kv_b_0",
    "w_scale_kv_b_0",
    "w_scale2_kv_b_0",
    "w_kv_b_1",
    "w_scale_kv_b_1",
    "w_scale2_kv_b_1",
    "w_o",
    "w_scale_o",
    "w_scale2_o",
]


def rmsnorm_sqrt(x, r=None, eps=1e-6):
    """Record the FP32 ``sqrt(mean(x**2) + eps)`` RMSNorm denominator."""
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden_size = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=x.shape[-1])
    square_mean = ax.dsl.div(square_sum, hidden_size)
    epsilon = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=eps)
    return ax.dsl.sqrt(ax.dsl.add(square_mean, epsilon), r=r)


def rmsnorm(x, weight=None, eps=1e-6, output_dtype=None, r=None):
    """Record RMSNorm with FP32 reduction and a compatible output boundary."""
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden_size = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=x.shape[-1])
    square_mean = ax.dsl.div(square_sum, hidden_size)
    epsilon = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=eps)
    denominator = ax.dsl.sqrt(ax.dsl.add(square_mean, epsilon))
    target_dtype = x.dtype if output_dtype is None else output_dtype
    if weight is None:
        return ax.dsl.div(x, denominator, output_dtype=target_dtype, r=r)
    normalized = ax.dsl.div(x, denominator, output_dtype=ax.fp32)
    return ax.dsl.mul(normalized, weight, output_dtype=target_dtype, r=r)


def dsv4_query_rmsnorm_bf16(x, eps=1e-6, r=None):
    """Match model.py's BF16 rounding boundaries for the post-wq_b query norm."""
    squared = ax.dsl.mul(x, x, output_dtype=x.dtype)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    hidden_size = ax.dsl.full((1,), ax.fp32, fill_value=x.shape[-1])
    mean_square = ax.dsl.div(square_sum, hidden_size, output_dtype=x.dtype)
    epsilon = ax.dsl.full((1,), ax.fp32, fill_value=eps)
    stabilized = ax.dsl.add(mean_square, epsilon, output_dtype=x.dtype)

    # Keep the BF16 input to rsqrt, but evaluate sqrt/reciprocal in FP32 before
    # rounding the inverse RMS to BF16; AX DSL has no dedicated rsqrt primitive.
    stabilized_fp32 = ax.dsl.cast(stabilized, dtype=ax.fp32)
    denominator = ax.dsl.sqrt(stabilized_fp32, output_dtype=ax.fp32)
    inverse_rms = ax.dsl.reciprocal(denominator, output_dtype=x.dtype)
    return ax.dsl.mul(x, inverse_rms, output_dtype=x.dtype, r=r)


def rope_impl(
    x,
    cos,
    sin,
    position_ids=None,
    r=None,
    head_less=False,
    head_first=False,
    rotary_dim=None,
    banks=None,
):
    """Record rotary embedding for indexed or already-indexed position tables.

    ``position_ids`` may be omitted when ``cos`` and ``sin`` already contain
    the rows for ``x``. ``head_less`` and ``head_first`` preserve the layouts
    used by MLA/DSA and paged attention respectively; ``rotary_dim`` allows a
    partial rotary prefix while copying the remaining channels unchanged.
    """
    assert not (head_less and head_first)
    if position_ids is not None:
        cos = gather_rows(cos, position_ids, dim=0, banks=banks)
        sin = gather_rows(sin, position_ids, dim=0, banks=banks)
    if not head_less:
        if head_first:
            cos = cos.reshape(1, cos.shape[0], cos.shape[1])
            sin = sin.reshape(1, sin.shape[0], sin.shape[1])
        else:
            cos = cos.reshape(cos.shape[0], 1, cos.shape[1])
            sin = sin.reshape(sin.shape[0], 1, sin.shape[1])
    head_dim = x.shape[-1]
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert cos.shape[-1] == sin.shape[-1] == rotary_dim // 2

    first_half = x[..., slice(0, rotary_dim // 2)]
    second_half = x[..., slice(rotary_dim // 2, rotary_dim)]
    first_cos = ax.dsl.mul(first_half, cos, output_dtype=ax.fp32)
    second_sin = ax.dsl.mul(second_half, sin, output_dtype=ax.fp32)
    second_cos = ax.dsl.mul(second_half, cos, output_dtype=ax.fp32)
    first_sin = ax.dsl.mul(first_half, sin, output_dtype=ax.fp32)

    if r is None:
        rotated_first = ax.dsl.sub(first_cos, second_sin, output_dtype=x.dtype)
        rotated_second = ax.dsl.add(second_cos, first_sin, output_dtype=x.dtype)
        rotated = ax.dsl.concat(rotated_first, rotated_second, dim=-1)
        if rotary_dim < head_dim:
            return ax.dsl.concat(rotated, x[..., slice(rotary_dim, head_dim)], dim=-1)
        return rotated

    ax.dsl.sub(first_cos, second_sin, r=r[..., slice(0, rotary_dim // 2)])
    ax.dsl.add(second_cos, first_sin, r=r[..., slice(rotary_dim // 2, rotary_dim)])
    if rotary_dim < head_dim:
        ax.dsl.move(x[..., slice(rotary_dim, head_dim)], r=r[..., slice(rotary_dim, head_dim)])
    return r


def normalize_mrope_section(mrope_section):
    """Return a three-entry ``(t, h, w)`` M-RoPE section tuple."""
    if mrope_section is None:
        raise ValueError("mrope_section is required")
    section = tuple(int(value) for value in mrope_section)
    if len(section) != 3:
        raise ValueError(f"mrope_section must have 3 entries [t, h, w], got {mrope_section}")
    if any(value < 0 for value in section):
        raise ValueError(f"mrope_section values must be non-negative, got {section}")
    return section


def mrope_channel_masks(mrope_section, interleaved=False, dtype=np.float32):
    """Build per-channel T/H/W masks for chunked or interleaved M-RoPE."""
    t_size, h_size, w_size = normalize_mrope_section(mrope_section)
    rotary_half = t_size + h_size + w_size
    t_mask = np.zeros((rotary_half,), dtype=dtype)
    h_mask = np.zeros((rotary_half,), dtype=dtype)
    w_mask = np.zeros((rotary_half,), dtype=dtype)
    if interleaved:
        # Match the interleaved RoPE layout: T, H, and W occupy channels 0, 1, and 2.
        t_mask[:] = 1
        t_mask[1 : h_size * 3 : 3] = 0
        t_mask[2 : w_size * 3 : 3] = 0
        h_mask[1 : h_size * 3 : 3] = 1
        w_mask[2 : w_size * 3 : 3] = 1
    else:
        t_mask[:t_size] = 1
        h_mask[t_size : t_size + h_size] = 1
        w_mask[t_size + h_size :] = 1
    return t_mask, h_mask, w_mask


def mix_mrope_tables(t_table, h_table, w_table, mrope_section, interleaved=False):
    """Mix T/H/W cosine or sine rows into one M-RoPE table."""
    t_mask, h_mask, w_mask = mrope_channel_masks(mrope_section, interleaved=interleaved, dtype=np.float32)
    if isinstance(t_table, torch.Tensor):
        t_m = torch.as_tensor(t_mask, device=t_table.device, dtype=t_table.dtype)
        h_m = torch.as_tensor(h_mask, device=h_table.device, dtype=h_table.dtype)
        w_m = torch.as_tensor(w_mask, device=w_table.device, dtype=w_table.dtype)
        return t_table * t_m + h_table * h_m + w_table * w_m
    return t_table * t_mask + h_table * h_mask + w_table * w_mask


def gather_mrope_cos_sin(
    cos,
    sin,
    position_ids,
    mrope_section,
    interleaved=False,
    cos_sin_cache=None,
    banks=None,
):
    """Index T/H/W positions and mix them into one rotary cosine/sine table."""
    section = normalize_mrope_section(mrope_section)
    rotary_half = section[0] + section[1] + section[2]
    if cos_sin_cache is None:
        assert cos.shape[-1] == sin.shape[-1] == rotary_half
    else:
        assert cos_sin_cache.shape[-1] == 2 * rotary_half
    assert position_ids.shape[0] == 3

    t_pos = position_ids[0:1].reshape(position_ids.shape[-1])
    h_pos = position_ids[1:2].reshape(position_ids.shape[-1])
    w_pos = position_ids[2:3].reshape(position_ids.shape[-1])
    if cos_sin_cache is None:
        t_cos = gather_rows(cos, t_pos, dim=0, banks=banks)
        h_cos = gather_rows(cos, h_pos, dim=0, banks=banks)
        w_cos = gather_rows(cos, w_pos, dim=0, banks=banks)
        t_sin = gather_rows(sin, t_pos, dim=0, banks=banks)
        h_sin = gather_rows(sin, h_pos, dim=0, banks=banks)
        w_sin = gather_rows(sin, w_pos, dim=0, banks=banks)
    else:
        # Gather complete contiguous rows before splitting on chip. Slicing
        # the full DDR cache first would require strided external inputs.
        t_rows = gather_rows(cos_sin_cache, t_pos, dim=0, banks=banks)
        h_rows = gather_rows(cos_sin_cache, h_pos, dim=0, banks=banks)
        w_rows = gather_rows(cos_sin_cache, w_pos, dim=0, banks=banks)
        t_cos, t_sin = t_rows[:, :rotary_half], t_rows[:, rotary_half:]
        h_cos, h_sin = h_rows[:, :rotary_half], h_rows[:, rotary_half:]
        w_cos, w_sin = w_rows[:, :rotary_half], w_rows[:, rotary_half:]

    t_mask, h_mask, w_mask = mrope_channel_masks(section, interleaved=interleaved, dtype=np.float32)
    t_m = ax.dsl.from_numpy(t_mask)
    h_m = ax.dsl.from_numpy(h_mask)
    w_m = ax.dsl.from_numpy(w_mask)
    rope_cos = ax.dsl.add(
        ax.dsl.add(ax.dsl.mul(t_cos, t_m), ax.dsl.mul(h_cos, h_m)),
        ax.dsl.mul(w_cos, w_m),
    )
    rope_sin = ax.dsl.add(
        ax.dsl.add(ax.dsl.mul(t_sin, t_m), ax.dsl.mul(h_sin, h_m)),
        ax.dsl.mul(w_sin, w_m),
    )
    return rope_cos, rope_sin


def record_mla_shared_projections(
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
    *,
    core_id,
    core_num,
    seq_slice,
    q_lora_slice,
    kv_pe_slice,
    kv_lora_rank,
    qk_rope_head_dim,
    seq_lens=None,
    name_prefix="mla",
):
    """Record the dense and sparse MLA shared projection prefix.

    This is the algorithmic fork point shared by dense MLA and DSA. It records
    input RMSNorm, query down-projection and RMSNorm, compressed KV/PE
    projection, KV RMSNorm, and PE RoPE. Attention selection and output
    projection remain with their dense or sparse caller.

    Returns:
        ``(x_shared, x_norm, q_lora, kv_pe, position_ids, seq_mask)``. The
        first four tensors use OCM for one core and SHOCM for multiple cores.
        ``seq_mask`` is INT32 ``[B, S, 1]`` when ``seq_lens`` is supplied and
        otherwise ``None``.
    """
    multi_core = core_num > 1
    batch, seq_len, _ = x.shape
    seq_count = seq_slice.stop - seq_slice.start
    q_lora_rank = w_dq.shape[0]
    kv_pe_dim = w_dkv.shape[0]
    assert kv_pe_dim == kv_lora_rank + qk_rope_head_dim

    memory = ax.MemoryType.SHARED_OCM if multi_core else ax.MemoryType.OCM
    x_shared = ax.dsl.create_tensor(x.shape, x.dtype, memory, name=f"{name_prefix}_x_shared")
    if seq_count > 0:
        ax.dsl.load(r=x_shared[:, seq_slice, :], x=x[:, seq_slice, :])
    if multi_core:
        axd.sync_all()

    attn_norm = ax.dsl.create_tensor(
        w_attn_norm.shape,
        w_attn_norm.dtype,
        memory,
        name=f"{name_prefix}_attn_norm",
    )
    if seq_count > 0:
        ax.dsl.load(r=attn_norm, x=w_attn_norm)
    if multi_core:
        axd.sync_all()

    x_norm = ax.dsl.create_tensor(x.shape, x.dtype, memory, name=f"{name_prefix}_x_norm")
    if seq_count > 0:
        ax.dsl.move(
            rmsnorm(x_shared[:, seq_slice, :], attn_norm),
            r=x_norm[:, seq_slice, :],
        )
    if multi_core:
        axd.sync_all()

    q_down = ax.dsl.create_tensor(
        (batch, seq_len, q_lora_rank),
        x.dtype,
        memory,
        name=f"{name_prefix}_q_down",
    )
    ax.dsl.matmul(
        x_norm,
        w_dq[q_lora_slice, :],
        w_scale=w_scale_dq[q_lora_slice, :],
        w_scale2=w_scale2_dq,
        transposed=True,
        output_dtype=ax.bf16,
        const_inputs=["y"],
        r=q_down[:, :, q_lora_slice],
    )
    if multi_core:
        axd.sync_all()

    q_lora = ax.dsl.create_tensor(
        q_down.shape,
        q_down.dtype,
        memory,
        name=f"{name_prefix}_q_lora",
    )
    if seq_count > 0:
        ax.dsl.move(
            rmsnorm(q_down[:, seq_slice, :], ax.dsl.load(w_q_norm)),
            r=q_lora[:, seq_slice, :],
        )
    if multi_core:
        axd.sync_all()

    loaded_position_ids = ax.dsl.load(position_ids)
    kv_pe_raw = ax.dsl.create_tensor(
        (batch, seq_len, kv_pe_dim),
        x.dtype,
        memory,
        name=f"{name_prefix}_kv_pe_raw",
    )
    ax.dsl.matmul(
        x_norm,
        w_dkv[kv_pe_slice, :],
        w_scale=w_scale_dkv[kv_pe_slice, :],
        w_scale2=w_scale2_dkv,
        transposed=True,
        output_dtype=ax.bf16,
        const_inputs=["y"],
        r=kv_pe_raw[:, :, kv_pe_slice],
    )
    if multi_core:
        axd.sync_all()

    kv_pe = ax.dsl.create_tensor(
        (batch, seq_len, kv_pe_dim),
        x.dtype,
        memory,
        name=f"{name_prefix}_kv_pe",
    )
    kv = kv_pe[:, :, :kv_lora_rank]
    if seq_count > 0:
        ax.dsl.move(
            rmsnorm(
                kv_pe_raw[:, seq_slice, :kv_lora_rank],
                ax.dsl.load(w_kv_norm),
            ),
            r=kv[:, seq_slice, :],
        )
    if multi_core:
        axd.sync_all()
    rope_impl(
        kv_pe_raw[:, :, kv_lora_rank:],
        cos,
        sin,
        loaded_position_ids,
        kv_pe[:, :, kv_lora_rank:],
        True,
    )

    seq_mask = None
    if seq_lens is not None:
        loaded_seq_lens = ax.dsl.load(seq_lens)
        positions = ax.dsl.from_numpy(np.arange(seq_len, dtype=np.int32)).reshape(1, seq_len, 1)
        one = ax.dsl.from_numpy(np.ones((1,), dtype=np.int32))
        zero = ax.dsl.from_numpy(np.zeros((1,), dtype=np.int32))
        seq_mask = loaded_seq_lens.reshape(batch, 1, 1) - positions
        seq_mask = ax.dsl.minimum(seq_mask, one)
        seq_mask = ax.dsl.maximum(seq_mask, zero)
        masked_kv_pe = ax.dsl.create_tensor(
            kv_pe.shape,
            kv_pe.dtype,
            memory,
            name=f"{name_prefix}_kv_pe_masked",
        )
        ax.dsl.mul(
            kv_pe,
            ax.dsl.cast(seq_mask, dtype=kv_pe.dtype),
            output_dtype=kv_pe.dtype,
            r=masked_kv_pe,
        )
        kv_pe = masked_kv_pe

    return x_shared, x_norm, q_lora, kv_pe, loaded_position_ids, seq_mask


def record_mla_absorbed_query(
    q_lora,
    w_uq,
    w_scale_uq,
    w_scale2_uq,
    w_kv_b_key,
    w_scale_kv_b_key,
    w_scale2_kv_b_key,
    cos,
    sin,
    position_ids,
    *,
    global_head_start,
    global_head_stop,
    name_prefix="mla",
):
    """Record shared MLA Q-Up, RoPE, and key absorption for one head block."""
    batch, seq_len, _ = q_lora.shape
    _, kv_lora_rank, qk_nope_head_dim = w_kv_b_key.shape
    q_head_width = w_uq.shape[0] // w_kv_b_key.shape[0]
    qk_rope_head_dim = q_head_width - qk_nope_head_dim
    head_count = global_head_stop - global_head_start
    output_slice = slice(
        global_head_start * q_head_width,
        global_head_stop * q_head_width,
    )
    q_up = ax.dsl.matmul(
        q_lora,
        w_uq[output_slice, :],
        w_scale=w_scale_uq[output_slice, :],
        w_scale2=w_scale2_uq,
        transposed=True,
        output_dtype=ax.bf16,
        const_inputs=["y"],
    ).reshape(batch, seq_len, head_count, q_head_width)
    q_nope = q_up[:, :, :, :qk_nope_head_dim]
    q_pe = q_up[:, :, :, qk_nope_head_dim:]

    absorbed = ax.dsl.create_tensor(
        (batch, seq_len, head_count, kv_lora_rank + qk_rope_head_dim),
        q_lora.dtype,
        name=f"{name_prefix}_absorbed_query_h{global_head_start}",
    )
    rope_impl(
        q_pe,
        cos,
        sin,
        position_ids,
        absorbed[:, :, :, kv_lora_rank:],
    )
    for global_head in range(global_head_start, global_head_stop):
        local_head = global_head - global_head_start
        ax.dsl.matmul(
            q_nope[:, :, local_head : local_head + 1, :].reshape(batch, seq_len, qk_nope_head_dim),
            w_kv_b_key[global_head : global_head + 1, :, :].reshape(kv_lora_rank, qk_nope_head_dim),
            w_scale=w_scale_kv_b_key[global_head : global_head + 1, :, :].reshape(kv_lora_rank, -1),
            w_scale2=w_scale2_kv_b_key,
            transposed=True,
            output_dtype=ax.bf16,
            const_inputs=["y"],
            r=absorbed[:, :, local_head : local_head + 1, :kv_lora_rank].reshape(batch, seq_len, kv_lora_rank),
        )
    return absorbed


def normalize_fc_scale(scale, weight):
    """Return a contiguous 2-D scale for a projection weight."""
    output_channels = weight.shape[0]
    if scale is None:
        return np.ones((output_channels, 1), dtype=np.float32)
    scale = np.asarray(scale)
    if scale.size == 1:
        return np.full((output_channels, 1), scale.reshape(-1)[0], dtype=np.float32)
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    if scale.shape[0] != output_channels:
        raise ValueError(f"scale shape {scale.shape} is incompatible with weight shape " f"{weight.shape}")
    return np.ascontiguousarray(scale)


def normalize_head_fc_scale(scale, weight):
    """Return scales for a per-head ``[head, output, input]`` weight."""
    head_count, output_channels, _ = weight.shape
    if scale is None:
        return np.ones((head_count, output_channels, 1), dtype=np.float32)
    scale = np.asarray(scale)
    if scale.size == 1:
        return np.full(
            (head_count, output_channels, 1),
            scale.reshape(-1)[0],
            dtype=np.float32,
        )
    if scale.ndim == 3 and scale.shape[:2] == (head_count, output_channels):
        return np.ascontiguousarray(scale)
    if scale.ndim == 2 and scale.shape == (head_count, output_channels):
        return np.ascontiguousarray(scale.reshape(head_count, output_channels, 1), dtype=np.float32)
    if scale.ndim == 1 and scale.shape[0] == head_count:
        return np.ascontiguousarray(
            np.repeat(scale.reshape(head_count, 1, 1), output_channels, axis=1),
            dtype=np.float32,
        )
    raise ValueError(f"scale shape {scale.shape} is incompatible with weight shape {weight.shape}")


def normalize_scale2(scale):
    """Return a numeric FP32 secondary scale, using identity when absent."""
    if scale is None:
        return np.ones((1,), dtype=np.float32)
    scale = np.asarray(scale).reshape(-1)
    if scale.size != 1:
        raise ValueError(f"secondary scale must contain one value, got shape {scale.shape}")
    # A structured low-bit dtype is only a generator placeholder: the format
    # has no numeric secondary scale, so its runtime meaning is identity.
    if scale.dtype.fields is not None:
        return np.ones((1,), dtype=np.float32)
    return np.ascontiguousarray(scale, dtype=np.float32)


def scale_or_default(scale):
    """Return the scalar identity scale when an optional FC scale is absent."""
    return 1.0 if scale is None else scale


def slice_fc_scale(scale, slice_oc, slice_ic=None, full_ic=None):
    """Slice per-output or grouped FC scales consistently with an ``[OC, IC]`` weight shard."""
    if scale is None or not isinstance(scale, np.ndarray) or scale.size <= 1:
        return scale_or_default(scale)
    if slice_ic is None or full_ic is None:
        return scale[slice_oc, :]

    scale_group_count = scale.shape[1]
    if scale_group_count == 1:
        return scale[slice_oc, :]
    if scale_group_count == full_ic // 16:
        group_size = 16
    elif scale_group_count == full_ic // 32:
        group_size = 32
    elif scale_group_count == full_ic:
        group_size = 1
    else:
        raise AssertionError(f"unsupported FC scale shape {scale.shape} for input width {full_ic}")

    ic_start = 0 if slice_ic.start is None else slice_ic.start
    ic_stop = full_ic if slice_ic.stop is None else slice_ic.stop
    assert ic_start % group_size == 0, "FC input slice start must align to its scale group"
    assert ic_stop % group_size == 0, "FC input slice stop must align to its scale group"
    return scale[slice_oc, ic_start // group_size : ic_stop // group_size]


def const_fc_scale(weights: dict, name: str):
    """Scale for a constant-weight projection: numpy quant scales stay numpy.

    Quantized weights (NVFP4 fp8 per-group, FP8 block, AWQ/GPTQ) reach the
    kernel as numpy constants and are dequantized by the constant-weight FC op,
    which needs a numpy scale of the original dtype. Only a genuinely missing
    scale falls back to a torch scalar for the plain DDR matmul path.
    """
    value = weights.get(name)
    if value is None:
        return torch.ones((1,), dtype=torch.float32)
    if not isinstance(value, torch.Tensor):
        return np.asarray(value, dtype=np.float32)
    return value


def zeropoint_arg(weights: dict, name: str):
    """Per-group zeropoint const for AWQ/GPTQ; a zero scalar for the rest.

    Symmetric formats (NVFP4/MXFP4/FP8) and unquantized weights carry no
    zeropoint, so the kernel receives a size-1 numpy zero and takes the
    symmetric matmul path.
    """
    value = weights.get(name)
    if value is None or np.asarray(value).size <= 1:
        return np.zeros((1,), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def weight_num_bits(weight) -> int:
    """Bit width of a projection weight, for numpy consts and DSL tensors alike.

    Quantized numpy weights (NVFP4/MXFP4/INT4) carry an axnn datatype; BF16/FP16
    DSL tensors expose ``dtype.num_bits``.
    """
    num_bits = getattr(getattr(weight, "dtype", None), "num_bits", None)
    if num_bits is not None:
        return int(num_bits)
    if isinstance(weight, np.ndarray):
        return int(datatype.DType(weight.dtype).num_bits)
    return 16


def normalize_group_fc_param(weight, value, name):
    """Lay out a group param like ``[groups, OC]`` to match a ``[OC, IC]`` weight.

    AWQ/GPTQ checkpoints deliver per-group scales and zeropoints as ``[OC,
    groups]``; the constant-weight FC op expects the transposed layout.
    """
    value = np.asarray(value, dtype=np.float32)
    output_channels, input_channels = weight.shape
    if value.ndim != 2:
        raise ValueError(f"group-quantized FC requires 2-D {name}, got {value.shape}")
    if value.shape[1] == output_channels and input_channels % value.shape[0] == 0:
        return np.ascontiguousarray(value)
    if value.shape[0] == output_channels and input_channels % value.shape[1] == 0:
        return np.ascontiguousarray(value.transpose(1, 0))
    raise ValueError(f"{name} shape {value.shape} does not match weight shape {weight.shape}")


def group_quant_matmul(
    x,
    weight,
    *,
    w_scale=1.0,
    w_scale2=1.0,
    w_zeropoint=None,
    output_dtype=None,
    transposed=True,
    const_inputs=None,
    r=None,
):
    """MatMul that reproduces the checkpoint's weight-only quant contract.

    NVFP4/MXFP4 use the BF16/FP16 activation directly and dequantize only the
    weight through the MatMul scale arguments. Asymmetric INT4 (AWQ/GPTQ) adds a
    per-group zeropoint, which the group-quant FC applies as ``(w - zeropoint) *
    scale``. A ``None`` or scalar ``w_zeropoint`` selects the symmetric path.
    """
    if const_inputs is None:
        const_inputs = ["y"]
    has_zeropoint = w_zeropoint is not None and isinstance(weight, np.ndarray) and np.asarray(w_zeropoint).size > 1
    if not has_zeropoint:
        return ax.dsl.matmul(
            x,
            weight,
            w_scale=w_scale,
            w_scale2=w_scale2,
            output_dtype=output_dtype,
            transposed=transposed,
            const_inputs=const_inputs,
            r=r,
        )
    oriented_weight = weight if transposed else weight.transpose(0, 1)
    group_scale = normalize_group_fc_param(oriented_weight, np.asarray(w_scale), "w_scale")
    group_zeropoint = normalize_group_fc_param(oriented_weight, np.asarray(w_zeropoint), "w_zeropoint")
    attrs = {
        "output_dtype": output_dtype,
        "const_inputs": {
            "w": oriented_weight,
            "b": np.zeros((oriented_weight.shape[0],), dtype=np.float32),
            "w_scale": group_scale,
            "w_zeropoint": group_zeropoint,
        },
    }
    return ax_util.op_convert(
        "AxGroupQuantizedWeightFullyConnected",
        attrs,
        {"x": x},
        {"r": r},
    )


def load_attention_prefix(q_residual, kv, cos, sin, meta):
    """Publish split-attention projection tensors in SHARED_OCM.

    SWA, CSA, and HCA core entries consume the same four tensors.  Keeping
    this publication sequence in the shared helper prevents one variant from
    changing the token ownership or barrier order of the others.
    """
    local_token_slice = token_slice(meta.core_id, meta.core_num, meta.seq_len)
    q_residual_shared = ax.dsl.create_tensor(
        q_residual.shape,
        q_residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="split_q_residual",
    )
    kv_shared = ax.dsl.create_tensor(kv.shape, kv.dtype, ax.MemoryType.SHARED_OCM, name="split_kv")
    cos_shared = ax.dsl.create_tensor(cos.shape, cos.dtype, ax.MemoryType.SHARED_OCM, name="split_cos")
    sin_shared = ax.dsl.create_tensor(sin.shape, sin.dtype, ax.MemoryType.SHARED_OCM, name="split_sin")
    if local_token_slice.start < local_token_slice.stop:
        ax.dsl.load(
            q_residual[:, local_token_slice, :],
            r=q_residual_shared[:, local_token_slice, :],
        )
        ax.dsl.load(kv[:, local_token_slice, :], r=kv_shared[:, local_token_slice, :])
        ax.dsl.load(cos[:, local_token_slice, :], r=cos_shared[:, local_token_slice, :])
        ax.dsl.load(sin[:, local_token_slice, :], r=sin_shared[:, local_token_slice, :])
    axd.sync_all()
    return AttentionPrefix(
        state=None,
        normalized=None,
        q_residual=q_residual_shared,
        kv=kv_shared,
        cos=cos_shared,
        sin=sin_shared,
    )


def token_slice(core_id, core_num, seq_len):
    rows = (seq_len + core_num - 1) // core_num
    start = min(core_id * rows, seq_len)
    return slice(start, min(start + rows, seq_len))


def constant(value, name):
    # Output: [*value.shape], value.dtype.
    return ax.dsl.from_numpy(np.ascontiguousarray(value), prefix=name)


def dsl_dtype(dtype):
    for torch_dtype, dsl_type in (
        (torch.float32, ax.fp32),
        (torch.bfloat16, ax.bf16),
        (torch.float8_e4m3fn, ax.mxfp8_e4m3),
        (torch.float8_e8m0fnu, ax.mxe8m0),
        (torch.float4_e2m1fn_x2, ax.mxfp4),
    ):
        if dtype is torch_dtype:
            return dsl_type
    if isinstance(dtype, type(ax.bf16)):
        return dtype
    try:
        dtype_type = datatype.DType(dtype).type
    except (AssertionError, KeyError, TypeError, ValueError):
        dtype_type = None
    dtype_map = {
        datatype.DataType.FP32: ax.fp32,
        datatype.DataType.BF16: ax.bf16,
        datatype.DataType.FP8_E4M3: ax.mxfp8_e4m3,
        datatype.DataType.MXFP8_E4M3: ax.mxfp8_e4m3,
        datatype.DataType.NVFP4: ax.nvfp4,
        datatype.DataType.MXFP4: ax.mxfp4,
        datatype.DataType.MXE8M0: ax.mxe8m0,
    }
    if dtype_type in dtype_map:
        return dtype_map[dtype_type]
    try:
        dtype = np.dtype(dtype)
    except TypeError:
        raise AssertionError(f"unsupported DeepSeek V4 dtype: {dtype}") from None
    if dtype == BF16_DTYPE:
        return ax.bf16
    if dtype == MXFP4_DTYPE:
        return ax.mxfp4
    if dtype == MXFP8_E4M3_DTYPE:
        return ax.mxfp8_e4m3
    if dtype == MXE8M0_DTYPE:
        return ax.mxe8m0
    raise AssertionError(f"unsupported DeepSeek V4 dtype: {dtype}")


def mx_scale_slice(weight, scale, output_slice=None, input_slice=None):
    """Slice one E8M0 MX scale matrix alongside a contiguous ``[OC, IC]`` weight view."""
    if dsl_dtype(weight.dtype) is ax.bf16:
        return 1.0
    # Some checkpoint projections use one tensor-wide scale instead of an MX
    # matrix.  A scalar scale applies to every tile and must not be indexed as
    # a two-dimensional ``[OC, IC/group]`` tensor.
    if len(scale.shape) < 2:
        return scale
    output_slice = slice(0, weight.shape[0]) if output_slice is None else output_slice
    input_slice = slice(0, weight.shape[1]) if input_slice is None else input_slice
    output_start, output_stop, _ = output_slice.indices(weight.shape[0])
    input_start, input_stop, _ = input_slice.indices(weight.shape[1])
    return scale[
        output_start:output_stop,
        input_start // MX_GROUP_SIZE : input_stop // MX_GROUP_SIZE,
    ]


def validate_weight_scales(weights, weight_scales):
    """Validate one entry kernel's BF16 or MX constant-weight bundle."""
    for name, weight in weights.items():
        weight_dtype = np.dtype(weight.dtype)
        assert (
            weight_dtype == BF16_DTYPE or weight_dtype in MX_WEIGHT_DTYPES
        ), f"unsupported {name} dtype: {weight_dtype}"
        if weight_dtype == BF16_DTYPE:
            continue
        assert name in weight_scales, f"missing MX scale for {name}"
        scale = weight_scales[name]
        assert scale.dtype == np.uint8
        assert weight.shape[-1] % MX_GROUP_SIZE == 0
        assert scale.shape == (
            *weight.shape[:-1],
            weight.shape[-1] // MX_GROUP_SIZE,
        ), f"{name} scale must be {(*weight.shape[:-1], weight.shape[-1] // MX_GROUP_SIZE)}, got {scale.shape}"


def constant_linear(x, weight, weight_scale=None, *, output_dtype=None, r=None):
    scale = mx_scale_slice(weight, weight_scale)
    # Output: [*x.shape[:-1], weight.shape[0]], output_dtype or x.dtype.
    return ax.dsl.matmul(
        x,
        np.ascontiguousarray(weight),
        w_scale=scale,
        transposed=True,
        output_dtype=x.dtype if output_dtype is None else output_dtype,
        const_inputs=["y"],
        r=r,
    )


def matmul_optional_scale2(
    x,
    weight,
    *,
    w_scale=None,
    w_scale2=None,
    output_dtype=None,
    transposed=False,
    output_add_assign=False,
    r=None,
):
    """Record a runtime-weight matmul without materializing an unused scale2.

    ``ax.dsl.matmul`` currently inserts a ``from_numpy([1])`` tensor whenever
    a scaled runtime weight is used with ``w_scale2=None``. MXFP4/MXFP8 and
    the DSV4 FP4 projections have no secondary scale, so that tensor is only a
    graph input and becomes a useless CRUN load job. Calling the same public
    op conversion with the optional input omitted preserves the AxMatMul
    contract while keeping the graph free of that synthetic input.
    """
    if w_scale2 is None and isinstance(w_scale, ax_util.DslTensor):
        attrs = {
            "transposed": transposed,
            "output_dtype": output_dtype,
            "output_add_assign": output_add_assign,
        }
        return ax_util.op_convert(
            "AxMatMul",
            attrs,
            {"x": x, "y": weight, "y_scale": w_scale},
            {"r": r},
        )
    return ax.dsl.matmul(
        x,
        weight,
        w_scale=1.0 if w_scale is None else w_scale,
        w_scale2=w_scale2,
        output_dtype=output_dtype,
        transposed=transposed,
        output_add_assign=output_add_assign,
        r=r,
    )


def runtime_weight_scale_slice(weight, weight_scale, output_slice=None, input_slice=None):
    """Slice a runtime E8M0 scale using the weight's logical IC dimension."""
    if dsl_dtype(weight.dtype) is ax.bf16:
        return weight_scale
    if len(weight_scale.shape) < 2:
        return weight_scale
    output_slice = slice(0, weight.shape[0]) if output_slice is None else output_slice
    input_slice = slice(0, weight.shape[1]) if input_slice is None else input_slice
    output_start, output_stop, _ = output_slice.indices(weight.shape[0])
    input_start, input_stop, _ = input_slice.indices(weight.shape[1])
    return weight_scale[
        output_start:output_stop,
        input_start // MX_GROUP_SIZE : input_stop // MX_GROUP_SIZE,
    ]


def uses_model_activation_quant(weight):
    """Return whether ``model.py::linear`` quantizes inputs for this weight."""
    return dsl_dtype(weight.dtype) in (ax.nvfp4, ax.mxfp4, ax.mxfp8_e4m3, ax.fp8_e4m3)


def model_activation_quantize(x):
    """Return model.py's UE8M0, K=128 FP8 activation and per-block scales."""
    width = x.shape[-1]
    if width % ACTIVATION_QUANT_BLOCK_SIZE:
        raise ValueError(
            "model activation quantization requires the K dimension to be "
            f"divisible by {ACTIVATION_QUANT_BLOCK_SIZE}, got {width}"
        )

    rows = x.shape_size // width
    groups = width // ACTIVATION_QUANT_BLOCK_SIZE
    x_fp32 = ax.dsl.cast(x.reshape(rows, width), dtype=ax.fp32)
    grouped = x_fp32.reshape(rows, groups, ACTIVATION_QUANT_BLOCK_SIZE)
    negative_one = ax.dsl.full((1,), ax.fp32, fill_value=-1.0)
    absolute = ax.dsl.maximum(
        grouped,
        ax.dsl.mul(grouped, negative_one, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    amax = ax.dsl.max(absolute, dim=-1, keepdim=True)
    minimum_amax = ax.dsl.full(amax.shape, ax.fp32, fill_value=1e-4)
    amax = ax.dsl.maximum(amax, minimum_amax, output_dtype=ax.fp32)

    fp8_max = ax.dsl.full((1,), ax.fp32, fill_value=448.0)
    scale = ax.dsl.div(amax, fp8_max, output_dtype=ax.fp32)
    scale_exponent = ax.dsl.ceil(ax.dsl.log2(scale, output_dtype=ax.fp32))
    ln_two = ax.dsl.full((1,), ax.fp32, fill_value=float(np.log(2.0)))
    scale = ax.dsl.exp(
        ax.dsl.mul(scale_exponent, ln_two, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )

    scaled = ax.dsl.div(grouped, scale, output_dtype=ax.fp32)
    lower = ax.dsl.full((1,), ax.fp32, fill_value=-448.0)
    upper = ax.dsl.full((1,), ax.fp32, fill_value=448.0)
    scaled = ax.dsl.minimum(
        ax.dsl.maximum(scaled, lower, output_dtype=ax.fp32),
        upper,
        output_dtype=ax.fp32,
    )
    quantized = ax.dsl.cast(scaled, dtype=ax.fp8_e4m3)
    return quantized.reshape(rows, width), scale.reshape(rows, groups)


def model_activation_dequantize(x):
    """Materialize model.py's quantized activation at one BF16 boundary.

    ``model.py`` feeds the FP8 payload and its UE8M0 scale to GEMM.  AXERA's
    runtime-weight MatMul has no row/group activation-scale input, so the
    equivalent backend representation is the dequantized BF16 activation. It
    preserves the FP8 rounding and scale exponent while allowing one MatMul
    per output tile instead of one MatMul per K=128 group.
    """
    quantized, scale = model_activation_quantize(x)
    rows = quantized.shape_size // quantized.shape[-1]
    groups = quantized.shape[-1] // ACTIVATION_QUANT_BLOCK_SIZE
    grouped = ax.dsl.cast(
        quantized.reshape(rows, groups, ACTIVATION_QUANT_BLOCK_SIZE),
        dtype=ax.fp32,
    )
    scaled = ax.dsl.mul(
        grouped,
        scale.reshape(rows, groups, 1),
        output_dtype=ax.bf16,
    )
    return scaled.reshape(rows, quantized.shape[-1])


def prepare_model_activation(x, weight):
    """Prepare one activation for a model.py-compatible MX projection.

    The recorded activation quantization is independent of the weight values.
    Callers that apply several projections to the same activation can therefore
    record this subgraph once and pass the returned BF16 tensor to
    ``runtime_linear(..., activation_input=...)``.
    """
    if uses_model_activation_quant(weight) and x.shape[-1] >= ACTIVATION_QUANT_FAST_PATH_MIN_WIDTH:
        return model_activation_dequantize(x)
    return x


def model_activation_matmul(x, weight, weight_scale, activation_scale, *, output_dtype=None):
    """Apply model.py's K=128 activation scales around MX-scaled MatMuls."""
    input_dim = x.shape[-1]
    if input_dim % ACTIVATION_QUANT_BLOCK_SIZE:
        raise ValueError(
            "model activation quantization requires the K dimension to be "
            f"divisible by {ACTIVATION_QUANT_BLOCK_SIZE}, got {input_dim}"
        )
    weight_groups = input_dim // MX_GROUP_SIZE
    activation_groups = input_dim // ACTIVATION_QUANT_BLOCK_SIZE
    groups_per_activation = ACTIVATION_QUANT_BLOCK_SIZE // MX_GROUP_SIZE
    if weight_scale.shape != (weight.shape[0], weight_groups):
        raise ValueError(
            "model activation quantization requires one K=32 weight scale per "
            f"group, expected {(weight.shape[0], weight_groups)}, got {weight_scale.shape}"
        )
    if dsl_dtype(weight_scale.dtype) is not ax.mxe8m0:
        raise ValueError(
            "model activation quantization requires UE8M0 weight scales, " f"got {dsl_dtype(weight_scale.dtype)}"
        )

    result_dtype = x.dtype if output_dtype is None else output_dtype
    result = None
    for group in range(activation_groups):
        k_slice = slice(
            group * ACTIVATION_QUANT_BLOCK_SIZE,
            (group + 1) * ACTIVATION_QUANT_BLOCK_SIZE,
        )
        weight_group_slice = slice(
            group * groups_per_activation,
            (group + 1) * groups_per_activation,
        )
        partial = matmul_optional_scale2(
            x[:, k_slice],
            weight[:, k_slice],
            w_scale=weight_scale[:, weight_group_slice],
            transposed=True,
            output_dtype=ax.fp32,
        )
        partial = ax.dsl.mul(
            partial,
            activation_scale[:, group : group + 1],
            output_dtype=ax.fp32,
        )
        result = partial if result is None else ax.dsl.add(result, partial, output_dtype=ax.fp32)

    return result if result_dtype == ax.fp32 else ax.dsl.cast(result, dtype=result_dtype)


def runtime_linear(
    x,
    weight,
    weight_scale,
    *,
    output_dtype=None,
    r=None,
    output_tile=128,
    input_load_tile=1024,
    row_tile=None,
    activation_input=None,
    banks=None,
):
    """Record a runtime-weight linear projection with bounded OCM weight tiles.

    A weight-load tile is wider than one MatMul tile for MX weights.  This keeps
    the weight in OCM while reusing each DDR slice for four output-channel
    MatMuls, matching the established Qwen projection layout.  BF16 weights use
    a smaller resident tile to avoid excessive OCM pressure.
    """
    input_dim = x.shape[-1]
    output_dim = weight.shape[0]
    row_count = x.shape_size // input_dim
    row_tile = row_count if row_tile is None else row_tile
    result_shape = (*x.shape[:-1], output_dim)
    result_dtype = x.dtype if output_dtype is None else output_dtype
    if r is None:
        result = ax.dsl.create_tensor(result_shape, result_dtype, name="runtime_linear_output")
    else:
        result = r
    x_rows = (x if activation_input is None else activation_input).reshape(row_count, input_dim)
    result_rows = result.reshape(row_count, output_dim)
    activation_scales = None
    dequantized_activation = None
    physical_result_rows = None
    physical_output_tiles = []
    # Keep the physical output layout in lockstep with the resident weight
    # tile. A mismatch leaves the final crop lookup without a matching tile
    # and raises StopIteration after the expensive FinishKernel build.
    # The compact activation path keeps one BF16 K-row resident alongside the
    # runtime MX weight.  A 512-OC resident tile stays within SWA OCM after
    # scheduler metadata and DDR-swap buffers are accounted for; larger tiles
    # can make otherwise valid attention builders fail OCM allocation.
    load_output_tile = 512 if dsl_dtype(weight.dtype) is not ax.bf16 else 64
    if (activation_input is None or activation_input is x) and uses_model_activation_quant(weight):
        if input_dim >= ACTIVATION_QUANT_FAST_PATH_MIN_WIDTH:
            # Keep model.py's FP8/UE8M0 rounding, then consume one BF16 tile in
            # AX MatMul. This is the production path that prevents CRUN graph
            # expansion for DSV4 attention and MoE projections.
            dequantized_activation = model_activation_dequantize(x)
            x_rows = dequantized_activation
        else:
            x_rows, activation_scales = model_activation_quantize(x)
        if activation_scales is not None:
            physical_output_dim = 0
            for load_start in range(0, output_dim, load_output_tile):
                load_stop = min(load_start + load_output_tile, output_dim)
                for output_start in range(load_start, load_stop, output_tile):
                    output_stop = min(output_start + output_tile, load_stop)
                    tile_width = output_stop - output_start
                    physical_output_tiles.append((output_start, output_stop, physical_output_dim))
                    physical_output_dim += ((tile_width + 15) // 16) * 16
            physical_result_rows = ax.dsl.create_tensor(
                (row_count, physical_output_dim),
                result_dtype,
                ax.MemoryType.OCM,
                name="runtime_linear_mxfp4_output",
            )

    # A 1024-channel MX tile is 2 MiB for K=4096 and remains within the
    # per-core OCM budget.  Keeping the whole K row in one resident tile
    # avoids the repeated DDR slice loads that dominated DSV4 MoE graph
    # construction and CRUN simulation.
    for load_start in range(0, output_dim, load_output_tile):
        load_stop = min(load_start + load_output_tile, output_dim)
        load_count = load_stop - load_start
        load_slice = slice(load_start, load_stop)
        resident_load_count = max(load_count, 16) if activation_scales is not None else load_count
        if resident_load_count == load_count and input_load_tile >= input_dim:
            # A complete K tile is already bounded by the caller.  Loading the
            # runtime input directly avoids an otherwise redundant DDR->OCM
            # load followed by an OCM move for every expert projection.
            loaded_weight = load_tile(weight[load_slice, :], banks=banks)
        else:
            loaded_weight = ax.dsl.create_tensor(
                (resident_load_count, input_dim),
                weight.dtype,
                name=f"runtime_weight_oc{load_start}",
            )
            for input_start in range(0, input_dim, input_load_tile):
                input_stop = min(input_start + input_load_tile, input_dim)
                input_slice = slice(input_start, input_stop)
                loaded_slice = load_tile(weight[load_slice, input_slice], banks=banks)
                ax.dsl.move(r=loaded_weight[:load_count, input_slice], x=loaded_slice)
        # Torch packs FP4 runtime inputs with one shared dtype, so the graph
        # declares them NVFP4.  AX8860 reclassifies NVFP4+E8M0 during CRUN
        # lowering, while PYRUN needs the explicit MXFP4 view to apply the
        # per-block E8M0 scale numerically.  Cast only the PYRUN consumer;
        # changing the producer dtype would make CRUN AxLoad validation fail.
        matmul_weight = loaded_weight
        if dsl_dtype(weight.dtype) is ax.nvfp4 and configured_run_mode(ax.RunMode.PYRUN) is not ax.RunMode.CRUN:
            matmul_weight = ax.dsl.cast(loaded_weight, dtype=ax.mxfp4)
        for output_start in range(load_start, load_stop, output_tile):
            output_stop = min(output_start + output_tile, load_stop)
            output_slice = slice(output_start, output_stop)
            local_slice = slice(output_start - load_start, output_stop - load_start)
            scale_slice = runtime_weight_scale_slice(
                weight,
                weight_scale,
                output_slice,
                slice(0, input_dim),
            )
            loaded_scale = 1.0 if dsl_dtype(weight.dtype) is ax.bf16 else load_tile(scale_slice, banks=banks)
            # The reference path does not use a secondary scale.  Keep this
            # unset so CRUN and PYRUN share the same weight semantics.
            secondary_scale = None
            # Activation scales are indexed by input row, so rows can be
            # evaluated together without changing the model.py semantics.
            # Keeping the caller's tile avoids one graph sub-tree per token.
            matmul_row_tile = row_tile
            for row_start in range(0, row_count, matmul_row_tile):
                row_stop = min(row_start + matmul_row_tile, row_count)
                row_slice = slice(row_start, row_stop)
                if activation_scales is not None:
                    logical_width = output_stop - output_start
                    physical_weight = matmul_weight[local_slice, :]
                    physical_scale = loaded_scale
                    # MXFP4's AX8860 OCB is 16 channels.  A short final tile
                    # must therefore be materialized as one complete OCB;
                    # padding only to eight still leaves the scale K-buffer
                    # in small-OC mode and reproduces the signed-scale bug.
                    if logical_width < 16:
                        pad_oc = 16 - logical_width
                        physical_weight = ax.dsl.pad(
                            physical_weight,
                            padding=(0, pad_oc, 0, 0),
                            value=0,
                        )
                        physical_scale = ax.dsl.pad(
                            physical_scale,
                            padding=(0, pad_oc, 0, 0),
                            value=127,
                        )
                        physical_weight_storage = ax.dsl.create_tensor(
                            physical_weight.shape,
                            physical_weight.dtype,
                            ax.MemoryType.OCM,
                            name=f"runtime_linear_mxfp4_weight_{output_start}_{row_start}",
                        )
                        physical_scale_storage = ax.dsl.create_tensor(
                            physical_scale.shape,
                            physical_scale.dtype,
                            ax.MemoryType.OCM,
                            name=f"runtime_linear_mxfp4_scale_{output_start}_{row_start}",
                        )
                        ax.dsl.move(r=physical_weight_storage, x=physical_weight)
                        ax.dsl.move(r=physical_scale_storage, x=physical_scale)
                        physical_weight = physical_weight_storage
                        physical_scale = physical_scale_storage
                    # AX8860's MX scale K-buffer assumes a 16-group row
                    # stride.  Keep the logical `[OC, K/32]` view consumed by
                    # MatMul, but give each row a physically aligned stride
                    # so adjacent OC scale rows cannot alias the K buffer.
                    scale_groups = input_dim // MX_GROUP_SIZE
                    aligned_scale_groups = max(16, scale_groups)
                    if aligned_scale_groups > scale_groups:
                        physical_scale = ax.dsl.pad(
                            physical_scale,
                            padding=(0, 0, 0, aligned_scale_groups - scale_groups),
                            value=127,
                        )
                        physical_scale = physical_scale[:, :scale_groups]
                    projected = model_activation_matmul(
                        x_rows[row_slice, :],
                        physical_weight,
                        physical_scale,
                        activation_scales[row_slice, :],
                        output_dtype=result_dtype,
                    )
                    if physical_result_rows is None:
                        ax.dsl.move(
                            projected[:, :logical_width],
                            r=result_rows[row_slice, output_slice],
                        )
                    else:
                        physical_width = projected.shape[-1]
                        ax.dsl.move(
                            projected,
                            r=physical_result_rows[
                                row_slice,
                                next(
                                    tile_start
                                    for tile_begin, tile_end, tile_start in physical_output_tiles
                                    if tile_begin == output_start and tile_end == output_stop
                                ) : next(
                                    tile_start
                                    for tile_begin, tile_end, tile_start in physical_output_tiles
                                    if tile_begin == output_start and tile_end == output_stop
                                )
                                + physical_width,
                            ],
                        )
                else:
                    matmul_optional_scale2(
                        x_rows[row_slice, :],
                        matmul_weight[local_slice, :],
                        w_scale=loaded_scale,
                        transposed=True,
                        output_dtype=result_dtype,
                        r=result_rows[row_slice, output_slice],
                        w_scale2=secondary_scale,
                    )
    if physical_result_rows is not None:
        # Crop each tile only after all MX MatMul tiles have been materialized
        # at full OCB width; direct short-view writes trigger CRUN small-OC
        # scale addressing for MXFP4.
        for tile_begin, tile_end, tile_start in physical_output_tiles:
            ax.dsl.move(
                physical_result_rows[:, tile_start : tile_start + tile_end - tile_begin],
                r=result_rows[:, tile_begin:tile_end],
            )
    return result


def rmsnorm_with_constants(x, weight, inverse_divisor, epsilon, *, result_dtype, r=None):
    """Record RMSNorm with scalar tensors whose lifetime is owned by the caller."""
    # squared: [*x.shape], FP32.
    squared = ax.dsl.mul(x, x, output_dtype=ax.fp32)
    # square_sum: [*x.shape[:-1], 1], FP32.
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    # mean_square: [*x.shape[:-1],1], FP32. The following add is lowered as one FMA.
    mean_square = ax.dsl.mul(square_sum, inverse_divisor, output_dtype=ax.fp32)
    # stabilized: [*x.shape[:-1],1], FP32.
    stabilized = ax.dsl.add(mean_square, epsilon, output_dtype=ax.fp32)
    # denominator: [*x.shape[:-1],1], FP32.
    denominator = ax.dsl.sqrt(stabilized, output_dtype=ax.fp32)
    # normalized: [*x.shape], FP32 with a norm weight or result_dtype without one.
    normalized = ax.dsl.div(
        x,
        denominator,
        output_dtype=ax.fp32 if weight is not None else result_dtype,
        r=r if weight is None else None,
    )
    if weight is not None:
        # normalized: [*x.shape], result_dtype (BF16 in production).
        normalized = ax.dsl.mul(normalized, weight, output_dtype=result_dtype, r=r)
    return normalized


def mhc_controls(mix, base, scale, epsilon, two, post_output, comb_output, *, sinkhorn_iters):
    """Compute the token-local mHC gates and Sinkhorn matrix."""
    batch, token_count, _ = mix.shape
    # pre_scaled/pre_logits: [B,T,4], FP32.
    pre_scaled = ax.dsl.mul(mix[..., :4], scale[:1], output_dtype=ax.fp32)
    pre_logits = ax.dsl.add(pre_scaled, base[:4], output_dtype=ax.fp32)
    # pre_probability/pre: [B,T,4], FP32.
    pre_probability = ax.dsl.sigmoid(pre_logits, output_dtype=ax.fp32)
    pre = ax.dsl.add(pre_probability, epsilon, output_dtype=ax.fp32)
    # post_scaled/post_logits/post: [B,T,4], FP32.
    post_scaled = ax.dsl.mul(mix[..., 4:8], scale[1:2], output_dtype=ax.fp32)
    post_logits = ax.dsl.add(post_scaled, base[4:8], output_dtype=ax.fp32)
    post_probability = ax.dsl.sigmoid(post_logits, output_dtype=ax.fp32)
    post = ax.dsl.mul(post_probability, two, output_dtype=ax.fp32, r=post_output)

    # comb_logits/comb: [B,T,4,4], FP32.
    comb_scaled = ax.dsl.mul(mix[..., 8:24], scale[2:3], output_dtype=ax.fp32)
    comb_logits = ax.dsl.add(comb_scaled, base[8:24], output_dtype=ax.fp32).reshape(
        batch,
        token_count,
        4,
        4,
    )
    comb_probability = ax.dsl.softmax(comb_logits, dim=-1, output_dtype=ax.fp32)
    comb = ax.dsl.add(comb_probability, epsilon, output_dtype=ax.fp32)
    column_sum = ax.dsl.sum(comb, dim=-2, keepdim=True, output_dtype=ax.fp32)
    column_denominator = ax.dsl.add(column_sum, epsilon, output_dtype=ax.fp32)
    comb = ax.dsl.div(
        comb,
        column_denominator,
        output_dtype=ax.fp32,
        r=comb_output if sinkhorn_iters == 1 else None,
    )
    for iteration in range(sinkhorn_iters - 1):
        row_sum = ax.dsl.sum(comb, dim=-1, keepdim=True, output_dtype=ax.fp32)
        row_denominator = ax.dsl.add(row_sum, epsilon, output_dtype=ax.fp32)
        comb = ax.dsl.div(comb, row_denominator, output_dtype=ax.fp32)
        column_sum = ax.dsl.sum(comb, dim=-2, keepdim=True, output_dtype=ax.fp32)
        column_denominator = ax.dsl.add(column_sum, epsilon, output_dtype=ax.fp32)
        comb = ax.dsl.div(
            comb,
            column_denominator,
            output_dtype=ax.fp32,
            r=comb_output if iteration == sinkhorn_iters - 2 else None,
        )
    return pre, post, comb


def mhc_residual_outputs(
    residual_rows,
    pre,
    comb,
    collapsed,
    *,
    name_prefix,
    residual_path_out=None,
    residual_path_shared=None,
):
    """Compute token-local collapsed and residual-path rows."""
    batch, token_count, streams, hidden = residual_rows.shape
    pre_expanded = pre.reshape(batch, token_count, streams, 1)
    for hidden_start in range(0, hidden, MHC_COLLAPSED_HIDDEN_TILE):
        hidden_stop = min(hidden_start + MHC_COLLAPSED_HIDDEN_TILE, hidden)
        hidden_slice = slice(hidden_start, hidden_stop)
        weighted_residual = ax.dsl.mul(
            residual_rows[:, :, :, hidden_slice],
            pre_expanded,
            output_dtype=ax.fp32,
        )
        ax.dsl.sum(
            weighted_residual,
            dim=2,
            output_dtype=residual_rows.dtype,
            r=collapsed[:, :, hidden_slice],
        )

    # Match model.py::Block.hc_post: retain FP32 comb values and accumulate the
    # residual path in FP32. Elementwise multiply/reduce avoids AxMatMul's
    # activation-dtype restriction.
    transposed_comb = ax.dsl.transpose(comb, perm=(0, 1, 3, 2))
    residual_path_target = residual_path_shared if residual_path_out is None else residual_path_out

    # Store completed hidden tiles immediately so their SDMA writes can overlap later TENG tiles.
    for hidden_start in range(0, hidden, MHC_RESIDUAL_HIDDEN_TILE):
        hidden_stop = min(hidden_start + MHC_RESIDUAL_HIDDEN_TILE, hidden)
        hidden_slice = slice(hidden_start, hidden_stop)
        tile_width = hidden_stop - hidden_start
        residual_path_tile = ax.dsl.create_tensor(
            (batch, token_count, streams, tile_width),
            ax.fp32,
            ax.MemoryType.OCM,
            name=f"{name_prefix}_residual_path_{hidden_start}",
        )
        for output_stream in range(streams):
            stream_weights = transposed_comb[:, :, output_stream : output_stream + 1, :].reshape(
                batch,
                token_count,
                streams,
                1,
            )
            weighted_residual = ax.dsl.mul(
                residual_rows[:, :, :, hidden_slice],
                stream_weights,
                output_dtype=ax.fp32,
            )
            ax.dsl.sum(
                weighted_residual,
                dim=2,
                keepdim=True,
                output_dtype=ax.fp32,
                r=residual_path_tile[:, :, output_stream : output_stream + 1, :],
            )
        target_tile = residual_path_target[:, :, :, hidden_slice]
        if target_tile.dtype != residual_path_tile.dtype:
            residual_path_tile = ax.dsl.cast(residual_path_tile, dtype=target_tile.dtype)
        ax.dsl.store(residual_path_tile, r=target_tile)
    return residual_path_target


def mhc(
    residual,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    meta,
    *,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    name_prefix="mhc",
    residual_path_out=None,
):
    """Record the four-core stream/IC/token mHC schedule from ``parallel-plan.md``."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len, streams, hidden = residual.shape
    is_crun = configured_run_mode(ax.RunMode.PYRUN) is ax.RunMode.CRUN
    local_token_slice = token_slice(core_id, core_num, seq_len)

    # residual_shared: [B, S, 4, D], residual.dtype (BF16 in production).
    residual_shared = ax.dsl.create_tensor(
        residual.shape,
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_residual",
    )
    # A one-core launch owns all four mHC streams; the production four-core
    # launch keeps the original one-stream-per-core ownership.
    if core_num == 1:
        ax.dsl.load(residual, r=residual_shared)
    else:
        ax.dsl.load(
            residual[:, :, core_id : core_id + 1, :],
            r=residual_shared[:, :, core_id : core_id + 1, :],
        )

    # square_partials: [B, S, 4], FP32; each core writes one stream slot.
    square_partials = ax.dsl.create_tensor(
        (batch, seq_len, streams),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_square_partials",
    )
    if core_num == 1:
        stream_squared = ax.dsl.mul(residual_shared, residual_shared, output_dtype=ax.fp32)
        ax.dsl.sum(stream_squared, dim=-1, output_dtype=ax.fp32, r=square_partials)
    else:
        stream = residual_shared[:, :, core_id : core_id + 1, :]
        stream_squared = ax.dsl.mul(stream, stream, output_dtype=ax.fp32)
        ax.dsl.sum(
            stream_squared,
            dim=-1,
            output_dtype=ax.fp32,
            r=square_partials[:, :, core_id : core_id + 1],
        )
    # Synchronization only; no tensor output.
    axd.sync_all()

    # denominator: [B, S, 1], FP32, assembled by token owners.
    denominator = ax.dsl.create_tensor(
        (batch, seq_len, 1),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_denominator",
    )
    if local_token_slice.start < local_token_slice.stop:
        # total: [B, T_core, 1], FP32, reduced across four residual streams.
        total = ax.dsl.sum(
            square_partials[:, local_token_slice, :],
            dim=-1,
            keepdim=True,
            output_dtype=ax.fp32,
        )
        # rms_scale: [1], FP32.
        rms_scale = ax.dsl.full((1,), ax.fp32, fill_value=1.0 / (streams * hidden))
        # mean: [B, T_core, 1], FP32.
        mean = ax.dsl.mul(total, rms_scale, output_dtype=ax.fp32)
        # rms_epsilon: [1], FP32.
        rms_epsilon = ax.dsl.full((1,), ax.fp32, fill_value=rms_eps)
        # stabilized_mean: [B,T_core,1], FP32. The preceding multiply and this add fuse as one FMA.
        stabilized_mean = ax.dsl.add(mean, rms_epsilon)
        # Denominator slice: [B, T_core, 1], FP32.
        ax.dsl.sqrt(
            stabilized_mean,
            output_dtype=ax.fp32,
            r=denominator[:, local_token_slice, :],
        )
    # Synchronization only; no tensor output.
    axd.sync_all()

    # status_partials: [B, S, 4, 24], FP32; stream/core is the third axis.
    status_partials = ax.dsl.create_tensor(
        (batch, seq_len, streams, 24),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_status_partials",
    )
    # Split the core-local D-wide IC shard across four MatMul EUs. Each runtime
    # weight shard is loaded once and reused by every sequence tile.
    eu_hidden = hidden // CONV_EUS_PER_CORE
    owner_streams = range(streams) if core_num == 1 else (core_id,)
    status_weights = []
    status_scales = []
    for stream_id in owner_streams:
        stream_weights = []
        stream_scales = []
        for ic_part_id in range(CONV_EUS_PER_CORE):
            local_ic_start = ic_part_id * eu_hidden
            local_ic_stop = local_ic_start + eu_hidden
            global_ic_start = stream_id * hidden + local_ic_start
            global_ic_stop = global_ic_start + eu_hidden
            status_weight = ax.dsl.create_tensor(
                (24, eu_hidden),
                hc_fn.dtype,
                name=f"{name_prefix}_status_weight_s{stream_id}_{ic_part_id}",
            )
            for load_start in range(0, eu_hidden, 1024):
                load_stop = min(load_start + 1024, eu_hidden)
                ax.dsl.load(
                    hc_fn[:, global_ic_start + load_start : global_ic_start + load_stop],
                    r=status_weight[:, load_start:load_stop],
                )
            matmul_status_weight = status_weight
            if dsl_dtype(hc_fn.dtype) is ax.nvfp4 and not is_crun:
                matmul_status_weight = ax.dsl.cast(status_weight, dtype="mxfp4")
            stream_weights.append(matmul_status_weight)
            if dsl_dtype(hc_fn.dtype) is ax.bf16:
                stream_scales.append(None)
            else:
                status_scale = runtime_weight_scale_slice(
                    hc_fn,
                    hc_fn_scale,
                    slice(0, 24),
                    slice(global_ic_start, global_ic_stop),
                )
                stream_scales.append(ax.dsl.load(status_scale))
        status_weights.append(stream_weights)
        status_scales.append(stream_scales)

    for seq_start in range(0, seq_len, MHC_STATUS_SEQ_TILE):
        seq_stop = min(seq_start + MHC_STATUS_SEQ_TILE, seq_len)
        seq_slice = slice(seq_start, seq_stop)
        tile_tokens = seq_stop - seq_start
        # normalized_tile: [B,T_status,D], residual.dtype (BF16 in production).
        for stream_index, stream_id in enumerate(owner_streams):
            stream = residual_shared[:, :, stream_id : stream_id + 1, :]
            normalized_tile = ax.dsl.div(
                stream.reshape(batch, seq_len, hidden)[:, seq_slice, :],
                denominator[:, seq_slice, :],
                output_dtype=residual.dtype,
            )
            status_eu_partials = ax.dsl.create_tensor(
                (batch, tile_tokens, CONV_EUS_PER_CORE, 24),
                ax.fp32,
                ax.MemoryType.OCM,
                name=f"{name_prefix}_status_eu_partials_s{stream_id}_{seq_start}",
            )
            for ic_part_id in range(CONV_EUS_PER_CORE):
                local_ic_start = ic_part_id * eu_hidden
                local_ic_stop = local_ic_start + eu_hidden
                status_input = normalized_tile[:, :, local_ic_start:local_ic_stop]
                matmul_scales = {}
                if dsl_dtype(hc_fn.dtype) is not ax.bf16:
                    scale = status_scales[stream_index][ic_part_id]
                    matmul_scales["w_scale"] = scale
                    matmul_scales["w_scale2"] = scale if is_crun else None
                matmul_optional_scale2(
                    status_input.reshape(batch * tile_tokens, eu_hidden),
                    status_weights[stream_index][ic_part_id],
                    transposed=True,
                    output_dtype=ax.fp32,
                    r=status_eu_partials[:, :, ic_part_id : ic_part_id + 1, :].reshape(batch * tile_tokens, 24),
                    **matmul_scales,
                )
            ax.dsl.sum(
                status_eu_partials,
                dim=2,
                output_dtype=ax.fp32,
                r=status_partials[:, seq_slice, stream_id : stream_id + 1, :].reshape(batch, tile_tokens, 24),
            )
    # Synchronization only; no tensor output.
    axd.sync_all()

    # mix_shared: [B,S,24], FP32.
    mix_shared = ax.dsl.create_tensor(
        (batch, seq_len, 24),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_mix",
    )
    # post_shared: [B,S,4], FP32.
    post_shared = ax.dsl.create_tensor(
        (batch, seq_len, 4),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_post",
    )
    # comb_shared: [B,S,4,4], FP32.
    comb_shared = ax.dsl.create_tensor(
        (batch, seq_len, 4, 4),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_comb",
    )
    # collapsed_shared: [B, S, D], residual.dtype (BF16 in production).
    collapsed_shared = ax.dsl.create_tensor(
        (batch, seq_len, hidden),
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name=f"{name_prefix}_collapsed",
    )
    residual_path_shared = None
    if residual_path_out is None:
        # residual_path_shared: [B,S,4,D], FP32 for the final mHC accumulation.
        residual_path_shared = ax.dsl.create_tensor(
            residual.shape,
            ax.fp32,
            ax.MemoryType.SHARED_OCM,
            name=f"{name_prefix}_residual_path",
        )

    if local_token_slice.start < local_token_slice.stop:
        # mix: [B, T_core, 24], FP32, reduced directly into shared state.
        mix = ax.dsl.sum(
            status_partials[:, local_token_slice, :, :],
            dim=2,
            output_dtype=ax.fp32,
            r=mix_shared[:, local_token_slice, :],
        )
        # base/scale/epsilon: [24]/[3]/[1], FP32.
        base = ax.dsl.load(hc_base)
        scale = ax.dsl.load(hc_scale)
        # epsilon: [1], FP32.
        epsilon = ax.dsl.full((1,), ax.fp32, fill_value=hc_eps)
        # two: [1], FP32.
        two = ax.dsl.full((1,), ax.fp32, fill_value=2.0)
        comb_output = comb_shared[:, local_token_slice, :, :]
        pre, _, comb = mhc_controls(
            mix,
            base,
            scale,
            epsilon,
            two,
            post_shared[:, local_token_slice, :],
            comb_output,
            sinkhorn_iters=sinkhorn_iters,
        )
        # residual_rows: [B, T_core, 4, D], residual.dtype.
        residual_rows = residual_shared[:, local_token_slice, :, :]
        mhc_residual_outputs(
            residual_rows,
            pre,
            comb,
            collapsed_shared[:, local_token_slice, :],
            name_prefix=name_prefix,
            residual_path_out=(None if residual_path_out is None else residual_path_out[:, local_token_slice, :, :]),
            residual_path_shared=(
                None if residual_path_shared is None else residual_path_shared[:, local_token_slice, :, :]
            ),
        )
    # The same token owner consumes ``collapsed_shared`` next. Both attention
    # and FFN publish their normalized token shards before any cross-core
    # consumer reads collapsed/post, so a trailing mHC collective is redundant.
    return MHCState(post_shared, comb_shared, collapsed_shared, residual_path_shared)


def tail_rope(x, cos, sin, rotary_dim, *, r=None):
    """Record trailing interleaved-pair RoPE for headless or token-major head tensors."""
    prefix = x.shape[:-1]
    # pairs: [*prefix, Dr/2, 2], x.dtype.
    pairs = x[..., -rotary_dim:].reshape(*prefix, rotary_dim // 2, 2)
    swap_last_two = (*range(len(prefix)), len(prefix) + 1, len(prefix))
    # pairs_transposed: [*prefix, 2, Dr/2], x.dtype; Dr/2 becomes the TENG channel axis.
    pairs_transposed = ax.dsl.transpose(pairs, perm=swap_last_two)
    # first/second: [*prefix, 1, Dr/2], x.dtype.
    first = pairs_transposed[..., 0:1, :]
    second = pairs_transposed[..., 1:2, :]
    head_dims = len(prefix) - 2
    angle_shape = (*cos.shape[:2], *(1 for _ in range(head_dims)), 1, cos.shape[-1])
    # cos_view/sin_view: [B, S, (1 per head axis), 1, Dr/2], cos/sin dtype (BF16).
    cos_view = cos.reshape(*angle_shape)
    sin_view = sin.reshape(*angle_shape)
    # rotated_transposed: [*prefix, 2, Dr/2], x.dtype. The two arithmetic
    # branches write their disjoint pair-axis slices without an AxConcat.
    rotated_transposed = ax.dsl.create_tensor((*prefix, 2, rotary_dim // 2), x.dtype)
    # first_cos/second_sin: [*prefix,1,Dr/2], FP32.
    first_cos = ax.dsl.mul(first, cos_view, output_dtype=ax.fp32)
    # second_sin: [*prefix,1,Dr/2], FP32.
    second_sin = ax.dsl.mul(second, sin_view, output_dtype=ax.fp32)
    # first_out: [*prefix,1,Dr/2], x.dtype.
    ax.dsl.sub(
        first_cos,
        second_sin,
        output_dtype=x.dtype,
        r=rotated_transposed[..., :1, :],
    )
    # second_cos/first_sin: [*prefix,1,Dr/2], FP32.
    second_cos = ax.dsl.mul(second, cos_view, output_dtype=ax.fp32)
    # first_sin: [*prefix,1,Dr/2], FP32.
    first_sin = ax.dsl.mul(first, sin_view, output_dtype=ax.fp32)
    # second_out: [*prefix,1,Dr/2], x.dtype.
    ax.dsl.add(
        second_cos,
        first_sin,
        output_dtype=x.dtype,
        r=rotated_transposed[..., 1:2, :],
    )
    # rotated_pairs: [*prefix, Dr/2, 2], x.dtype, written directly to the trailing target slice.
    rotated_target = None if r is None else r[..., -rotary_dim:].reshape(*prefix, rotary_dim // 2, 2)
    # rotated_pairs: [*prefix,Dr/2,2], x.dtype.
    rotated_pairs = ax.dsl.transpose(rotated_transposed, perm=swap_last_two, r=rotated_target)
    # rotated: [*prefix, Dr], x.dtype.
    rotated = rotated_pairs.reshape(*prefix, rotary_dim)
    if rotary_dim == x.shape[-1]:
        return r if r is not None else rotated
    if r is x:
        # The transpose above has replaced the trailing rotary slice in place; the prefix
        # channels were never modified and do not need a copy/concat round trip.
        return r
    # Output: [*prefix,Dh], x.dtype; only the trailing Dr channels are replaced.
    return ax.dsl.concat(x[..., :-rotary_dim], rotated, dim=-1, r=r)


def dsv4_kv_fp8_qat(x, rotary_dim, *, group_size=64, inplace=False):
    """Match model.py's post-RoPE FP8 QAT on non-RoPE KV channels."""
    non_rope_dim = x.shape[-1] - rotary_dim
    if non_rope_dim <= 0 or non_rope_dim % group_size:
        raise ValueError(
            "DSV4 KV QAT requires the non-RoPE width to be divisible by " f"{group_size}, got {non_rope_dim}"
        )

    rows = x.shape_size // x.shape[-1]
    groups = non_rope_dim // group_size
    non_rope = x[..., :non_rope_dim].reshape(rows, groups, group_size)
    non_rope_fp32 = ax.dsl.cast(non_rope, dtype=ax.fp32)
    negative_one = ax.dsl.full((1,), ax.fp32, fill_value=-1.0)
    absolute = ax.dsl.maximum(
        non_rope_fp32,
        ax.dsl.mul(non_rope_fp32, negative_one, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    amax = ax.dsl.max(absolute, dim=-1, keepdim=True)
    minimum_amax = ax.dsl.full(amax.shape, ax.fp32, fill_value=1e-4)
    amax = ax.dsl.maximum(amax, minimum_amax, output_dtype=ax.fp32)

    fp8_max = ax.dsl.full((1,), ax.fp32, fill_value=448.0)
    scale = ax.dsl.div(amax, fp8_max, output_dtype=ax.fp32)
    exponent = ax.dsl.ceil(ax.dsl.log2(scale, output_dtype=ax.fp32))
    ln_two = ax.dsl.full((1,), ax.fp32, fill_value=float(np.log(2.0)))
    scale = ax.dsl.exp(
        ax.dsl.mul(exponent, ln_two, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )

    scaled = ax.dsl.div(non_rope_fp32, scale, output_dtype=ax.fp32)
    lower = ax.dsl.full((1,), ax.fp32, fill_value=-448.0)
    upper = ax.dsl.full((1,), ax.fp32, fill_value=448.0)
    scaled = ax.dsl.minimum(
        ax.dsl.maximum(scaled, lower, output_dtype=ax.fp32),
        upper,
        output_dtype=ax.fp32,
    )
    quantized = ax.dsl.cast(scaled, dtype=ax.fp8_e4m3)
    dequantized = (
        ax.dsl.mul(
            ax.dsl.cast(quantized, dtype=ax.fp32),
            scale,
            output_dtype=x.dtype,
        )
        .reshape(rows, non_rope_dim)
        .reshape(*x.shape[:-1], non_rope_dim)
    )
    if inplace:
        # Preserve RoPE positional channels and replace only the non-RoPE KV groups.
        ax.dsl.move(dequantized, r=x[..., :non_rope_dim])
        return x
    return ax.dsl.concat(dequantized, x[..., non_rope_dim:], dim=-1)


def project_attention_query_rows(
    q_residual,
    cos,
    sin,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    meta,
    *,
    head_dim,
    rotary_dim,
    rms_eps,
    r,
):
    """Project, normalize, and rotate one core's heads into ``r``."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len = meta.batch, meta.seq_len
    heads_per_core = q_b_weight_chip.shape[0] // (core_num * head_dim)
    head_begin = core_id * heads_per_core
    head_end = head_begin + heads_per_core
    q_rows = slice(head_begin * head_dim, head_end * head_dim)
    q_weight = q_b_weight_chip[q_rows, :]
    q_weight_scale = runtime_weight_scale_slice(
        q_b_weight_chip,
        q_b_weight_chip_scale,
        q_rows,
        slice(0, q_b_weight_chip.shape[1]),
    )
    for seq_start in range(0, seq_len, ATTENTION_PROJECTION_SEQ_TILE):
        seq_stop = min(seq_start + ATTENTION_PROJECTION_SEQ_TILE, seq_len)
        seq_slice = slice(seq_start, seq_stop)
        tile_tokens = seq_stop - seq_start
        query_tile = runtime_linear(
            q_residual[:, seq_slice, :],
            q_weight,
            q_weight_scale,
            output_dtype=q_residual.dtype,
        ).reshape(batch, tile_tokens, heads_per_core, head_dim)
        query_tile = dsv4_query_rmsnorm_bf16(query_tile, eps=rms_eps)
        tail_rope(
            query_tile,
            cos[:, seq_slice, :],
            sin[:, seq_slice, :],
            rotary_dim,
            r=r[:, seq_slice, :, :],
        )
    return r


def project_attention_query(
    q_residual,
    cos,
    sin,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    meta,
    *,
    head_dim,
    rotary_dim,
    rms_eps,
):
    """Project one core's private query-head shard."""
    heads_per_core = q_b_weight_chip.shape[0] // (meta.core_num * head_dim)
    # query: [B,S,Hq_core,Dh], q_residual.dtype private OCM; production
    # [1,128,8,512] BF16.
    query = ax.dsl.create_tensor(
        (meta.batch, meta.seq_len, heads_per_core, head_dim),
        q_residual.dtype,
        ax.MemoryType.OCM,
        name="query",
    )
    return project_attention_query_rows(
        q_residual,
        cos,
        sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        r=query,
    )


def project_shared_attention_query(
    q_residual,
    cos,
    sin,
    q_b_weight_chip,
    q_b_weight_chip_scale,
    meta,
    *,
    head_dim,
    rotary_dim,
    rms_eps,
):
    """Publish one core's query-head shard in a chip-visible query tensor."""
    heads = q_b_weight_chip.shape[0] // head_dim
    heads_per_core = heads // meta.core_num
    head_start = meta.core_id * heads_per_core
    head_slice = slice(head_start, head_start + heads_per_core)
    # query: [B,S,Hq_chip,Dh], q_residual.dtype SHARED_OCM; production
    # [1,128,32,512] BF16.
    query = ax.dsl.create_tensor(
        (meta.batch, meta.seq_len, heads, head_dim),
        q_residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="shared_query",
    )
    project_attention_query_rows(
        q_residual,
        cos,
        sin,
        q_b_weight_chip,
        q_b_weight_chip_scale,
        meta,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rms_eps=rms_eps,
        r=query[:, :, head_slice, :],
    )
    return query


def attention_prefix(
    residual,
    hc_fn,
    hc_fn_scale,
    hc_base,
    hc_scale,
    input_norm_weight,
    q_a_weight,
    q_a_weight_scale,
    q_a_norm_weight,
    kv_weight,
    kv_weight_scale,
    kv_norm_weight,
    cos,
    sin,
    kv_out,
    meta,
    *,
    rotary_dim=64,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    residual_path_out,
):
    """Record the variant-independent mHC, low-rank Q, and KV projections."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len, hidden = meta.batch, meta.seq_len, meta.hidden
    head_dim = kv_weight.shape[0]
    local_token_slice = token_slice(core_id, core_num, seq_len)

    # state.post/collapsed: [B,S,4] FP32 / [B,S,D] BF16. Attention writes
    # residual-path token shards directly to the public [B,S,4,D] BF16 output.
    state = mhc(
        residual,
        hc_fn,
        hc_fn_scale,
        hc_base,
        hc_scale,
        meta,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        name_prefix="attn_mhc",
        residual_path_out=residual_path_out,
    )

    # normalized: [B, S, D], residual.dtype; production [1,128,4096] BF16.
    normalized = ax.dsl.create_tensor(
        (batch, seq_len, hidden),
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="attn_normalized",
    )
    # q_a_projected: [B,S,Rq], residual.dtype, assembled from four core-owned OC shards.
    q_a_projected = ax.dsl.create_tensor(
        (batch, seq_len, q_a_weight.shape[0]),
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="q_a_projected",
    )
    # q_residual: [B, S, Rq], residual.dtype; production [1,128,1024] BF16.
    q_residual = ax.dsl.create_tensor(
        (batch, seq_len, q_a_weight.shape[0]),
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="q_residual",
    )
    # A single token belongs to one core.  Keep parameter loads on the owner
    # core so empty cores do not emit DDR->OCM jobs with no consumer.
    token_owner = local_token_slice.start < local_token_slice.stop
    if token_owner:
        # input_norm: [D], BF16 model parameter.
        input_norm = ax.dsl.load(input_norm_weight)
        for seq_start in range(local_token_slice.start, local_token_slice.stop, ATTENTION_PREFIX_SEQ_TILE):
            seq_stop = min(seq_start + ATTENTION_PREFIX_SEQ_TILE, local_token_slice.stop)
            seq_slice = slice(seq_start, seq_stop)
            # Normalized tile: [B,T_prefix,D], residual.dtype; production [1,16,4096] BF16.
            rmsnorm(
                state.collapsed[:, seq_slice, :],
                input_norm,
                eps=rms_eps,
                r=normalized[:, seq_slice, :],
            )
    # Synchronization only; no tensor output.
    axd.sync_all()

    q_rank = q_a_weight.shape[0]
    q_a_oc_per_core = q_rank // core_num
    assert q_a_oc_per_core >= 16
    q_a_rows = slice(core_id * q_a_oc_per_core, (core_id + 1) * q_a_oc_per_core)
    local_q_a_weight_scale = runtime_weight_scale_slice(
        q_a_weight,
        q_a_weight_scale,
        q_a_rows,
        slice(0, hidden),
    )
    # Core OC shard: [B,S,Rq/4], residual.dtype.
    runtime_linear(
        normalized,
        q_a_weight[q_a_rows, :],
        local_q_a_weight_scale,
        output_dtype=residual.dtype,
        r=q_a_projected[:, :, q_a_rows],
        row_tile=ATTENTION_PROJECTION_SEQ_TILE,
    )
    # Synchronization only; no tensor output.
    axd.sync_all()

    if token_owner:
        # q_a_norm: [Rq], BF16 model parameter.
        q_a_norm = ax.dsl.load(q_a_norm_weight)
        for seq_start in range(local_token_slice.start, local_token_slice.stop, ATTENTION_PREFIX_SEQ_TILE):
            seq_stop = min(seq_start + ATTENTION_PREFIX_SEQ_TILE, local_token_slice.stop)
            seq_slice = slice(seq_start, seq_stop)
            # q_a tile: [B,T_prefix,Rq], residual.dtype after FP32 RMS statistics.
            rmsnorm(
                q_a_projected[:, seq_slice, :],
                q_a_norm,
                eps=rms_eps,
                r=q_residual[:, seq_slice, :],
            )
    # Synchronization only; no tensor output.
    axd.sync_all()

    # cos_local/sin_local: [B, S, Dr/2], input dtype; production [1,128,32] BF16.
    cos_local = ax.dsl.load(cos)
    # sin_local: [B,S,Dr/2], sin.dtype.
    sin_local = ax.dsl.load(sin)
    # kv_projected: [B,S,Dh], residual.dtype.
    kv_projected = ax.dsl.create_tensor(
        (batch, seq_len, head_dim),
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="projected_kv",
    )
    # kv: [B,S,Dh], residual.dtype.
    kv = ax.dsl.create_tensor(
        (batch, seq_len, head_dim),
        residual.dtype,
        ax.MemoryType.SHARED_OCM,
        name="shared_kv",
    )
    kv_oc_per_core = head_dim // core_num
    assert kv_oc_per_core >= 16
    kv_rows = slice(core_id * kv_oc_per_core, (core_id + 1) * kv_oc_per_core)
    local_kv_weight_scale = runtime_weight_scale_slice(
        kv_weight,
        kv_weight_scale,
        kv_rows,
        slice(0, hidden),
    )
    # Core OC shard: [B,S,Dh/4], residual.dtype.
    runtime_linear(
        normalized,
        kv_weight[kv_rows, :],
        local_kv_weight_scale,
        output_dtype=residual.dtype,
        r=kv_projected[:, :, kv_rows],
        row_tile=ATTENTION_PROJECTION_SEQ_TILE,
    )
    # Synchronization only; no tensor output.
    axd.sync_all()

    if token_owner:
        # kv_norm: [Dh], BF16 model parameter.
        kv_norm = ax.dsl.load(kv_norm_weight)
        for seq_start in range(local_token_slice.start, local_token_slice.stop, ATTENTION_PREFIX_SEQ_TILE):
            seq_stop = min(seq_start + ATTENTION_PREFIX_SEQ_TILE, local_token_slice.stop)
            seq_slice = slice(seq_start, seq_stop)
            # kv_value: [B,T_prefix,Dh], residual.dtype after weighted RMSNorm.
            kv_value = rmsnorm(kv_projected[:, seq_slice, :], kv_norm, eps=rms_eps)
            # kv_value: [B,T_prefix,Dh], residual.dtype after trailing-Dr RoPE.
            kv_value = tail_rope(
                kv_value,
                cos_local[:, seq_slice, :],
                sin_local[:, seq_slice, :],
                rotary_dim,
                r=kv[:, seq_slice, :],
            )
            # Match model.py: quantize only non-RoPE KV channels after RoPE.
            kv_value = dsv4_kv_fp8_qat(
                kv_value,
                rotary_dim,
                inplace=True,
            )
            # kv_out slice: [B,T_prefix,Dh], kv_value.dtype.
            ax.dsl.store(kv_value, r=kv_out[:, seq_slice, :])
    # The first variant-specific collective publishes these token-owned KV
    # writes before any cross-core consumer. SWA first copies the same owner's
    # KV rows and then publishes its assembled source; CSA/HCA synchronize in
    # their compressors before attention reads another core's rows.

    return AttentionPrefix(
        state,
        normalized,
        q_residual,
        kv,
        cos_local,
        sin_local,
    )


def project_attention_output_slice(
    prefix,
    context,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    meta,
    *,
    rotary_dim=64,
    banks=None,
):
    """Record grouped Woa/Wob and return this core's D/4 pre-Post output."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len, heads_per_core, head_dim = context.shape
    hidden = o_b_weight_chip.shape[0]
    groups = o_a_weight_chip.shape[0]
    rank = o_a_weight_chip.shape[1]
    heads_per_group = o_a_weight_chip.shape[2] // head_dim
    groups_per_core = groups // core_num

    # minus_one: [1], FP32.
    minus_one = ax.dsl.full((1,), ax.fp32, fill_value=-1.0)
    # negative_sin: [B,S,Dr/2], prefix.sin dtype; production [1,128,32] BF16.
    negative_sin = ax.dsl.mul(
        prefix.sin,
        minus_one,
        output_dtype=prefix.sin.dtype,
    )
    # inverse: [B,S,Hq_core,Dh], context.dtype after inverse trailing-Dr RoPE;
    # TP2 production [1,128,8,512] BF16.
    inverse = tail_rope(context, prefix.cos, negative_sin, rotary_dim, r=context)
    # Every core owns complete adjacent attention groups, matching each Woa
    # shard without an on-chip IC reduction. TP2 owns one group per core;
    # TP1 owns two groups per core.
    # group_values: [B,S,G,Ro], context.dtype; TP1 production
    # [1,128,8,1024] BF16.
    group_values = ax.dsl.create_tensor(
        (batch, seq_len, groups, rank),
        context.dtype,
        ax.MemoryType.SHARED_OCM,
        name="oa_groups",
    )
    for local_group in range(groups_per_core):
        group = core_id * groups_per_core + local_group
        head_start = local_group * heads_per_group
        head_stop = head_start + heads_per_group
        group_input = inverse[:, :, head_start:head_stop, :].reshape(
            batch * seq_len,
            heads_per_group * head_dim,
        )
        if dsl_dtype(o_a_weight_chip.dtype) is ax.bf16:
            group_weight_scale = o_a_weight_chip_scale
        else:
            group_weight_scale = o_a_weight_chip_scale[group : group + 1, :, :].reshape(
                rank,
                heads_per_group * head_dim // MX_GROUP_SIZE,
            )
        # Group result: [B*S,Ro], written directly into its complete group slot.
        runtime_linear(
            group_input,
            o_a_weight_chip[group : group + 1, :, :].reshape(
                rank,
                heads_per_group * head_dim,
            ),
            group_weight_scale,
            output_dtype=context.dtype,
            r=group_values[:, :, group : group + 1, :].reshape(batch * seq_len, rank),
            row_tile=ATTENTION_PROJECTION_SEQ_TILE,
            banks=banks,
        )
    # Synchronization only; no tensor output.
    axd.sync_all()

    # Official Wob is row-parallel across chips. Inside one chip, every core
    # consumes all local group ranks but owns disjoint output rows.
    hidden_quarter = hidden // core_num
    branch_hidden_slice = slice(core_id * hidden_quarter, (core_id + 1) * hidden_quarter)
    local_groups = group_values.reshape(batch * seq_len, groups * rank)
    # local_output: [B,S,D/4], context.dtype; TP2 production [1,128,1024] BF16.
    local_o_b_scale = runtime_weight_scale_slice(
        o_b_weight_chip,
        o_b_weight_chip_scale,
        branch_hidden_slice,
        slice(0, groups * rank),
    )
    local_output = runtime_linear(
        local_groups,
        o_b_weight_chip[branch_hidden_slice, :],
        local_o_b_scale,
        output_dtype=context.dtype,
        row_tile=ATTENTION_PROJECTION_SEQ_TILE,
        banks=banks,
    ).reshape(batch, seq_len, hidden_quarter)
    return local_output, branch_hidden_slice


def project_attention_output(
    prefix,
    context,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    branch_partial_out,
    meta,
    *,
    rotary_dim=64,
    residual_path_out=None,
    banks=None,
):
    """Store the Post-applied branch, optionally completed with one residual path."""
    batch, seq_len = context.shape[:2]
    local_output, branch_hidden_slice = project_attention_output_slice(
        prefix,
        context,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        meta,
        rotary_dim=rotary_dim,
        banks=banks,
    )
    # post: [B,S,4,1], FP32; output: [B,S,1,D/4], context.dtype.
    post = prefix.state.post.reshape(batch, seq_len, 4, 1)
    output = local_output.reshape(batch, seq_len, 1, local_output.shape[-1])
    # Match model.py::Block.hc_post and round only after the branch/residual sum.
    branch = ax.dsl.mul(post, output, output_dtype=ax.fp32)
    if residual_path_out is not None:
        # residual_path is FP32 and was initialized by token owners before the
        # prefix projection barriers.
        residual_path = load_tile(residual_path_out[:, :, :, branch_hidden_slice], banks=banks)
        branch = ax.dsl.add(branch, residual_path, output_dtype=context.dtype)
    else:
        branch = ax.dsl.cast(branch, dtype=context.dtype)
    # branch_partial_out slice: [B,S,4,D/4], branch.dtype.
    store_tile(branch, r=branch_partial_out[:, :, :, branch_hidden_slice], banks=banks)


def project_attention_local_output(
    prefix,
    context,
    o_a_weight_chip,
    o_a_weight_chip_scale,
    o_b_weight_chip,
    o_b_weight_chip_scale,
    local_output_out,
    meta,
    *,
    rotary_dim=64,
):
    """Store the D-wide chip contribution consumed by the inter-chip all-reduce."""
    local_output, hidden_slice = project_attention_output_slice(
        prefix,
        context,
        o_a_weight_chip,
        o_a_weight_chip_scale,
        o_b_weight_chip,
        o_b_weight_chip_scale,
        meta,
        rotary_dim=rotary_dim,
    )
    # local_output_out slice: [B,S,D/4], context.dtype.
    ax.dsl.store(local_output, r=local_output_out[:, :, hidden_slice])


def positive_i32(value, zero_i32, one_i32):
    """Return an INT32 0/1 indicator with the same shape as ``value``."""
    # nonnegative: [...same as value...], INT32.
    nonnegative = ax.dsl.maximum(value, zero_i32, output_dtype=ax.int32)
    # indicator: [...same as value...], INT32.
    return ax.dsl.minimum(nonnegative, one_i32, output_dtype=ax.int32)


def paged_swa_source(
    prefix,
    swa_kv_cache,
    swa_page_indices,
    swa_topk_lengths,
    meta,
    *,
    window,
    name_prefix,
    mask_query_slice,
    source_alignment=32,
    cooperative=True,
):
    """Build one aligned SWA source and the mask rows owned by this core."""
    core_id, core_num = meta.core_id, meta.core_num
    batch, seq_len = meta.batch, meta.seq_len
    head_dim = prefix.kv.shape[-1]
    assert batch == 1
    history_capacity = window - 1
    swa_capacity = history_capacity + seq_len
    source_capacity = ((swa_capacity + source_alignment - 1) // source_alignment) * source_alignment
    padding = source_capacity - swa_capacity
    assert history_capacity > 0

    source_memory = ax.MemoryType.SHARED_OCM if cooperative else ax.MemoryType.OCM
    # source: [B,T_aligned,Dh], prefix.kv.dtype. Shared callers use SHOCM and
    # partition source rows; private callers materialize the complete OCM
    # source independently per core. Production is [1,256,512] BF16.
    source = ax.dsl.create_tensor(
        (batch, source_capacity, head_dim),
        prefix.kv.dtype,
        source_memory,
        name=f"{name_prefix}_source",
    )
    # history_mask: [W-1], FP32 source_memory. The query-major page row stores
    # valid history at the beginning; it is right-aligned in source.
    history_mask = ax.dsl.create_tensor(
        (history_capacity,),
        ax.fp32,
        source_memory,
        name=f"{name_prefix}_history_mask",
    )
    history_slice = token_slice(core_id, core_num, history_capacity) if cooperative else slice(0, history_capacity)
    local_history = history_slice.stop - history_slice.start
    assert local_history > 0
    # zero_i32: [1], INT32.
    zero_i32 = ax.dsl.zeros((1,), ax.int32)
    one_i32 = constant(np.array([1], dtype=np.int32), f"{name_prefix}_history_one")
    # first_swa_length: [1], INT32.
    first_swa_length = ax.dsl.load(swa_topk_lengths[:1])
    # history_length: [1], INT32.
    history_length = ax.dsl.sub(first_swa_length, one_i32, output_dtype=ax.int32)
    history_capacity_i32 = constant(
        np.array([history_capacity], dtype=np.int32),
        f"{name_prefix}_history_capacity",
    )
    # history_padding: [1], INT32.
    history_padding = ax.dsl.sub(
        history_capacity_i32,
        history_length,
        output_dtype=ax.int32,
    )
    history_offsets = constant(
        np.arange(history_slice.start, history_slice.stop, dtype=np.int32),
        f"{name_prefix}_history_offsets",
    )
    # source_offsets: [W_core] or [W-1], INT32.
    source_offsets = ax.dsl.sub(history_offsets, history_padding, output_dtype=ax.int32)
    # safe_source_offsets: source_offsets.shape, INT32.
    safe_source_offsets = ax.dsl.maximum(source_offsets, zero_i32, output_dtype=ax.int32)
    # local_indices: source_offsets.shape, INT32.
    local_indices = ax.dsl.index_select(
        swa_page_indices[0:1, :].reshape(swa_page_indices.shape[1]),
        safe_source_offsets,
        dim=0,
    )
    # safe_indices: local_indices.shape, INT32.
    safe_indices = ax.dsl.maximum(local_indices, zero_i32, output_dtype=ax.int32)
    # source history slice: [B,W_core,Dh] or [B,W-1,Dh], prefix.kv.dtype.
    ax.dsl.index_select(
        swa_kv_cache.reshape(-1, head_dim),
        safe_indices,
        dim=0,
        r=source[:, history_slice, :].reshape(local_history, head_dim),
    )
    # active_distance: source_offsets.shape, INT32.
    active_distance = ax.dsl.add(source_offsets, one_i32, output_dtype=ax.int32)
    active = positive_i32(active_distance, zero_i32, one_i32)
    # inactive: source_offsets.shape, INT32.
    inactive = ax.dsl.sub(one_i32, active, output_dtype=ax.int32)
    # negative_bias: [1], FP32.
    negative_bias = ax.dsl.full((1,), ax.fp32, fill_value=-1.0e30)
    # history_mask slice: source_offsets.shape, FP32.
    ax.dsl.mul(
        inactive,
        negative_bias,
        output_dtype=ax.fp32,
        r=history_mask[history_slice],
    )

    sourcetoken_slice = token_slice(core_id, core_num, seq_len) if cooperative else slice(0, seq_len)
    source_token_count = sourcetoken_slice.stop - sourcetoken_slice.start
    current_slice = slice(
        history_capacity + sourcetoken_slice.start,
        history_capacity + sourcetoken_slice.stop,
    )
    # source current-token slice: [B,S_core,Dh] or [B,S,Dh], prefix.kv.dtype.
    if source_token_count > 0:
        ax.dsl.move(prefix.kv[:, sourcetoken_slice, :], r=source[:, current_slice, :])

    if padding and (not cooperative or core_id == core_num - 1):
        # padding_rows: [B,T_padding,Dh], prefix.kv.dtype.
        padding_rows = ax.dsl.zeros((batch, padding, head_dim), prefix.kv.dtype)
        # source padding slice: [B,T_padding,Dh], prefix.kv.dtype.
        ax.dsl.move(
            padding_rows,
            r=source[:, swa_capacity:, :],
        )
    if cooperative:
        # Synchronization only; no tensor output.
        axd.sync_all()
    mask_query_count = mask_query_slice.stop - mask_query_slice.start
    assert mask_query_count > 0

    # The fixed right-aligned layout makes each query's SWA window a compile-time
    # contiguous interval [query_index, query_index + W). Only early-history
    # padding remains a runtime mask.
    static_swa_mask = np.full((mask_query_count, swa_capacity), -1.0e30, dtype=np.float32)
    for local_query, query_index in enumerate(range(mask_query_slice.start, mask_query_slice.stop)):
        static_swa_mask[local_query, query_index : query_index + window] = 0.0
    swa_window_mask = constant(
        static_swa_mask,
        f"{name_prefix}_window_mask_c{core_id}",
    )
    # history_window_mask: [T_query,W-1], FP32.
    history_window_mask = ax.dsl.add(
        swa_window_mask[:, :history_capacity],
        history_mask.reshape(1, history_capacity),
        output_dtype=ax.fp32,
    )
    # source_mask: [T_query,W-1+S], FP32 before alignment padding.
    source_mask = ax.dsl.concat(
        history_window_mask,
        swa_window_mask[:, history_capacity:],
        dim=1,
    )
    if padding:
        # padding_mask: [T_query,T_padding], FP32.
        padding_mask = ax.dsl.full((mask_query_count, padding), ax.fp32, fill_value=-1.0e30)
        # source_mask: [T_query,T_aligned], FP32.
        source_mask = ax.dsl.concat(source_mask, padding_mask, dim=1)
    return source, source_mask.reshape(batch, mask_query_count, source_capacity)


__all__ = [
    "AttentionPhase",
    "AttentionShardDim",
    "TensorParallelLayout",
    "DSAIndexNorm",
    "DSASparseAttentionImpl",
    "as_phase",
    "as_shard_dim",
    "as_tp_layout",
    "as_dsa_index_norm",
    "as_dsa_sparse_impl",
    "configured_run_mode",
    "env_flag",
    "MLA_CONST_INPUTS",
    "normalize_fc_scale",
    "normalize_head_fc_scale",
    "normalize_scale2",
    "record_mla_absorbed_query",
    "record_mla_shared_projections",
    "rmsnorm",
    "rmsnorm_sqrt",
    "rope_impl",
    "gather_mrope_cos_sin",
    "mix_mrope_tables",
    "mrope_channel_masks",
    "normalize_mrope_section",
    "slice_fc_scale",
    "MX_GROUP_SIZE",
    "MX_WEIGHT_DTYPES",
    "ACTIVATION_QUANT_BLOCK_SIZE",
    "CONV_EUS_PER_CORE",
    "MHC_STATUS_SEQ_TILE",
    "MHC_FFN_QUERY_TILE",
    "MHC_COLLAPSED_HIDDEN_TILE",
    "MHC_RESIDUAL_HIDDEN_TILE",
    "ATTENTION_PREFIX_SEQ_TILE",
    "ATTENTION_PROJECTION_SEQ_TILE",
    "ATTENTION_QUERY_SEQ_TILE",
    "CSA_TOPK_SOURCE_CAPACITY",
    "CSA_INDEX_SOURCE_BLOCK",
    "MOE_EXPERT_CAPACITY",
    "MOE_ROUTER_TOKEN_TILE",
    "MOE_DISPATCH_EXPERT_GROUP",
    "MOE_SLOT_LOOKUP_TILE",
    "MOE_WEIGHTED_SUM_TILE",
    "MOE_WEIGHTED_SUM_HIDDEN_PARTS",
    "SHARED_EXPERT_GATE_UP_IC_TILE",
    "SHARED_EXPERT_DOWN_OC_TILE",
    "SHARED_EXPERT_DECODE_PRELOAD_WAVES_MXFP8",
    "SHARED_EXPERT_DECODE_PRELOAD_WAVES_MXFP4",
    "kernel_cache_enabled",
    "kernel_cache_path",
    "MHCState",
    "KernelMeta",
    "AttentionPrefix",
    "load_attention_prefix",
    "token_slice",
    "constant",
    "dsl_dtype",
    "mx_scale_slice",
    "validate_weight_scales",
    "constant_linear",
    "runtime_weight_scale_slice",
    "uses_model_activation_quant",
    "model_activation_quantize",
    "model_activation_matmul",
    "runtime_linear",
    "rmsnorm_with_constants",
    "mhc_controls",
    "mhc_residual_outputs",
    "mhc",
    "tail_rope",
    "dsv4_kv_fp8_qat",
    "project_attention_query_rows",
    "project_attention_query",
    "project_shared_attention_query",
    "attention_prefix",
    "project_attention_output_slice",
    "project_attention_output",
    "project_attention_local_output",
    "positive_i32",
    "paged_swa_source",
    "weight_num_bits",
    "normalize_group_fc_param",
    "group_quant_matmul",
    "const_fc_scale",
    "zeropoint_arg",
]
