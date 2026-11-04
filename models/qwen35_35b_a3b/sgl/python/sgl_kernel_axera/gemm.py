import math
import os
from typing import Optional

import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from axnn.backend.ax8860.spec import SHOCM_SIZE

M_SPLIT_FACTOR = 8
LM_HEAD_SPLIT_FACTOR = 256
UNROLL_N = 2
_PROJECTION_DTYPES = {None: None, torch.float16: ax.fp16, torch.bfloat16: ax.bf16, torch.float32: ax.fp32}


def _normalize_scale(scale: torch.Tensor, size: int, name: str, device: torch.device) -> torch.Tensor:
    scale = scale.reshape(-1)
    if scale.numel() == 1:
        return scale.to(device=device, dtype=torch.float32).expand(size).clone()
    if scale.numel() != size:
        raise ValueError(f"{name} shape {tuple(scale.shape)} is incompatible with size={size}")
    return scale.to(device=device, dtype=torch.float32)


def _prepare_bias_output(
    n: int,
    m: int,
    device: torch.device,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
):
    if bias is None:
        bias = torch.zeros((n,), device=device, dtype=out_dtype)
    elif tuple(bias.shape) != (n,):
        raise ValueError(f"bias shape {tuple(bias.shape)} is incompatible with N={n}")
    else:
        bias = bias.to(device=device, dtype=out_dtype)

    if output is None:
        output = torch.empty((m, n), device=device, dtype=out_dtype)
    elif tuple(output.shape) != (m, n):
        raise ValueError(f"output shape {tuple(output.shape)} is incompatible with {(m, n)}")

    return bias, output


# ── Plain GEMM (fp16 / bf16, no scale) ──────────────────────────────
@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def gemm_kernel(a, b_t, bias, output, tile_size, projection_dtype=None):
    offset_m = ax.program_id(0) * tile_size[0]
    offset_n = ax.program_id(1) * tile_size[1]

    m_end = min(a.shape[0], offset_m + tile_size[0])
    n_end = min(b_t.shape[0], offset_n + tile_size[1])

    slice_a = (slice(offset_m, m_end), slice(0, None))
    slice_b = (slice(offset_n, n_end), slice(0, None))
    slice_out = (slice(offset_m, m_end), slice(offset_n, n_end))

    a_tile = ax.dsl.load(a[slice_a])
    b_tile = ax.dsl.load(b_t[slice_b])

    out = ax.dsl.matmul(a_tile, b_tile, transposed=True, output_dtype=ax.fp32)

    bias_tile = ax.dsl.load(bias[slice(offset_n, n_end)])
    bias_tile = bias_tile.reshape(1, bias_tile.shape[0])
    projection_dtype = output.dtype if projection_dtype is None else projection_dtype
    out = ax.dsl.add(out, bias_tile, output_dtype=projection_dtype)
    if projection_dtype != output.dtype:
        out = ax.dsl.cast(out, dtype=output.dtype)
    ax.dsl.store(r=output[slice_out], x=out)

    return output


def fp16_gemm(
    a: torch.Tensor,
    b_t: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    projection_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Project ``a[M,K]`` through checkpoint weights ``b_t[N,K]``.

    ``projection_dtype`` selects projection rounding before the on-chip
    conversion to the output dtype; it defaults to the output dtype.
    """
    if a.dim() != 2 or b_t.dim() != 2:
        raise ValueError("a and b_t must be 2D tensors")

    m, k = a.shape
    if b_t.shape[1] != k:
        raise ValueError(f"b_t must have shape [N, K], got {tuple(b_t.shape)} for K={k}")
    n = b_t.shape[0]

    bias, output = _prepare_bias_output(n, m, a.device, out_dtype, bias, output)

    # CRUN keeps the loaded B tile and an internal transposed tile in OCM.
    # For LM-head shapes such as K=4096, N=151936, Ntile=512 makes each
    # B-side tile 4 MiB and OCM allocation can fail before DDR swap helps.
    max_b_tile_bytes = 1 << 20
    tile_n = max(16, max_b_tile_bytes // (k * b_t.element_size()))
    tile_n = max(16, (tile_n // 16) * 16)
    tile_size = [min(16, m), min(512, n, tile_n)]
    grid_x = math.ceil(m / tile_size[0])
    grid_y = math.ceil(n / tile_size[1])
    gemm_kernel[(grid_x, grid_y), 1](a, b_t, bias, output, tile_size, _PROJECTION_DTYPES[projection_dtype])
    return output


# ── Scaled GEMM (fp8 / int8) ─────────────────────────────────────────
@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def scaled_gemm_kernel(a, b_t, scale_a, scale_b, bias, output, tile_size):
    offset_m = ax.program_id(0) * tile_size[0]
    offset_n = ax.program_id(1) * tile_size[1]

    m_end = min(a.shape[0], offset_m + tile_size[0])
    n_end = min(b_t.shape[0], offset_n + tile_size[1])

    slice_a = (slice(offset_m, m_end), slice(0, None))
    slice_b = (slice(offset_n, n_end), slice(0, None))
    slice_out = (slice(offset_m, m_end), slice(offset_n, n_end))

    a_tile = ax.dsl.load(a[slice_a])
    b_tile = ax.dsl.load(b_t[slice_b])

    scale_a_tile = ax.dsl.load(scale_a[slice(offset_m, m_end)])
    scale_b_tile = ax.dsl.load(scale_b[slice(offset_n, n_end)])
    scale_a_tile = scale_a_tile.reshape(scale_a_tile.shape[0], 1)
    # Weights are [N, K]; each output-channel scale broadcasts across K.
    scale_b_tile = scale_b_tile.reshape(scale_b_tile.shape[0], 1)

    a_fp32 = ax.dsl.mul(a_tile, scale_a_tile, output_dtype=output.dtype)
    b_fp32 = ax.dsl.mul(b_tile, scale_b_tile, output_dtype=output.dtype)
    out = ax.dsl.matmul(a_fp32, b_fp32, transposed=True, output_dtype=ax.fp32)

    bias_tile = ax.dsl.load(bias[slice(offset_n, n_end)])
    bias_tile = bias_tile.reshape(1, bias_tile.shape[0])
    # accumulate in fp32, cast to output dtype only at the final add-bias step
    out = ax.dsl.add(out, bias_tile, output_dtype=output.dtype)
    ax.dsl.store(r=output[slice_out], x=out)

    return output


def _scaled_gemm(
    a: torch.Tensor,
    b_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    if a.dim() != 2 or b_t.dim() != 2:
        raise ValueError("a and b_t must be 2D tensors")

    m, k = a.shape
    if b_t.shape[1] != k:
        raise ValueError(f"b_t must have shape [N, K], got {tuple(b_t.shape)} for K={k}")
    n = b_t.shape[0]

    scale_a = _normalize_scale(scale_a, m, "scale_a", a.device)
    scale_b = _normalize_scale(scale_b, n, "scale_b", a.device)
    bias, output = _prepare_bias_output(n, m, a.device, out_dtype, bias, output)

    tile_size = [min(16, m), min(16, n)]
    grid_x = math.ceil(m / tile_size[0])
    grid_y = math.ceil(n / tile_size[1])
    scaled_gemm_kernel[(grid_x, grid_y), 1](a, b_t, scale_a, scale_b, bias, output, tile_size)
    return output


# ── RMSNorm + GEMM (dense FP16/BF16 or runtime NVFP4 weights) ───────
@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def rmsnorm_gemm_kernel(a, norm_weight, b_t, bias, normalized_ddr, output, tile_size, eps, b_scale=None, b_scale2=None):
    """RMSNorm and LM head projection in one tensor-parallel launch."""
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    tile_m, tile_n, shared_m = tile_size
    if a.shape[-1] % core_num != 0:
        raise ValueError(f"hidden size {a.shape[-1]} must be divisible by core_num={core_num}")
    shared_mem_type = ax.MemoryType.SHARED_OCM
    hidden_per_core = a.shape[-1] // core_num
    hidden_slice = slice(core_id * hidden_per_core, (core_id + 1) * hidden_per_core)
    hidden_size = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=a.shape[-1])
    eps_value = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=eps)
    total_n_tiles = math.ceil(b_t.shape[0] / tile_n)
    # AX8860's NVFP4 datapath consumes BF16 operands. Produce that dtype in
    # the final RMSNorm multiply, including when the source activation is FP16.
    projection_dtype = ax.bf16 if b_scale is not None else b_t.dtype
    # NVFP4 scales stay runtime inputs, alongside the packed weight matrix.
    scale2 = ax.dsl.load(b_scale2) if b_scale2 is not None else None  # [1] FP32 DDR -> OCM
    # Keep each normalized chunk on chip when it fits; oversized fmaps use the
    # DDR scratch tensor while preserving the same N-outer weight reuse below.
    normalized_shared = (
        ax.dsl.create_tensor(
            (shared_m, a.shape[-1]),
            projection_dtype,
            shared_mem_type,
            name="rmsnorm_gemm_normalized_shared",
        )
        if a.shape[0] <= shared_m
        else None
    )

    # Stage 1: normalize every M tile. Each core owns a hidden shard and the
    # local square sums are reduced through shared OCM in FP32.
    for chunk_offset in range(0, a.shape[0], shared_m):
        chunk_end = min(a.shape[0], chunk_offset + shared_m)
        for offset_m in range(chunk_offset, chunk_end, tile_m):
            m_end = min(chunk_end, offset_m + tile_m)
            row_slice = slice(offset_m, m_end)
            shared_row_slice = slice(offset_m - chunk_offset, m_end - chunk_offset)
            a_tile = ax.dsl.load(a[row_slice, hidden_slice])
            norm_weight_tile = ax.dsl.load(norm_weight[hidden_slice])

            square = ax.dsl.mul(a_tile, a_tile, output_dtype=ax.fp32)
            # Independent FP32 [1, tile_m] slots avoid ordered in-place
            # accumulation: a core with no vocab tile may reach the sum early.
            partial_sums = ax.dsl.create_tensor(
                (core_num, m_end - offset_m),
                ax.fp32,
                shared_mem_type,
                name=f"rmsnorm_gemm_partial_sums_{offset_m}",
            )  # [cores, tile_m] FP32 SHOCM
            ax.dsl.sum(
                square,
                dim=-1,
                keepdim=True,
                output_dtype=ax.fp32,
                r=partial_sums[core_id : core_id + 1, :].reshape(m_end - offset_m, 1),
            )
            axd.pub(partial_sums)
            axd.sub(partial_sums)
            square_sum_shared = ax.dsl.sum(partial_sums, dim=0, keepdim=True, output_dtype=ax.fp32)
            square_sum_shared = square_sum_shared.reshape(m_end - offset_m, 1)  # [tile_m, 1] FP32 OCM view

            variance = ax.dsl.div(square_sum_shared, hidden_size, output_dtype=ax.fp32)
            rms = ax.dsl.sqrt(ax.dsl.add(variance, eps_value, output_dtype=ax.fp32), output_dtype=ax.fp32)
            inv_rms = ax.dsl.reciprocal(rms, output_dtype=ax.fp32)
            normalized = ax.dsl.mul(a_tile, inv_rms, output_dtype=ax.fp32)
            normalized = ax.dsl.mul(
                normalized,
                norm_weight_tile,
                output_dtype=projection_dtype,
                r=normalized_shared[shared_row_slice, hidden_slice] if a.shape[0] <= shared_m else None,
            )
            if a.shape[0] > shared_m:
                ax.dsl.store(r=normalized_ddr[row_slice, hidden_slice], x=normalized)

    # Stage 2: reuse each group of weight tiles across all M chunks.
    # Projection cores consume every hidden shard, including shards from
    # cores that own no vocabulary tiles. Publish the actual shared input.
    normalized_input = normalized_shared if a.shape[0] <= shared_m else normalized_ddr
    axd.pub(normalized_input)
    axd.sub(normalized_input)
    unroll_n = UNROLL_N
    total_groups = total_n_tiles // unroll_n
    for group_idx in range(core_id, total_groups, core_num):
        group_base = group_idx * unroll_n * tile_n
        n_slices = []
        b_tiles = []
        scale_tiles = []
        bias_tiles = [] if bias is not None else None
        for u in range(unroll_n):
            offset_n = group_base + u * tile_n
            n_end = min(b_t.shape[0], offset_n + tile_n)
            n_slice = slice(offset_n, n_end)
            n_slices.append(n_slice)
            b_tiles.append(ax.dsl.load(b_t[n_slice, :]))
            # [tile_n, K/16] E4M3 DDR -> OCM; full K groups for these output rows.
            scale_tiles.append(ax.dsl.load(b_scale[n_slice, :]) if b_scale is not None else None)
            if bias is not None:
                bias_tiles.append(ax.dsl.load(bias[n_slice]).reshape(1, n_end - offset_n))
        for chunk_offset in range(0, a.shape[0], shared_m):
            chunk_end = min(a.shape[0], chunk_offset + shared_m)
            for offset_m in range(chunk_offset, chunk_end, tile_m):
                m_end = min(chunk_end, offset_m + tile_m)
                row_slice = slice(offset_m, m_end)
                shared_row_slice = slice(offset_m - chunk_offset, m_end - chunk_offset)
                if a.shape[0] > shared_m:
                    normalized_tile = ax.dsl.load(normalized_ddr[row_slice, :])
                else:
                    normalized_tile = normalized_shared[shared_row_slice, :]
                for u in range(unroll_n):
                    # [tile_m, K] BF16 x [tile_n, K/2] packed NVFP4
                    # (or [tile_n, K] dense) + block/global scales -> [tile_m, tile_n] FP32.
                    out = ax.dsl.matmul(
                        normalized_tile,
                        b_tiles[u],
                        w_scale=scale_tiles[u],
                        w_scale2=scale2,
                        transposed=True,
                        output_dtype=ax.fp32,
                    )
                    if bias is not None:
                        out = ax.dsl.add(out, bias_tiles[u], output_dtype=ax.fp32)
                    ax.dsl.store(r=output[row_slice, n_slices[u]], x=out)
    # Assign any unpaired final N tile to its original round-robin core.
    for tile_idx in range(total_groups * unroll_n, total_n_tiles):
        if tile_idx % core_num != core_id:
            continue
        offset_n = tile_idx * tile_n
        n_end = min(b_t.shape[0], offset_n + tile_n)
        n_slice = slice(offset_n, n_end)
        b_tile = ax.dsl.load(b_t[n_slice, :])
        scale_tile = ax.dsl.load(b_scale[n_slice, :]) if b_scale is not None else None  # [tail_n, K/16] E4M3
        if bias is not None:
            bias_tile = ax.dsl.load(bias[n_slice]).reshape(1, n_end - offset_n)
        for chunk_offset in range(0, a.shape[0], shared_m):
            chunk_end = min(a.shape[0], chunk_offset + shared_m)
            for offset_m in range(chunk_offset, chunk_end, tile_m):
                m_end = min(chunk_end, offset_m + tile_m)
                row_slice = slice(offset_m, m_end)
                shared_row_slice = slice(offset_m - chunk_offset, m_end - chunk_offset)
                if a.shape[0] > shared_m:
                    normalized_tile = ax.dsl.load(normalized_ddr[row_slice, :])
                else:
                    normalized_tile = normalized_shared[shared_row_slice, :]
                # Tail projection has the same input/scale dtypes and writes [tile_m, tail_n] FP32.
                out = ax.dsl.matmul(
                    normalized_tile,
                    b_tile,
                    w_scale=scale_tile,
                    w_scale2=scale2,
                    transposed=True,
                    output_dtype=ax.fp32,
                )
                if bias is not None:
                    out = ax.dsl.add(out, bias_tile, output_dtype=ax.fp32)
                ax.dsl.store(r=output[row_slice, n_slice], x=out)

    return output


def rmsnorm_fp16_gemm(
    a: torch.Tensor,
    norm_weight: torch.Tensor,
    b_t: torch.Tensor,
    eps: float = 1e-6,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    core_num=int(os.getenv("SGLANG_AXERA_KERNEL_CORE_NUM", "4")),
    *,
    b_scale=None,
    b_scale2=None,
) -> torch.Tensor:
    """Apply RMSNorm to ``a`` and project through ``b_t`` into FP32 logits.

    ``norm_weight`` uses plain RMSNorm semantics (``x * weight``). Callers with
    zero-centered weights must pass ``1 + weight``. FP32 coefficients retain
    their precision until the normalized activation is rounded.
    RMSNorm reductions and GEMM accumulation use FP32. Normalized operands use
    BF16 for NVFP4, otherwise the dense weight dtype. Logits are written in FP32.
    NVFP4 weights use packed [N, K/2] storage, E4M3 ``b_scale`` [N, K/16]
    and FP32 ``b_scale2`` [1], all supplied at runtime.
    """
    if not isinstance(core_num, int) or core_num < 1:
        raise ValueError(f"core_num must be positive, got {core_num}")
    if a.dim() != 2 or b_t.dim() != 2:
        raise ValueError("a and b_t must be 2D tensors")

    m, k = a.shape
    packed = b_t.dtype == torch.float4_e2m1fn_x2
    if b_t.shape[1] * (2 if packed else 1) != k:
        raise ValueError(f"b_t has incompatible shape {tuple(b_t.shape)} for K={k}")
    if packed:
        if k % 16 or b_scale is None or tuple(b_scale.shape) != (b_t.shape[0], k // 16):
            raise ValueError("NVFP4 requires block scales [N, K/16]")
        if (
            b_scale.dtype != torch.float8_e4m3fn
            or b_scale2 is None
            or b_scale2.shape != (1,)
            or b_scale2.dtype != torch.float32
        ):
            raise ValueError("NVFP4 requires E4M3 block scales and a FP32 global scale [1]")
    elif b_scale is not None or b_scale2 is not None:
        raise ValueError("block/global scales require packed NVFP4 weights")
    if tuple(norm_weight.shape) != (k,):
        raise ValueError(f"norm_weight shape {tuple(norm_weight.shape)} is incompatible with K={k}")
    n = b_t.shape[0]

    if bias is not None:
        if tuple(bias.shape) != (n,):
            raise ValueError(f"bias shape {tuple(bias.shape)} is incompatible with N={n}")
        bias = bias.to(device=a.device, dtype=torch.float32)
    if output is None:
        output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    elif tuple(output.shape) != (m, n) or output.dtype != torch.float32:
        raise ValueError(f"output must be FP32 with shape {(m, n)}")
    projection_dtype = torch.bfloat16 if packed else b_t.dtype
    # Keep FP32 norm coefficients until the kernel rounds the normalized input.
    norm_weight = norm_weight.to(device=a.device)

    cache_m = max(M_SPLIT_FACTOR, int(os.environ.get("SGLANG_AXERA_RMSNORM_GEMM_M", str(M_SPLIT_FACTOR))))
    kernel_m = cache_m if m <= cache_m else m
    kernel_a = a
    kernel_output = output
    if kernel_m != m:
        kernel_a = a.new_empty((kernel_m, k))
        kernel_a[:m] = a
        kernel_a[m:] = a[-1]
        kernel_output = torch.empty((kernel_m, n), device=output.device, dtype=output.dtype)

    if k % core_num != 0:
        raise ValueError(f"hidden size {k} must be divisible by core_num={core_num} for RMSNorm")

    # Reserve the per-core FP32 partial sums and use the remaining SHARED_OCM for
    # normalized rows. Physical row bytes are aligned to the hardware's
    # 32-byte shared-memory alignment.
    row_bytes = math.ceil(k * projection_dtype.itemsize / 32) * 32
    square_sum_bytes = core_num * M_SPLIT_FACTOR * torch.float32.itemsize
    max_shared_m = (SHOCM_SIZE - square_sum_bytes) // row_bytes
    max_shared_m = (max_shared_m // M_SPLIT_FACTOR) * M_SPLIT_FACTOR
    if max_shared_m < M_SPLIT_FACTOR:
        raise ValueError(
            f"one {M_SPLIT_FACTOR}-row normalized tile requires more than {SHOCM_SIZE} bytes of SHARED_OCM"
        )
    shared_m = min(kernel_m, max_shared_m)
    normalized_ddr = None
    if kernel_m > shared_m:
        normalized_ddr = torch.empty((kernel_m, k), device=a.device, dtype=projection_dtype)

    args = (
        kernel_a,
        norm_weight,
        b_t,
        bias,
        normalized_ddr,
        kernel_output,
        [min(M_SPLIT_FACTOR, kernel_m), LM_HEAD_SPLIT_FACTOR, shared_m],
        eps,
        b_scale,
        b_scale2,
    )
    rmsnorm_gemm_kernel[core_num, core_num](*args)
    if kernel_m != m:
        output.copy_(kernel_output[:m])
    return output


def fp8_gemm(
    a_fp8: torch.Tensor,
    b_fp8_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _scaled_gemm(a_fp8, b_fp8_t, scale_a, scale_b, out_dtype, bias, output)


def bf16_gemm(
    a: torch.Tensor,
    b_t: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return fp16_gemm(a, b_t, out_dtype, bias, output)


def int8_gemm(
    a_int8: torch.Tensor,
    b_int8_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _scaled_gemm(a_int8, b_int8_t, scale_a, scale_b, out_dtype, bias, output)


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def nvfp4_gemm_kernel(a, b_fp4, b_scale, b_scale2, bias, output, tile_size):
    offset_m = ax.program_id(0) * tile_size[0]
    offset_n = ax.program_id(1) * tile_size[1]

    m_end = min(a.shape[0], offset_m + tile_size[0])
    n_end = min(output.shape[1], offset_n + tile_size[1])

    slice_a = (slice(offset_m, m_end), slice(0, None))
    slice_b_fp4 = (slice(offset_n, n_end), slice(0, None))
    slice_b_scale = (slice(offset_n, n_end), slice(0, None))
    slice_out = (slice(offset_m, m_end), slice(offset_n, n_end))

    a_tile = ax.dsl.load(a[slice_a])

    # hardware dequantizes: w_float = fp4_to_float(b_fp4) * fp8_to_float(b_scale) * b_scale2
    out = ax.dsl.matmul(
        a_tile,
        b_fp4[slice_b_fp4],
        w_scale=b_scale[slice_b_scale],
        w_scale2=b_scale2,
        transposed=True,
        output_dtype=ax.fp32,
        const_inputs=["b"],
    )

    bias_tile = ax.dsl.load(bias[slice(offset_n, n_end)])
    bias_tile = bias_tile.reshape(1, bias_tile.shape[0])
    # accumulate in fp32, cast to output dtype at the final add-bias step
    out = ax.dsl.add(out, bias_tile, output_dtype=output.dtype)
    ax.dsl.store(r=output[slice_out], x=out)

    return output


def nvfp4_gemm(
    a: torch.Tensor,
    b_fp4,
    b_scale,
    b_scale2,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NVFP4 GEMM: output = a @ dequant(b_fp4, b_scale, b_scale2).T + bias

    a        : (m, k) bf16/fp16 activation
    b_fp4    : (n, k) NVFP4 weight (1 fp4 per byte, ml_dtypes layout)
    b_scale  : (n, k//16) FP8_E4M3 per-block scale (block_size=16)
    b_scale2 : (1,) float32 global scale
    """
    if a.dim() != 2:
        raise ValueError("a must be a 2D tensor")

    m, k = a.shape
    n = b_fp4.shape[0]
    if b_fp4.shape[1] != k:
        raise ValueError(f"b_fp4 shape {b_fp4.shape} is incompatible with a having k={k}")

    bias, output = _prepare_bias_output(n, m, a.device, out_dtype, bias, output)

    tile_size = [min(16, m), min(16, n)]
    grid_m = math.ceil(m / tile_size[0])
    grid_n = math.ceil(n / tile_size[1])
    nvfp4_gemm_kernel[(grid_m, grid_n), 1](a, b_fp4, b_scale, b_scale2, bias, output, tile_size)
    return output


# ──MX-format GEMM (mxfp4 / mxfp8 / int4) ─────────────────────────
@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def mx_gemm_kernel(a, b_w, b_scale, bias, output, tile_size):
    offset_m = ax.program_id(0) * tile_size[0]
    offset_n = ax.program_id(1) * tile_size[1]

    m_end = min(a.shape[0], offset_m + tile_size[0])
    n_end = min(output.shape[1], offset_n + tile_size[1])

    slice_a = (slice(offset_m, m_end), slice(0, None))
    slice_b_w = (slice(offset_n, n_end), slice(0, None))
    slice_b_scale = (slice(offset_n, n_end), slice(0, None))
    slice_out = (slice(offset_m, m_end), slice(offset_n, n_end))

    a_tile = ax.dsl.load(a[slice_a])

    # hardware dequantizes: w_float = w_to_float(b_w) * scale_to_float(b_scale)
    out = ax.dsl.matmul(
        a_tile,
        b_w[slice_b_w],
        w_scale=b_scale[slice_b_scale],
        transposed=True,
        output_dtype=ax.fp32,
        const_inputs=["b"],
    )

    bias_tile = ax.dsl.load(bias[slice(offset_n, n_end)])
    bias_tile = bias_tile.reshape(1, bias_tile.shape[0])
    # accumulate in fp32, cast to output dtype at the final add-bias step
    out = ax.dsl.add(out, bias_tile, output_dtype=output.dtype)
    ax.dsl.store(r=output[slice_out], x=out)

    return output


def _mx_gemm(
    a: torch.Tensor,
    b_w,
    b_scale,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    if a.dim() != 2:
        raise ValueError("a must be a 2D tensor")

    m, k = a.shape
    n = b_w.shape[0]
    if b_w.shape[1] != k:
        raise ValueError(f"b_w shape {b_w.shape} is incompatible with a having k={k}")

    bias, output = _prepare_bias_output(n, m, a.device, out_dtype, bias, output)

    tile_size = [min(16, m), min(16, n)]
    grid_m = math.ceil(m / tile_size[0])
    grid_n = math.ceil(n / tile_size[1])
    mx_gemm_kernel[(grid_m, grid_n), 1](a, b_w, b_scale, bias, output, tile_size)
    return output


def mxfp4_gemm(
    a: torch.Tensor,
    b_fp4,
    b_scale,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MXFP4 GEMM: output = a @ dequant(b_fp4, b_scale).T + bias

    a       : (m, k) bf16/fp16 activation
    b_fp4   : (n, k) MXFP4 weight (1 fp4 per byte, ml_dtypes layout)
    b_scale : (n, k//32) uint8 E8M0 per-block scale (block_size=32)
    """
    return _mx_gemm(a, b_fp4, b_scale, out_dtype, bias, output)


def mxfp8_gemm(
    a: torch.Tensor,
    b_fp8,
    b_scale,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MXFP8 GEMM: output = a @ dequant(b_fp8, b_scale).T + bias

    a       : (m, k) bf16/fp16 activation
    b_fp8   : (n, k) MXFP8 weight (fp8_e4m3 or fp8_e5m2)
    b_scale : (n, k//32) uint8 E8M0 per-block scale (block_size=32)
    """
    return _mx_gemm(a, b_fp8, b_scale, out_dtype, bias, output)


def int4_gemm(
    a: torch.Tensor,
    b_int4,
    b_scale,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    INT4 GEMM: output = a @ dequant(b_int4, b_scale).T + bias

    a       : (m, k) bf16/fp16 activation
    b_int4  : (n, k) INT4 weight (1 int4 per byte)
    b_scale : (n, k//group_size) float32 per-group scale
    """
    return _mx_gemm(a, b_int4, b_scale, out_dtype, bias, output)
