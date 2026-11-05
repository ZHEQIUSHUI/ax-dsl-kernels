import math
import os
from typing import Optional

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd

SAMPLING_TILE_BATCH = 16
# Ax8860 reduce cycle model requires vocab>65536 to be a multiple of 65536.
ARGMAX_VOCAB_ALIGN = 65536
# Budget each core's loaded logits and reduction scratch in FP32 bytes,
# including when the caller passes fp16/bf16 logits.
MAX_ARGMAX_TILE_BYTES = 4 << 20


def _sampling_kernel_batch(batch_size: int) -> int:
    cache_batch = max(
        SAMPLING_TILE_BATCH,
        int(os.environ.get("SGLANG_AXERA_SAMPLING_BATCH", str(SAMPLING_TILE_BATCH))),
    )
    return cache_batch if batch_size <= cache_batch else batch_size


def _pad_rows(tensor: torch.Tensor, kernel_batch: int) -> torch.Tensor:
    # Repeat the last valid row. Filling with 0/-inf lets NPU reduce/argmax
    # treat the pad as a real competitor (inf often becomes 0 after quant).
    if tensor.shape[0] == kernel_batch:
        return tensor
    padded = tensor.new_empty((kernel_batch, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    padded[tensor.shape[0] :] = tensor[-1]
    return padded


def _empty_rows(tensor: torch.Tensor, kernel_batch: int) -> torch.Tensor:
    if tensor.shape[0] == kernel_batch:
        return tensor
    return tensor.new_empty((kernel_batch, *tensor.shape[1:]))


def _scalar_fp32(value: float, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32, device=ref.device)


def _padded_launch_tensors(
    src: torch.Tensor,
    dst: torch.Tensor,
    kernel_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if src.shape[0] == kernel_batch:
        return src, dst
    return _pad_rows(src, kernel_batch), _empty_rows(dst, kernel_batch)


def _argmax_chunk_plan(vocab: int) -> tuple[int, int]:
    """Split vocabulary across cores with at most 65536 elements per reduce."""
    chunk_count = max(4, math.ceil(vocab / ARGMAX_VOCAB_ALIGN))
    chunk_width = math.ceil(vocab / chunk_count)
    # Ceil division can leave an empty last chunk for tiny vocabularies.
    return math.ceil(vocab / chunk_width), chunk_width


def _argmax_tile_batch(vocab: int) -> int:
    # Large-vocab ArgMax shards chunks across four cores. Each core
    # keeps at most ceil(n_chunks / 4) loaded chunks live, plus the
    # transpose scratch. Keep the conservative FP32 budget because AxArgMax
    # lowers through the float reduce path for fp16/bf16 inputs.
    n_chunks, chunk_cols = _argmax_chunk_plan(vocab)
    chunks_per_core = math.ceil(n_chunks / 4)
    bytes_per_row = chunk_cols * 4 * (chunks_per_core + 1)
    max_rows = max(1, MAX_ARGMAX_TILE_BYTES // max(bytes_per_row, 1))
    return min(SAMPLING_TILE_BATCH, max_rows)


def _launch_row_kernel(kernel, kernel_batch: int, *args, **kwargs):
    tile_size = [SAMPLING_TILE_BATCH]
    grid_m = math.ceil(kernel_batch / tile_size[0])
    kernel[grid_m, 1](*args, tile_size=tile_size, **kwargs)


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def argmax_kernel(logits, output, tile_size, max_id):
    """Emit INT32[M] IDs or INT32[M,2] words backed by the caller's INT64[M] buffer."""
    assert ax.num_programs(0) == 4, "this kernel requires four cores"
    # All cores work on the same row tile and own disjoint vocabulary chunks.
    del tile_size
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    end_m = logits.shape[0]
    row_slice = slice(0, end_m)
    vocab = logits.shape[1]

    shared_mem_type = ax.MemoryType.SHARED_OCM

    chunk_count, chunk_width = _argmax_chunk_plan(vocab)
    # Keep one candidate per vocab chunk in shared OCM. The candidate columns
    # are indexed by chunk_id rather than core_id so first-index tie handling
    # remains identical to the original serial chunk order even when a core
    # owns more than one chunk.
    shared_vals = ax.dsl.create_tensor(
        (end_m, chunk_count),
        logits.dtype,
        shared_mem_type,
        name="argmax_chunk_values_shared",
    )
    shared_idxs = ax.dsl.create_tensor(
        (end_m, chunk_count),
        ax.int32,
        shared_mem_type,
        name="argmax_chunk_indices_shared",
    )

    # Each core handles a disjoint set of chunks. The strided form also keeps
    # the kernel correct when the vocabulary needs more chunks than cores.
    for chunk_id in range(core_id, chunk_count, core_num):
        vocab_start = chunk_id * chunk_width
        vocab_end = min(vocab, vocab_start + chunk_width)
        tile = ax.dsl.load(logits[row_slice, vocab_start:vocab_end])
        destination = shared_idxs[:, chunk_id : chunk_id + 1]
        # ArgMax needs a contiguous result for multi-row reductions; a shared
        # candidate column has a strided layout. Gather and Add accept that view.
        idx = ax.dsl.argmax(tile, dim=1, keepdim=1)
        ax.dsl.gather(tile, idx, dim=1, r=shared_vals[:, chunk_id : chunk_id + 1])
        if vocab_start != 0:
            idx = ax.dsl.add(
                idx,
                ax.dsl.full(shape=(1,), dtype=ax.int32, fill_value=vocab_start),
                r=destination,
            )
        else:
            ax.dsl.move(idx, r=destination)

    axd.pub(shared_vals)
    axd.pub(shared_idxs)
    axd.sub(shared_vals)
    axd.sub(shared_idxs)

    # The second reduction is tiny: it only compares one candidate per vocab
    # chunk. Restrict it to core 0 so output ownership is unambiguous.
    if core_id == 0:
        winner = ax.dsl.argmax(shared_vals, dim=1, keepdim=1)
        result = ax.dsl.gather(shared_idxs, winner, dim=1)
        result = ax.dsl.reshape(result, shape=(end_m,))
        # Keep the token id inside the real vocabulary.
        result = ax.dsl.clip(result, min=0, max=max_id)
        if len(output.shape) == 2:
            # result[M] int32 -> low/high words[M,2] of INT64 token IDs.
            result = ax.dsl.concat(result.reshape(end_m, 1), ax.dsl.zeros((end_m, 1), dtype=ax.int32), dim=1)
        ax.dsl.store(r=output[row_slice], x=result)

    return output


def _pad_logits_for_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Pad vocab dim so large reductions satisfy Ax8860 cycle-model alignment."""
    vocab = logits.shape[1]
    if vocab <= ARGMAX_VOCAB_ALIGN or vocab % ARGMAX_VOCAB_ALIGN == 0:
        return logits

    padded_vocab = math.ceil(vocab / ARGMAX_VOCAB_ALIGN) * ARGMAX_VOCAB_ALIGN
    # Use the finite dtype min. -inf is flushed to 0 on some NPU quant paths
    # and then beats typical negative logits, emitting out-of-range token ids.
    padded = logits.new_full((logits.shape[0], padded_vocab), torch.finfo(logits.dtype).min)
    padded[:, :vocab] = logits
    return padded


def argmax(
    logits: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    core_num: int = int(os.getenv("SGLANG_AXERA_KERNEL_CORE_NUM", "4")),
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("logits must be a 2D tensor")

    m = logits.shape[0]
    orig_vocab = logits.shape[1]

    if output is None:
        output = torch.empty((m,), device=logits.device, dtype=torch.int32)
    elif tuple(output.shape) != (m,):
        raise ValueError(f"output shape {tuple(output.shape)} is incompatible with batch size {m}")

    logits = _pad_logits_for_argmax(logits)
    tile_batch = _argmax_tile_batch(logits.shape[1])
    # One cached (tile_batch, vocab) kernel at a time. Four cores share all
    # vocabulary chunks, including Qwen's three natural 65536-element windows.
    for row_start in range(0, m, tile_batch):
        row_end = min(m, row_start + tile_batch)
        kernel_logits, kernel_output = _padded_launch_tensors(
            logits[row_start:row_end], output[row_start:row_end], tile_batch
        )
        argmax_kernel[core_num, core_num](kernel_logits, kernel_output, tile_size=[tile_batch], max_id=orig_vocab - 1)
        if kernel_output.shape[0] != row_end - row_start:
            output[row_start:row_end].copy_(kernel_output[: row_end - row_start])
    return output


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def min_p_sampling_from_probs_kernel(probs, output, tile_size, p, eps):

    offset_m = ax.program_id(0) * tile_size[0]
    end_m = min(probs.shape[0], offset_m + tile_size[0])
    slice_in = (slice(offset_m, end_m), slice(0, None))

    probs_tile = ax.dsl.load(probs[slice_in])
    max_per_row = ax.dsl.max(probs_tile, dim=-1, keepdim=True)  # (tile_batch, 1)

    p_const = ax.dsl.load(p)
    eps_const = ax.dsl.load(eps)
    threshold = ax.dsl.mul(max_per_row, p_const)
    threshold_strict = ax.dsl.add(threshold, eps_const)

    BIG = 1.0e10
    big_const = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=BIG)
    diff = ax.dsl.sub(probs_tile, threshold_strict)
    scaled = ax.dsl.mul(diff, big_const)
    mask = ax.dsl.clip(scaled, min=0.0, max=1.0)

    kept = ax.dsl.mul(probs_tile, mask)
    kept_sum = ax.dsl.sum(kept, dim=-1, keepdim=True)
    norm_kept = ax.dsl.div(kept, kept_sum)

    ax.dsl.store(r=output[slice_in], x=norm_kept)
    return output


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p,
    eps: float = 1e-9,
    generator: Optional[torch.Generator] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if probs.dim() != 2:
        raise ValueError("probs must be a 2D tensor")

    if isinstance(min_p, torch.Tensor):
        if min_p.numel() == 0:
            raise ValueError("min_p tensor is empty")
        p_min = float(min_p.min().item())
        p_max = float(min_p.max().item())
        if p_min != p_max:
            raise ValueError(f"min_p tensor must be uniform across the batch (got min={p_min}, max={p_max})")
        p = p_min
    else:
        p = float(min_p)

    if not (0.0 <= p <= 1.0):
        raise ValueError(f"min_p must be in [0, 1], got {p}")

    batch_size, vocab_size = probs.shape
    if p == 1.0:
        samples = torch.argmax(probs, dim=-1).to(torch.int32)
        if output is None:
            return samples
        output.copy_(samples)
        return output

    # Special case: p == 0 keeps everything. Skip the kernel and sample directly.
    if p == 0.0:
        renorm = probs
    else:
        renorm = torch.empty_like(probs)
        kernel_batch = _sampling_kernel_batch(batch_size)
        kernel_probs, kernel_renorm = _padded_launch_tensors(probs, renorm, kernel_batch)
        _launch_row_kernel(
            min_p_sampling_from_probs_kernel,
            kernel_batch,
            kernel_probs,
            kernel_renorm,
            p=_scalar_fp32(p, probs),
            eps=_scalar_fp32(eps, probs),
        )
        if kernel_batch != batch_size:
            renorm = kernel_renorm[:batch_size]

    samples = torch.multinomial(renorm, num_samples=1, generator=generator).squeeze(-1)
    samples = samples.to(torch.int32)

    if output is None:
        return samples
    output.copy_(samples)
    return output


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def top_k_renorm_prob_kernel(probs, output, tile_size, k):

    offset_m = ax.program_id(0) * tile_size[0]
    end_m = min(probs.shape[0], offset_m + tile_size[0])

    slice_in = (slice(offset_m, end_m), slice(0, None))

    probs_tile = ax.dsl.load(probs[slice_in])

    top_values, top_indices = ax.dsl.topk(probs_tile, k=k, dim=-1)
    top_sum = ax.dsl.sum(top_values, dim=-1, keepdim=True)
    top_norm = ax.dsl.div(top_values, top_sum)

    tile_batch = end_m - offset_m
    out_zero = ax.dsl.zeros(shape=(tile_batch, probs.shape[1]), dtype=top_norm.dtype)
    ax.dsl.scatter(r=out_zero, dim=1, indices=top_indices, x=top_norm)

    ax.dsl.store(r=output[slice_in], x=out_zero)
    return output


def top_k_renorm_prob(probs: torch.Tensor, k, output: Optional[torch.Tensor] = None) -> torch.Tensor:
    if probs.dim() != 2:
        raise ValueError("probs must be a 2D tensor")

    if isinstance(k, torch.Tensor):
        if k.numel() == 0:
            raise ValueError("top_k tensor is empty")
        k_min = int(k.min().item())
        k_max = int(k.max().item())
        if k_min != k_max:
            raise ValueError(f"top_k tensor must be uniform across the batch (got min={k_min}, max={k_max})")
        k = k_min
    else:
        k = int(k)

    batch_size, vocab_size = probs.shape

    if k >= vocab_size:
        k = vocab_size

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if output is None:
        output = torch.empty_like(probs)
    elif tuple(output.shape) != (batch_size, vocab_size):
        raise ValueError(f"output shape {tuple(output.shape)} != probs shape {(batch_size, vocab_size)}")

    kernel_batch = _sampling_kernel_batch(batch_size)
    kernel_probs, kernel_output = _padded_launch_tensors(probs, output, kernel_batch)
    _launch_row_kernel(top_k_renorm_prob_kernel, kernel_batch, kernel_probs, kernel_output, k=k)
    if kernel_batch != batch_size:
        output.copy_(kernel_output[:batch_size])
    return output


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def top_k_top_p_sampling_from_probs_kernel(probs, output, tile_size, k, p, eps):

    offset_m = ax.program_id(0) * tile_size[0]
    end_m = min(probs.shape[0], offset_m + tile_size[0])
    slice_in = (slice(offset_m, end_m), slice(0, None))

    V = probs.shape[1]
    tile_batch = end_m - offset_m

    probs_tile = ax.dsl.load(probs[slice_in])  # (tile_batch, V)

    top_vals, top_idx = ax.dsl.topk(probs_tile, k=k, dim=-1)
    lt_np = np.tril(np.ones((k, k), dtype=np.float32), k=-1)
    lt_const = ax.dsl.from_numpy(lt_np)

    top_vals = ax.dsl.cast(top_vals, dtype=ax.fp16)
    lt_const = ax.dsl.cast(lt_const, dtype=ax.fp16)
    excl_cumsum = ax.dsl.matmul(top_vals, lt_const, transposed=True, output_dtype=ax.fp32)

    BIG = 1.0e10
    threshold_const = ax.dsl.add(ax.dsl.load(p), ax.dsl.load(eps))
    big_const = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=BIG)
    diff = ax.dsl.sub(threshold_const, excl_cumsum)
    scaled = ax.dsl.mul(diff, big_const)
    mask = ax.dsl.clip(scaled, min=0.0, max=1.0)

    top_vals = ax.dsl.cast(top_vals, dtype=ax.fp32)
    kept = ax.dsl.mul(top_vals, mask)
    kept_sum = ax.dsl.sum(kept, dim=-1, keepdim=True)
    norm_kept = ax.dsl.div(kept, kept_sum)

    out_zero = ax.dsl.zeros(shape=(tile_batch, V), dtype=top_vals.dtype)
    ax.dsl.scatter(r=out_zero, dim=1, indices=top_idx, x=norm_kept)

    ax.dsl.store(r=output[slice_in], x=out_zero)
    return output


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k,
    top_p,
    filter_apply_order: str = "joint",
    eps: float = 1e-4,
    generator: Optional[torch.Generator] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if filter_apply_order != "joint":
        raise ValueError(f"only filter_apply_order='joint' is supported, got {filter_apply_order!r}")
    if probs.dim() != 2:
        raise ValueError("probs must be a 2D tensor")

    if isinstance(top_k, torch.Tensor):
        if top_k.numel() == 0:
            raise ValueError("top_k tensor is empty")
        k_min = int(top_k.min().item())
        k_max = int(top_k.max().item())
        if k_min != k_max:
            raise ValueError(f"top_k tensor must be uniform across the batch (got min={k_min}, max={k_max})")
        k = k_min
    else:
        k = int(top_k)

    if isinstance(top_p, torch.Tensor):
        if top_p.numel() == 0:
            raise ValueError("top_p tensor is empty")
        p_min = float(top_p.min().item())
        p_max = float(top_p.max().item())
        if p_min != p_max:
            raise ValueError(f"top_p tensor must be uniform across the batch (got min={p_min}, max={p_max})")
        p = p_min
    else:
        p = float(top_p)

    batch_size, vocab_size = probs.shape
    if k <= 0 or k > vocab_size:
        raise ValueError(f"top_k ({k}) must be in (0, vocab_size={vocab_size}]")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {p}")

    renorm = torch.empty_like(probs)
    kernel_batch = _sampling_kernel_batch(batch_size)
    kernel_probs, kernel_renorm = _padded_launch_tensors(probs, renorm, kernel_batch)
    _launch_row_kernel(
        top_k_top_p_sampling_from_probs_kernel,
        kernel_batch,
        kernel_probs,
        kernel_renorm,
        k=k,
        p=_scalar_fp32(p, probs),
        eps=_scalar_fp32(eps, probs),
    )
    if kernel_batch != batch_size:
        renorm = kernel_renorm[:batch_size]

    samples = torch.multinomial(renorm, num_samples=1, generator=generator).squeeze(-1)
    samples = samples.to(torch.int32)

    if output is None:
        return samples
    output.copy_(samples)
    return output


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def top_p_renorm_prob_kernel(probs, output, tile_size, p):
    offset_m = ax.program_id(0) * tile_size[0]
    end_m = min(probs.shape[0], offset_m + tile_size[0])
    slice_in = (slice(offset_m, end_m), slice(0, None))

    V = probs.shape[1]
    tile_batch = end_m - offset_m

    probs_tile = ax.dsl.load(probs[slice_in])  # (tile_batch, V)
    sorted_desc, indices = ax.dsl.topk(probs_tile, k=V, dim=-1)

    # Exclusive prefix sum via AxCumSum: cumsum(x) - x. The old construction built a
    # V x V lower-triangular constant and used matmul as the prefix sum, which
    # allocates V*V*4 host bytes (86 GiB at vocab 151936). The fp16 round trip keeps
    # the cutoff numerics of the old fp16 matmul path.
    sorted_fp16 = ax.dsl.cast(sorted_desc, dtype=ax.fp16)
    sorted_fp32 = ax.dsl.cast(sorted_fp16, dtype=ax.fp32)
    incl_cumsum = ax.dsl.cumsum(sorted_fp32, dim=1, output_dtype=ax.fp32)
    excl_cumsum = ax.dsl.sub(incl_cumsum, sorted_fp32)

    BIG = 1.0e10
    p_const = ax.dsl.load(p)
    big_const = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=BIG)
    diff = ax.dsl.sub(p_const, excl_cumsum)
    scaled = ax.dsl.mul(diff, big_const)
    mask = ax.dsl.clip(scaled, min=0.0, max=1.0)

    kept = ax.dsl.mul(sorted_fp32, mask)
    kept_sum = ax.dsl.sum(kept, dim=-1, keepdim=True)
    norm_kept = ax.dsl.div(kept, kept_sum)

    out_zero = ax.dsl.zeros(shape=(tile_batch, V), dtype=sorted_fp32.dtype)
    ax.dsl.scatter(r=out_zero, dim=1, indices=indices, x=norm_kept)

    ax.dsl.store(r=output[slice_in], x=out_zero)
    return output


def top_p_renorm_prob(probs: torch.Tensor, p, output: Optional[torch.Tensor] = None) -> torch.Tensor:
    if probs.dim() != 2:
        raise ValueError("probs must be a 2D tensor")

    if isinstance(p, torch.Tensor):
        if p.numel() == 0:
            raise ValueError("top_p tensor is empty")
        p_min = float(p.min().item())
        p_max = float(p.max().item())
        if p_min != p_max:
            raise ValueError(f"top_p tensor must be uniform across the batch (got min={p_min}, max={p_max})")
        p = p_min
    else:
        p = float(p)

    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    batch_size, vocab_size = probs.shape

    if output is None:
        output = torch.empty_like(probs)
    elif tuple(output.shape) != (batch_size, vocab_size):
        raise ValueError(f"output shape {tuple(output.shape)} != probs shape {(batch_size, vocab_size)}")

    if p == 1.0:
        output.copy_(probs)
        return output

    kernel_batch = _sampling_kernel_batch(batch_size)
    kernel_probs, kernel_output = _padded_launch_tensors(probs, output, kernel_batch)
    _launch_row_kernel(
        top_p_renorm_prob_kernel,
        kernel_batch,
        kernel_probs,
        kernel_output,
        p=_scalar_fp32(p, probs),
    )
    if kernel_batch != batch_size:
        output.copy_(kernel_output[:batch_size])
    return output
