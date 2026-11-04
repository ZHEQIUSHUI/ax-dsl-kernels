import os
from typing import Optional

from sgl_kernel_axera.attention.attn_utils.common import rmsnorm
import torch

import axnn.dsl as ax

DEFAULT_POOL_ROWS = 8


@ax.builder(
    enable_cache=True,
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def pool_norm_l2_kernel(pooled, norm_weight, output, eps, dims):
    """RMSNorm + optional matryoshka truncation + L2 normalize in one launch.

    ``pooled`` holds pre-selected last-token rows; the gather itself is a
    host-side ``index_select`` so the kernel only sees the small
    (rows, hidden) tensor whose padded shape keeps the compile cache stable.
    ``eps`` and ``dims`` are trace-time constants: the RMSNorm variance is
    always over the full hidden width, the L2 norm over the truncated width.
    """
    x = ax.dsl.load(pooled)
    weight = ax.dsl.load(norm_weight)
    # FP32 RMSNorm over the full hidden width (shared attention helper).
    normalized = rmsnorm(x, weight, eps=eps, output_dtype=ax.fp32)
    if dims < pooled.shape[-1]:
        normalized = normalized[:, :dims]
    # L2-normalize the (possibly truncated) rows, FP32 accumulation.
    squared = ax.dsl.mul(normalized, normalized, output_dtype=ax.fp32)
    square_sum = ax.dsl.sum(squared, dim=-1, keepdim=True, output_dtype=ax.fp32)
    inv_norm = ax.dsl.reciprocal(ax.dsl.sqrt(square_sum, output_dtype=ax.fp32), output_dtype=ax.fp32)
    out = ax.dsl.mul(normalized, inv_norm, output_dtype=ax.fp32)
    ax.dsl.store(r=output, x=out)
    return output


def pool_norm_l2(
    hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    dims: Optional[int] = None,
) -> torch.Tensor:
    """Fused embedding tail: gather last-token rows, RMSNorm, truncate, L2.

    Equivalent to ``F.normalize(rmsnorm(hidden_states[idx])[:, :dims], dim=-1)``
    in FP32 — the eager tail's ``pow``/``mean``/``rsqrt``/``linalg_vector_norm``
    are ATen ops torch_axera does not register.

    Args:
        hidden_states: (num_tokens, hidden) pre-norm rows, residual already
            merged (bf16/fp16).
        last_token_indices: (num_requests,) row index of each request's last
            token.
        norm_weight: (hidden,) RMSNorm weight; cast to FP32 here.
        eps: RMSNorm epsilon.
        dims: optional matryoshka output width in [1, hidden]. ``None`` keeps
            the full hidden width.

    Returns:
        (num_requests, dims or hidden) FP32 tensor, each row L2-normalized.
    """
    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be 2D, got {tuple(hidden_states.shape)}")
    _, hidden = hidden_states.shape
    if tuple(norm_weight.shape) != (hidden,):
        raise ValueError(f"norm_weight shape {tuple(norm_weight.shape)} is incompatible with hidden={hidden}")
    if dims is not None and not 1 <= dims <= hidden:
        raise ValueError(f"dims must be in [1, {hidden}], got {dims}")

    num_requests = last_token_indices.shape[0]
    out_width = hidden if dims is None else dims
    output = torch.empty((num_requests, out_width), device=hidden_states.device, dtype=torch.float32)
    if num_requests == 0:
        return output

    # The gather stays on the host side: feeding the (num_tokens, hidden)
    # source into the kernel would bake a fresh shape into every compile.
    indices = last_token_indices.to(device=hidden_states.device)
    pooled = hidden_states.index_select(0, indices)
    # The weight arrives as an nn.Parameter (requires_grad); the NPU launch
    # path converts tensors to NumPy, which grad-tracking tensors forbid.
    weight = norm_weight.detach().to(device=hidden_states.device, dtype=torch.float32)

    # Pad the row count up to a cache floor (duplicate the last row) so the
    # compiled graph — and thus the BRUN cache — stays shape-stable.
    cache_rows = max(1, int(os.environ.get("SGLANG_AXERA_EMBED_ROWS", str(DEFAULT_POOL_ROWS))))
    kernel_rows = cache_rows if num_requests <= cache_rows else num_requests
    kernel_pooled, kernel_output = pooled, output
    if kernel_rows != num_requests:
        kernel_pooled = pooled.new_empty((kernel_rows, hidden))
        kernel_pooled[:num_requests] = pooled
        kernel_pooled[num_requests:] = pooled[-1]
        kernel_output = torch.empty((kernel_rows, out_width), device=output.device, dtype=torch.float32)

    ax.init(chip_num=1, core_num=1)
    pool_norm_l2_kernel[(1, 1), 1](kernel_pooled, weight, kernel_output, eps, out_width)
    if kernel_rows != num_requests:
        output.copy_(kernel_output[:num_requests])
    return output
