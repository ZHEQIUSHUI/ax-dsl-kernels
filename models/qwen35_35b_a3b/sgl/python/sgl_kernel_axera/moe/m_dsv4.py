"""DeepSeek-V4 HASH and ROUTED MoE kernels plus compact validation helpers."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd
from sgl_kernel_axera.attention.parallel_utils import linear_program_slice, validate_core_num

from ..attention.attn_utils.common import (
    CONV_EUS_PER_CORE,
    MOE_WEIGHTED_SUM_HIDDEN_PARTS,
    MOE_WEIGHTED_SUM_TILE,
    prepare_model_activation,
    runtime_linear,
    token_slice,
)
from .moe_helper import (
    ffn_prefill,
    pack_tokens_to_local_experts,
    prepare_routed_weighted_sum,
    prepare_token_dispatch,
    route_tokens_with_router,
    routed_weighted_sum_tile,
    run_shared_expert,
)


def run_mode() -> ax.RunMode:
    return ax.RunMode(os.environ.get("RUN_MODE", "CRUN"))


def kernel_weight(value: torch.Tensor) -> torch.Tensor:
    if run_mode() == ax.RunMode.CRUN and value.dtype == torch.float32:
        return value.to(torch.bfloat16)
    return value


def constant(value, name=None):
    """Create a contiguous DSL constant for route and dispatch metadata."""
    array = np.asarray(value)
    if array.dtype.kind == "f":
        array = array.astype(np.float32)
    array = np.ascontiguousarray(array)
    return ax.dsl.from_numpy(array, prefix=name) if name is not None else ax.dsl.from_numpy(array)


def tiled_matmul(
    x,
    weight,
    weight_scale=None,
    *,
    output_dtype,
    output_tile: int = 1024,
    activation_input=None,
):
    if weight_scale is not None:
        return runtime_linear(
            x,
            weight,
            weight_scale,
            output_dtype=output_dtype,
            output_tile=output_tile,
            # Keep one complete K row for each 512-channel output tile
            # resident.  The old 32-wide slicing emitted four DDR-load jobs
            # per tile and multiplied into thousands of FinishKernel nodes.
            input_load_tile=4096,
            activation_input=activation_input,
        )
    output_width = weight.shape[0]
    parts = []
    for start in range(0, output_width, output_tile):
        stop = min(start + output_tile, output_width)
        weight_part = weight[start:stop, :]
        if weight_part.memory_type == ax.MemoryType.DDR:
            weight_part = ax.dsl.load(weight_part)
        if weight_part.dtype in (ax.fp8_e4m3, ax.fp8_e5m2):
            weight_part = ax.dsl.cast(weight_part, dtype=ax.bf16)
        part = ax.dsl.matmul(x, weight_part, transposed=True, output_dtype=output_dtype)
        parts.append(part)
    return parts[0] if len(parts) == 1 else ax.dsl.concat(*parts, dim=-1)


def router_scores(x, gate_weight):
    logits = ax.dsl.matmul(x, gate_weight, transposed=True, output_dtype=ax.fp32)
    return ax.dsl.sqrt(ax.dsl.softplus(logits, output_dtype=ax.fp32), output_dtype=ax.fp32)


def normalize_weights(weights, route_scale: float):
    denominator = ax.dsl.add(
        ax.dsl.sum(weights, dim=-1, keepdim=True, output_dtype=ax.fp32),
        ax.dsl.full((1,), ax.fp32, fill_value=1e-20),
        output_dtype=ax.fp32,
    )
    return ax.dsl.mul(
        ax.dsl.div(weights, denominator, output_dtype=ax.fp32), constant(route_scale), output_dtype=ax.fp32
    )


def normalize_selected_weights(scores, indices, route_scale: float):
    return normalize_weights(ax.dsl.gather(scores, indices, dim=1), route_scale)


def expert_forward(
    x,
    w1,
    w2,
    w3,
    swiglu_limit: float,
    output_dtype,
    route_weight=None,
    route_after_down: bool = True,
    weight_scales=None,
    activation_input=None,
):
    weight_scales = {} if weight_scales is None else weight_scales
    # Gate and up projections consume the same activation.  The model.py
    # compatible FP8 activation dequantization is graph work, so record it
    # once per FFN input instead of once for every projection and expert.
    projected_input = prepare_model_activation(x, w1) if activation_input is None else activation_input
    gate = ax.dsl.clip(
        ax.dsl.cast(
            tiled_matmul(
                x,
                w1,
                weight_scales.get("w1"),
                output_dtype=output_dtype,
                activation_input=projected_input,
            ),
            dtype=ax.fp32,
        ),
        min=-3.389531389e38,
        max=swiglu_limit,
    )
    up = ax.dsl.clip(
        ax.dsl.cast(
            tiled_matmul(
                x,
                w3,
                weight_scales.get("w3"),
                output_dtype=output_dtype,
                activation_input=projected_input,
            ),
            dtype=ax.fp32,
        ),
        min=-swiglu_limit,
        max=swiglu_limit,
    )
    # Match the reference Expert.forward: keep SwiGLU and pre-down route
    # weighting in FP32, then round once for the down-projection input.
    hidden = ax.dsl.mul(ax.dsl.silu(gate, output_dtype=ax.fp32), up, output_dtype=ax.fp32)
    if route_weight is not None and not route_after_down:
        hidden = ax.dsl.mul(hidden, route_weight, output_dtype=ax.fp32)
    hidden = ax.dsl.cast(hidden, dtype=output_dtype)
    value = tiled_matmul(hidden, w2, weight_scales.get("w2"), output_dtype=output_dtype)
    if route_weight is not None and route_after_down:
        value = ax.dsl.mul(value, route_weight, output_dtype=output_dtype)
    return value


def selected_expert_forward(
    x,
    w1,
    w2,
    w3,
    swiglu_limit: float,
    output_dtype,
    route_weight=None,
    weight_scales=None,
):
    weight_scales = {} if weight_scales is None else weight_scales
    gate = tiled_matmul(x, w1, weight_scales.get("w1"), output_dtype=output_dtype)
    up = tiled_matmul(x, w3, weight_scales.get("w3"), output_dtype=output_dtype)
    gate = ax.dsl.clip(gate, min=-3.389531389e38, max=swiglu_limit)
    up = ax.dsl.clip(up, min=-swiglu_limit, max=swiglu_limit)
    hidden = ax.dsl.mul(ax.dsl.silu(gate, output_dtype=output_dtype), up, output_dtype=output_dtype)
    value = tiled_matmul(hidden, w2, weight_scales.get("w2"), output_dtype=output_dtype)
    return ax.dsl.mul(value, route_weight, output_dtype=output_dtype) if route_weight is not None else value


def expert_weight_scales(weight_scales, prefix, index, intermediate_dim, hidden_dim):
    if weight_scales is None:
        return None
    return {
        "w1": weight_scales[f"{prefix}_w1"][index : index + 1].reshape(intermediate_dim, hidden_dim // 32),
        "w2": weight_scales[f"{prefix}_w2"][index : index + 1].reshape(hidden_dim, intermediate_dim // 32),
        "w3": weight_scales[f"{prefix}_w3"][index : index + 1].reshape(intermediate_dim, hidden_dim // 32),
    }


def shared_weight_scales(weight_scales):
    if weight_scales is None:
        return None
    return {
        "w1": weight_scales["shared_w1"],
        "w2": weight_scales["shared_w2"],
        "w3": weight_scales["shared_w3"],
    }


def moe_partial_value(
    x,
    gate_weight,
    selected_indices,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    expert_start: int,
    expert_stop: int,
    include_shared: bool,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
    weight_scales=None,
):
    batch, seq_len, hidden_dim = x.shape
    flat = x.reshape(batch * seq_len, hidden_dim)
    scores = router_scores(flat, gate_weight)
    indices = selected_indices.reshape(batch * seq_len, selected_indices.shape[-1])
    selected = normalize_selected_weights(scores, indices, route_scale)
    return weighted_moe_partial_value(
        x,
        selected.reshape(batch, seq_len, selected.shape[-1]),
        selected_indices,
        expert_w1,
        expert_w2,
        expert_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        expert_start=expert_start,
        expert_stop=expert_stop,
        include_shared=include_shared,
        swiglu_limit=swiglu_limit,
        output_dtype=output_dtype,
        route_after_down=route_after_down,
        weight_scales=weight_scales,
    )


def weighted_moe_partial_value(
    x,
    topk_weights,
    topk_ids,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    expert_start: int,
    expert_stop: int,
    include_shared: bool,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
    weight_scales=None,
):
    """Evaluate a compact expert set using SGLang-style Top-K outputs."""
    batch, seq_len, hidden_dim = x.shape
    expert_num, intermediate_dim, _ = expert_w1.shape
    if expert_start < 0 or expert_start > expert_stop or expert_stop > expert_num:
        raise ValueError("routed expert slice is out of range")
    if topk_weights.shape != topk_ids.shape or topk_ids.shape[:2] != (batch, seq_len):
        raise ValueError("topk_weights and topk_ids must have shape [batch, sequence, topk]")
    flat = x.reshape(batch * seq_len, hidden_dim)
    indices = topk_ids.reshape(batch * seq_len, topk_ids.shape[-1])
    selected = topk_weights.reshape(batch * seq_len, topk_weights.shape[-1])
    shared_input = prepare_model_activation(flat, expert_w1) if expert_num else flat
    sparse = ax.dsl.zeros((batch * seq_len, expert_num), dtype=ax.fp32)
    ax.dsl.scatter(selected, indices, dim=1, r=sparse)
    combined = ax.dsl.zeros((batch * seq_len, hidden_dim), dtype=ax.fp32)
    for expert in range(expert_start, expert_stop):
        value = expert_forward(
            flat,
            expert_w1[expert : expert + 1].reshape(intermediate_dim, hidden_dim),
            expert_w2[expert : expert + 1].reshape(hidden_dim, intermediate_dim),
            expert_w3[expert : expert + 1].reshape(intermediate_dim, hidden_dim),
            swiglu_limit,
            output_dtype,
            sparse[:, expert : expert + 1],
            route_after_down=route_after_down,
            weight_scales=expert_weight_scales(weight_scales, "expert", expert, intermediate_dim, hidden_dim),
            activation_input=shared_input,
        )
        combined = ax.dsl.add(combined, value, output_dtype=ax.fp32)
    if include_shared:
        combined = ax.dsl.add(
            combined,
            expert_forward(
                flat,
                shared_w1,
                shared_w2,
                shared_w3,
                swiglu_limit,
                output_dtype,
                weight_scales=shared_weight_scales(weight_scales),
                activation_input=shared_input,
            ),
            output_dtype=ax.fp32,
        )
    return combined.reshape(batch, seq_len, hidden_dim)


def weighted_moe_value(
    x,
    topk_weights,
    topk_ids,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
    weight_scales=None,
):
    """Evaluate all compact routed experts and one shared expert."""
    return ax.dsl.cast(
        weighted_moe_partial_value(
            x,
            topk_weights,
            topk_ids,
            expert_w1,
            expert_w2,
            expert_w3,
            shared_w1,
            shared_w2,
            shared_w3,
            expert_start=0,
            expert_stop=expert_w1.shape[0],
            include_shared=True,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            route_after_down=route_after_down,
            weight_scales=weight_scales,
        ),
        dtype=output_dtype,
    )


def moe_value(
    x,
    gate_weight,
    selected_indices,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
    weight_scales=None,
):
    return ax.dsl.cast(
        moe_partial_value(
            x,
            gate_weight,
            selected_indices,
            expert_w1,
            expert_w2,
            expert_w3,
            shared_w1,
            shared_w2,
            shared_w3,
            expert_start=0,
            expert_stop=expert_w1.shape[0],
            include_shared=True,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            route_after_down=route_after_down,
            weight_scales=weight_scales,
        ),
        dtype=output_dtype,
    )


def selected_moe_partial_value(
    x,
    gate_weight,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    slot_start: int,
    slot_stop: int,
    include_shared: bool,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    weight_scales=None,
):
    batch, seq_len, hidden_dim = x.shape
    if batch * seq_len != 1:
        raise ValueError("preselected expert validation requires one input token")
    topk, intermediate_dim, _ = expert_w1.shape
    flat = x.reshape(1, hidden_dim)
    selected = normalize_weights(router_scores(flat, gate_weight), route_scale)
    combined = ax.dsl.zeros((1, hidden_dim), dtype=ax.fp32)
    for slot in range(slot_start, slot_stop):
        value = selected_expert_forward(
            flat,
            expert_w1[slot : slot + 1].reshape(intermediate_dim, hidden_dim),
            expert_w2[slot : slot + 1].reshape(hidden_dim, intermediate_dim),
            expert_w3[slot : slot + 1].reshape(intermediate_dim, hidden_dim),
            swiglu_limit,
            output_dtype,
            selected[:, slot : slot + 1],
            weight_scales=expert_weight_scales(weight_scales, "expert", slot, intermediate_dim, hidden_dim),
        )
        combined = ax.dsl.add(combined, value, output_dtype=ax.fp32)
    if include_shared:
        combined = ax.dsl.add(
            combined,
            selected_expert_forward(
                flat,
                shared_w1,
                shared_w2,
                shared_w3,
                swiglu_limit,
                output_dtype,
                weight_scales=shared_weight_scales(weight_scales),
            ),
            output_dtype=ax.fp32,
        )
    return combined.reshape(batch, seq_len, hidden_dim)


def selected_moe_value(
    x,
    gate_weight,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    layer_id: int = 0,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    weight_scales=None,
):
    return selected_moe_partial_value(
        x,
        gate_weight,
        expert_w1,
        expert_w2,
        expert_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        slot_start=0,
        slot_stop=expert_w1.shape[0],
        include_shared=True,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        output_dtype=output_dtype,
        weight_scales=weight_scales,
    )


def selected_moe_multicore_value(
    x,
    gate_weight,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    layer_id: int,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
):
    batch, seq_len, hidden_dim = x.shape
    if batch * seq_len != 1:
        raise ValueError("preselected expert validation requires one input token")
    topk, intermediate_dim, _ = expert_w1.shape
    core_id, core_num = ax.program_id(0), ax.num_programs(0)
    validate_core_num(core_num, topk, "selected expert slots")
    # Keep each expert contribution in FP32, matching the single-core
    # reduction order before the final activation-dtype cast.
    contributions = ax.dsl.create_tensor(
        (topk + 1, batch, seq_len, hidden_dim),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name=f"ds_v4_layer_{layer_id}_moe_contributions",
    )
    flat = x.reshape(1, hidden_dim)
    selected = normalize_weights(router_scores(flat, gate_weight), route_scale)
    slot_start, slot_stop = core_id * topk // core_num, (core_id + 1) * topk // core_num
    for slot in range(slot_start, slot_stop):
        value = selected_expert_forward(
            flat,
            expert_w1[slot : slot + 1].reshape(intermediate_dim, hidden_dim),
            expert_w2[slot : slot + 1].reshape(hidden_dim, intermediate_dim),
            expert_w3[slot : slot + 1].reshape(intermediate_dim, hidden_dim),
            swiglu_limit,
            output_dtype,
            selected[:, slot : slot + 1],
        )
        ax.dsl.move(
            ax.dsl.cast(value, dtype=ax.fp32).reshape(1, batch, seq_len, hidden_dim),
            r=contributions[slot : slot + 1],
        )
    if core_id == 0:
        shared = selected_expert_forward(flat, shared_w1, shared_w2, shared_w3, swiglu_limit, output_dtype)
        ax.dsl.move(
            ax.dsl.cast(shared, dtype=ax.fp32).reshape(1, batch, seq_len, hidden_dim),
            r=contributions[topk : topk + 1],
        )
    axd.sync_all()
    combined = ax.dsl.zeros((batch, seq_len, hidden_dim), dtype=ax.fp32)
    for slot in range(topk + 1):
        combined = ax.dsl.add(
            combined, contributions[slot : slot + 1].reshape(batch, seq_len, hidden_dim), output_dtype=ax.fp32
        )
    return ax.dsl.cast(combined, dtype=output_dtype)


def hash_moe_value(
    x,
    gate_weight,
    hash_indices,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
):
    return (
        moe_value(
            x,
            gate_weight,
            hash_indices,
            expert_w1,
            expert_w2,
            expert_w3,
            shared_w1,
            shared_w2,
            shared_w3,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            route_after_down=route_after_down,
        ),
        hash_indices,
    )


def hash_moe_partial_value(
    x,
    gate_weight,
    hash_indices,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    expert_start: int,
    expert_stop: int,
    include_shared: bool,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
):
    return (
        moe_partial_value(
            x,
            gate_weight,
            hash_indices,
            expert_w1,
            expert_w2,
            expert_w3,
            shared_w1,
            shared_w2,
            shared_w3,
            expert_start=expert_start,
            expert_stop=expert_stop,
            include_shared=include_shared,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            route_after_down=route_after_down,
        ),
        hash_indices,
    )


def score_selected_indices(x, gate_weight, gate_bias, topk: int):
    batch, seq_len, hidden_dim = x.shape
    flat = x.reshape(batch * seq_len, hidden_dim)
    logits = ax.dsl.add(
        router_scores(flat, gate_weight), gate_bias.reshape(1, gate_weight.shape[0]), output_dtype=ax.fp32
    )
    _, indices = ax.dsl.topk(logits, k=int(topk), dim=-1)
    return indices.reshape(batch, seq_len, int(topk))


def score_moe_value(
    x,
    gate_weight,
    gate_bias,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    topk: int,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
):
    indices = score_selected_indices(x, gate_weight, gate_bias, topk)
    return (
        moe_value(
            x,
            gate_weight,
            indices,
            expert_w1,
            expert_w2,
            expert_w3,
            shared_w1,
            shared_w2,
            shared_w3,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            route_after_down=route_after_down,
        ),
        indices,
    )


def score_moe_partial_value(
    x,
    gate_weight,
    gate_bias,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    *,
    topk: int,
    expert_start: int,
    expert_stop: int,
    include_shared: bool,
    route_scale: float,
    swiglu_limit: float,
    output_dtype,
    route_after_down: bool = True,
):
    indices = score_selected_indices(x, gate_weight, gate_bias, topk)
    return (
        moe_partial_value(
            x,
            gate_weight,
            indices,
            expert_w1,
            expert_w2,
            expert_w3,
            shared_w1,
            shared_w2,
            shared_w3,
            expert_start=expert_start,
            expert_stop=expert_stop,
            include_shared=include_shared,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            route_after_down=route_after_down,
        ),
        indices,
    )


@ax.builder(run_mode=run_mode())
def hash_moe_kernel(
    x,
    gate_weight,
    hash_indices,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    output,
    output_indices,
    route_scale,
    swiglu_limit,
):
    batch, seq_len, hidden_dim = x.shape
    topk = hash_indices.shape[-1]
    rows = batch * seq_len
    row_slice = linear_program_slice(rows)
    local_rows = row_slice.stop - row_slice.start
    value, indices = hash_moe_value(
        ax.dsl.load(x.reshape(rows, hidden_dim)[row_slice, :]).reshape(1, local_rows, hidden_dim),
        ax.dsl.load(gate_weight),
        ax.dsl.load(hash_indices.reshape(rows, topk)[row_slice, :]).reshape(1, local_rows, topk),
        ax.dsl.load(expert_w1),
        ax.dsl.load(expert_w2),
        ax.dsl.load(expert_w3),
        ax.dsl.load(shared_w1),
        ax.dsl.load(shared_w2),
        ax.dsl.load(shared_w3),
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        output_dtype=output.dtype,
    )
    ax.dsl.store(value.reshape(local_rows, hidden_dim), r=output.reshape(rows, hidden_dim)[row_slice, :])
    ax.dsl.store(indices.reshape(local_rows, topk), r=output_indices.reshape(rows, topk)[row_slice, :])
    return output, output_indices


@ax.builder(run_mode=run_mode())
def score_moe_kernel(
    x,
    gate_weight,
    gate_bias,
    expert_w1,
    expert_w2,
    expert_w3,
    shared_w1,
    shared_w2,
    shared_w3,
    output,
    output_indices,
    topk,
    route_scale,
    swiglu_limit,
):
    batch, seq_len, hidden_dim = x.shape
    rows = batch * seq_len
    row_slice = linear_program_slice(rows)
    local_rows = row_slice.stop - row_slice.start
    value, indices = score_moe_value(
        ax.dsl.load(x.reshape(rows, hidden_dim)[row_slice, :]).reshape(1, local_rows, hidden_dim),
        ax.dsl.load(gate_weight),
        ax.dsl.load(gate_bias),
        ax.dsl.load(expert_w1),
        ax.dsl.load(expert_w2),
        ax.dsl.load(expert_w3),
        ax.dsl.load(shared_w1),
        ax.dsl.load(shared_w2),
        ax.dsl.load(shared_w3),
        topk=topk,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        output_dtype=output.dtype,
    )
    ax.dsl.store(value.reshape(local_rows, hidden_dim), r=output.reshape(rows, hidden_dim)[row_slice, :])
    ax.dsl.store(indices.reshape(local_rows, topk), r=output_indices.reshape(rows, topk)[row_slice, :])
    return output, output_indices


def validate_moe_weights(x, gate_weight, expert_w1, expert_w2, expert_w3, shared_w1, shared_w2, shared_w3):
    if x.ndim != 3 or gate_weight.ndim != 2:
        raise ValueError("MoE expects x:[B,S,D] and gate_weight:[E,D]")
    experts, hidden_dim = gate_weight.shape
    if hidden_dim != x.shape[-1] or expert_w1.ndim != 3:
        raise ValueError("MoE gate and expert dimensions are incompatible")
    intermediate = expert_w1.shape[1]
    if (
        tuple(expert_w1.shape) != (experts, intermediate, hidden_dim)
        or tuple(expert_w3.shape) != (experts, intermediate, hidden_dim)
        or tuple(expert_w2.shape) != (experts, hidden_dim, intermediate)
    ):
        raise ValueError("MoE expert dimensions are incompatible")
    if (
        tuple(shared_w1.shape) != (intermediate, hidden_dim)
        or tuple(shared_w3.shape) != tuple(shared_w1.shape)
        or tuple(shared_w2.shape) != (hidden_dim, intermediate)
    ):
        raise ValueError("shared expert dimensions are incompatible")
    return experts, hidden_dim


def deepseek_moe(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_w1: torch.Tensor,
    expert_w2: torch.Tensor,
    expert_w3: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w3: torch.Tensor,
    *,
    topk: int,
    route_scale: float,
    swiglu_limit: float,
    hash_indices: Optional[torch.Tensor] = None,
    gate_bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    core_num: int = 1,
):
    experts, _ = validate_moe_weights(x, gate_weight, expert_w1, expert_w2, expert_w3, shared_w1, shared_w2, shared_w3)
    if topk <= 0 or topk > experts or swiglu_limit <= 0:
        raise ValueError("MoE topk and swiglu_limit must be positive and valid")
    if (hash_indices is None) == (gate_bias is None):
        raise ValueError("provide exactly one of hash_indices or gate_bias")
    output = output if output is not None else torch.empty_like(x)
    selected_indices = torch.empty((*x.shape[:2], topk), dtype=torch.int32, device=x.device)
    converted = tuple(
        kernel_weight(v) for v in (gate_weight, expert_w1, expert_w2, expert_w3, shared_w1, shared_w2, shared_w3)
    )
    validate_core_num(core_num, x.shape[0] * x.shape[1], "MoE token rows")
    ax.init(chip_num=1, core_num=core_num)
    if hash_indices is not None:
        if tuple(hash_indices.shape) != (*x.shape[:2], topk):
            raise ValueError("hash_indices must have shape [B,S,K]")
        hash_moe_kernel[core_num](
            x,
            converted[0],
            hash_indices.to(torch.int32),
            *converted[1:],
            output,
            selected_indices,
            route_scale,
            swiglu_limit,
        )
    else:
        if tuple(gate_bias.shape) != (experts,):
            raise ValueError("gate_bias must have one value per routed expert")
        score_moe_kernel[core_num](
            x, converted[0], gate_bias, *converted[1:], output, selected_indices, topk, route_scale, swiglu_limit
        )
    return output, selected_indices


@ax.builder(
    const_inputs=[
        "hc_fn",
        "hc_base",
        "hc_scale",
        "weight_scales",
        "norm_weight",
        "router_weight",
        "gate_up_weight_local",
        "down_weight_local",
        "shared_gate_up_weight_chip",
        "shared_down_weight_chip",
    ],
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    # Layer-local expert weights must stay constants.  Exposing them as
    # runtime inputs expands every expert load into the FinishKernel graph.
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_hash_prefill(
    residual,
    token_ids,
    tid2eid,
    hc_fn,
    hc_base,
    hc_scale,
    weight_scales,
    norm_weight,
    router_weight,
    gate_up_weight_local,
    down_weight_local,
    shared_gate_up_weight_chip,
    shared_down_weight_chip,
    expert_output_workspace,
    expert_slot_workspace,
    branch_partial,
    residual_path,
    selected_ids_out,
    route_weights_out,
    expert_offset=0,
    top_k=6,
    route_scale=1.5,
    swiglu_limit=10.0,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record HASH selection plus local experts and shared-expert TP contribution."""
    ffn_prefill(
        residual,
        hc_fn,
        hc_base,
        hc_scale,
        norm_weight,
        router_weight,
        shared_gate_up_weight_chip,
        shared_down_weight_chip,
        expert_output_workspace,
        expert_slot_workspace,
        branch_partial,
        residual_path,
        selected_ids_out,
        route_weights_out,
        weight_scales,
        gate_up_weight_local=gate_up_weight_local,
        down_weight_local=down_weight_local,
        token_ids=token_ids,
        tid2eid=tid2eid,
        expert_offset=expert_offset,
        top_k=top_k,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    return branch_partial, residual_path, selected_ids_out, route_weights_out


@ax.builder(
    const_inputs=[
        "hc_fn",
        "hc_base",
        "hc_scale",
        "weight_scales",
        "norm_weight",
        "router_weight",
        "shared_gate_up_weight_chip",
        "shared_down_weight_chip",
    ],
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_hash_static_prefill(
    residual,
    token_ids,
    tid2eid,
    hc_fn,
    hc_base,
    hc_scale,
    weight_scales,
    norm_weight,
    router_weight,
    shared_gate_up_weight_chip,
    shared_down_weight_chip,
    expert_output_workspace,
    expert_slot_workspace,
    branch_partial,
    residual_path,
    selected_ids_out,
    route_weights_out,
    expert_offset=0,
    top_k=6,
    route_scale=1.5,
    swiglu_limit=10.0,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Consume precomputed expert outputs using HASH-selected routes."""
    ffn_prefill(
        residual,
        hc_fn,
        hc_base,
        hc_scale,
        norm_weight,
        router_weight,
        shared_gate_up_weight_chip,
        shared_down_weight_chip,
        expert_output_workspace,
        expert_slot_workspace,
        branch_partial,
        residual_path,
        selected_ids_out,
        route_weights_out,
        weight_scales,
        token_ids=token_ids,
        tid2eid=tid2eid,
        expert_offset=expert_offset,
        top_k=top_k,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    return branch_partial, residual_path, selected_ids_out, route_weights_out


@ax.builder(
    const_inputs=[
        "hc_fn",
        "hc_base",
        "hc_scale",
        "weight_scales",
        "norm_weight",
        "router_weight",
        "correction_bias",
        "gate_up_weight_local",
        "down_weight_local",
        "shared_gate_up_weight_chip",
        "shared_down_weight_chip",
    ],
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_routed_prefill(
    residual,
    hc_fn,
    hc_base,
    hc_scale,
    weight_scales,
    norm_weight,
    router_weight,
    correction_bias,
    gate_up_weight_local,
    down_weight_local,
    shared_gate_up_weight_chip,
    shared_down_weight_chip,
    expert_output_workspace,
    expert_slot_workspace,
    branch_partial,
    residual_path,
    selected_ids_out,
    route_weights_out,
    expert_offset=0,
    top_k=6,
    route_scale=1.5,
    swiglu_limit=10.0,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record correction-biased Top-K plus local experts and shared-expert TP contribution."""
    ffn_prefill(
        residual,
        hc_fn,
        hc_base,
        hc_scale,
        norm_weight,
        router_weight,
        shared_gate_up_weight_chip,
        shared_down_weight_chip,
        expert_output_workspace,
        expert_slot_workspace,
        branch_partial,
        residual_path,
        selected_ids_out,
        route_weights_out,
        weight_scales,
        gate_up_weight_local=gate_up_weight_local,
        down_weight_local=down_weight_local,
        correction_bias=correction_bias,
        expert_offset=expert_offset,
        top_k=top_k,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    return branch_partial, residual_path, selected_ids_out, route_weights_out


@ax.builder(
    const_inputs=[
        "hc_fn",
        "hc_base",
        "hc_scale",
        "weight_scales",
        "norm_weight",
        "router_weight",
        "correction_bias",
        "shared_gate_up_weight_chip",
        "shared_down_weight_chip",
    ],
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_routed_static_prefill(
    residual,
    hc_fn,
    hc_base,
    hc_scale,
    weight_scales,
    norm_weight,
    router_weight,
    correction_bias,
    shared_gate_up_weight_chip,
    shared_down_weight_chip,
    expert_output_workspace,
    expert_slot_workspace,
    branch_partial,
    residual_path,
    selected_ids_out,
    route_weights_out,
    expert_offset=0,
    top_k=6,
    route_scale=1.5,
    swiglu_limit=10.0,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Consume precomputed expert outputs and retain the non-expert-MLP FFN stages."""
    ffn_prefill(
        residual,
        hc_fn,
        hc_base,
        hc_scale,
        norm_weight,
        router_weight,
        shared_gate_up_weight_chip,
        shared_down_weight_chip,
        expert_output_workspace,
        expert_slot_workspace,
        branch_partial,
        residual_path,
        selected_ids_out,
        route_weights_out,
        weight_scales,
        correction_bias=correction_bias,
        expert_offset=expert_offset,
        top_k=top_k,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    return branch_partial, residual_path, selected_ids_out, route_weights_out


@ax.builder(
    const_inputs=["weight_scales", "router_weight", "correction_bias"],
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_routed_dispatch_prefill(
    token_rows,
    selected_ids_out,
    route_weights_out,
    expert_input_workspace,
    expert_slot_workspace,
    weight_scales,
    router_weight,
    correction_bias,
    expert_offset=0,
    top_k=6,
    route_scale=1.5,
):
    """Run learned routing and pack tokens for this chip's experts."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    tokens, hidden = token_rows.shape
    experts_per_chip, _, workspace_hidden = expert_input_workspace.shape
    assert selected_ids_out.shape == route_weights_out.shape == (tokens, top_k)
    assert router_weight.shape[0] == 2 * experts_per_chip
    assert workspace_hidden == hidden
    assert expert_slot_workspace.shape == (experts_per_chip + 1, tokens)

    route_tokens_with_router(
        token_rows,
        router_weight,
        selected_ids_out,
        route_weights_out,
        weight_scales,
        core_id=core_id,
        core_num=core_num,
        correction_bias=correction_bias,
        route_scale=route_scale,
    )
    expert_token_map, core_expert_begin, experts_per_core = prepare_token_dispatch(
        selected_ids_out,
        expert_slot_workspace,
        core_id=core_id,
        core_num=core_num,
        expert_offset=expert_offset,
    )
    pack_tokens_to_local_experts(
        token_rows,
        expert_token_map,
        expert_input_workspace,
        core_expert_begin=core_expert_begin,
        experts_per_core=experts_per_core,
    )
    return selected_ids_out, route_weights_out, expert_input_workspace, expert_slot_workspace


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_routed_weighted_sum_prefill(
    expert_output_workspace,
    expert_slot_workspace,
    selected_ids,
    route_weights,
    shared_output,
    routed_output,
    expert_offset=0,
):
    """Accumulate this chip's routed rows and shared TP partial for every token."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    experts_with_sentinel, _, hidden = expert_output_workspace.shape
    tokens, top_k = selected_ids.shape
    experts_per_chip = experts_with_sentinel - 1
    global_experts = 2 * experts_per_chip
    assert core_num == 4
    assert expert_slot_workspace.shape == (experts_with_sentinel, tokens)
    assert route_weights.shape == (tokens, top_k)
    assert shared_output.shape == (tokens, hidden)
    assert routed_output.shape == (tokens, hidden)
    assert 0 <= expert_offset <= global_experts - experts_per_chip

    chip_expert_lookup = np.full((global_experts,), experts_per_chip, dtype=np.int32)
    chip_expert_lookup[expert_offset : expert_offset + experts_per_chip] = np.arange(
        experts_per_chip,
        dtype=np.int32,
    )
    chip_expert_lookup = constant(chip_expert_lookup, "routed_sum_chip_expert_lookup")
    core_token_slice = slice_tokens(core_id, core_num, tokens)
    packed_indices, selected_weights = prepare_routed_weighted_sum(
        expert_output_workspace,
        expert_slot_workspace,
        chip_expert_lookup,
        selected_ids,
        route_weights,
        core_token_slice,
        routed_output.dtype,
    )
    for token_start in range(core_token_slice.start, core_token_slice.stop, MOE_WEIGHTED_SUM_TILE):
        token_stop = min(token_start + MOE_WEIGHTED_SUM_TILE, core_token_slice.stop)
        localtoken_slice = slice(
            token_start - core_token_slice.start,
            token_stop - core_token_slice.start,
        )
        # local_output: [T_tile,D], routed_output.dtype.
        local_output = routed_weighted_sum_tile(
            expert_output_workspace,
            packed_indices,
            selected_weights,
            shared_output[token_start:token_stop, :],
            localtoken_slice,
            routed_output.dtype,
            token_offset=token_start,
        )
        # Keep writeback on the same hidden shards used by Gather and reduction.
        for hidden_part in range(MOE_WEIGHTED_SUM_HIDDEN_PARTS):
            hidden_slice = token_slice(hidden_part, MOE_WEIGHTED_SUM_HIDDEN_PARTS, hidden)
            ax.dsl.store(
                local_output[:, hidden_slice],
                r=routed_output[token_start:token_stop, hidden_slice],
            )
    return routed_output


@ax.builder(
    const_inputs=["weight_scales", "gate_up_weight", "down_weight"],
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    enable_cache=False,
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def ffn_shared_expert_prefill(
    hidden_states,
    output,
    weight_scales,
    gate_up_weight,
    down_weight,
    swiglu_limit=10.0,
):
    """Compute one chip's intermediate-axis shared-expert TP contribution."""
    tokens, hidden = hidden_states.shape
    intermediate = gate_up_weight.shape[0] // 2
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    assert core_num == CONV_EUS_PER_CORE
    assert tokens % core_num == 0
    assert output.shape == hidden_states.shape
    assert gate_up_weight.shape == (2 * intermediate, hidden)
    assert down_weight.shape == (hidden, intermediate)

    _runshared_expert(
        hidden_states,
        gate_up_weight,
        down_weight,
        output,
        weight_scales,
        core_id=core_id,
        core_num=core_num,
        limit=swiglu_limit,
    )
    return output
