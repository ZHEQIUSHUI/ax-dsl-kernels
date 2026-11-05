"""HASH and routed MoE graph helpers."""

from __future__ import annotations

import numpy as np

import axnn.dsl as ax
from axnn.dsl import parallel as axd

from ..attention.attn_utils.common import *  # noqa: F401,F403

from ..attention.attn_utils.numeric import dtype_bits


def clipped_swiglu(gate, up, limit, *, r=None):
    """Apply DeepSeek's clipped SwiGLU activation to one expert tile."""
    # upper/lower: [1], gate.dtype; both inputs share the activation dtype.
    upper = ax.dsl.full((1,), gate.dtype, fill_value=limit)
    lower = ax.dsl.full((1,), gate.dtype, fill_value=-limit)
    # gate/up: [M,Dm], gate.dtype after clipping.
    gate = ax.dsl.minimum(gate, upper)
    up = ax.dsl.minimum(up, upper)
    up = ax.dsl.maximum(up, lower)
    # activated_gate: [M,Dm], gate.dtype.
    activated_gate = ax.dsl.silu(gate, output_dtype=gate.dtype)
    # activated: [M,Dm], gate.dtype.
    return ax.dsl.mul(activated_gate, up, output_dtype=gate.dtype, r=r)


def expert_activation(x, gate_up_weight, gate_up_scale, *, limit):
    """Compute the clipped SwiGLU activation shared by both expert schedules."""
    rows, hidden = x.shape
    intermediate = gate_up_weight.shape[0] // 2
    # One full AX8860 weight path holds 1,024 output-channel bits.
    full_mac_oc = 1024 // dtype_bits(gate_up_weight.dtype)
    gate_up_parts = min(CONV_EUS_PER_CORE, max(1, 2 * intermediate // full_mac_oc))

    # gate_up: [M,2*Dm], x.dtype. Use only OC shards large enough to fill the
    # weight path; larger model dimensions still expose all four Conv EUs.
    gate_up = ax.dsl.create_tensor((rows, 2 * intermediate), x.dtype)
    for part_id in range(gate_up_parts):
        output_slice = token_slice(part_id, gate_up_parts, 2 * intermediate)
        # gate_up shard: [M,ceil(2*Dm/gate_up_parts)], x.dtype.
        constant_linear(
            x,
            gate_up_weight[output_slice, :],
            mx_scale_slice(
                gate_up_weight,
                gate_up_scale,
                output_slice,
                slice(0, hidden),
            ),
            output_dtype=x.dtype,
            r=gate_up[:, output_slice],
        )
    gate = gate_up[:, :intermediate]
    up = gate_up[:, intermediate:]
    # activated: [M,Dm], x.dtype.
    return clipped_swiglu(gate, up, limit)


def full_expert(x, gate_up_weight, gate_up_scale, down_weight, down_scale, *, limit):
    """Compute one complete clipped SwiGLU expert."""
    _, hidden = x.shape
    intermediate = gate_up_weight.shape[0] // 2
    activated = expert_activation(x, gate_up_weight, gate_up_scale, limit=limit)
    full_mac_oc = 1024 // dtype_bits(gate_up_weight.dtype)
    down_parts = min(CONV_EUS_PER_CORE, max(1, hidden // full_mac_oc))

    # expert: [M,D], x.dtype. Avoid underfilled OC shards for the down projection.
    expert = ax.dsl.create_tensor((x.shape[0], hidden), x.dtype)
    for part_id in range(down_parts):
        output_slice = token_slice(part_id, down_parts, hidden)
        # expert shard: [M,ceil(D/down_parts)], x.dtype.
        constant_linear(
            activated,
            down_weight[output_slice, :],
            mx_scale_slice(
                down_weight,
                down_scale,
                output_slice,
                slice(0, intermediate),
            ),
            output_dtype=x.dtype,
            r=expert[:, output_slice],
        )
    return expert


def preload_shared_gate_up_decode(
    gate_up_weight,
    gate_up_scale,
    *,
    core_id,
    core_num,
):
    """Stage the first decode Gate/Up waves while Router work is still in flight."""
    hidden = gate_up_weight.shape[1]
    intermediate = gate_up_weight.shape[0] // 2
    intermediate_per_core = intermediate // core_num
    local_intermediate_slice = slice(
        core_id * intermediate_per_core,
        (core_id + 1) * intermediate_per_core,
    )
    local_slices = (
        (local_intermediate_slice, slice(0, intermediate_per_core), "gate"),
        (
            slice(
                intermediate + local_intermediate_slice.start,
                intermediate + local_intermediate_slice.stop,
            ),
            slice(intermediate_per_core, 2 * intermediate_per_core),
            "up",
        ),
    )
    preloaded = []
    input_starts = list(range(0, hidden, SHARED_EXPERT_GATE_UP_IC_TILE))
    preload_waves = (
        SHARED_EXPERT_DECODE_PRELOAD_WAVES_MXFP4
        if np.dtype(gate_up_weight.dtype) == MXFP4_DTYPE
        else SHARED_EXPERT_DECODE_PRELOAD_WAVES_MXFP8
    )
    for input_start in input_starts[:preload_waves]:
        input_stop = min(input_start + SHARED_EXPERT_GATE_UP_IC_TILE, hidden)
        input_slice = slice(input_start, input_stop)
        input_width = input_stop - input_start
        weight_buffer = ax.dsl.create_tensor(
            (2 * intermediate_per_core, input_width),
            dsl_dtype(gate_up_weight.dtype),
            name=f"shared_gate_up_weight_ic{input_start}",
        )
        scale_buffer = None
        if gate_up_scale is not None:
            scale_buffer = ax.dsl.create_tensor(
                (2 * intermediate_per_core, input_width // MX_GROUP_SIZE),
                ax.mxe8m0,
                name=f"shared_gate_up_scale_ic{input_start}",
            )
        for source_slice, buffer_slice, name in local_slices:
            constant(
                gate_up_weight[source_slice, input_slice],
                f"shared_{name}_weight_ic{input_start}_core{core_id}",
                r=weight_buffer[buffer_slice, :input_width],
            )
            if scale_buffer is not None:
                input_scale_slice = slice(input_start // MX_GROUP_SIZE, input_stop // MX_GROUP_SIZE)
                scale_width = input_width // MX_GROUP_SIZE
                core_scale = np.ascontiguousarray(gate_up_scale[source_slice, input_scale_slice]).view(MXE8M0_DTYPE)
                constant(
                    core_scale,
                    f"shared_{name}_scale_ic{input_start}_core{core_id}",
                    r=scale_buffer[buffer_slice, :scale_width],
                )
        preloaded.append((weight_buffer, scale_buffer))
    return preloaded


def shared_expert(
    x,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    scale2,
    *,
    core_id,
    core_num,
    cross_core_tp=False,
    preloaded_gate_up=(),
    limit,
):
    """Compute one chip's intermediate-axis TP shard with four local channel pipelines."""
    rows, hidden = x.shape
    intermediate = gate_up_weight.shape[0] // 2
    assert core_num == CONV_EUS_PER_CORE
    assert intermediate % CONV_EUS_PER_CORE == 0
    intermediate_per_eu = intermediate // CONV_EUS_PER_CORE

    # Scaled test profiles can be narrower than one useful dynamic down-IC shard. Keep their
    # constant-weight path instead of padding a diagnostic shape into the production schedule.
    if hidden < 128 or intermediate_per_eu < 64:
        activated = expert_activation(x, gate_up_weight, gate_up_scale, limit=limit)
        expert = ax.dsl.create_tensor((rows, hidden), x.dtype)
        for part_id in range(CONV_EUS_PER_CORE):
            intermediate_slice = token_slice(part_id, CONV_EUS_PER_CORE, intermediate)
            if part_id == 0:
                constant_linear(
                    activated[:, intermediate_slice],
                    down_weight[:, intermediate_slice],
                    mx_scale_slice(down_weight, down_scale, slice(0, hidden), intermediate_slice),
                    output_dtype=x.dtype,
                    r=expert,
                )
                continue
            partial = constant_linear(
                activated[:, intermediate_slice],
                down_weight[:, intermediate_slice],
                mx_scale_slice(down_weight, down_scale, slice(0, hidden), intermediate_slice),
                output_dtype=x.dtype,
            )
            ax.dsl.add(expert, partial, output_dtype=x.dtype, r=expert)
        return expert

    local_intermediate_slice = slice(
        core_id * intermediate_per_eu,
        (core_id + 1) * intermediate_per_eu,
    )

    compute_part_count = 1 if cross_core_tp else CONV_EUS_PER_CORE
    compute_parts = ((core_id, 0),) if cross_core_tp else tuple((eu_id, eu_id) for eu_id in range(CONV_EUS_PER_CORE))
    gate_parts = ax.dsl.create_tensor(
        (compute_part_count, rows, intermediate_per_eu),
        x.dtype,
        name="shared_gate_parts",
    )
    up_parts = ax.dsl.create_tensor(
        (compute_part_count, rows, intermediate_per_eu),
        x.dtype,
        name="shared_up_parts",
    )
    gate_up_ic_tile = min(SHARED_EXPERT_GATE_UP_IC_TILE, hidden)
    local_gate_slice = local_intermediate_slice
    local_up_slice = slice(
        intermediate + local_intermediate_slice.start,
        intermediate + local_intermediate_slice.stop,
    )
    gate_up_input_starts = list(range(0, hidden, gate_up_ic_tile))
    gate_up_weight_buffers = [buffers[0] for buffers in preloaded_gate_up]
    gate_up_scale_buffers = [buffers[1] for buffers in preloaded_gate_up]
    gate_up_buffer_outputs = 2 * intermediate_per_eu if cross_core_tp else 2 * intermediate
    gate_up_buffer_memory = ax.MemoryType.OCM if cross_core_tp else ax.MemoryType.SHARED_OCM
    for input_start in gate_up_input_starts[len(preloaded_gate_up) :]:
        input_stop = min(input_start + gate_up_ic_tile, hidden)
        input_width = input_stop - input_start
        gate_up_weight_buffers.append(
            ax.dsl.create_tensor(
                (gate_up_buffer_outputs, input_width),
                dsl_dtype(gate_up_weight.dtype),
                gate_up_buffer_memory,
                name=f"shared_gate_up_weight_ic{input_start}",
            )
        )
        if gate_up_scale is not None:
            gate_up_scale_buffers.append(
                ax.dsl.create_tensor(
                    (gate_up_buffer_outputs, input_width // MX_GROUP_SIZE),
                    ax.mxe8m0,
                    gate_up_buffer_memory,
                    name=f"shared_gate_up_scale_ic{input_start}",
                )
            )

    # Barrier liveness retains the previous, current, and next logical waves together. A 448-wide
    # tile keeps all three MXFP8 weight-and-scale waves below 6 MiB SHOCM. Loading the next wave
    # before computing the current one exposes SDMA/CONV overlap.
    for pipeline_step in range(len(gate_up_input_starts) + 1):
        if len(preloaded_gate_up) <= pipeline_step < len(gate_up_input_starts):
            load_start = gate_up_input_starts[pipeline_step]
            load_stop = min(load_start + gate_up_ic_tile, hidden)
            load_slice = slice(load_start, load_stop)
            load_width = load_stop - load_start
            gate_up_weight_buffer = gate_up_weight_buffers[pipeline_step]
            gate_up_scale_buffer = None if gate_up_scale is None else gate_up_scale_buffers[pipeline_step]
            load_slices = ((local_gate_slice, local_gate_slice, "gate"), (local_up_slice, local_up_slice, "up"))
            if cross_core_tp:
                load_slices = (
                    (local_gate_slice, slice(0, intermediate_per_eu), "gate"),
                    (local_up_slice, slice(intermediate_per_eu, 2 * intermediate_per_eu), "up"),
                )
            for source_slice, buffer_slice, name in load_slices:
                constant(
                    gate_up_weight[source_slice, load_slice],
                    f"shared_{name}_weight_ic{load_start}_core{core_id}",
                    r=gate_up_weight_buffer[buffer_slice, :load_width],
                )
                if gate_up_scale_buffer is not None:
                    input_scale_slice = slice(load_start // MX_GROUP_SIZE, load_stop // MX_GROUP_SIZE)
                    scale_width = load_width // MX_GROUP_SIZE
                    core_scale = np.ascontiguousarray(gate_up_scale[source_slice, input_scale_slice]).view(MXE8M0_DTYPE)
                    constant(
                        core_scale,
                        f"shared_{name}_scale_ic{load_start}_core{core_id}",
                        r=gate_up_scale_buffer[buffer_slice, :scale_width],
                    )

        compute_step = pipeline_step - 1
        if compute_step >= 0:
            input_start = gate_up_input_starts[compute_step]
            input_stop = min(input_start + gate_up_ic_tile, hidden)
            input_slice = slice(input_start, input_stop)
            input_width = input_stop - input_start
            gate_up_weight_buffer = gate_up_weight_buffers[compute_step]
            gate_up_scale_buffer = None if gate_up_scale is None else gate_up_scale_buffers[compute_step]
            gate_partials = []
            up_partials = []
            for intermediate_part, output_part in compute_parts:
                intermediate_slice = slice(
                    intermediate_part * intermediate_per_eu,
                    (intermediate_part + 1) * intermediate_per_eu,
                )
                gate_slice = intermediate_slice
                up_slice = slice(intermediate + intermediate_slice.start, intermediate + intermediate_slice.stop)
                gate_weight_slice = gate_slice
                up_weight_slice = up_slice
                if cross_core_tp:
                    gate_weight_slice = slice(0, intermediate_per_eu)
                    up_weight_slice = slice(intermediate_per_eu, 2 * intermediate_per_eu)
                gate_output = gate_parts[output_part : output_part + 1].reshape(rows, intermediate_per_eu)
                up_output = up_parts[output_part : output_part + 1].reshape(rows, intermediate_per_eu)
                first_input_tile = compute_step == 0
                for output_slice, output, partials in (
                    (gate_weight_slice, gate_output, gate_partials),
                    (up_weight_slice, up_output, up_partials),
                ):
                    partial = output if first_input_tile else None
                    partial = ax.dsl.matmul(
                        x[:, input_slice],
                        gate_up_weight_buffer[output_slice, :input_width],
                        w_scale=(
                            1.0
                            if gate_up_scale_buffer is None
                            else gate_up_scale_buffer[output_slice, : input_width // MX_GROUP_SIZE]
                        ),
                        w_scale2=1.0 if scale2 is None else scale2,
                        transposed=True,
                        output_dtype=x.dtype,
                        r=partial,
                    )
                    if not first_input_tile:
                        partials.append((partial, output))
            for partial, output in gate_partials + up_partials:
                ax.dsl.move(partial, r=output, op="add")
        if not cross_core_tp:
            axd.sync_all()

    activated_parts = ax.dsl.create_tensor(
        gate_parts.shape,
        x.dtype,
        name="shared_activated_parts",
    )
    for output_part in range(compute_part_count):
        gate_part = gate_parts[output_part : output_part + 1].reshape(rows, intermediate_per_eu)
        up_part = up_parts[output_part : output_part + 1].reshape(rows, intermediate_per_eu)
        activated_part = activated_parts[output_part : output_part + 1].reshape(rows, intermediate_per_eu)
        clipped_swiglu(
            gate_part,
            up_part,
            limit,
            r=activated_part,
        )

    expert = ax.dsl.create_tensor((rows, hidden), x.dtype, name="shared_expert_sum")
    down_oc_tile = min(SHARED_EXPERT_DOWN_OC_TILE, hidden)
    down_output_starts = list(range(0, hidden, down_oc_tile))
    down_weight_buffers = []
    down_scale_buffers = []
    down_buffer_inputs = intermediate_per_eu if cross_core_tp else intermediate
    down_buffer_memory = ax.MemoryType.OCM if cross_core_tp else ax.MemoryType.SHARED_OCM
    for output_start in down_output_starts:
        output_stop = min(output_start + down_oc_tile, hidden)
        output_width = output_stop - output_start
        down_weight_buffers.append(
            ax.dsl.create_tensor(
                (output_width, down_buffer_inputs),
                dsl_dtype(down_weight.dtype),
                down_buffer_memory,
                name=f"shared_down_weight_oc{output_start}",
            )
        )
        if down_scale is not None:
            down_scale_buffers.append(
                ax.dsl.create_tensor(
                    (output_width, down_buffer_inputs // MX_GROUP_SIZE),
                    ax.mxe8m0,
                    down_buffer_memory,
                    name=f"shared_down_scale_oc{output_start}",
                )
            )

    # Two stages share one barrier: load the next logical weight wave while CONV consumes the
    # current one. Distinct tensor identities keep cross-core producers stable. The 960-wide tile
    # also leaves room when barrier liveness temporarily retains three logical waves in SHOCM.
    for pipeline_step in range(len(down_output_starts) + 1):
        if pipeline_step < len(down_output_starts):
            load_start = down_output_starts[pipeline_step]
            load_stop = min(load_start + down_oc_tile, hidden)
            load_slice = slice(load_start, load_stop)
            load_width = load_stop - load_start
            down_weight_buffer = down_weight_buffers[pipeline_step]
            down_buffer_input_slice = local_intermediate_slice
            if cross_core_tp:
                down_buffer_input_slice = slice(0, intermediate_per_eu)
            constant(
                down_weight[load_slice, local_intermediate_slice],
                f"shared_down_weight_oc{load_start}_core{core_id}",
                r=down_weight_buffer[:load_width, down_buffer_input_slice],
            )
            if down_scale is not None:
                local_scale_slice = slice(
                    local_intermediate_slice.start // MX_GROUP_SIZE,
                    local_intermediate_slice.stop // MX_GROUP_SIZE,
                )
                buffer_scale_slice = local_scale_slice
                if cross_core_tp:
                    buffer_scale_slice = slice(0, intermediate_per_eu // MX_GROUP_SIZE)
                core_scale = np.ascontiguousarray(down_scale[load_slice, local_scale_slice]).view(MXE8M0_DTYPE)
                constant(
                    core_scale,
                    f"shared_down_scale_oc{load_start}_core{core_id}",
                    r=down_scale_buffers[pipeline_step][
                        :load_width,
                        buffer_scale_slice,
                    ],
                )

        compute_step = pipeline_step - 1
        if compute_step >= 0:
            output_start = down_output_starts[compute_step]
            output_stop = min(output_start + down_oc_tile, hidden)
            output_slice = slice(output_start, output_stop)
            output_width = output_stop - output_start
            down_weight_buffer = down_weight_buffers[compute_step]
            down_scale_buffer = None if down_scale is None else down_scale_buffers[compute_step]
            partials = []
            expert_output = expert[:, output_slice]
            for intermediate_part, output_part in compute_parts:
                intermediate_slice = slice(
                    intermediate_part * intermediate_per_eu,
                    (intermediate_part + 1) * intermediate_per_eu,
                )
                scale_slice = slice(
                    intermediate_slice.start // MX_GROUP_SIZE,
                    intermediate_slice.stop // MX_GROUP_SIZE,
                )
                weight_slice = intermediate_slice
                if cross_core_tp:
                    weight_slice = slice(0, intermediate_per_eu)
                    scale_slice = slice(0, intermediate_per_eu // MX_GROUP_SIZE)
                partial = expert_output if output_part == 0 else None
                activated_part = activated_parts[output_part : output_part + 1].reshape(rows, intermediate_per_eu)
                partial = ax.dsl.matmul(
                    activated_part,
                    down_weight_buffer[:output_width, weight_slice],
                    w_scale=(1.0 if down_scale_buffer is None else down_scale_buffer[:output_width, scale_slice]),
                    w_scale2=1.0 if scale2 is None else scale2,
                    transposed=True,
                    output_dtype=x.dtype,
                    r=partial,
                )
                if output_part != 0:
                    partials.append(partial)
            for partial in partials:
                ax.dsl.move(partial, r=expert_output, op="add")
        if not cross_core_tp:
            axd.sync_all()
    if cross_core_tp:
        expert_partials = ax.dsl.create_tensor(
            (core_num, rows, hidden),
            x.dtype,
            ax.MemoryType.SHARED_OCM,
            name="shared_decode_partials",
        )
        ax.dsl.move(
            expert,
            r=expert_partials[core_id : core_id + 1].reshape(rows, hidden),
        )
        axd.sync_all()
        if core_id == 0:
            reduced = ax.dsl.create_tensor(
                (rows, hidden),
                x.dtype,
                name="shared_decode_sum",
            )
            ax.dsl.move(expert_partials[0:1].reshape(rows, hidden), r=reduced)
            for source_core in range(1, core_num):
                ax.dsl.move(
                    expert_partials[source_core : source_core + 1].reshape(rows, hidden),
                    r=reduced,
                    op="add",
                )
            return reduced
    return expert


def run_shared_expert(
    input_rows,
    gate_up_weight,
    down_weight,
    output_rows,
    weight_scales,
    *,
    core_id,
    core_num,
    preloaded_gate_up=(),
    limit,
):
    """Run one chip's shared-expert TP partial for one disjoint token shard per core."""
    tokens, hidden = input_rows.shape
    coretoken_slice = token_slice(core_id, core_num, tokens)
    gate_up_scale = weight_scales.get("shared_gate_up")
    down_scale = weight_scales.get("shared_down")
    scale2 = None
    if gate_up_scale is not None or down_scale is not None:
        scale2 = constant(
            np.ones((1,), dtype=np.float32),
            "shared_scale2",
        )

    # With fewer tokens than cores, split one shared-expert row across all cores. Every core must
    # enter the same weight-pipeline barriers even though CORE0 publishes the reduced rows.
    if tokens < core_num:
        if input_rows.memory_type == ax.MemoryType.DDR:
            shared_tokens = ax.dsl.load(input_rows)
        else:
            shared_tokens = ax.dsl.create_tensor(
                input_rows.shape,
                input_rows.dtype,
                ax.MemoryType.OCM,
                name="shared_decode_input",
            )
            ax.dsl.move(input_rows, r=shared_tokens)
        shared_rows = shared_expert(
            shared_tokens,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            scale2,
            core_id=core_id,
            core_num=core_num,
            cross_core_tp=True,
            preloaded_gate_up=preloaded_gate_up,
            limit=limit,
        )
        if core_id == 0:
            if output_rows.memory_type == ax.MemoryType.DDR:
                ax.dsl.store(shared_rows, r=output_rows)
            else:
                ax.dsl.move(shared_rows, r=output_rows)
        if tokens > 1:
            # CORE0 writes every row in this underfilled TP path. Publish those rows before the
            # other token owners enter the final routed/shared accumulation.
            axd.sync_all()
        return

    for token_start in range(coretoken_slice.start, coretoken_slice.stop, MOE_EXPERT_CAPACITY):
        token_stop = min(token_start + MOE_EXPERT_CAPACITY, coretoken_slice.stop)
        # shared_tokens: [M<=128,D], input_rows.dtype.
        shared_tokens = ax.dsl.load(input_rows[token_start:token_stop, :])
        # shared_rows: [M<=128,D], input_rows.dtype.
        shared_rows = shared_expert(
            shared_tokens,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            scale2,
            core_id=core_id,
            core_num=core_num,
            limit=limit,
        )
        # output_rows slice: [M<=128,D], input_rows.dtype chip-local TP partial.
        ax.dsl.store(shared_rows, r=output_rows[token_start:token_stop, :])


def stage_router_weight(router_weight, router_scale, *, core_id, core_num):
    """Publish one core's Router IC shard and return the shared weight tensors."""
    _, hidden = router_weight.shape
    router_dtype = np.dtype(router_weight.dtype)
    router_ic_per_core = hidden // core_num
    router_core_ic_slice = slice(core_id * router_ic_per_core, (core_id + 1) * router_ic_per_core)
    router_weight_shared = ax.dsl.create_tensor(
        router_weight.shape,
        dsl_dtype(router_dtype),
        ax.MemoryType.SHARED_OCM,
        name="router_weight_shared",
    )
    constant(
        router_weight[:, router_core_ic_slice],
        f"router_weight_core{core_id}",
        r=router_weight_shared[:, router_core_ic_slice],
    )

    router_scale_shared = None
    router_scale2 = None
    if router_dtype != BF16_DTYPE:
        router_scale_per_core = router_ic_per_core // MX_GROUP_SIZE
        router_core_scale_slice = slice(
            core_id * router_scale_per_core,
            (core_id + 1) * router_scale_per_core,
        )
        router_scale_shared = ax.dsl.create_tensor(
            router_scale.shape,
            ax.mxe8m0,
            ax.MemoryType.SHARED_OCM,
            name="router_scale_shared",
        )
        core_router_scale = np.ascontiguousarray(router_scale[:, router_core_scale_slice]).view(MXE8M0_DTYPE)
        constant(
            core_router_scale,
            f"router_scale_core{core_id}",
            r=router_scale_shared[:, router_core_scale_slice],
        )
        router_scale2 = constant(
            np.ones((1,), dtype=np.float32),
            "router_scale2",
        )
    return router_weight_shared, router_scale_shared, router_scale2


def route_single_token_decode_logits(
    token_rows,
    router_weight,
    router_scale,
    *,
    core_id,
    core_num,
):
    """Publish one decode token's core-local Router logits to shared OCM."""
    _, hidden = token_rows.shape
    global_experts = router_weight.shape[0]
    assert global_experts % core_num == 0
    experts_per_core = global_experts // core_num
    expert_slice = slice(
        core_id * experts_per_core,
        (core_id + 1) * experts_per_core,
    )
    local_weight = constant(
        router_weight[expert_slice, :],
        f"router_decode_weight_core{core_id}",
    )
    local_scale = None
    scale2 = None
    if router_scale is not None:
        local_scale = constant(
            np.ascontiguousarray(router_scale[expert_slice, :]).view(MXE8M0_DTYPE),
            f"router_decode_scale_core{core_id}",
        )
        scale2 = constant(
            np.ones((1,), dtype=np.float32),
            "router_scale2",
        )

    if token_rows.memory_type == ax.MemoryType.DDR:
        router_rows = ax.dsl.load(token_rows)
    else:
        router_rows = ax.dsl.create_tensor(
            token_rows.shape,
            token_rows.dtype,
            ax.MemoryType.OCM,
            name="router_decode_input",
        )
        ax.dsl.move(token_rows, r=router_rows)
    local_logits = ax.dsl.create_tensor(
        (1, experts_per_core),
        ax.fp32,
        name="router_decode_local_logits",
    )
    partials = []
    ic_per_eu = hidden // CONV_EUS_PER_CORE
    for eu_id in range(CONV_EUS_PER_CORE):
        ic_slice = slice(eu_id * ic_per_eu, (eu_id + 1) * ic_per_eu)
        scale_slice = slice(ic_slice.start // MX_GROUP_SIZE, ic_slice.stop // MX_GROUP_SIZE)
        partial = local_logits if eu_id == 0 else None
        partial = ax.dsl.matmul(
            router_rows[:, ic_slice],
            local_weight[:, ic_slice],
            w_scale=1.0 if local_scale is None else local_scale[:, scale_slice],
            w_scale2=1.0 if scale2 is None else scale2,
            transposed=True,
            output_dtype=ax.fp32,
            r=partial,
        )
        if eu_id != 0:
            partials.append(partial)
    for partial in partials:
        ax.dsl.move(partial, r=local_logits, op="add")

    router_logits = ax.dsl.create_tensor(
        (1, global_experts),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name="router_decode_logits",
    )
    ax.dsl.move(local_logits, r=router_logits[:, expert_slice])
    return router_logits


def finish_single_token_decode(
    router_logits,
    correction_bias,
    selected_ids_out,
    route_weights_out,
    *,
    token_ids=None,
    tid2eid=None,
    route_scale,
):
    """Select experts and normalize route weights after all logits are published."""
    scores = ax.dsl.sqrt(
        ax.dsl.softplus(router_logits, output_dtype=ax.fp32),
        output_dtype=ax.fp32,
    )
    if token_ids is None:
        correction = constant(correction_bias, "router_correction_bias")
        _, selected = ax.dsl.topk(
            ax.dsl.add(scores, correction, output_dtype=ax.fp32),
            k=selected_ids_out.shape[1],
            dim=-1,
        )
    else:
        selected = ax.dsl.index_select(
            tid2eid,
            ax.dsl.load(token_ids.reshape(1)),
            dim=0,
        )
    chosen = ax.dsl.gather(scores, selected, dim=-1)
    chosen_sum = ax.dsl.sum(chosen, dim=-1, keepdim=True, output_dtype=ax.fp32)
    normalized_weights = ax.dsl.div(
        chosen,
        ax.dsl.add(
            chosen_sum,
            ax.dsl.full((1,), ax.fp32, fill_value=1e-20),
            output_dtype=ax.fp32,
        ),
        output_dtype=ax.fp32,
    )
    weights = ax.dsl.mul(
        normalized_weights,
        ax.dsl.full((1,), ax.fp32, fill_value=route_scale),
        output_dtype=ax.fp32,
    )
    ax.dsl.store(selected, r=selected_ids_out)
    ax.dsl.store(weights, r=route_weights_out)


def route_tokens_with_router(
    token_rows,
    router_weight,
    selected_ids_out,
    route_weights_out,
    weight_scales,
    *,
    core_id,
    core_num,
    correction_bias=None,
    token_ids=None,
    tid2eid=None,
    route_scale=1.5,
):
    """Compute Router selection and weights for disjoint token shards."""
    tokens, hidden = token_rows.shape
    global_experts, weight_hidden = router_weight.shape
    top_k = selected_ids_out.shape[1]
    hash_mode = token_ids is not None
    assert core_num == CONV_EUS_PER_CORE
    assert weight_hidden == hidden
    assert route_weights_out.shape == selected_ids_out.shape
    assert hidden % core_num == 0
    assert hidden % CONV_EUS_PER_CORE == 0
    validate_weight_scales({"router": router_weight}, weight_scales)

    router_scale = weight_scales.get("router")
    router_weight_shared, router_scale_shared, router_scale2 = stage_router_weight(
        router_weight,
        router_scale,
        core_id=core_id,
        core_num=core_num,
    )
    axd.sync_all()

    correction = None if hash_mode else constant(correction_bias, "router_correction_bias")
    flat_token_ids = None if token_ids is None else token_ids.reshape(tokens)
    coretoken_slice = token_slice(core_id, core_num, tokens)
    router_ic_per_eu = hidden // CONV_EUS_PER_CORE
    for token_start in range(coretoken_slice.start, coretoken_slice.stop, MOE_ROUTER_TOKEN_TILE):
        token_stop = min(token_start + MOE_ROUTER_TOKEN_TILE, coretoken_slice.stop)
        # Each token owner computes all global logits. Four independent IC partials occupy the
        # four CONV EUs, then OCM NMC additions form the FP32 Router accumulator.
        router_rows = ax.dsl.load(token_rows[token_start:token_stop, :])
        router_logits = ax.dsl.create_tensor(
            (token_stop - token_start, global_experts),
            ax.fp32,
            ax.MemoryType.OCM,
            name=f"router_logits_{token_start}",
        )
        router_partials = []
        for eu_id in range(CONV_EUS_PER_CORE):
            ic_slice = slice(eu_id * router_ic_per_eu, (eu_id + 1) * router_ic_per_eu)
            scale_slice = slice(ic_slice.start // MX_GROUP_SIZE, ic_slice.stop // MX_GROUP_SIZE)
            partial = router_logits if eu_id == 0 else None
            partial = ax.dsl.matmul(
                router_rows[:, ic_slice],
                router_weight_shared[:, ic_slice],
                w_scale=1.0 if router_scale_shared is None else router_scale_shared[:, scale_slice],
                w_scale2=1.0 if router_scale2 is None else router_scale2,
                transposed=True,
                output_dtype=ax.fp32,
                r=partial,
            )
            if eu_id != 0:
                router_partials.append(partial)
        for partial in router_partials:
            ax.dsl.move(partial, r=router_logits, op="add")

        if hash_mode:
            selected = ax.dsl.index_select(
                tid2eid,
                ax.dsl.load(flat_token_ids[token_start:token_stop]),
                dim=0,
            )
            chosen_logits = ax.dsl.gather(router_logits, selected, dim=-1)
            chosen = ax.dsl.sqrt(
                ax.dsl.softplus(chosen_logits, output_dtype=ax.fp32),
                output_dtype=ax.fp32,
            )
        else:
            scores = ax.dsl.sqrt(
                ax.dsl.softplus(router_logits, output_dtype=ax.fp32),
                output_dtype=ax.fp32,
            )
            _, selected = ax.dsl.topk(
                ax.dsl.add(scores, correction, output_dtype=ax.fp32),
                k=top_k,
                dim=-1,
            )
            chosen = ax.dsl.gather(scores, selected, dim=-1)

        chosen_sum = ax.dsl.sum(chosen, dim=-1, keepdim=True, output_dtype=ax.fp32)
        normalized_weights = ax.dsl.div(
            chosen,
            ax.dsl.add(
                chosen_sum,
                ax.dsl.full((1,), ax.fp32, fill_value=1e-20),
                output_dtype=ax.fp32,
            ),
            output_dtype=ax.fp32,
        )
        route_weights = ax.dsl.mul(
            normalized_weights,
            ax.dsl.full((1,), ax.fp32, fill_value=route_scale),
            output_dtype=ax.fp32,
        )
        ax.dsl.store(selected, r=selected_ids_out[token_start:token_stop, :])
        ax.dsl.store(route_weights, r=route_weights_out[token_start:token_stop, :])

    # Dispatch reads every token's selected IDs on every expert-owning core.
    axd.sync_all()


def prepare_token_dispatch(
    selected_ids,
    expert_slot_workspace,
    *,
    core_id,
    core_num,
    expert_offset,
):
    """Build this core's local expert map and publish token-to-expert slots."""
    tokens, top_k = selected_ids.shape
    experts_per_chip = expert_slot_workspace.shape[0] - 1
    global_experts = 2 * experts_per_chip
    experts_per_core = experts_per_chip // core_num
    core_expert_begin = core_id * experts_per_core

    core_expert_lookup = np.full((global_experts,), experts_per_core, dtype=np.int32)
    global_expert_begin = expert_offset + core_expert_begin
    core_expert_lookup[global_expert_begin : global_expert_begin + experts_per_core] = np.arange(
        experts_per_core,
        dtype=np.int32,
    )
    # selected: [T,K], INT32 private OCM.
    selected = ax.dsl.load(selected_ids)
    # core_local_ids: [T,K], INT32 in [0,E_core].
    core_local_ids = ax.dsl.index_select(
        constant(core_expert_lookup, f"core{core_id}_expert_lookup"),
        selected,
        dim=0,
    )
    # core_local_ids_by_route: [K,T], INT32.
    core_local_ids_by_route = ax.dsl.transpose(core_local_ids, perm=(1, 0))
    # expert_token_map: [E_core+1,T], UINT8 private OCM.
    expert_token_map = ax.dsl.create_tensor(
        (experts_per_core + 1, tokens),
        ax.uint8,
        name=f"core{core_id}_expert_token_map",
    )
    ax.dsl.move(ax.dsl.zeros(expert_token_map.shape, ax.uint8), r=expert_token_map)
    # token_map_updates: [K,T], UINT8 ones.
    token_map_updates = ax.dsl.full((top_k, tokens), ax.uint8, fill_value=1)
    ax.dsl.scatter(token_map_updates, core_local_ids_by_route, dim=0, r=expert_token_map)
    # slot_counts: [E_core,T], INT32 inclusive token rank within each expert.
    slot_counts = ax.dsl.cumsum(
        expert_token_map[:experts_per_core, :],
        dim=1,
        output_dtype=ax.int32,
    )
    # expert_slots: [E_core,T], INT32. Only active map positions are consumed downstream.
    expert_slots = ax.dsl.sub(
        slot_counts,
        ax.dsl.full((1,), ax.int32, fill_value=1),
        output_dtype=ax.int32,
    )
    ax.dsl.store(
        expert_slots,
        r=expert_slot_workspace[core_expert_begin : core_expert_begin + experts_per_core, :],
    )
    if core_id == 0:
        ax.dsl.store(
            ax.dsl.zeros((1, tokens), ax.int32),
            r=expert_slot_workspace[experts_per_chip : experts_per_chip + 1, :],
        )
    return expert_token_map, core_expert_begin, experts_per_core


def pack_tokens_to_local_experts(
    token_rows,
    expert_token_map,
    expert_input_workspace,
    *,
    core_expert_begin,
    experts_per_core,
):
    """Pack one core's expert inputs with a four-expert Gather/Store OCM window."""
    tokens, hidden = token_rows.shape
    _, expert_capacity, workspace_hidden = expert_input_workspace.shape
    assert workspace_hidden == hidden
    capacity_positions = constant(
        np.arange(expert_capacity, dtype=np.int32),
        "dispatch_capacity_positions",
    )
    capacity_positions_u32 = ax.dsl.cast(capacity_positions, dtype=ax.uint32)
    one = constant(np.array([1], dtype=np.int32), "dispatch_one")
    zero = ax.dsl.zeros((1,), ax.int32)
    for expert_group_begin in range(0, experts_per_core, MOE_DISPATCH_EXPERT_GROUP):
        expert_outputs = []
        group_end = min(expert_group_begin + MOE_DISPATCH_EXPERT_GROUP, experts_per_core)
        for core_local_id in range(expert_group_begin, group_end):
            # token_indices/count: [T]/[1], UINT32.
            token_indices, token_count = ax.dsl.nonzero(
                expert_token_map[core_local_id : core_local_id + 1, :].reshape(tokens)
            )
            if tokens < expert_capacity:
                token_indices = ax.dsl.pad(
                    token_indices,
                    padding=(0, expert_capacity - tokens),
                    value=0,
                )
            token_indices = token_indices[:expert_capacity]
            # active_u32: [capacity], UINT32 in {0,1}.
            remaining = ax.dsl.sub(
                ax.dsl.cast(token_count, dtype=ax.int32),
                capacity_positions,
                output_dtype=ax.int32,
            )
            active = ax.dsl.minimum(remaining, one)
            active = ax.dsl.maximum(active, zero)
            active_u32 = ax.dsl.cast(active, dtype=ax.uint32)
            raw_token_indices = ax.dsl.index_select(token_indices, capacity_positions_u32, dim=0)
            input_indices = ax.dsl.mul(raw_token_indices, active_u32, output_dtype=ax.uint32)
            # Keep four expert outputs live so gather and store can use separate OCM buffers and
            # both SDMA engines stay fed across the dispatch window.
            expert_tokens = ax.dsl.index_select(token_rows, input_indices, dim=0)
            expert_outputs.append((core_local_id, expert_tokens))

        for core_local_id, expert_tokens in expert_outputs:
            expert_id = core_expert_begin + core_local_id
            ax.dsl.store(
                expert_tokens,
                r=expert_input_workspace[expert_id : expert_id + 1, :, :].reshape(
                    expert_capacity,
                    hidden,
                ),
            )


def routed_expert_branches(
    gate_up_weight_local,
    down_weight_local,
    gate_up_scales,
    down_scales,
    *,
    core_expert_begin,
    experts_per_core,
    tokens,
    expert_capacity,
    swiglu_limit,
):
    """Build the static expert bodies selected by one core's runtime expert range."""

    def make_branch(core_local_id):
        chip_local_id = core_expert_begin + core_local_id

        def expert_branch(
            normalized,
            expert_token_map,
            capacity_positions,
            capacity_positions_u32,
            one,
            zero,
            expert_workspace,
        ):
            # token_indices/count: [T]/[1], UINT32.
            token_indices, token_count = ax.dsl.nonzero(
                expert_token_map[core_local_id : core_local_id + 1, :].reshape(tokens)
            )
            if tokens < expert_capacity:
                # token_indices: [128], UINT32; padded values are masked below.
                token_indices = ax.dsl.pad(token_indices, padding=(0, expert_capacity - tokens), value=0)
            token_indices = token_indices[:expert_capacity]
            # token_count_i32: [1], INT32.
            token_count_i32 = ax.dsl.cast(token_count, dtype=ax.int32)
            # remaining: [128], INT32.
            remaining = ax.dsl.sub(token_count_i32, capacity_positions, output_dtype=ax.int32)
            # active_i32: [128], INT32 in {0,1}.
            active_i32 = ax.dsl.minimum(remaining, one)
            # active_i32: [128], INT32 in {0,1}.
            active_i32 = ax.dsl.maximum(active_i32, zero)
            # active_u32: [128], UINT32.
            active_u32 = ax.dsl.cast(active_i32, dtype=ax.uint32)
            # raw_token_indices: [128], UINT32.
            raw_token_indices = ax.dsl.index_select(token_indices, capacity_positions_u32, dim=0)
            # input_indices: [128], UINT32. Inactive rows read token zero but are weighted by zero.
            input_indices = ax.dsl.mul(raw_token_indices, active_u32, output_dtype=ax.uint32)
            # expert_tokens: [128,D], activation_dtype.
            expert_tokens = ax.dsl.index_select(normalized, input_indices, dim=0)
            # expert_workspace row: [128,D], activation_dtype DDR. The branch owns
            # a compile-time expert row, so the MLP can store without dynamic DDR IO.
            expert_output = full_expert(
                expert_tokens,
                gate_up_weight_local[chip_local_id],
                gate_up_scales[chip_local_id] if gate_up_scales is not None else None,
                down_weight_local[chip_local_id],
                down_scales[chip_local_id] if down_scales is not None else None,
                limit=swiglu_limit,
            )
            # expert_workspace row: [128,D], activation_dtype DDR.
            ax.dsl.store(
                expert_output,
                r=expert_workspace[core_local_id : core_local_id + 1, :, :].reshape(
                    expert_capacity, normalized.shape[-1]
                ),
            )
            return (
                normalized,
                expert_token_map,
                capacity_positions,
                capacity_positions_u32,
                one,
                zero,
                expert_workspace,
            )

        return expert_branch

    return {local_id: make_branch(local_id) for local_id in range(experts_per_core)}


def run_routed_experts(
    normalized,
    expert_token_map,
    gate_up_weight_local,
    down_weight_local,
    expert_output_workspace,
    weight_scales,
    *,
    core_expert_begin,
    experts_per_core,
    swiglu_limit,
):
    """Record this core's runtime routed-expert range."""
    tokens, hidden = normalized.shape
    capacity_positions = constant(
        np.arange(MOE_EXPERT_CAPACITY, dtype=np.int32),
        "expert_capacity_positions",
    )
    capacity_positions_u32 = ax.dsl.cast(capacity_positions, dtype=ax.uint32)
    one = constant(np.array([1], dtype=np.int32), "expert_one")
    zero = ax.dsl.zeros((1,), ax.int32)
    core_expert_workspace = expert_output_workspace[
        core_expert_begin : core_expert_begin + experts_per_core,
        :,
        :,
    ]
    branches = routed_expert_branches(
        gate_up_weight_local,
        down_weight_local,
        weight_scales.get("gate_up"),
        weight_scales.get("down"),
        core_expert_begin=core_expert_begin,
        experts_per_core=experts_per_core,
        tokens=tokens,
        expert_capacity=MOE_EXPERT_CAPACITY,
        swiglu_limit=swiglu_limit,
    )

    if experts_per_core == 1:
        branches[0](
            normalized,
            expert_token_map,
            capacity_positions,
            capacity_positions_u32,
            one,
            zero,
            core_expert_workspace,
        )
    else:
        range_index = ax.dsl.zeros((1,), ax.int32)
        range_start = ax.dsl.zeros((1,), ax.int32)
        range_stop = constant(np.array([experts_per_core], dtype=np.int32), "expert_range_stop")
        range_step = constant(np.array([1], dtype=np.int32), "expert_range_step")

        def expert_loop(
            expert_index,
            loop_normalized,
            loop_token_map,
            loop_positions,
            loop_positions_u32,
            loop_one,
            loop_zero,
            loop_expert_workspace,
        ):
            return ax.dsl.switch(
                expert_index,
                branches,
                loop_normalized,
                loop_token_map,
                loop_positions,
                loop_positions_u32,
                loop_one,
                loop_zero,
                loop_expert_workspace,
            )

        ax.dsl.range(
            range_index,
            range_start,
            range_stop,
            range_step,
            expert_loop,
            normalized,
            expert_token_map,
            capacity_positions,
            capacity_positions_u32,
            one,
            zero,
            core_expert_workspace,
            unroll_factor=1,
        )
    # Join the dynamic range on all cores before its expert rows leave the control-flow graph.
    axd.sync_all()


def prepare_routed_weighted_sum(
    expert_output_workspace,
    expert_slot_workspace,
    chip_expert_lookup,
    selected_ids,
    route_weights,
    token_slice,
    output_dtype,
):
    """Resolve one core's exact packed rows and weights before its tile loop."""
    tokens, top_k = selected_ids.shape
    experts_with_sentinel, expert_capacity, _ = expert_output_workspace.shape
    token_count = token_slice.stop - token_slice.start
    flat_expert_slots = expert_slot_workspace.reshape(experts_with_sentinel * tokens)

    # selected_rows/weight_rows: [T_core,K], INT32/FP32.
    selected_rows = ax.dsl.load(selected_ids[token_slice, :])
    weight_rows = ax.dsl.load(route_weights[token_slice, :])
    # token_positions: [T_core,1], INT32.
    token_positions = constant(
        np.arange(token_slice.start, token_slice.stop, dtype=np.int32),
        f"routed_sum_token_positions_{token_slice.start}",
    ).reshape(token_count, 1)
    # chip_local_ids: [T_core,K], INT32. Remote experts select the zero sentinel row.
    chip_local_ids = ax.dsl.index_select(chip_expert_lookup, selected_rows, dim=0)
    # slot_lookup_indices: [T_core,K], INT32.
    slot_lookup_indices = ax.dsl.add(
        ax.dsl.mul(
            chip_local_ids,
            ax.dsl.full((1,), ax.int32, fill_value=tokens),
            output_dtype=ax.int32,
        ),
        token_positions,
        output_dtype=ax.int32,
    )
    # Only the exact T_core*K entries are read. Preloading the dense slot table would transfer
    # 21.5x more data for the production topology. Keep each atomic DDR Gather bounded, then retain
    # the complete 24 KiB per-core result across the data/compute tile loop.
    expert_slots = ax.dsl.create_tensor(
        (token_count, top_k),
        ax.int32,
        ax.MemoryType.OCM,
        name=f"routed_sum_expert_slots_{token_slice.start}",
    )
    for slot_start in range(0, token_count, MOE_SLOT_LOOKUP_TILE):
        slot_stop = min(slot_start + MOE_SLOT_LOOKUP_TILE, token_count)
        slot_slice = slice(slot_start, slot_stop)
        ax.dsl.index_select(
            flat_expert_slots,
            slot_lookup_indices[slot_slice, :],
            dim=0,
            r=expert_slots[slot_slice, :],
        )
    # packed_indices/selected_weights: [T_core,K], UINT32/output_dtype private OCM.
    packed_indices = ax.dsl.add(
        ax.dsl.mul(
            chip_local_ids,
            ax.dsl.full((1,), ax.int32, fill_value=expert_capacity),
            output_dtype=ax.int32,
        ),
        expert_slots,
        output_dtype=ax.int32,
    )
    return (
        ax.dsl.cast(packed_indices, dtype=ax.uint32),
        ax.dsl.cast(weight_rows, dtype=output_dtype),
    )


def routed_weighted_sum_tile(
    expert_output_workspace,
    packed_indices,
    selected_weights,
    shared_rows,
    token_slice,
    output_dtype,
    *,
    token_offset,
):
    """Reduce three token-and-hidden-sharded two-route waves for one token tile."""
    experts_with_sentinel, expert_capacity, hidden = expert_output_workspace.shape
    _, top_k = packed_indices.shape
    token_count = token_slice.stop - token_slice.start
    assert shared_rows.shape == (token_count, hidden)
    hidden_parts = MOE_WEIGHTED_SUM_HIDDEN_PARTS
    assert hidden % hidden_parts == 0
    hidden_per_part = hidden // hidden_parts
    packed_expert_output = expert_output_workspace.reshape(
        experts_with_sentinel * expert_capacity * hidden_parts,
        hidden_per_part,
    )
    routes_per_wave = min(2, top_k)
    wave_count = (top_k + routes_per_wave - 1) // routes_per_wave
    expert_row_buffers = [
        [
            ax.dsl.create_tensor(
                (token_count, routes_per_wave, hidden_per_part),
                output_dtype,
                ax.MemoryType.OCM,
                name=f"routed_sum_expert_rows_{token_offset}_buffer{buffer_id}_h{hidden_part}",
            )
            for hidden_part in range(hidden_parts)
        ]
        for buffer_id in range(wave_count)
    ]
    hidden_parts_value = constant(
        np.array([hidden_parts], dtype=np.uint32),
        f"routed_sum_hidden_parts_{token_offset}",
    )
    hidden_part_offsets = [
        constant(
            np.array([hidden_part], dtype=np.uint32),
            f"routed_sum_hidden_part_{token_offset}_{hidden_part}",
        )
        for hidden_part in range(1, hidden_parts)
    ]
    wave_metadata = []
    for wave_id, route_start in enumerate(range(0, top_k, routes_per_wave)):
        route_stop = min(route_start + routes_per_wave, top_k)
        route_slice = slice(route_start, route_stop)
        wave_routes = route_stop - route_start
        # Route-column slices are strided in the token-major core tensors. Materialize the small
        # wave metadata so Gather and FMR consume compact rows with unambiguous offsets.
        wave_indices = ax.dsl.create_tensor(
            (token_count, wave_routes),
            ax.uint32,
            ax.MemoryType.OCM,
            name=f"routed_sum_indices_{token_offset}_wave{wave_id}",
        )
        ax.dsl.move(packed_indices[token_slice, route_slice], r=wave_indices)
        wave_weights = ax.dsl.create_tensor(
            (token_count, wave_routes),
            output_dtype,
            ax.MemoryType.OCM,
            name=f"routed_sum_weights_{token_offset}_wave{wave_id}",
        )
        ax.dsl.move(selected_weights[token_slice, route_slice], r=wave_weights)

        # Treat each packed D-wide row as contiguous D/N rows. This keeps both the DDR source and
        # compact [T,2,D/N] OCM destinations contiguous for each hidden shard.
        hidden_row_indices = ax.dsl.mul(
            wave_indices,
            hidden_parts_value,
            output_dtype=ax.uint32,
        )
        indices_by_hidden = [hidden_row_indices]
        for hidden_part in range(1, hidden_parts):
            indices_by_hidden.append(
                ax.dsl.add(
                    hidden_row_indices,
                    hidden_part_offsets[hidden_part - 1],
                    output_dtype=ax.uint32,
                )
            )
        wave_metadata.append(
            (
                wave_routes,
                wave_weights,
                indices_by_hidden,
            )
        )

    # The shared-expert TP partial is the seventh route with an implicit weight of one.
    accumulator = ax.dsl.create_tensor(
        (token_count, hidden),
        output_dtype,
        ax.MemoryType.OCM,
        name=f"routed_sum_accumulator_{token_offset}",
    )
    if shared_rows.memory_type == ax.MemoryType.DDR:
        ax.dsl.load(shared_rows, r=accumulator)
    else:
        ax.dsl.move(shared_rows, r=accumulator)
    for wave_id, (wave_routes, wave_weights, hidden_indices_by_part) in enumerate(wave_metadata):
        expert_rows = [buffer[:, :wave_routes, :] for buffer in expert_row_buffers[wave_id]]
        for hidden_part in range(hidden_parts):
            ax.dsl.index_select(
                packed_expert_output,
                hidden_indices_by_part[hidden_part],
                dim=0,
                r=expert_rows[hidden_part],
            )

        weights = wave_weights.reshape(token_count, wave_routes, 1)
        next_accumulator = ax.dsl.create_tensor(
            (token_count, hidden),
            output_dtype,
            ax.MemoryType.OCM,
            name=f"routed_sum_accumulator_{token_offset}_wave{wave_id}",
        )
        for hidden_part in range(hidden_parts):
            weighted_rows = ax.dsl.mul(
                expert_rows[hidden_part],
                weights,
                output_dtype=output_dtype,
            )
            partial = ax.dsl.sum(
                weighted_rows,
                dim=1,
                output_dtype=output_dtype,
            )
            hidden_slice = token_slice(hidden_part, hidden_parts, hidden)
            ax.dsl.add(
                accumulator[:, hidden_slice],
                partial,
                output_dtype=output_dtype,
                r=next_accumulator[:, hidden_slice],
            )
        accumulator = next_accumulator
    return accumulator


def routed_weighted_sum_decode_part(
    expert_output_workspace,
    chip_expert_lookup,
    selected_ids,
    route_weights,
    shared_rows,
    *,
    hidden_part,
    hidden_parts,
    output_dtype,
):
    """Accumulate one decode token's routed rows for one core-owned hidden shard."""
    experts_with_sentinel, expert_capacity, hidden = expert_output_workspace.shape
    hidden_slice = token_slice(hidden_part, hidden_parts, hidden)
    hidden_width = hidden_slice.stop - hidden_slice.start
    selected = ax.dsl.load(selected_ids)
    weights = ax.dsl.cast(ax.dsl.load(route_weights), dtype=output_dtype)
    chip_local_ids = ax.dsl.index_select(chip_expert_lookup, selected, dim=0)
    packed_rows = ax.dsl.mul(
        chip_local_ids,
        ax.dsl.full((1,), ax.int32, fill_value=expert_capacity),
        output_dtype=ax.int32,
    )
    hidden_rows = ax.dsl.add(
        ax.dsl.mul(
            ax.dsl.cast(packed_rows, dtype=ax.uint32),
            ax.dsl.full((1,), ax.uint32, fill_value=hidden_parts),
            output_dtype=ax.uint32,
        ),
        ax.dsl.full((1,), ax.uint32, fill_value=hidden_part),
        output_dtype=ax.uint32,
    )
    packed_expert_output = expert_output_workspace.reshape(
        experts_with_sentinel * expert_capacity * hidden_parts,
        hidden_width,
    )
    accumulator = ax.dsl.create_tensor(
        (1, hidden_width),
        output_dtype,
        ax.MemoryType.OCM,
        name=f"routed_sum_decode_accumulator_h{hidden_part}",
    )
    ax.dsl.move(shared_rows[:, hidden_slice], r=accumulator)
    routes_per_wave = 3
    for route_start in range(0, selected_ids.shape[1], routes_per_wave):
        route_stop = min(route_start + routes_per_wave, selected_ids.shape[1])
        route_slice = slice(route_start, route_stop)
        expert_rows = ax.dsl.index_select(
            packed_expert_output,
            hidden_rows[:, route_slice],
            dim=0,
        )
        weighted_rows = ax.dsl.mul(
            expert_rows,
            weights[:, route_slice].reshape(1, route_stop - route_start, 1),
            output_dtype=output_dtype,
        )
        partial = ax.dsl.sum(weighted_rows, dim=1, output_dtype=output_dtype)
        # Two balanced waves keep both SDMAs busy. Accumulating in the stationary OCM buffer also
        # removes standalone AxAdd output tensors from the decode critical path.
        ax.dsl.move(partial, r=accumulator, op="add")
    return accumulator, hidden_slice


def stage_ffn_mhc_weight(hc_fn, hc_fn_scale, *, hidden, core_id, core_num):
    """Load one stream's mHC status weight shard into shared OCM on each core."""
    input_slice = slice(core_id * hidden, (core_id + 1) * hidden)
    shared_weight = ax.dsl.create_tensor(
        hc_fn.shape,
        dsl_dtype(hc_fn.dtype),
        ax.MemoryType.SHARED_OCM,
        name="ffn_mhc_weight_shared",
    )
    constant(
        hc_fn[:, input_slice],
        f"ffn_mhc_weight_core{core_id}",
        r=shared_weight[:, input_slice],
    )

    shared_scale = None
    scale2 = None
    if hc_fn_scale is not None:
        scale_slice = slice(input_slice.start // MX_GROUP_SIZE, input_slice.stop // MX_GROUP_SIZE)
        shared_scale = ax.dsl.create_tensor(
            hc_fn_scale.shape,
            ax.mxe8m0,
            ax.MemoryType.SHARED_OCM,
            name="ffn_mhc_scale_shared",
        )
        core_scale = np.ascontiguousarray(hc_fn_scale[:, scale_slice]).view(MXE8M0_DTYPE)
        constant(
            core_scale,
            f"ffn_mhc_scale_core{core_id}",
            r=shared_scale[:, scale_slice],
        )
        scale2 = constant(np.ones((1,), dtype=np.float32), "ffn_mhc_scale2")
    return shared_weight, shared_scale, scale2


def ffn_mhc_token_controls(
    residual_rows,
    hc_fn,
    hc_fn_scale,
    hc_fn_scale2,
    mhc_inverse_divisor,
    rms_epsilon,
    control_base,
    control_scale,
    control_epsilon,
    control_two,
    post_out,
    *,
    sinkhorn_iters,
    name_prefix,
):
    """Normalize one FFN query tile and compute its mHC controls."""
    batch, query_count, streams, hidden = residual_rows.shape

    # Each query owner sees all four streams, so RMS and status projection need no collective.
    normalized_residual = rmsnorm_with_constants(
        residual_rows.reshape(batch, query_count, streams * hidden),
        None,
        mhc_inverse_divisor,
        rms_epsilon,
        result_dtype=residual_rows.dtype,
    )
    status_partials = ax.dsl.create_tensor(
        (batch, query_count, streams, 24),
        ax.fp32,
        ax.MemoryType.OCM,
        name=f"{name_prefix}_status_partials",
    )
    for stream_id in range(streams):
        input_slice = slice(stream_id * hidden, (stream_id + 1) * hidden)
        scale_slice = slice(input_slice.start // MX_GROUP_SIZE, input_slice.stop // MX_GROUP_SIZE)
        ax.dsl.matmul(
            normalized_residual[:, :, input_slice].reshape(batch * query_count, hidden),
            hc_fn[:, input_slice],
            w_scale=1.0 if hc_fn_scale is None else hc_fn_scale[:, scale_slice],
            w_scale2=1.0 if hc_fn_scale2 is None else hc_fn_scale2,
            transposed=True,
            output_dtype=ax.fp32,
            r=status_partials[:, :, stream_id : stream_id + 1, :].reshape(batch * query_count, 24),
        )
    mix = ax.dsl.sum(status_partials, dim=2, output_dtype=ax.fp32)
    comb_output = ax.dsl.create_tensor(
        (batch, query_count, streams, streams),
        ax.fp32,
        ax.MemoryType.OCM,
        name=f"{name_prefix}_comb",
    )
    pre, _, comb = mhc_controls(
        mix,
        control_base,
        control_scale,
        control_epsilon,
        control_two,
        post_out,
        comb_output,
        sinkhorn_iters=sinkhorn_iters,
    )
    return pre, comb


def ffn_prefill(
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
    branch_partial_out,
    residual_path_out,
    selected_ids_out,
    route_weights_out,
    weight_scales,
    *,
    gate_up_weight_local=None,
    down_weight_local=None,
    token_ids=None,
    tid2eid=None,
    correction_bias=None,
    expert_offset=0,
    top_k=6,
    route_scale=1.5,
    swiglu_limit=10.0,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
):
    """Record one chip of the two-chip HASH/learned routed-MoE deployment.

    Omitting both local routed-expert weight tensors consumes the supplied packed expert output
    workspace and records the routing metadata, shared expert, and final accumulation stages.
    """
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    batch, seq_len, streams, hidden = residual.shape
    tokens = batch * seq_len
    global_experts = router_weight.shape[0]
    experts_per_chip = expert_output_workspace.shape[0] - 1
    has_routed_mlp = gate_up_weight_local is not None

    # These constraints define the four-core, two-chip ownership. Tensor operations
    # below enforce the remaining compatible shapes directly.
    assert batch == 1
    assert core_num == streams == 4
    assert experts_per_chip * 2 == global_experts
    assert experts_per_chip % core_num == 0
    assert 0 <= expert_offset <= global_experts - experts_per_chip
    assert has_routed_mlp == (down_weight_local is not None)
    assert selected_ids_out.shape == route_weights_out.shape == (tokens, top_k)
    decode_static = not has_routed_mlp and tokens == 1
    weights = {
        "hc_fn": hc_fn,
        "shared_gate_up": shared_gate_up_weight_chip,
        "shared_down": shared_down_weight_chip,
    }
    if has_routed_mlp:
        assert gate_up_weight_local.shape[0] == experts_per_chip
        weights.update(gate_up=gate_up_weight_local, down=down_weight_local)
    validate_weight_scales(weights, weight_scales)

    coretoken_slice = token_slice(core_id, core_num, seq_len)
    # post_shared: [B,S,4], FP32 SHOCM; every core writes only its token slice.
    post_shared = ax.dsl.create_tensor(
        (batch, seq_len, 4),
        ax.fp32,
        ax.MemoryType.SHARED_OCM,
        name="ffn_post",
    )

    staged_mhc_weight = stage_ffn_mhc_weight(
        hc_fn,
        weight_scales.get("hc_fn"),
        hidden=hidden,
        core_id=core_id,
        core_num=core_num,
    )
    axd.sync_all()

    # FFN mHC is query-owned. Each core consumes all four shared weight shards for its disjoint
    # token slice, which removes stream/status exchanges and duplicate DDR weight reads.
    # norm: [D], norm_weight.dtype.
    norm = constant(norm_weight, "ffn_norm")
    mhc_inverse_divisor = constant(
        np.array([1.0 / (streams * hidden)], dtype=np.float32),
        "ffn_mhc_inverse_divisor",
    )
    norm_inverse_divisor = constant(
        np.array([1.0 / hidden], dtype=np.float32),
        "ffn_norm_inverse_divisor",
    )
    rms_epsilon = constant(np.array([rms_eps], dtype=np.float32), "ffn_rms_epsilon")
    control_base = constant(hc_base.astype(np.float32), "ffn_mhc_base")
    control_scale = constant(hc_scale.astype(np.float32), "ffn_mhc_scale")
    control_epsilon = constant(np.array([hc_eps], dtype=np.float32), "ffn_mhc_epsilon")
    control_two = constant(np.array([2.0], dtype=np.float32), "ffn_mhc_two")
    if decode_static:
        normalized_storage = ax.dsl.create_tensor(
            (batch, seq_len, hidden),
            residual.dtype,
            ax.MemoryType.SHARED_OCM,
            name="ffn_decode_normalized",
        )
    else:
        normalized_storage = branch_partial_out[:, :, 0:1, :].reshape(batch, seq_len, hidden)

    if decode_static:
        preloaded_shared_gate_up = preload_shared_gate_up_decode(
            shared_gate_up_weight_chip,
            weight_scales.get("shared_gate_up"),
            core_id=core_id,
            core_num=core_num,
        )
    loaded_tile = None
    controlled_tile = None
    tile_starts = range(coretoken_slice.start, coretoken_slice.stop, MHC_FFN_QUERY_TILE)
    # Keep adjacent tiles in load, control, and output stages so their SDMA, Conv, and TENG work
    # can overlap. Two final sentinels drain the latter stages through the same loop body.
    for next_start in [*tile_starts, None, None]:
        next_loaded_tile = None
        if next_start is not None:
            next_stop = min(next_start + MHC_FFN_QUERY_TILE, coretoken_slice.stop)
            next_residual_rows = ax.dsl.create_tensor(
                (batch, next_stop - next_start, streams, hidden),
                residual.dtype,
                ax.MemoryType.OCM,
                name=f"ffn_mhc_{next_start}_residual",
            )
            ax.dsl.load(residual[:, next_start:next_stop, :, :], r=next_residual_rows)
            next_loaded_tile = next_start, next_stop, next_residual_rows

        next_controlled_tile = None
        if loaded_tile is not None:
            seq_start, seq_stop, residual_rows = loaded_tile
            pre, comb = ffn_mhc_token_controls(
                residual_rows,
                *staged_mhc_weight,
                mhc_inverse_divisor,
                rms_epsilon,
                control_base,
                control_scale,
                control_epsilon,
                control_two,
                post_shared[:, seq_start:seq_stop, :],
                sinkhorn_iters=sinkhorn_iters,
                name_prefix=f"ffn_mhc_{seq_start}",
            )
            next_controlled_tile = seq_start, seq_stop, residual_rows, pre, comb

        # Record the next tile's controls before the current tile's output stage. Its status
        # MatMul can then run while TENG folds and normalizes the current residual rows.
        if controlled_tile is not None:
            seq_start, seq_stop, residual_rows, pre, comb = controlled_tile
            query_count = seq_stop - seq_start
            collapsed = ax.dsl.create_tensor(
                (batch, query_count, hidden),
                residual.dtype,
                ax.MemoryType.OCM,
                name=f"ffn_mhc_{seq_start}_collapsed",
            )
            mhc_residual_outputs(
                residual_rows,
                pre,
                comb,
                collapsed,
                name_prefix=f"ffn_mhc_{seq_start}",
                residual_path_out=residual_path_out[:, seq_start:seq_stop, :, :],
            )
            normalized = rmsnorm_with_constants(
                collapsed,
                norm,
                norm_inverse_divisor,
                rms_epsilon,
                result_dtype=residual.dtype,
            )
            normalized_out = normalized_storage[:, seq_start:seq_stop, :]
            if normalized_out.memory_type == ax.MemoryType.DDR:
                ax.dsl.store(normalized, r=normalized_out)
            else:
                ax.dsl.move(normalized, r=normalized_out)
        loaded_tile = next_loaded_tile
        controlled_tile = next_controlled_tile

    flat_normalized = normalized_storage.reshape(tokens, hidden)
    if decode_static:
        # CORE0 owns the only mHC query. Publish its normalized row before the other cores read it;
        # the weight preloads above remain independent and execute while mHC is in flight.
        axd.sync_all()
        router_logits = route_single_token_decode_logits(
            flat_normalized,
            router_weight,
            weight_scales.get("router"),
            core_id=core_id,
            core_num=core_num,
        )
        # A decode token can occupy only slot zero of every selected expert. The static kernel
        # consumes prepacked expert outputs, so it does not need the general prefix/cumsum map.
        if core_id == 0:
            ax.dsl.store(
                ax.dsl.zeros(expert_slot_workspace.shape, ax.int32),
                r=expert_slot_workspace,
            )
            # Remote routes always use sentinel slot zero; no other sentinel row is read.
            ax.dsl.store(
                ax.dsl.zeros((1, hidden), residual.dtype),
                r=expert_output_workspace[experts_per_chip : experts_per_chip + 1, 0:1, :,].reshape(
                    1,
                    hidden,
                ),
            )

        # Router logits and the shared expert are independent consumers of the normalized row.
        # The shared-expert reduction barrier also publishes every core's local Router logits.
        run_shared_expert(
            flat_normalized,
            shared_gate_up_weight_chip,
            shared_down_weight_chip,
            normalized_storage.reshape(tokens, hidden),
            weight_scales,
            core_id=core_id,
            core_num=core_num,
            preloaded_gate_up=preloaded_shared_gate_up,
            limit=swiglu_limit,
        )
        if core_id == 0:
            finish_single_token_decode(
                router_logits,
                correction_bias,
                selected_ids_out,
                route_weights_out,
                token_ids=token_ids,
                tid2eid=tid2eid,
                route_scale=route_scale,
            )
        axd.sync_all()

        chip_expert_lookup = np.full((global_experts,), experts_per_chip, dtype=np.int32)
        chip_expert_lookup[expert_offset : expert_offset + experts_per_chip] = np.arange(
            experts_per_chip,
            dtype=np.int32,
        )
        chip_expert_lookup = constant(chip_expert_lookup, "chip_expert_lookup")
        local_output, hidden_slice = routed_weighted_sum_decode_part(
            expert_output_workspace,
            chip_expert_lookup,
            selected_ids_out,
            route_weights_out,
            normalized_storage.reshape(tokens, hidden),
            hidden_part=core_id,
            hidden_parts=core_num,
            output_dtype=residual.dtype,
        )
        post = post_shared.reshape(batch, 1, streams, 1)
        branch = ax.dsl.mul(
            post,
            local_output.reshape(batch, 1, 1, local_output.shape[-1]),
            output_dtype=residual.dtype,
        )
        ax.dsl.store(branch, r=branch_partial_out[:, :, :, hidden_slice])
        return

    # Router and dispatch
    route_tokens_with_router(
        flat_normalized,
        router_weight,
        selected_ids_out,
        route_weights_out,
        weight_scales,
        core_id=core_id,
        core_num=core_num,
        correction_bias=correction_bias,
        token_ids=token_ids,
        tid2eid=tid2eid,
        route_scale=route_scale,
    )
    expert_token_map, core_expert_begin, experts_per_core = prepare_token_dispatch(
        selected_ids_out,
        expert_slot_workspace,
        core_id=core_id,
        core_num=core_num,
        expert_offset=expert_offset,
    )
    if core_id == 0:
        # Remote routes always use sentinel slot zero; no other sentinel row is read.
        ax.dsl.store(
            ax.dsl.zeros((1, hidden), residual.dtype),
            r=expert_output_workspace[experts_per_chip : experts_per_chip + 1, 0:1, :,].reshape(
                1,
                hidden,
            ),
        )

    # Run local routed experts when this chip owns their weights.
    if has_routed_mlp:
        run_routed_experts(
            flat_normalized,
            expert_token_map,
            gate_up_weight_local,
            down_weight_local,
            expert_output_workspace,
            weight_scales,
            core_expert_begin=core_expert_begin,
            experts_per_core=experts_per_core,
            swiglu_limit=swiglu_limit,
        )
    # The range-local barrier above joins its dynamic branches. This separate barrier publishes
    # every core's completed routed rows and slot table before shared-expert work starts.
    axd.sync_all()
    run_shared_expert(
        flat_normalized,
        shared_gate_up_weight_chip,
        shared_down_weight_chip,
        normalized_storage.reshape(tokens, hidden),
        weight_scales,
        core_id=core_id,
        core_num=core_num,
        limit=swiglu_limit,
    )

    # Resolve each core's exact routed rows once. The tile loop pipelines six routed rows plus the
    # shared-expert TP partial as a seventh route before applying mHC post mixing.
    chip_expert_lookup = np.full((global_experts,), experts_per_chip, dtype=np.int32)
    chip_expert_lookup[expert_offset : expert_offset + experts_per_chip] = np.arange(
        experts_per_chip,
        dtype=np.int32,
    )
    chip_expert_lookup = constant(chip_expert_lookup, "chip_expert_lookup")
    if coretoken_slice.start < coretoken_slice.stop:
        packed_indices, selected_weights = prepare_routed_weighted_sum(
            expert_output_workspace,
            expert_slot_workspace,
            chip_expert_lookup,
            selected_ids_out,
            route_weights_out,
            coretoken_slice,
            residual.dtype,
        )
        for seq_start in range(coretoken_slice.start, coretoken_slice.stop, MOE_WEIGHTED_SUM_TILE):
            seq_stop = min(seq_start + MOE_WEIGHTED_SUM_TILE, coretoken_slice.stop)
            token_count = seq_stop - seq_start
            localtoken_slice = slice(
                seq_start - coretoken_slice.start,
                seq_stop - coretoken_slice.start,
            )
            # local_output: [T_final,D], residual.dtype. Shared load and routed FMR waves are pipelined.
            local_output = routed_weighted_sum_tile(
                expert_output_workspace,
                packed_indices,
                selected_weights,
                normalized_storage[:, seq_start:seq_stop, :].reshape(token_count, hidden),
                localtoken_slice,
                residual.dtype,
                token_offset=seq_start,
            )
            # post: [B,T_final,4,1], FP32.
            post = post_shared[:, seq_start:seq_stop, :].reshape(batch, token_count, 4, 1)
            # output_rows: [B,T_final,1,D], residual.dtype.
            output_rows = local_output.reshape(batch, token_count, 1, hidden)
            # Keep mHC post mixing and DDR writeback on the same hidden quarters as the routed Gather.
            for hidden_part in range(MOE_WEIGHTED_SUM_HIDDEN_PARTS):
                hidden_slice = token_slice(hidden_part, MOE_WEIGHTED_SUM_HIDDEN_PARTS, hidden)
                # branch: [B,T_final,4,D/4], residual.dtype.
                branch = ax.dsl.mul(
                    post,
                    output_rows[:, :, :, hidden_slice],
                    output_dtype=residual.dtype,
                )
                # branch_partial_out slice: [B,T_final,4,D/4], residual.dtype DDR.
                ax.dsl.store(
                    branch,
                    r=branch_partial_out[:, seq_start:seq_stop, :, hidden_slice],
                )
