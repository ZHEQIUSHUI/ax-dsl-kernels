"""Shared weight storage and contiguous kernel arguments for dynamic blocks."""

import math

import torch

# AXCL allocations are limited to UINT32_MAX bytes. Torch rounds large
# allocations up to 2 MiB, so reserve that final rounding interval.
MAX_WEIGHT_ALLOCATION_BYTES = (1 << 32) - (2 << 20)


def bind_weight_banks(layers):
    """Allocate each contiguous weight bank once, then load checkpoint shards into it."""
    plans = []
    device = next(layers[0].parameters()).device
    for phase in range(4):
        group = [dict(layer.named_parameters()) for layer in layers[phase::4]]
        for name, first in group[0].items():
            if first.dtype not in (torch.uint8, torch.float8_e4m3fn):
                continue
            parameters = [layer[name] for layer in group]
            size = len(group) * first.numel() * first.element_size()
            if size > MAX_WEIGHT_ALLOCATION_BYTES:
                raise ValueError(f"weight bank {name} needs {size} bytes, exceeding the AXCL allocation limit")
            plans.append(
                (
                    (phase, name),
                    (len(group), *first.shape),
                    first.dtype,
                    parameters,
                    size,
                )
            )
    # These buffers have no checkpoint values yet. Release all of them before
    # allocating the banks, and return unused device-cache blocks to CMM.
    for _, _, dtype, parameters, _ in plans:
        for parameter in parameters:
            parameter.data = torch.empty(0, dtype=dtype, device=device)
    if device.type != "cpu":
        torch.get_device_module(device).empty_cache()
    banks = {}
    # A kernel binds banks individually. Packing unrelated banks into a larger
    # arena needs a larger physical CMM hole without reducing kernel IO or copies.
    # Place the largest banks first while those contiguous regions are available.
    for key, shape, dtype, parameters, _ in sorted(plans, key=lambda plan: plan[-1], reverse=True):
        bank = torch.empty(shape, dtype=dtype, device=device)
        for index, parameter in enumerate(parameters):
            # Retain Parameter subclasses, shard metadata and weight_loader.
            parameter.data = bank[index]
        banks[key] = bank
    return banks


def block_arguments(layers, mlp_kind, weight_banks=()):
    """Build four block descriptors and contiguous kernel IO, preserving aliases."""
    blocks = len(layers) // 4
    tensors = {}
    arguments, inputs = {}, {}

    def input_name(label, bank):
        key = bank.data_ptr(), bank.dtype, tuple(bank.shape)
        if key not in inputs:
            assert bank.is_contiguous(), f"kernel input {label} must be contiguous"
            inputs[key] = label
            tensors[label] = bank
        return inputs[key]

    def argument(label, values):
        first = values[0]
        if not torch.is_tensor(first):
            return first, tuple(values)
        key = (
            tuple(value.data_ptr() for value in values),
            first.dtype,
            tuple(first.shape),
            tuple(first.stride()),
        )
        if key in arguments:
            return arguments[key]
        elements = first.numel()
        offsets = [value.storage_offset() for value in values]
        stride = offsets[1] - offsets[0] if blocks > 1 else elements
        shared = all(value.untyped_storage().data_ptr() == first.untyped_storage().data_ptr() for value in values)
        parent = next(
            (bank for bank in weight_banks if bank.data_ptr() <= first.data_ptr() < bank.data_ptr() + bank.nbytes),
            None,
        )
        if (
            label.endswith(("_cos", "_sin", "_cos_sin_cache"))
            and shared
            and all(offset == offsets[0] for offset in offsets)
        ):
            # RoPE is shared by every block. Reuse its storage as a one-row
            # bank; its explicit layer operand always selects row 0.
            bank = first.unsqueeze(0)
            expression = input_name(label, bank), 0, 0
            views = (first,) * blocks
        elif label.endswith(("_in", "_out", "_kv")) and shared:
            # GDN state pools interleave three layers per block. Bind the full
            # pool and retain the phase + block*3 index rather than copying state.
            count = first.untyped_storage().nbytes() // first.nbytes
            bank = first.as_strided((count, *first.shape), (elements, *first.stride()), storage_offset=0)
            expression = (
                input_name(label, bank),
                offsets[0] // elements,
                stride // elements,
            )
            views = tuple(values)
        elif parent is not None:
            row_elements = elements // first.shape[0]
            rows = parent.numel() // blocks // row_elements
            bank = parent.view(first.dtype).reshape(blocks, rows, *first.shape[1:])
            start = (first.data_ptr() - bank.data_ptr()) // (row_elements * first.element_size())
            # Torch strides are not external DSL IO metadata. Pass the full
            # contiguous fused bank and slice QKV/gate/up inside the graph.
            expression = input_name(label, bank), (start, start + first.shape[0])
            views = tuple(values)
        else:
            # Each layer is also passed directly to standalone prefill kernels.
            # Their external DDR pointers must stay 32-byte aligned, including
            # scalar scale2 values that would otherwise have a four-byte stride.
            row_bytes = math.prod(first.shape[1:]) * first.element_size()
            alignment = 32 // math.gcd(32, row_bytes)
            rows = (first.shape[0] + alignment - 1) // alignment * alignment if blocks > 1 else first.shape[0]
            if (
                rows == first.shape[0]
                and shared
                and first.data_ptr() % 32 == 0
                and offsets == [offsets[0] + i * elements for i in range(blocks)]
            ):
                bank = first.as_strided((blocks, *first.shape), (elements, *first.stride()))
            else:
                bank = torch.empty(
                    (blocks, rows, *first.shape[1:]),
                    dtype=first.dtype,
                    device=first.device,
                )
                for index, value in enumerate(values):
                    bank[index, : first.shape[0]].reshape(-1).view(torch.uint8).copy_(
                        value.reshape(-1).view(torch.uint8)
                    )
                torch.get_device_module(bank.device).synchronize(bank.device)
            expression = input_name(label, bank)
            if rows != first.shape[0]:
                expression = expression, (0, first.shape[0])
            views = tuple(bank[index, : first.shape[0]] for index in range(blocks))
        arguments[key] = expression, views
        return expression, views

    layout = []
    for phase in range(4):
        group = layers[phase::4]
        kv_arg, kv_views = argument(f"b{phase}_kv", [kv for kv, _, _ in group])
        mappings = []
        for slot, kind in ((1, "attn"), (2, mlp_kind)):
            fields = {}
            for name in sorted(group[0][slot]):
                expression, views = argument(f"b{phase}_{kind}_{name}", [layer[slot][name] for layer in group])
                for layer, view in zip(group, views, strict=True):
                    layer[slot][name] = view
                fields[name] = expression
            mappings.append(fields)
        for index, view in enumerate(kv_views):
            _, attention, mlp = layers[4 * index + phase]
            layers[4 * index + phase] = view, attention, mlp
        layout.append((kv_arg, *mappings))
    return tuple(layout), tensors


def block_stages(layout, weights):
    """Resolve bank names and projection slices into DSL tensor views."""

    def resolve(value):
        if isinstance(value, str):
            return weights[value]
        if isinstance(value, tuple):
            if len(value) == 2:
                name, (start, stop) = value
                return weights[name][:, start:stop]
            name, offset, stride = value
            return weights[name], offset, stride
        return value

    return tuple(
        (
            resolve(kv),
            {name: resolve(value) for name, value in attention.items()},
            {name: resolve(value) for name, value in mlp.items()},
        )
        for kv, attention, mlp in layout
    )


def block_layer(stage, block_index, input_banks):
    """Pass static slices and explicit affine layer coordinates to consumers."""
    from axnn.dsl.util import DslTensor

    banks = {}

    def select(value):
        offset, stride = 0, 1
        if isinstance(value, tuple):
            value, offset, stride = value
        if isinstance(value, DslTensor):
            if stride and value.name not in banks:
                bank = input_banks[value.name]
                rows = bank.shape_size // bank.shape[0] // bank.shape[-1]
                banks[value.name] = bank, (block_index, stride * rows, offset * rows)
            start = 0 if stride else offset
            return value[start : start + 1].reshape(*value.shape[1:])
        return value

    kv, attention, mlp = stage
    return (
        select(kv),
        {name: select(value) for name, value in attention.items()},
        {name: select(value) for name, value in mlp.items()},
    ), banks
