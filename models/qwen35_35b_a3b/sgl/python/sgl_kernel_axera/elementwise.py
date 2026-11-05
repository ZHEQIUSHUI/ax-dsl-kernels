import math
import os

import torch

import axnn.dsl as ax


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert input.shape[:-1] == output.shape[:-1], f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert input.shape[-1] == 2 * output.shape[-1], f"{input.shape[-1]} != {2 * output.shape[-1]}"


@ax.builder(
    run_mode=ax.RunMode(os.environ.get("RUN_MODE", "CRUN")),
    dump_dsl_perf=os.environ.get("DUMP_DSL_PERF", "0") == "1",
)
def gelu_and_mul_kernel(input, out, tile_size):
    dim = input.shape[2] // 2

    offset_0 = ax.program_id(0) * tile_size[0]
    offset_1 = ax.program_id(1) * tile_size[1]
    offset_2 = ax.program_id(2) * tile_size[2]

    end_0 = min(input.shape[0], offset_0 + tile_size[0])
    end_1 = min(input.shape[1], offset_1 + tile_size[1])
    end_2 = min(dim, offset_2 + tile_size[2])

    slice_gate = (
        slice(offset_0, end_0),
        slice(offset_1, end_1),
        slice(offset_2, end_2),
    )
    slice_up = (
        slice(offset_0, end_0),
        slice(offset_1, end_1),
        slice(dim + offset_2, dim + end_2),
    )
    slice_out = (
        slice(offset_0, end_0),
        slice(offset_1, end_1),
        slice(offset_2, end_2),
    )

    gate_tile = ax.dsl.load(input[slice_gate])
    up_tile = ax.dsl.load(input[slice_up])
    out_tile = ax.dsl.fmul(up_tile, ax.dsl.gelu(gate_tile), output_dtype=out.dtype)

    ax.dsl.store(r=out[slice_out], x=out_tile)
    return out


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor = None):
    ax.init(chip_num=1, core_num=1)
    if input.shape[2] % 2 != 0:
        raise ValueError(f"input.shape[2] must be even, but got {input.shape[2]}")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )

    dim = input.shape[2] // 2
    expected_out_shape = (input.shape[0], input.shape[1], dim)
    if tuple(out.shape) != expected_out_shape:
        raise ValueError(f"out shape must be {expected_out_shape}, but got {tuple(out.shape)}")

    tile_size = [min(input.shape[0], 4), min(input.shape[1], 16), min(dim, 1024)]

    grid_x = math.ceil(input.shape[0] / tile_size[0])
    grid_y = math.ceil(input.shape[1] / tile_size[1])
    grid_z = math.ceil(dim / tile_size[2])

    gelu_and_mul_kernel[(grid_x, grid_y, grid_z), 1](input, out, tile_size)
    return out
