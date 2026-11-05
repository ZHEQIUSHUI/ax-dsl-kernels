"""Tiled runtime-bank transfers expressed with existing DSL operators.

String keys in ``banks`` map storage names to ``(bank, (index, stride, offset))``.
The affine expression selects a layer in physical bank rows. Tuple keys cache
row-index tensors and mark read-only banks for fixed-slice dynamic loads.
"""
import math
import numpy as np
import axnn.dsl as ax
from axnn.dsl import util
from axnn.dsl import parallel as axd


def _integer_scale(x, factor):
    """Multiply unsigned row addresses exactly, including values above 2**24."""
    assert factor >= 0
    if factor == 1:
        return x
    zero = ax.dsl.zeros((1,), ax.uint32)
    if factor == 0:
        return zero
    parts = [
        x
        if bit == 0
        else util.op_convert(
            "elemwise_int",
            {"op": "add", "shift_bit": -bit, "output_dtype": ax.uint32},
            {"x": x, "y": zero},
            {"r": None},
        )
        for bit in range(factor.bit_length())
        if factor & (1 << bit)
    ]
    result = parts[0]
    for part in parts[1:]:
        result = util.op_convert(
            "elemwise_int", {"op": "add", "output_dtype": ax.uint32}, {"x": result, "y": part}, {"r": None}
        )
    return result


def _linear(index, factor, offset, *, exact_float):
    if factor == 1 and isinstance(offset, np.ndarray) and not np.any(offset):
        return index
    if exact_float:
        return util.op_convert(
            "fma_float",
            {"output_dtype": ax.uint32},
            {
                "x": index,
                "y": ax.dsl.full((1,), ax.uint32, factor),
                "z": ax.dsl.from_numpy(offset) if isinstance(offset, np.ndarray) else offset,
            },
            {"r": None},
        )
    base = _integer_scale(index, factor)
    if isinstance(offset, np.ndarray):
        if not np.any(offset):
            return base
        offset = ax.dsl.from_numpy(offset)
    return util.op_convert(
        "elemwise_int", {"op": "add", "output_dtype": ax.uint32}, {"x": base, "y": offset}, {"r": None}
    )


def _rows(tile, banks, indices=None, *, grouped=False):
    bank, (index, factor, start) = banks[tile.name]
    columns = math.gcd(bank.shape[-1], *(stride // tile.dtype.num_bits for stride in tile.bit_strides[:-1]))
    # Down-projection IC waves may split one original physical matrix row.
    # Keep every static/runtime coordinate in that smaller contiguous row unit.
    ratio = bank.shape[-1] // columns
    factor *= ratio
    start *= ratio
    if indices is not None and len(tile.shape) == 1:
        # Expert scalar scales index elements instead of matrix rows.
        factor *= columns
        start *= columns
        columns = 1
        tile = tile.reshape(tile.shape[0], 1)
    if tile.shape[-1] > columns:
        assert tile.shape[-1] % columns == 0
        tile = tile.reshape(*tile.shape[:-1], tile.shape[-1] // columns, columns)
    width = tile.shape[-1]
    bits = tile.dtype.num_bits
    element_offset = (tile.offset - bank.offset) * 8 // bits
    row_offset, column = divmod(element_offset, columns)
    assert column + width <= columns
    shape = tile.shape[:-1]
    offsets_shape = (1, *shape[1:]) if indices is not None else shape
    offsets = np.full(offsets_shape or (1,), start + row_offset, dtype=np.uint32)
    for axis, size in enumerate(offsets_shape):
        assert tile.bit_strides[axis] % (columns * bits) == 0
        stride = tile.bit_strides[axis] // (columns * bits)
        offsets += (
            np.arange(size, dtype=np.uint32).reshape((1,) * axis + (size,) + (1,) * (len(shape) - axis - 1)) * stride
        )
    group = 1
    flat = offsets.reshape(-1)
    expert_stride = None if indices is None else tile.bit_strides[0] // (columns * bits)
    # A complete contiguous payload can use fewer, larger gather rows. Writes
    # stay two-dimensional so partial state/KV updates preserve their aliases.
    if grouped and width == columns and flat.size > 1 and np.all(np.diff(flat) == 1):
        divisor = math.gcd(
            factor,
            int(flat[0]),
            flat.size,
            bank.shape_size // columns,
            0 if expert_stride is None else expert_stride,
        )
        # A cropped/strided output can make the existing gather builder expand
        # a group using UINT16 coordinates; 32768 is the largest safe power of two.
        group = min(divisor & -divisor, 32768)
        factor //= group
        offsets = flat[::group] // group
        if expert_stride is not None:
            expert_stride //= group
    # FP32 represents every possible row address exactly in this range. Larger
    # banks retain integer shifts/adds, including odd offsets beyond 2**24.
    exact_float = bank.shape_size // columns // group <= 2**24
    selected = None if indices is None else util.get_name(indices)
    key = ("row_ids", util.get_name(index), factor, tuple(offsets.reshape(-1)), selected, expert_stride)
    if key not in banks:
        if indices is not None:
            assert shape
            local = _linear(
                indices.reshape(-1, *([1] * (len(shape) - 1))), expert_stride, offsets, exact_float=exact_float
            )
        else:
            local = offsets.reshape(-1)
        banks[key] = _linear(index, factor, local, exact_float=exact_float).reshape(-1)
    source = (
        bank.reshape(-1, group, columns)[:, :, column : column + width]
        if group > 1
        else bank.reshape(-1, columns)[:, column : column + width]
    )
    return source, banks[key]


def _dynamic_weight_tile(x, banks, r):
    bank, (index, stride, start) = banks[x.name]
    columns = bank.shape[-1]
    offset = (x.offset - bank.offset) * 8 // x.dtype.num_bits + start * columns
    if x.is_contiguous():
        source = bank.reshape(-1, stride * columns)[:, offset : offset + x.shape_size]
        source = source.reshape(source.shape[0], *x.shape)
    else:
        assert x.ndim == 2 and x.bit_strides == (columns * x.dtype.num_bits, x.dtype.num_bits)
        row, column = divmod(offset, columns)
        rows, width = x.shape
        source = bank.reshape(-1, stride, columns)[:, row : row + rows, column : column + width]
    extent = ("weight_slice_extent",)
    if extent not in banks:
        banks[extent] = ax.dsl.full((1,), ax.int32, 1)
    # Only the layer coordinate is dynamic. Static matrix slices retain their
    # physical row strides, and a one-layer result fixes the copy extent to one.
    result = ax.dsl.dynamic_ld(
        source,
        index,
        banks[extent],
        dim=(0,),
        output_shape=(1, *x.shape),
        r=None if r is None else r.reshape(1, *x.shape),
    )
    return result.reshape(*x.shape) if r is None else r


def load_tile(x, *, banks=None, r=None):
    if not banks or x.name not in banks:
        return ax.dsl.load(x, r=r)
    if ("dynamic_ld", x.name) in banks and x.shape_size * x.dtype.num_bits >= 256:
        return _dynamic_weight_tile(x, banks, r)
    source, rows = _rows(x, banks, grouped=True)
    result = ax.dsl.index_select(source, rows, dim=0, r=None if r is None else r.reshape(-1, *source.shape[1:]))
    return result.reshape(*x.shape) if r is None else r


def load_shared_scale_rows(scale, *, banks=None):
    """Read each complete scale row once, shared by all IC-sharded consumers."""
    core, cores = ax.program_id(0), ax.num_programs(0)
    assert scale.ndim == 2 and scale.shape[0] % cores == 0
    rows = scale.shape[0] // cores
    owned = slice(core * rows, (core + 1) * rows)
    result = ax.dsl.create_tensor(scale.shape, scale.dtype, ax.MemoryType.SHARED_OCM)
    load_tile(scale[owned, :], r=result[owned, :], banks=banks)
    axd.pub(result)
    axd.sub(result)
    return result


def store_tile(x, *, r, banks=None):
    if not banks or r.name not in banks:
        return ax.dsl.store(x, r=r)
    target, rows = _rows(r, banks)
    ax.dsl.select_scatter(x.reshape(-1, target.shape[-1]), rows, dim=0, r=target)


def gather_rows(x, indices, *, dim, r=None, banks=None):
    if not banks or x.name not in banks:
        return ax.dsl.index_select(x, indices, dim=dim, r=r)
    assert dim == 0
    if indices.memory_type == ax.MemoryType.DDR:
        indices = ax.dsl.load(indices)
    source, rows = _rows(x, banks, indices, grouped=True)
    result = ax.dsl.index_select(source, rows, dim=0, r=None if r is None else r.reshape(-1, *source.shape[1:]))
    return result.reshape(*indices.shape, *x.shape[1:]) if r is None else r


def scatter_rows(x, indices, *, dim, r, banks=None):
    if not banks or r.name not in banks:
        return ax.dsl.select_scatter(x, indices, dim=dim, r=r)
    assert dim == 0
    if indices.memory_type == ax.MemoryType.DDR:
        indices = ax.dsl.load(indices)
    target, rows = _rows(r, banks, indices)
    ax.dsl.select_scatter(x.reshape(-1, target.shape[-1]), rows, dim=0, r=target)
    return r
