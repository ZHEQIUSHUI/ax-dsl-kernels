"""NumPy dtype and host quantization helpers for DeepSeek V4 Flash."""

import ml_dtypes
import numpy as np


BF16_DTYPE = np.dtype(ml_dtypes.bfloat16)
MXFP4_DTYPE = np.dtype([("MXFP4", np.uint8)])
MXFP8_E4M3_DTYPE = np.dtype([("MXFP8_E4M3", np.uint8)])
MXE8M0_DTYPE = np.dtype([("MXE8M0", np.uint8)])

_DTYPE_NAMES = {
    BF16_DTYPE: "BF16",
    MXFP4_DTYPE: "MXFP4",
    MXFP8_E4M3_DTYPE: "MXFP8_E4M3",
    MXE8M0_DTYPE: "MXE8M0",
}
_DTYPE_BITS = {
    BF16_DTYPE: 16,
    MXFP4_DTYPE: 4,
    MXFP8_E4M3_DTYPE: 8,
    MXE8M0_DTYPE: 8,
}


def dtype_name(dtype) -> str:
    return _DTYPE_NAMES[np.dtype(dtype)]


def dtype_bits(dtype) -> int:
    return _DTYPE_BITS[np.dtype(dtype)]


def fp32_to_mxfp4(values) -> np.ndarray:
    values = np.clip(np.asarray(values), -6.0, 6.0)
    return values.astype(ml_dtypes.float4_e2m1fn).view(MXFP4_DTYPE)


def mxfp4_to_fp32(values) -> np.ndarray:
    return np.asarray(values).view(ml_dtypes.float4_e2m1fn).astype(np.float32)


def fp32_to_mxfp8_e4m3(values) -> np.ndarray:
    values = np.asarray(values)
    overflow = np.isinf(values) | (np.isfinite(values) & (np.abs(values) > 448.0))
    if np.any(overflow):
        values = values.copy()
        values[overflow] = np.nan
    return values.astype(ml_dtypes.float8_e4m3fn).view(MXFP8_E4M3_DTYPE)


def mxfp8_e4m3_to_fp32(values) -> np.ndarray:
    return np.asarray(values).view(ml_dtypes.float8_e4m3fn).astype(np.float32)


def torch_to_numpy(tensor) -> np.ndarray:
    """Convert host Torch tensors to NumPy while preserving DSL constant dtypes."""
    import torch

    dtype_map = {
        torch.bfloat16: ml_dtypes.bfloat16,
        torch.float8_e4m3fn: ml_dtypes.float8_e4m3fn,
        torch.float8_e5m2: ml_dtypes.float8_e5m2,
    }
    if tensor.dtype in dtype_map:
        return tensor.to(torch.float32).numpy().copy().astype(dtype_map[tensor.dtype])
    return tensor.numpy().copy()
