"""Shared helpers for the MTP/verify board-vs-reference checks."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

# One-byte storage types: fp4 pairs, fp8, uint8.  view(uint8) keeps the exact bytes.
_PACKED = {
    torch.float4_e2m1fn_x2,
    torch.float8_e4m3fn,
    torch.float8_e8m0fnu,
    torch.uint8,
    torch.int8,
}


def raw_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    elif t.dtype in _PACKED:
        t = t.view(torch.uint8)
    return t.numpy().tobytes()


def write_bin(path: Path, t: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    b = raw_bytes(t)
    path.write_bytes(b)
    return len(b)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 and float(nb) == 0.0:
        return 1.0
    return float((a * b).sum() / (na * nb).clamp_min(1e-20))


def cos_rows(a: torch.Tensor, b: torch.Tensor, dim=0) -> list[float]:
    """Cosine per leading row for tensors whose first dim is the row index."""
    n = a.shape[dim]
    out = []
    for i in range(n):
        sl = [slice(None)] * a.ndim
        sl[dim] = i
        out.append(cos(a[tuple(sl)], b[tuple(sl)]))
    return out


def exact_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.cpu(), b.cpu())


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)
