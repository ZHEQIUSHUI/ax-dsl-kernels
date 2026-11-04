"""MXFP4 packing / unpacking exactly as ggml-axera's axera_quantize_mxfp4."""
from __future__ import annotations

import numpy as np
import torch

GROUP = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def quantize_mxfp4(weight: torch.Tensor, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (packed codes [N, K/2] uint8 low nibble first, E8M0 [N, K/32] uint8).

    Mirrors quant.cpp: e = floor(log2(amax)) - 2 (0 when amax == 0), scale byte
    clamp(e + 127, 0, 254), nearest E2M1 by strict-less comparison (ties low).
    Rows are processed in chunks so the N x K x 8 nearest-code temporary stays small.
    """
    w = weight.detach().cpu().float().contiguous()
    n, k = w.shape
    assert k % GROUP == 0, (n, k)
    pk = torch.empty((n, k // 2), dtype=torch.uint8)
    sc = torch.empty((n, k // GROUP), dtype=torch.uint8)
    for r0 in range(0, n, chunk):
        g = w[r0:r0 + chunk].reshape(-1, k // GROUP, GROUP)
        amax = g.abs().amax(-1)
        e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2,
                        torch.zeros_like(amax))
        x = g / torch.exp2(e).unsqueeze(-1)
        best = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1).to(torch.uint8)
        code = best | ((x < 0).to(torch.uint8) << 3)
        pk[r0:r0 + chunk] = (code[..., 0::2] | (code[..., 1::2] << 4)).reshape(-1, k // 2)
        sc[r0:r0 + chunk] = (e + 127).clamp(0, 254).to(torch.uint8)
    return pk.contiguous(), sc.contiguous()


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Packed [N, K/2] uint8 + E8M0 [N, K/32] uint8 -> [N, K] f32."""
    p = packed.detach().cpu().contiguous().view(torch.uint8)
    s = scale.detach().cpu().contiguous().view(torch.uint8)
    n = p.shape[0]
    out = torch.empty((n, p.shape[1] * 2), dtype=torch.float32)
    for r0 in range(0, n, chunk):
        pc = p[r0:r0 + chunk]
        code = torch.stack((pc & 0x0F, pc >> 4), dim=-1).reshape(pc.shape[0], -1).long()
        value = E2M1[code & 0x7] * torch.where(code >= 8, -1.0, 1.0)
        factor = torch.exp2(s[r0:r0 + chunk].float() - 127.0)
        out[r0:r0 + chunk] = (value.reshape(value.shape[0], -1, GROUP) *
                              factor.unsqueeze(-1)).reshape(value.shape)
    return out


def load_output_weight(gguf_path: str, name: str = "output.weight") -> torch.Tensor:
    """Q8_0 (or any GGUF type) weight -> f32 [N, K] using gguf's exact dequantizer."""
    from gguf import GGUFReader, quants
    r = GGUFReader(gguf_path)
    t = next(t for t in r.tensors if t.name == name)
    n, k = int(t.shape[1]), int(t.shape[0])
    dense = quants.dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(dense)).reshape(n, k).float()


def torch_logits(x: torch.Tensor, pk: torch.Tensor, sc: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """x [M,K] f32 @ dequant(pk,sc).T per row, computed in row chunks."""
    x = x.float()
    out = torch.empty((x.shape[0], pk.shape[0]), dtype=torch.float32)
    for r0 in range(0, pk.shape[0], chunk):
        dq = dequantize_mxfp4(pk[r0:r0 + chunk], sc[r0:r0 + chunk])
        out[:, r0:r0 + chunk] = x @ dq.T
    return out
