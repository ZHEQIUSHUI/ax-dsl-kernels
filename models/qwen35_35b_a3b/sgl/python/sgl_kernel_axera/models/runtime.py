"""Framework-independent state shared by inference mechanisms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class SamplingBuffers:
    """Model-owned projection and double-buffered token output storage.

    Results retain this same record until sampling completes. Each queue entry
    pairs a token buffer with its word view; rotation reuses both without copies.
    """

    projection_output: torch.Tensor
    logits: torch.Tensor
    outputs: deque[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class InferenceRuntime:
    inference: Any
    weight_banks: dict = field(default_factory=dict)
    layers: list = field(default_factory=list)
    workspace: Any = None
    kernel: Any = None
    gdn_states: tuple = ()
    lm_norm_weight: Any = None
    logits: SamplingBuffers | None = None
    decode_sequences: dict = field(default_factory=dict)
