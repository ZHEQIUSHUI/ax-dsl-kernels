"""Model-specific MoE kernel implementations."""

from .m_dsv4 import (
    deepseek_moe,
    ffn_hash_prefill,
    ffn_hash_static_prefill,
    ffn_routed_dispatch_prefill,
    ffn_routed_prefill,
    ffn_routed_static_prefill,
    ffn_routed_weighted_sum_prefill,
    ffn_shared_expert_prefill,
)
from .m_qwen3 import qwen3_moe

__all__ = [
    "deepseek_moe",
    "ffn_hash_prefill",
    "ffn_hash_static_prefill",
    "ffn_routed_dispatch_prefill",
    "ffn_routed_prefill",
    "ffn_routed_static_prefill",
    "ffn_routed_weighted_sum_prefill",
    "ffn_shared_expert_prefill",
    "qwen3_moe",
]
