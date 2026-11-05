"""Dynamic inference implementations for hybrid Transformer models.

Each mechanism provides forward(model, metadata, num_tokens).
Implementations also provide prepare(layers, ...). Selection happens before loading
weights so parameters can be bound into contiguous weight banks.

qwen38_27b fuses each four-layer block for decode and complete 256-token prefill chunks.
qwen36_35b_a3b fuses decode blocks and uses host-chunked attention followed by
a 4096-row MoE kernel for prefill. Both require repeating blocks of three GDN
layers and one GQA layer.
"""

from importlib import import_module


def select_inference(layers, *, is_moe):
    block_pattern = (
        bool(layers)
        and len(layers) % 4 == 0
        and all(hasattr(layer, "linear_attn") == (index % 4 != 3) for index, layer in enumerate(layers))
    )
    if not block_pattern:
        raise ValueError("dynamic inference requires repeating blocks of three GDN layers and one GQA layer")
    name = "qwen36_35b_a3b" if is_moe else "qwen38_27b"
    return import_module(f"{__name__}.{name}.forward")
