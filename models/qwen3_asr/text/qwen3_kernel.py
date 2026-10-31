"""One config-driven whole-depth Qwen3 dense text kernel family.

The same forward records a 256-row prefill graph and a one-row decode graph.
Both use four cores, runtime layer banks indexed by ``ax.dsl.range`` and the
same paged BF16 KV cache.  M-RoPE is selected from the model configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import axnn.dsl as ax
from axnn.dsl import parallel as axd

from sgl_kernel_axera.attention.flash_attn import (
    ATTENTION_WEIGHT_NAMES,
    decode_paged_kv_kernel,
    flash_attn_with_kvcache,
)
from sgl_kernel_axera.models.qwen38_27b.mlp import mlp_kernel
from sgl_kernel_axera.models.weights import block_arguments, block_layer, block_stages


PREFILL_ROWS = 256
DECODE_ROWS = 1
CTX = 4096
PAGES = CTX // PREFILL_ROWS
MX_BLOCK = 32


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    mrope_section: tuple[int, int, int] | None = None
    mrope_interleaved: bool = False
    context: int = CTX

    @property
    def has_mrope(self) -> bool:
        return self.mrope_section is not None

    @property
    def pages(self) -> int:
        return self.context // PREFILL_ROWS

    def as_dict(self) -> dict:
        value = asdict(self)
        value["mrope_section"] = list(self.mrope_section) if self.mrope_section else None
        value["pages"] = self.pages
        return value


MODEL_PATHS = {
    "embedding_0.6b": Path("/data/junda/work/qsmall/embedding/hf/Qwen3-0.6B/config.json"),
    "asr_0.6b": Path("hf/Qwen3-ASR-0.6B/config.json"),
    "asr_1.7b": Path("hf/Qwen3-ASR-1.7B/config.json"),
    "tts_0.6b": Path("hf/Qwen3-TTS-12Hz-0.6B-Base/config.json"),
    "tts_1.7b": Path("hf/Qwen3-TTS-12Hz-1.7B-Base/config.json"),
}


def _rope_fields(config: dict):
    scaling = config.get("rope_scaling") or config.get("rope_parameters") or {}
    section = scaling.get("mrope_section")
    interleaved = bool(scaling.get("mrope_interleaved", scaling.get("interleaved", False)))
    theta = float(config.get("rope_theta", scaling.get("rope_theta", 1_000_000.0)))
    return theta, tuple(section) if section else None, interleaved


def load_spec(name: str, *, config_path: str | Path | None = None) -> ModelSpec:
    path = Path(config_path) if config_path else MODEL_PATHS[name]
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    import json

    root = json.loads(path.read_text())
    if name.startswith("asr_"):
        config = root["thinker_config"]["text_config"]
    elif name.startswith("tts_"):
        config = root["talker_config"]
    else:
        config = root
    theta, section, interleaved = _rope_fields(config)
    spec = ModelSpec(
        name=name,
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        num_layers=int(config["num_hidden_layers"]),
        num_q_heads=int(config["num_attention_heads"]),
        num_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]),
        rope_theta=theta,
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        mrope_section=section,
        mrope_interleaved=interleaved,
        context=CTX,
    )
    assert spec.hidden_size in (1024, 2048)
    assert spec.num_q_heads == 16 and spec.num_kv_heads == 8 and spec.head_dim == 128
    assert spec.num_layers % 4 == 0
    if spec.has_mrope:
        assert sum(spec.mrope_section) * 2 == spec.head_dim
    return spec


def make_cache(spec: ModelSpec, layers: int) -> torch.Tensor:
    return torch.zeros(
        (layers, spec.pages, 2, PREFILL_ROWS, spec.num_kv_heads, spec.head_dim),
        dtype=torch.bfloat16,
    )


def make_rope_tables(spec: ModelSpec):
    half = spec.head_dim // 2
    angles = torch.arange(spec.context, dtype=torch.float32)[:, None] / (
        spec.rope_theta
        ** (torch.arange(0, spec.head_dim, 2, dtype=torch.float32) / spec.head_dim)[None, :]
    )
    assert angles.shape[-1] == half
    return angles.cos().contiguous(), angles.sin().contiguous()


@torch.no_grad()
def forward_qwen3_blocks(
    x,
    position_ids,
    indptr,
    indices,
    last_page_len,
    cache_location,
    output,
    final_norm,
    layout,
    mrope_section,
    valid_tokens=None,
    **weights,
):
    """Run all decoder layers and the final RMSNorm in one AXModel.

    ``x`` is [256,H] for prefill or [1,H] for decode.  Prefill masks padded
    output rows with ``valid_tokens``; decode returns its one final row.  Both
    modes read and write the same [layers,pages,2,256,8,128] BF16 cache.
    """
    input_banks = {value.name: value for value in weights.values() if hasattr(value, "name")}
    stages = block_stages(layout, weights)
    rows, hidden_dim = x.shape
    assert rows in (DECODE_ROWS, PREFILL_ROWS)
    assert hidden_dim == stages[0][2]["w_norm"].shape[-1]
    assert output.shape == x.shape
    assert (valid_tokens is not None) == (rows == PREFILL_ROWS)

    blocks = stages[0][2]["w_norm"].shape[0]
    core = ax.program_id(0)
    cores = ax.num_programs(0)
    assert cores == 4, "the fused Qwen3 forward requires four cores"
    width = hidden_dim // cores
    owned = slice(core * width, (core + 1) * width)

    hidden = ax.dsl.load(x[:, owned])
    zero = ax.dsl.zeros((1,), ax.int32)
    one = ax.dsl.full((1,), ax.int32, 1)
    stop = ax.dsl.full((1,), ax.int32, blocks)
    has_mrope = position_ids.ndim == 2

    def body(block_index, hidden):
        current = ax.dsl.create_tensor(
            x.shape, x.dtype, ax.MemoryType.SHARED_OCM, name="block_input"
        )
        ax.dsl.move(hidden, r=current[:, owned])
        axd.pub(current)
        axd.sub(current)

        for phase, stage in enumerate(stages):
            (kv, attention_weights, mlp_weights), banks = block_layer(
                stage, block_index, input_banks
            )
            activation = ax.dsl.create_tensor(
                x.shape,
                x.dtype,
                ax.MemoryType.SHARED_OCM,
                name=f"attention_handoff_{phase}",
            )
            options = dict(
                attn_output_gate=False,
                partial_rotary_factor=attention_weights["partial_rotary_factor"],
                mrope_section=mrope_section,
                mrope_interleaved=attention_weights.get("mrope_interleaved", False),
                has_mrope=has_mrope,
                cache_location=cache_location,
                banks=banks,
                cos_sin_cache=attention_weights.get("cos_sin_cache"),
            )
            if rows == DECODE_ROWS:
                decode_paged_kv_kernel.func(
                    current.reshape(1, 1, hidden_dim),
                    *(attention_weights[name] for name in ATTENTION_WEIGHT_NAMES[:17]),
                    position_ids,
                    *(attention_weights[name] for name in ATTENTION_WEIGHT_NAMES[17:]),
                    None,
                    None,
                    activation.reshape(1, 1, hidden_dim),
                    kv,
                    indptr,
                    indices,
                    last_page_len,
                    PREFILL_ROWS,
                    **options,
                )
            else:
                args = (
                    current.reshape(1, rows, hidden_dim),
                    *(attention_weights[name] for name in ATTENTION_WEIGHT_NAMES[:17]),
                    position_ids,
                    *(attention_weights[name] for name in ATTENTION_WEIGHT_NAMES[17:]),
                )
                flash_attn_with_kvcache.func(
                    *args,
                    None,
                    activation.reshape(1, rows, hidden_dim),
                    kv,
                    indptr,
                    indices,
                    None,
                    **options,
                )
            axd.pub(activation)
            axd.sub(activation)

            mlp_kernel.func(
                activation.reshape(1, rows, hidden_dim),
                residual=None,
                output=activation.reshape(1, rows, hidden_dim),
                tile_size=[rows],
                residual_buffer=hidden,
                banks=banks,
                **mlp_weights,
            )
            axd.pub(activation)
            axd.sub(activation)
            current = activation

        ax.dsl.move(current[:, owned], r=hidden)
        return (hidden,)

    (result,) = ax.dsl.range(zero, zero, stop, one, body, hidden, unroll_factor=1)

    square_sum_shared = ax.dsl.create_tensor(
        (cores, rows, 1), ax.fp32, ax.MemoryType.SHARED_OCM, name="final_rms_squares"
    )
    ax.dsl.sum(
        ax.dsl.mul(result, result, output_dtype=ax.fp32),
        dim=-1,
        keepdim=True,
        output_dtype=ax.fp32,
        r=square_sum_shared[core : core + 1].reshape(rows, 1),
    )
    axd.pub(square_sum_shared)
    axd.sub(square_sum_shared)
    square_sum = square_sum_shared[0:1].reshape(rows, 1)
    for owner in range(1, cores):
        square_sum = ax.dsl.add(
            square_sum,
            square_sum_shared[owner : owner + 1].reshape(rows, 1),
        )
    rms = ax.dsl.sqrt(
        ax.dsl.add(
            ax.dsl.div(
                square_sum,
                ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=hidden_dim),
                output_dtype=ax.fp32,
            ),
            ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=1e-6),
            output_dtype=ax.fp32,
        ),
        output_dtype=ax.fp32,
    )
    inverse_rms = ax.dsl.reciprocal(rms, output_dtype=ax.fp32)
    normalized = ax.dsl.mul(result, inverse_rms, output_dtype=ax.bf16)
    final_weight = ax.dsl.load(final_norm[owned])
    weighted = ax.dsl.mul(normalized, final_weight, output_dtype=ax.bf16)

    count = (
        ax.dsl.full((1,), ax.int32, 1)
        if rows == DECODE_ROWS
        else ax.dsl.load(valid_tokens)
    )
    positions = ax.dsl.from_numpy(np.arange(rows, dtype=np.int32))
    keep = ax.dsl.minimum(
        ax.dsl.maximum(ax.dsl.sub(count, positions), ax.dsl.zeros((1,), ax.int32)),
        ax.dsl.full((1,), ax.int32, 1),
    )
    masked = ax.dsl.mul(
        weighted,
        ax.dsl.cast(keep.reshape(rows, 1), dtype=ax.bf16),
        output_dtype=ax.bf16,
    )
    ax.dsl.store(masked, r=output[:, owned])
    return output


@torch.no_grad()
def prepare(
    spec: ModelSpec,
    layers,
    *,
    final_norm,
    cache_path,
    run_mode=None,
    weight_banks=(),
):
    """Bind one set of layer banks and record the prefill and decode graphs."""
    from qwen3_fidelity import install as install_fidelity

    install_fidelity()
    assert len(layers) % 4 == 0
    assert all(kv is not None for kv, _, _ in layers)
    layout, weights = block_arguments(layers, "mlp", weight_banks)
    cache_path = Path(cache_path)
    mrope_section = spec.mrope_section or (0, 0, 0)
    kernels = {
        rows: ax.builder(
            run_mode=ax.RunMode(run_mode or "CRUN"),
            const_inputs=["layout", "mrope_section"],
            enable_cache=True,
            unique_axmodel=True,
            cache_path=cache_path / f"prefill{rows}",
            dump_path=cache_path / f"prefill{rows}" / "artifacts",
        )(forward_qwen3_blocks)
        for rows in (DECODE_ROWS, PREFILL_ROWS)
    }
    return kernels, dict(
        layout=layout,
        mrope_section=tuple(mrope_section),
        final_norm=final_norm,
        **weights,
    )


def quantize_mxfp8(weight, *, block_size: int = MX_BLOCK):
    if weight.ndim != 2 or weight.shape[1] % block_size:
        raise ValueError("MXFP8 requires a matrix with K divisible by 32")
    values = weight.detach().cpu().float()
    n, k = values.shape
    blocks = values.reshape(n, k // block_size, block_size)
    amax = blocks.abs().amax(dim=-1)
    target = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    exponent = torch.ceil(torch.log2(target))
    encoded = (exponent + 127).clamp(1, 254).to(torch.uint8)
    scale_value = torch.pow(2.0, encoded.float() - 127.0)
    scaled = blocks / scale_value.unsqueeze(-1)
    codes = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(n, k)
    scale = encoded.view(torch.float8_e8m0fnu)
    return codes.contiguous(), scale.contiguous()


def dequantize_mxfp8(codes, scale, *, block_size: int = MX_BLOCK):
    enc = scale.contiguous().view(torch.uint8).float()
    decode = torch.pow(2.0, enc - 127.0)
    return codes.float() * decode.repeat_interleave(block_size, dim=-1)


def mxfp8_projection(weight):
    codes, scale = quantize_mxfp8(weight)
    return codes, scale, torch.ones(1, dtype=torch.float32), torch.zeros(
        1, dtype=torch.bfloat16
    )


def make_causal_mask(tokens: int, dtype=torch.float32):
    return torch.full((1, 1, tokens, tokens), float("-inf"), dtype=dtype).triu(1)
