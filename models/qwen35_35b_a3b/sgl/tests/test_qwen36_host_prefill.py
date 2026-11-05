"""Fixed 4096-row MoE prefill retains the independent-kernel numerical contract."""

import os
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from runtime_cases import axera_state_types, load_framework_inputs
from sgl_kernel_axera.models.runtime import InferenceRuntime
from test_qwen35_runtime import clone_layers, layer_case, moe_weights

import axnn.dsl as ax

AxeraModelState = axera_state_types()["AxeraModelState"]


@pytest.fixture
def padding_kernel(tmp_path, monkeypatch):
    module = import_module("sgl_kernel_axera.models.qwen36_35b_a3b.forward")
    kernel = ax.builder(
        run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")),
        cache_path=tmp_path / "cache",
        dump_path=tmp_path / "artifacts",
    )(module.clear_prefill_padding.func)
    monkeypatch.setattr(module, "clear_prefill_padding", kernel)
    return kernel


@pytest.mark.parametrize("kind", ["gdn", "gqa"])
def test_host_prefill_matches_standard_kernels_with_fixed_mlp_input(kind, padding_kernel):
    torch.manual_seed(364096)
    torch.set_num_threads(4)
    ax.init(chip_num=1)
    rows, hidden = 256, 256
    kv, attention, _ = layer_case(kind)
    # The four-core shared expert uses one complete 64-channel tile per core.
    # Compare each attention kind from identical inputs: the small and large
    # MoE kernels round BF16 sums in different orders, affecting later layers.
    layers = [(kv, attention, moe_weights(hidden, experts=16, intermediate=256))]
    workspace = load_framework_inputs().LayerWorkspace
    reference = InferenceRuntime(
        import_module("sgl_kernel_axera.models.standard.forward"),
        layers=clone_layers(layers),
        workspace=workspace(rows, hidden, dtype=torch.bfloat16, device="cpu"),
    )
    actual = InferenceRuntime(
        import_module("sgl_kernel_axera.models.qwen36_35b_a3b.forward"),
        layers=layers,
        workspace=workspace(4096, hidden, dtype=torch.bfloat16, device="cpu"),
    )
    values = (torch.randn(rows, hidden) * 0.03).bfloat16()
    reference.workspace.hidden[:rows].copy_(values)
    actual.workspace.hidden[:rows].copy_(values)
    reference.workspace.attention.zero_()
    # Stale padded activations must be cleared before each 4096-row MoE call.
    actual.workspace.attention.fill_(7)
    metadata = dict(
        kernel_tokens=rows,
        prefix=0,
        positions=torch.arange(rows, dtype=torch.int32),
        page=dict(
            indptr=torch.tensor([0, 0], dtype=torch.int32),
            indices=torch.arange(3, 35, dtype=torch.int32),
            last_page_len=torch.tensor([rows], dtype=torch.int32),
        ),
        cache_location=torch.tensor([3, 0], dtype=torch.int32),
        valid_tokens=torch.tensor([rows], dtype=torch.int32),
        num_tokens=torch.tensor([rows], dtype=torch.int32),
    )
    config = SimpleNamespace(model_type="qwen3_5_moe_text", rope_parameters={"mrope_section": (2, 3, 3)})
    reference.inference.forward(
        SimpleNamespace(config=config, axera_state=AxeraModelState(runtime=reference)),
        metadata,
        rows,
    )
    actual.inference.forward(
        SimpleNamespace(config=config, axera_state=AxeraModelState(runtime=actual)),
        metadata,
        rows,
    )
    torch.testing.assert_close(
        actual.workspace.hidden[:rows].float(), reference.workspace.hidden[:rows].float(), rtol=0.03, atol=0.015
    )
    assert torch.isfinite(actual.workspace.hidden[:rows]).all()
    for (kv, attention, _), (expected_kv, expected_attention, _) in zip(actual.layers, reference.layers, strict=True):
        if kv is None:
            for name in ("state_in", "q_conv_state_in", "k_conv_state_in", "v_conv_state_in"):
                torch.testing.assert_close(attention[name], expected_attention[name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(kv.float(), expected_kv.float(), rtol=0.03, atol=0.003)
    assert torch.count_nonzero(actual.workspace.attention[rows:4096]) == 0


@pytest.mark.parametrize("cores", [4])
def test_prefill_padding_clear_uses_runtime_length_and_preserves_live_values(cores, padding_kernel):
    clear_prefill_padding = padding_kernel

    ax.init(chip_num=1)
    torch.manual_seed(364097)
    source = torch.randn(4096, 256).bfloat16()
    # Clearing uses stores of zero, so stale NaNs must not survive masking.
    source[255::256] = float("nan")
    count = torch.empty(1, dtype=torch.int32)
    keys = set()
    for tokens in (1, 255, 256, 257, 4096):
        output = source.clone()
        count.fill_(tokens)
        clear_prefill_padding[cores, cores](output, count)
        if keys and clear_prefill_padding.run_mode != ax.RunMode.PYRUN:
            assert clear_prefill_padding._cache_hit
        keys.add(clear_prefill_padding.gen_cache_key())
        torch.testing.assert_close(output[:tokens], source[:tokens], rtol=0, atol=0, equal_nan=True)
        assert output[tokens:].eq(0).all()
    assert len(keys) == 1
