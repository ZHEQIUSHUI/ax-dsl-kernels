# Qwen3-Embedding-0.6B

ONE kernel: the whole model per page of 256 tokens (`qwen3_embedding_0.6b_prefill256_mx8_c4096_npu4`, role
`prefill`): 28 layers in a runtime loop over weight banks `[7, ...]` per phase (MXFP8, from the GGUF at load), paged
bf16 KV cache (context 4096), the valid rows at runtime, an RMS norm of unit weight at the end (ggml-axera applies the
model's final norm before the pooling). The ggml-axera adapter is `models/qwen3` (shared by the Qwen3 dense text models).

| kernel | source | build |
|---|---|---|
| `qwen3_embedding_0.6b_prefill256_mx8_c4096_npu4` | `text/qwen3_kernel.py` (on sgl-kernel-axera: `flash_attn_with_kvcache`, `mlp_kernel`, the bank machinery) | `text/compile_variant.sh embedding_0.6b` (npu-codebase cad0db3d6 + sgl-kernel-axera, `sgl_env.sh`), then `board_io.py` (E8M0 IO as u8); `text/collect_inputs.py` writes the IO |

Checks: `text/check_qwen3.py` (the kernel in PYRUN against transformers' layers with the same MXFP8 weights, paged
history). Board (.184, SDK V0.5.1, 2026-10-31), against the former 52 kernels: embeddings cos 0.9991-0.9997; 6 short
inputs 72 -> 62 ms, 2 long ones (884 tokens) 170 -> 69 ms.

The former kernels' sources (`src/<commit>/`) are no longer in the manifest.

Manifest (the contract with ggml-axera): `qwen3_embedding_0.6b.manifest.json`.
