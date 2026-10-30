# Qwen3.8-27B (AX8860, 4 cores)

The kernels of the llama.cpp ggml-axera adapter `qwen38_27b` (`ggml/src/ggml-axera/models/qwen38_27b`). Weights are
runtime inputs: the adapter converts them from the GGUF at load (MXFP4 GGUF rows as they are; other types, e.g. a
community Q4_K_M / Q8_0, quantized to MXFP4).

| file | role | source |
|---|---|---|
| `qwen38_27b_prefill256_mx4_kvc_c16384_npu4` | `prefill` | `prefill/build27.py`: sgl-kernel-axera `models/qwen38_27b/forward.py` (npu-codebase cad0db3d6) at the 27B shapes, MXFP4 weights. The whole model in one call per 256 cells: a runtime loop over the 16 blocks (3 delta-net + 1 attention layers, dense MLP), weights in banks `[16, ...]` indexed by the block, valid rows at runtime |
| `qwen38_27b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4` (+ `.layout`) | `decode.group` | `decode/q35_group.py` (ax-dsl-kernels ab3e90c + the prefill kernel's state layouts): 16 layers per call, 4 calls per token, weights in packs |
| `mul_mat_mx4_n248320_k5120_npu4` | `lm_head` | delivered directory (unchanged) |
| `qwen35_vit_*_h1152_npu4` (20) | `vit.*` | `qwen3_5/recipes/build_vit_4core_27b.sh` (unchanged) |

State layouts shared by prefill and decode (the adapter's CMM pools): KV cache paged `[16 layers][64 pages][K, V][256][4 x 256]`
bf16; delta-net state `[48 layers][48 heads][128][128]` f32; conv state q / k `[48][3][2048]`, v `[48][3][6144]` bf16,
oldest row first; value heads in HF (grouped) order. The decode group writes the new K / V row given by the host
(`pn = cell, blocks, page * 512 + offset`).

Build: `NPU_DECODE=<npu-codebase@7913a0599> NPU_PREFILL=<npu-codebase@cad0db3d6> ./build.sh <out>`, then the kernel
directory = the two kernels (+ layout), the output head and the ViT kernels, `manifest.json` (from
`tools/manifest/mkmanifest.py qwen38_27b <io.json> <layout dir>`; io.json: the kernels' IO tables read on the board).

Checks: `prefill/check27.py` runs block 0 of the prefill kernel (PYRUN) with weights converted from the GGUF as the
adapter does against transformers' layers (cos 0.99995 with the GGUF's weights); `decode/q35_group.py` compares the
group with the single-core layer kernels in the simulator.

Board (.184, SDK V0.5.1, 2026-10-30): short completion identical to the delivered path; 3908-token prompt 3.3-3.6 s
(delivered 4.76 s), decode 13.7-14.1 tok/s (delivered 13.4). The prefill kernel's results vary slightly between runs
(NMC accumulation order on the board), within the quality tolerance.

Reproducibility: the AXDSL compiler is deterministic for a fixed source and toolchain; `kernels.md5` holds the files of
the build above.
