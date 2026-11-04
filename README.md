# ax-dsl-kernels

NPU kernels for the AX8860 (4 cores) written in AXDSL (the axnn kernel DSL), one directory per model, and the tools
around them. The kernels take every weight as a runtime input: llama.cpp's ggml-axera backend
([ZHEQIUSHUI/llama.cpp](https://github.com/ZHEQIUSHUI/llama.cpp), `ggml/src/ggml-axera`) converts the GGUF's weights
at load and runs a kernel directory through the model adapter its `manifest.json` names.

| directory | model | ggml-axera adapter |
|---|---|---|
| `models/qwen38_27b` | Qwen3.8-27B: whole-model prefill kernel, 16-layer decode group, 16k / 128k / 256k | `qwen38_27b` |
| `models/qwen35_35b_a3b` | Qwen3.5-35B-A3B (MoE) | `qwen35_moe` |
| `models/qwen35_9b` | Qwen3.5-9B: whole-model prefill kernel, 16-layer decode group, 16k / 128k / 256k | `qwen35_whole` |
| `models/qwen35_2b` | Qwen3.5-2B: whole-model prefill kernel, 12-layer decode group, 16k / 128k / 256k | `qwen35_whole` |
| `models/qwen35_vit` | the Qwen3.5 vision encoder (27B, 35B-A3B, 9B, 2B), 4 cores | `qwen35_vit` |
| `models/qwen3_embedding` | Qwen3-Embedding-0.6B | `qwen3` |
| `models/qwen3_asr` | Qwen3-ASR-0.6B / 1.7B (text model + audio encoder) | `qwen3`, `qwen3_asr` |
| `models/qwen3_tts` | Qwen3-TTS-0.6B / 1.7B (talker, code predictor, codec decoder) | `qwen3`, `qwen3_tts` |
| `tools/manifest` | `mkmanifest.py`: a kernel directory's `manifest.json` from its kernels' IO tables and layouts | |
| `tools/board` | board programs: `axrun` (one kernel with inputs by name, timed), `iojson` (a kernel's IO table), multi-core tests | |

A model directory holds everything its kernels came from: the sources (`src/<commit>/` for the frozen models, the
commits that built the delivered files; a model in development keeps its sources directly), how they were built,
the manifests and `kernels.md5`. Models share no code: a change to one model's kernels does not touch another's.

## A new model

1. `models/<model>/` with its kernel sources (copy what you start from; no imports from another model's directory),
   `cfg/config.json` (the shapes; builds take random weights of them), `build.sh` (every kernel of the model, the
   toolchain pinned), the README (kernels, roles, layouts, checks).
2. Read the kernels' IO tables on the board (`tools/board/iojson`) and write the manifest
   (`tools/manifest/mkmanifest.py <model> <io.json> <layout dir>`); the kernels' FP4 / E8M0 IO declared u8 for the
   board runtime (`board_io.py` of the model).
3. ggml-axera: a new adapter `models/<model>/` there for a new model name in the manifest.

## Environment

Builds need the npu-codebase checkout a model's README names (axnn), its venv and `script/npu_dev`; board programs the
AX8860 BSP and the Arm GNU toolchain 13.2 (`make BSP=... CROSS=...`).
