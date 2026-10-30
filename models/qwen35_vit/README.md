# Qwen3.5 vision encoder (27B, 35B-A3B)

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| 4-core ViT: pre / embed / attn / post / merge for 256 / 1024 / 2048 / 3072 patches | `4bf12f3` | `src/4bf12f3/q35_vit.py` | recipes/build_vit_4core_27b.sh, build_vit_4core_35b.sh (head dim 72, h1152) |
