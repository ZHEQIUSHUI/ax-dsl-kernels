# Qwen3.5-9B

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| decode group: 32 layers + MXFP4 output head, c16384 | `ab3e90c` | `src/ab3e90c/q35_group.py` | 4090 lineA/kref2_go.sh `9g_16384` (DEC_SPLIT=1 LMAX=16384 G=32 LMHEAD=1, WFMT4=1) |
| GDN prefill P = 128 / 64 / 16 / 4 with a valid-row count (_pv) | `6f9b9b5` | `src/6f9b9b5/q35_layer_pk.py` | 4090 lineA/k9pv_go.sh (PAD=1 PRJ_ROWS=256 PRJ_KSPLIT=4096) |
| attention prefill P = 128 / 64 / 16 / 4 | `3f74c45` | `src/3f74c45/q35_layer_pk.py` | 4090 lineA/kpref3_go.sh |

Manifests (the contract with ggml-axera): `qwen35_9b.manifest.json`.
