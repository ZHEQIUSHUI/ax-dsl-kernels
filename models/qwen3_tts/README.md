# Qwen3-TTS-0.6B / 1.7B

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| talker decode group: 28 layers, c4096 | `4acb7f6` | `src/4acb7f6/q35_group.py` | ai-dev1 lineC/jobs/jkeep.txt `kt06_grp` / `kt17_grp` |
| talker attention prefill, output head | `64f9071` | `src/64f9071/q35_layer_pk.py`, `src/64f9071/q35_layer_mc.py` | ai-dev1 lineC/jobs/j70t.txt |
| code predictor: one frame per call (top-k 1 / 50) | `4acb7f6` | `src/4acb7f6/q3tts_cp_frame.py` | ai-dev1 lineC/jobs/jkeep.txt `kcf06` `kcf17` `kcf06k1` `kcf17k1` |
| codec decoder: conv1d_bf16_* (plain) | `64f9071` | `src/64f9071/conv1d_kernels.py` | shape lists in PROVENANCE.md (few files byte-verified) |
| codec decoder: conv1d_bf16_*_b / _br (bias, residual epilogues) | `0805ce0` | `src/0805ce0/conv1d_kernels.py` | EPI=b / br; see PROVENANCE.md |
| codec decoder GEMMs mm_bf16_* | `64f9071` | `src/64f9071/mm_kernels.py` | 46 shapes in PROVENANCE.md (2 byte-verified) |

Manifests (the contract with ggml-axera): `qwen3_tts_0.6b.manifest.json`, `qwen3_tts_1.7b.manifest.json`.
