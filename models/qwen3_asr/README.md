# Qwen3-ASR-0.6B / 1.7B

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| text model decode group: 28 layers + output head (MXFP8), c4096 | `4acb7f6` | `src/4acb7f6/q35_group.py` | ai-dev1 lineC/jobs/jkeep.txt `kq06_grp_4096` / `kq17_grp_4096` |
| text model attention prefill P = 128 / 64 / 16 / 4, c4096; output head mul_mat_mx8 | `64f9071` | `src/64f9071/q35_layer_pk.py`, `src/64f9071/q35_layer_mc.py` | ai-dev1 lineC/jobs/j702.txt (`q06_*` / `q17_*`, ONLY=attn / ONLY=lm) |
| audio encoder: conv front-end fe / fe2 / fe3, layers, projector (4 cores) | `127cc3a` | `src/127cc3a/q3a_fe.py`, `src/127cc3a/q3a_layer.py`, `src/127cc3a/q3a_proj.py` | branch wip/asr-enc (fe 15008e9, layer 83448fe, proj 127cc3a); originals in ai-dev1 lineC/k/{fe,fe2,alayer,aproj}; see PROVENANCE.md (only some files byte-verified) |
| audio encoder conv_out mm_bf16_m{13n}_n{H}_k7680 | `64f9071` | `src/64f9071/mm_kernels.py` | **source not established** (mm_kernels.py: the naming-compatible generator); see PROVENANCE.md |

Manifests (the contract with ggml-axera): `qwen3_asr_0.6b.manifest.json`, `qwen3_asr_1.7b.manifest.json`.
