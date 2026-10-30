# Qwen3-Embedding-0.6B

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| decode group: 28 layers + final norm (the hidden state), c16384 | `4acb7f6` | `src/4acb7f6/q35_group.py` | ai-dev1 lineC/jobs/jkeep.txt `kq06_grp_16384` (KEEP_SHM=1 DEC_SPLIT=1 LMAX=16384 G=28 LMHEAD=1); byte-identical rebuild not run |
| attention prefill P = 128 / 64 / 16 / 4, c16384 | `64f9071` | `src/64f9071/q35_layer_pk.py` | ai-dev1 lineC/jobs/j70.txt `q06_p<P>_16384` (ONLY=attn PCHUNK=<P> LMAX=16384); **rebuilt byte-identical (P=4) 2026-10-30** |

Manifests (the contract with ggml-axera): `qwen3_embedding_0.6b.manifest.json`.
