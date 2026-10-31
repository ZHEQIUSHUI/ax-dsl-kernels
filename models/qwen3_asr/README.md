# Qwen3-ASR-0.6B / 1.7B

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| text model decode group: 28 layers + output head (MXFP8), c4096 | `4acb7f6` | `src/4acb7f6/q35_group.py` | ai-dev1 lineC/jobs/jkeep.txt `kq06_grp_4096` / `kq17_grp_4096` |
| text model attention prefill P = 128 / 64 / 16 / 4, c4096; output head mul_mat_mx8 | `64f9071` | `src/64f9071/q35_layer_pk.py`, `src/64f9071/q35_layer_mc.py` | ai-dev1 lineC/jobs/j702.txt (`q06_*` / `q17_*`, ONLY=attn / ONLY=lm) |
| audio encoder: ONE kernel per model (`qwen3a_fused_<size>_h<H>_npu4`, role `audio.enc`): up to 8 chunks of 100 mel frames per call (the padding chunks hidden by the runtime `kmask`); conv front-end (im2col on chip), conv_out, positions, all 18 / 24 layers (runtime loop over weight banks), post norm, projector | this commit | `encoder/asr_kernel.py` | `encoder/build_asr.py --tags 0.6b,1.7b` (npu-codebase cad0db3d6 + sgl-kernel-axera, `sgl_env.sh`); checks `encoder/check_asr.py` (full depth vs the official qwen-asr tower, real speech, 1 / 3 / 5 / 8 chunks: cos min >= 0.9998); design and GGUF mapping in `encoder/NOTES.md` |

Manifests (the contract with ggml-axera): `qwen3_asr_0.6b.manifest.json`, `qwen3_asr_1.7b.manifest.json`.

Board (.184, SDK V0.5.1, 2026-10-31): the encoder of a 60 s recording (8 calls) 0.6B 255 -> 106 ms, 1.7B 296 -> 145 ms against the former fe / layer / proj kernels and host im2col (40 kernels -> 2); transcripts of the gate recordings (incl. a 27 s one whose last call pads 3 chunks to 8) identical to the former path, except the 0.6B 60 s one (one punctuation mark, one word). The former encoder kernels (`src/127cc3a`, PROVENANCE.md) are no longer in the manifests.
