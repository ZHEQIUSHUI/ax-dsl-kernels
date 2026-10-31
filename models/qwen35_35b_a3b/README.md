# Qwen3.5-35B-A3B

Frozen: the kernels of the delivered directory (`kernels.md5`, per manifest). The sources are those of the commits that built them (`src/<commit>/`, the scripts and the modules they import); toolchain: npu-codebase 7913a0599 (ai-dev1) / the 4090's of 2026-10-23. Builds take random weights of the config's shapes (QW_RANDOM=1); the backend binds the GGUF's weights at load.

| kernels | commit | entry | how they were built |
|---|---|---|---|
| decode group: 8 layers (MXFP8, routed experts MXFP4), c16384 | `70b0fbc` | `src/70b0fbc/q35_moe_group.py` | line B, x35f directory (.250); see the delivered kernels/MANIFEST.md |
| prefill: GDN / attention per chunk (P 256 with the router's top-K, 128, 64, 16, 4), expert slots, combine | `70b0fbc` | `src/70b0fbc/q35_moe_prefill.py`, `src/70b0fbc/q35_layer_mc.py` | line B, x35f |
| ViT `qwen35_35b_a3b_vit3072_npu4` (role `vit`): the whole vision encoder in one call (27 layers unrolled, post norm, 2 x 2 merger, projector width 2048), up to 3072 patches, bf16 weights stacked `[27, ...]`, tiles QB 512 / BC 256 | this commit | `vit/vit_full.py` (on `vit/q35_vit.py`) | `QB=512 BC=256 RUN_MODE=CRUN N=3072 GH=48 GW=64 MODEL=<HF Qwen3.5-35B-A3B> python vit/vit_full.py` (npu-codebase 7913a0599), stop once compiled; check `RUN_MODE=PYRUN N=256 VIT_L=27`: layer 26 cos 0.99969, output 0.99956 vs transformers |
| expert slots G128 (slots 8 / 16 / 32) | `3c013ec` | `src/3c013ec/q35_moe_prefill.py` | SLOTS=32/16/8 SLOT_G=128 ONLY=slots_px (.250 /data/k/x35h) |

Manifests (the contract with ggml-axera): `qwen35_35b_a3b.manifest.json`.

Board (.184, SDK V0.5.1, 2026-10-31): image (bus.jpg, 3072 patches) ViT 273 ms in one call + 6 ms cpu (the former 20 kernels: 337 ms + 55 ms cpu), the same description.
