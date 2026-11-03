# Qwen3-ASR-0.6B / 1.7B

Three kernels per model (the ggml-axera adapters `models/qwen3_asr` for the audio encoder, `models/qwen3` for the
text model); weights are runtime inputs, converted from the GGUFs at load.

| kernel | role | source | build |
|---|---|---|---|
| `qwen3a_fused_<size>_h<H>_npu4` | `audio.enc` | `encoder/asr_kernel.py`: the whole audio encoder per call of up to 8 chunks of 100 mel frames (conv front-end with on-chip im2col, conv_out, positions, all 18 / 24 layers in a runtime loop over weight banks, post norm, projector; padding chunks hidden by the runtime `kmask`), bf16 | `encoder/build_asr.py --tags 0.6b,1.7b` (npu-codebase cad0db3d6 + sgl-kernel-axera, `sgl_env.sh`) |
| `qwen3_asr_<size>_prefill256_mx8_c4096_npu4` | `prefill` | `text/qwen3_kernel.py`: the whole text model per page of 256 cells (28 layers in a runtime loop over MXFP8 weight banks, paged bf16 KV `[28,16,2,256,8,128]`, interleaved M-RoPE), unit-weight final norm | `text/compile_variant.sh asr_<size>` (cad0db3d6, `sgl_env.sh`), then `decode/board_io.py` (E8M0 IO as u8) |
| `qwen3_group28_attn_mx8_lm8_paged_c4096_npu4_asr_<size>` (+ `.layout`) | `decode` | `decode/q35_group.py`: all 28 layers + final norm + the output head (151936 rows, MXFP8) for one token, on the prefill kernel's paged KV (the current token's K / V exchanged through the shared OCM: KVX). `BANKS=1`: `w_q/k/v`, `w_gate/up` and their E8M0 scales read the prefill banks (`k_b<phase>_*`); `w_o`, `w_down`, their scales, norms and the head stay in packs | `BANKS=1 decode/build_group.py output --model asr_<size> --layers 28` (npu-codebase 7913a0599, `axdsl_env.sh`, `RUN_MODE=CRUN`), then `decode/board_io.py` and `tools/manifest/update_decode_banks.py` |

The adapter runs a prompt's last token through the decode kernel (its logits), so no separate output head kernel.

Checks: `encoder/check_asr.py` (full depth vs the official qwen-asr tower, real speech, 1 / 3 / 5 / 8 chunks: cos
min >= 0.9998), `text/check_qwen3.py` (prefill + paged history vs transformers), `decode/check_group.py` (the group
vs transformers, 28 layers), `decode/check_banks.py` (every `k_b*` input equals its prefill bank's bytes / shape; all
packed members in bounds; page-boundary positions in range). `kernels.md5`: a fresh build of the decode groups from
these sources is byte-identical.

Board (.184, SDK V0.5.1, 2026-11-01), against the former kernels (60+ per model): transcripts of the gate recordings
identical; the 60 s recording end to end on llama-server 0.6B 1.7 s, 1.7B 2.5 s (former 1.7B: 3.6 s); 1.7B decode
12.0 -> 9.9 ms/token, the 60 s recording's prompt 1391 -> 303 ms, the encoder 296 -> 145 ms.

The decode group's softmax reduces the 1 + 256 columns of a block only (no trailing sentinel; 2026-11-01): with it,
an all very negative score block gave 0 / 0 (seen on TTS-1.7B). Gate transcripts unchanged.

The former kernels' sources (`src/<commit>/`, PROVENANCE.md) are no longer in the manifests.

Manifests (the contract with ggml-axera): `qwen3_asr_0.6b.manifest.json`, `qwen3_asr_1.7b.manifest.json`.
