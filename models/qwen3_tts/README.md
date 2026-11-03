# Qwen3-TTS-0.6B / 1.7B

Four kernels per model (the ggml-axera adapters `models/qwen3` for the talker, `models/qwen3_tts` for the code
predictor and the codec decoder); weights are runtime inputs, converted from the GGUFs at load.

| kernel | role | source | build |
|---|---|---|---|
| `qwen3_tts_<size>_prefill256_mx8_c4096_npu4` | `prefill` | `text/qwen3_kernel.py`: the whole talker per page of 256 cells (28 layers in a runtime loop over MXFP8 weight banks, paged bf16 KV `[28,16,2,256,8,128]`), unit-weight final norm | `text/compile_variant.sh tts_<size>` (npu-codebase cad0db3d6, `sgl_env.sh`), then `decode/board_io.py` |
| `qwen3_group28_attn_mx8_lm8_paged_c4096_npu4_tts_<size>` (+ `.layout`) | `decode` | `decode/q35_group.py`: all 28 talker layers + final norm + the codec head (3072 rows, MXFP8) for one token, on the prefill kernel's paged KV (KVX). `BANKS=1`: `w_q/k/v`, `w_gate/up` and their E8M0 scales read the prefill banks (`k_b<phase>_*`); `w_o`, `w_down`, their scales, norms and the head stay in packs | `BANKS=1 decode/build_group.py output --model tts_<size> --layers 28` (npu-codebase 7913a0599, `axdsl_env.sh`, `RUN_MODE=CRUN`), then `decode/board_io.py` and `tools/manifest/update_decode_banks.py` |
| `qwen3tts_cp_frame5_mx8_h<H>_krt_c16_npu4` (+ `.layout`) | `cp.frame` | `cp/cp_kernel.py`: one frame of the code predictor (5 layers x 15 positions, the 15 codebook heads, on-chip top-k sampling with a runtime `top_k` <= 50), MXFP8 packs | `RUN_MODE=CRUN cp/build_cp.py <out> --model <size> --top-k 50` (7913a0599, `axdsl_env.sh`); layer modules in `cp/` (from `src/4acb7f6`) |
| `qwen3tts_codec_w72_npu4` | `gen.codec` | `codec/decoder_kernel.py`: the codec decoder (code2wav) for a window of 72 frames in one call: quantizer, pre-transformer, upsampling, the DAC decoder; the 0.6B and 1.7B files are the same | `codec/build_decoder.py` (7913a0599, `axdsl_env.sh`); weight banks packed by the adapter as `codec/pack_gguf.py` |

The adapter runs a prompt's last token through the decode kernel (its logits); the prompt's other rows' codec-head
logits (unused by the generation) on the CPU, so no separate output head kernel.

Checks: `text/check_qwen3.py` (prefill + paged history vs transformers), `decode/check_group.py` (28 layers vs
transformers), `decode/check_banks.py` (bank bytes / shapes, pack bounds, page-boundary indices), `cp/check_cp.py` (vs the former frame kernels: codes identical for top-k 1 and 50), `codec/check_decoder.py`
(vs the reference decoder). `kernels.md5` per model.

The talker decode group reads the prefill talker banks. The code predictor is a separate checkpoint submodel
(`a.gen.code.*`, `cp` params) and the codec is separate; neither duplicates talker layers.

Board (.209, SDK V0.5.1, 2026-11-01), against the former 6 kernels per model: speech transcribed back exactly
(greedy and sampled, zh / en); code predictor 9.1 -> 7.4 ms/frame (0.6B), 9.8 -> 7.8 ms (1.7B); talker with its head
4.3 -> 3.7 ms (0.6B), 8.8 -> 8.0 ms (1.7B).

The decode group's softmax reduces the 1 + 256 columns of a block only (no trailing sentinel): with it, an all very
negative score block gave 0 / 0 (TTS-1.7B: +inf at the first decode call).

The former kernels' sources (`src/<commit>/`, PROVENANCE.md) are no longer in the manifests.

Manifests (the contract with ggml-axera): `qwen3_tts_0.6b.manifest.json`, `qwen3_tts_1.7b.manifest.json`.
