# Qwen3-TTS codec decoder (code2wav) as one AX8860 kernel

Status: PYRUN numerics and CRUN compile+simulation pass.  One window variant (72
code frames -> 138240 samples at 24 kHz), one kernel, 4 NPU cores, all decoder
layers inside.  Weights are BF16 runtime inputs.  The final compiled axmodel is
`out_decoder/cache/decoder.298ac4dff2ddfbb72cea1a20e65e3cc60eacbbedfdeff2338af050fa45a7d0e6.compiled.axmodel`
(11,974,141 bytes, sha256 `f97614c501be0c62f3d1a014cfcc428ced9c2aa976e59fe9aa5c6a638a12986c`);
the full 4-core cmodel simulation completed with rc=0.

## 1. Environment and assets

- Toolchain: `source /data/junda/work/axdsl_env.sh` (npu-codebase `7913a0599`, the
  commit used by the delivered Qwen3-TTS kernels).  CUDA/GPU is not used.
  `RUN_MODE=PYRUN` for the numeric check, `RUN_MODE=CRUN` for the compile
  (the axmodel is written before the simulator starts).
- Model: ModelScope/HF `Qwen/Qwen3-TTS-Tokenizer-12Hz`, downloaded to `hf/`
  (`hf/model.safetensors`, 682 MB; `hf/config.json`).  The decoder is identical for
  Qwen3-TTS 0.6B and 1.7B; the frozen llama.cpp kernels are byte-identical, so one
  kernel serves both.
- Reference code: `ref_src/qwen_tts/.../modeling_qwen3_tts_tokenizer_v2.py`
  (Apache-2.0, copied from the model repo).  Two compatibility edits only, no math
  changes: bypass `check_model_inputs`, build the RoPE `inv_freq` directly, and use
  a direct 72-wide banded mask instead of the block-diagonal mask helper
  (transformers 5.12 incompatibilities).
- `artifacts/reference.pt` / `reference.json`: real speech
  (`/data/junda/work/qsmall/asrenc/audio/asr_en.wav`, resampled to 24 kHz) encoded
  with the tokenizer encoder on CPU -> `codes [148,16]`; decoded fully and with
  `chunked_decode(chunk=72, left=71)` using the same BF16 weights.  Full vs
  chunked: cosine `0.9999784`, SNR `42.268 dB`.  This is the streaming contract
  the kernel implements (KV prefix 71 frames, causal-conv tails, ConvTranspose
  col2im tails).

## 2. Kernel contract

One call (`decoder(...)`, 50 inputs, 1 output; read it from
`out_decoder/inputs.json`, which lists names/shapes/dtypes in kernel order):

- codes `[72,16]` int32, window index/position `pos [1]` int32,
  three flat BF16 weight banks (`quant_w`, `up_w`, `dac_w`, 223.43 MiB total),
  state inputs, state outputs, and `wav [138240]` float32.
- Window 0 starts from all-zero state and `pos=0`; later windows pass `pos += 72`
  and the returned `*_next` buffers (the caller swaps state/next in place).
- State: transformer KV prefix `[8,2,1024,71]` (layer, K/V, channel, 71);
  causal-conv left context: `pre_conv [2,512]`, up0/up1 dw `[6,1024]`, DAC entry
  `[6,1024]`, residual units `[6*d, C]` with dilations `d=1,3,9` for each of the
  4 DAC blocks, post conv `[6,96]`; ConvTranspose raw col2im tails
  `[8,768]`, `[5,384]`, `[4,192]`, `[3,96]`.
- Output: `72*1920 = 138240` samples, clamped to [-1,1] like the reference;
  float32.

Core layout (`decode_window`, `c = ax.program_id(0)`, 4 cores):

- The whole prefix (RVQ decode, causal pre-conv, 8-layer sliding-window transformer,
  output norm/projection, two 2x ConvNeXt upsample blocks) is replicated on every
  core because it is small and every core needs a prefix of the 4x-upsampled
  latent for its DAC quarter.  Prefix state is written channel-sharded
  (`c` owns 1/4 of channels / heads) so the four cores together produce the whole
  next state.
- DAC is time-sharded: core `c` computes the causal prefix `y[0 : 72*(c+1)]` of the
  288-row up1 latent and runs DAC entry + 4 blocks + post conv + final conv only
  up to that row; it writes output quarter `[c*34560, (c+1)*34560)`.  Only the last
  core reaches the end of the window, so only it writes DAC tails/residual/post
  state.  The four output quarters are disjoint.
- Internal DDR workspace: `create_tensor(3 * max_elem)` BF16 per core
  (`max_elem = 34560*(c+1)*96`; ~79.6 MB on core 3), used by the DAC time tiles.
  All intermediate tensors live inside the kernel; there are no host ops between
  layers.
- Tiling: causal convs are IM2COL-in-registers (`concat` of shifted rows) + GEMM
  with FP32 accumulation, one BF16 rounding per output.  The banks keep the
  checkpoint orientation `[OC,K*IC]`, so GEMMs use `matmul(..., transposed=True)`
  and never materialize a transposed weight tile (a 3.5 MB tile did not fit OCM).
  Per-layer transformer tensors and per-phase ConvTranspose weights are loaded
  one slice at a time (`_load_idx`) for the same reason.  ConvTranspose1d is split
  into `S` phase GEMMs and re-interleaved (`S` = 8,5,4,3), with the raw col2im
  tail carried between calls.  SnakeBeta, LayerNorm/RMSNorm, RoPE and attention
  follow the HF BF16 operation order (`codec_ref.py` / the reference module).
- All `div`s use `high_precision=True` (the AX8860 hardware divider).  The default
  reciprocal-LUT lowering materializes an FP32 `ty_inv` stream that the cmodel
  leaves pending in this large graph (`inner_check_output_residual`,
  `acc_stream_length=0x1e600`).  The hardware divider removes that stream and is
  at least as accurate.
- The attention mask is computed as `cold + warm*(band-cold)` instead of
  `ax.dsl.where`.  `AxWhere` hits the same cmodel residual assertion on the
  `[72,143]` mask even with a single consumer; the arithmetic form and a
  host-mask variant both simulate cleanly (`probe_where.py`).  This is a cmodel
  lowering workaround; the selected mask values are unchanged.
- Weights are looked up from the flat runtime banks by the generated
  `weight_layout.py` (offset+shape per tensor).  Banking and the 4-core
  `program_id`/state pattern follow the finished embedding pilot
  (`/data/junda/work/qsmall/embedding/emb_kernel.py`); the decoder's convs are
  hand-written DSL, so no sgl-kernel call is used.

## 3. Weight banks and GGUF provenance

The llama.cpp adapter (`ref/llama/qwen3_tts/qwen3_tts.cpp`) loads the GGUF produced
by `llama.cpp-axera/conversion/qwen3tts.py`; tensor names are from
`gguf-py/gguf/constants.py` (lines 2006-2038).  `pack_gguf.py` now builds all
three banks directly from `gguf/mmproj-qwen3-tts-0.6b-bf16.gguf` (raw BF16 in
`artifacts/gguf_banks/`); `pack_weights.py` remains the HF comparison path.
`GGUF_LAYOUT.md` has the per-tensor source/type/ne[]/index mapping for the C++
port.  Conv orientation in the bank is `[OC, K*IC]` (tap-major);
ConvTranspose is `[S, OC, J*IC]`, `J=2` because every transposed conv has
`K=2S`.

| bank tensor | GGUF tensor(s) | notes |
|---|---|---|
| `q_sem_embed`, `q_sem_usage` | `a.gen.wav.quant.first.codebook` | `q_sem_embed` is the GGUF's folded `embedding_sum/cluster_usage`; the kernel uses it directly and `q_sem_usage` is a constant-1 placeholder for layout compatibility |
| `q_sem_out` | `a.gen.wav.quant.first.out_proj` | |
| `q_ac_embed`, `q_ac_usage` | `a.gen.wav.quant.rest.codebook` | 15 folded codebooks, `[15,2048,256]`; `q_ac_usage` is a constant-1 placeholder |
| `q_ac_out` | `a.gen.wav.quant.rest.out_proj` | |
| `pre_conv_w/b` | `a.gen.wav.pre_conv.weight/.bias` | |
| `tfm_in_w/b`, `tfm_out_w/b`, `tfm_norm` | `a.gen.wav.tfm.in_proj`, `.out_proj`, `.output_norm` | |
| `tfm_{ln1,q,k,v,o,attn_scale,ln2,gate,up,down,ffn_scale}` | `a.gen.wav.tfm.blk.{0..7}.{ln1,attn_q,attn_k,attn_v,attn_out,ls1,ln2,ffn_gate,ffn_up,ffn_down,ls2}` | stacked `[8,...]` |
| `up{0,1}_tc_w/b` | `a.gen.wav.up.blk.{0,1}.conv.weight/.bias` | causal ConvTranspose1d, split by phase |
| `up{0,1}_dw_taps`, `up{0,1}_dw_b` | `a.gen.wav.up.blk.{0,1}.dwconv.weight/.bias` | `dwconv.weight` is `[C,1,7]`; stored as materialized `[7,T,C]` (`T=144,288`) because AX8860 `fma_float` cannot broadcast `[1,C]` against `[T,C]` |
| `up{0,1}_norm_w/b` | `a.gen.wav.up.blk.{0,1}.norm.weight/.bias` | |
| `up{0,1}_pw1_w/b` | `a.gen.wav.up.blk.{0,1}.pw1.weight/.bias` | |
| `up{0,1}_pw2_w/b` | `a.gen.wav.up.blk.{0,1}.pw2.weight/.bias` | |
| `up{0,1}_gamma` | `a.gen.wav.up.blk.{0,1}.gamma` | |
| `dac_entry_w/b` | `a.gen.wav.dac.entry.weight/.bias` | causal conv k=7, 1024->1536 |
| `dac{b}_snake_a/b` | `a.gen.wav.dac.blk.{b}.snake.alpha/.beta` | GGUF already stores `exp(alpha)` and `1/(exp(beta)+1e-9)`; the kernel consumes both directly |
| `dac{b}_tc_w/b` | `a.gen.wav.dac.blk.{b}.conv.weight/.bias` | ConvTranspose, `S=8,5,4,3` |
| `dac{b}_r{r}_a1/b1` | `a.gen.wav.dac.blk.{b}.res.{r}.act1.alpha/.beta` | `r=0..2`, dilations 1,3,9; GGUF values already folded |
| `dac{b}_r{r}_c1w/c1b` | `a.gen.wav.dac.blk.{b}.res.{r}.conv1.weight/.bias` | k=7 dilated causal conv |
| `dac{b}_r{r}_a2/b2` | `a.gen.wav.dac.blk.{b}.res.{r}.act2.alpha/.beta` | GGUF values already folded |
| `dac{b}_r{r}_c2w/c2b` | `a.gen.wav.dac.blk.{b}.res.{r}.conv2.weight/.bias` | k=1 + residual add |
| `post_snake_a/b` | `a.gen.wav.dac.post_snake.alpha/.beta` | GGUF values already folded |
| `post_conv_w/b` | `a.gen.wav.dac.post_conv.weight/.bias` | k=7, 96->1, clamp |

The GGUF converter forces some conv kernels to F16/F32 for ggml's conv paths;
the bank is BF16 for all of them.  BF16/F32 GGUF sources match the BF16 HF path
exactly after the cast; only the 29 F16-stored conv tensors differ (see C2 below).
`quant.*.in_proj` exists in the GGUF but the decode path does not use it (it
belongs to encode), so it is not in the banks.

The remaining kernel inputs are not GGUF tensors: `codes` are the tokenizer
encoder's RVQ indices (or embeddings), `pos` is the window position, the
`*_state`/`*_tail` buffers are the streaming state carried between calls, and
`wav` is the output.  `build_decoder.py` writes their exact order/shape/dtype to
`out_decoder/inputs.json`.

## 4. Checks (PYRUN vs PyTorch BF16, same packed weights)

Real codes from the encoder, chunked contract from the reference.  All cosine
values are per row unless noted.

- Reference streaming sanity: full vs chunked cosine `0.9999784`, SNR `42.268 dB`.
- Transformer per-layer (all 8, full depth; final-source log
  `debug_layers_final.log`):

| layer | cos mean | cos min |
|---|---|---|
| 0 | 0.9999939 | 0.9999899 |
| 1 | 0.9999915 | 0.9999881 |
| 2 | 0.9999890 | 0.9999822 |
| 3 | 0.9999854 | 0.9999722 |
| 4 | 0.9999830 | 0.9999729 |
| 5 | 0.9999799 | 0.9999706 |
| 6 | 0.9999782 | 0.9999701 |
| 7 | 0.9999769 | 0.9999327 |

- Prefix stages (`debug_stages.py`, final-source log `debug_stages_final.log`;
  per-time-row cosine over the 72-frame window):
  quantizer `1.0000004` mean (`0.9999999` min), pre-conv `1.0000000`
  (`0.9999998`), transformer `0.9999790` (`0.9999205`), up0 `0.9999592`
  (`0.9987997`), up1 `0.9999480` (`0.9986911`).  The upsampler rows have a few
  large-activation BF16 rounding differences (up1 absmax `1.22` at row 128) but
  the waveform-level error is small (below).
- Waveform, streaming two windows (`check_decoder.py --windows 2`, final log
  `check_decoder_final3.log`, after the divider/mask lowering workarounds):
  window 0 vs chunked cosine `0.999824762`, SNR
  `34.480 dB`, absmax `0.02637`; vs full `0.999826789`, `34.533 dB`.  Window 1 vs
  chunked cosine `0.999923766`, SNR `38.042 dB`, absmax `0.01758`; vs full
  `0.999926209`, `38.171 dB`.  Window 1 used window 0's returned state, so
  streaming continuity across two windows is verified.  The real-code decoder
  reference itself is full-vs-chunked cosine `0.9999784` / `42.268 dB`, so the
  kernel error is dominated by BF16 accumulation, not the streaming contract.

## 5. Build

- `build_decoder.py OUT` uses random weights with the real packed shapes, writes
  `OUT/inputs.json`, and calls `kernel[4](*args)`.  `check_*.py` are separate and
  use the real weights/codes.
- `RUN_MODE=CRUN nice python3 build_decoder.py out_decoder` (log
  `build_decoder_mask.log`).  Final model:
  `out_decoder/cache/decoder.298ac4dff2ddfbb72cea1a20e65e3cc60eacbbedfdeff2338af050fa45a7d0e6.compiled.axmodel`
  (11,974,141 bytes, sha256
  `f97614c501be0c62f3d1a014cfcc428ced9c2aa976e59fe9aa5c6a638a12986c`).  The run
  reached `ran`/`axmodels:` with rc=0; the full 4-core cmodel simulation took
  ~21.6 min (59m24s total build, 16.8 GB peak host RSS).

## 6. Code predictor (Task C remark)

`ref/qwen3_tts/src/4acb7f6/q3tts_cp_frame.py` is one kernel per frame (one call
emits the frame's 16 codes and logits), so it already meets the one-kernel rule.
Inside that kernel, however, the 16 positions and `NL` predictor layers are
statically unrolled (`for p in range(NP)` and `for j in range(NL)`, lines 98-102),
with per-layer packs indexed by `Qg.members(packs, j)`; the predictor layers are
not a runtime `ax.dsl.range` loop.

## 7. Open issues / deviations

- Two AX8860 cmodel lowering faults had to be avoided, not just tuned around: the
  default reciprocal-LUT `div` leaves an unconsumed FP32 `ty_inv` stream, and
  `AxWhere` leaves the mask output stream pending.  Both are reproduced in
  `probe_div16.py` / `probe_where.py`; the kernel uses the hardware divider and
  an arithmetic mask, with identical math.  A toolchain fix would allow the
  original formulations.
- Only the 72-frame window variant is implemented.  Other chunk sizes would need a
  different depthwise-tap materialization (`T=2*window`, `4*window`) and mask.
- Transformer layers and the 12 DAC residual units are statically unrolled inside
  the single kernel, not `ax.dsl.range` over a `[L,...]` weight bank.  This is why
  the graph is ~10.9k ops and CRUN is slow.  A runtime loop is possible for the
  residual units (same shapes, different dilation/state bank) but not required for
  the one-kernel contract; the transformer's per-layer KV state and static K/V
  slicing made a runtime loop riskier.
- `config.decoder_config.semantic_codebook_size` is 4096, but the checkpoint's
  first codebook tensor is `[2048,256]` and observed codes are <2048.  We use the
  actual tensor shape.
- GGUF's conv kernels are F16/F32 while this kernel consumes BF16; parity is
  against a GGUF-equivalent BF16 PyTorch reference (same F16 tensors and folded
  codebooks/snakes), not bit-exact against the ggml host path.
- PYRUN peak host RSS ~42 GB (many NumPy intermediates over four cores); CRUN
  workspace is the DDR allocation above.  The 3rd streaming window was not run
  (the 2-window check already exercises state continuity).


## 8. Task C2: GGUF bank adapter

- `pack_gguf.py` reads `gguf/mmproj-qwen3-tts-0.6b-bf16.gguf` with
  `PYTHONPATH=/data/junda/work/llama.cpp/gguf-py`, writes
  `artifacts/gguf_banks/{quant_w,up_w,dac_w}.bin` (raw little-endian BF16), the
  md5s, a per-tensor manifest, and keeps the exact `weight_layout.py` offsets.
  `GGUF_LAYOUT.md` documents all 164 bank tensors.
- The kernel now consumes the GGUF's folded forms directly: no codebook usage
  division and no SnakeBeta `exp`/reciprocal.  The `*_usage` tensors remain in
  the bank as constant 1.0 placeholders so its byte layout stays unchanged.
- GGUF-vs-HF bank parity (`check_gguf_banks.py`, log `check_gguf_banks.log`):
  135/164 tensors are identical; the 29 differing tensors are exactly the
  F16-stored convs.  Min cos `0.999998781`, max abs diff `0.015625`
  (`up{0,1}_dw_taps`).  Bank md5s:
  `490681368dfb0592842a92da6fe446ec` quant_w,
  `b5b7dc2ef8cb0045f3860d66add7242a` up_w,
  `e5981fc8b5d47395b423b8b38d99e9c1` dac_w.
- Folded codebooks change the PyTorch waveform relative to the old HF reference
  (GGUF-equivalent reference vs old chunked: window 0 cos `0.999566`, window 1
  `0.999800`; this is a storage floor, independent of the kernel).  The
  apples-to-apples reference is therefore `make_gguf_reference.py`, which
  substitutes the same F16 tensors and folded codebooks/snakes; its own
  chunked-vs-full cos is `0.999951839`.
- Final two-window kernel check
  (`check_decoder.py --windows 2`, which now defaults to the GGUF banks and
  GGUF reference; log `check_gguf_decoder_ref.log`):
  window 0 vs chunked cos `0.999828935`, SNR `34.573 dB`, absmax `0.03809`;
  window 1 vs chunked cos `0.999907613`, SNR `37.161 dB`, absmax `0.01334`.
  Against the old HF reference the same kernel output is cos `0.999639`
  (window 0), as expected from the folded-codebook floor.
- CRUN with the changed kernel compiled
  `out_decoder_gguf/cache/decoder.6f6f14347cc48474abfe6d90b2a9367b8e81e7f1b69fc1c2b734db4a0f6af952.compiled.axmodel`
  (11,068,533 bytes, sha256
  `51ffa2f5c57fb579e3377d3a3a6961be6949bf2ab04b0700d43951dc046f5759`);
  the run was stopped at simulator startup as requested.


## 9. Task C3: CRUN index_select lowering fix

- The board's wrong waveform was reproduced bit-for-bit by CRUN: the old
  window-0 output md5 was `54f48d333a85e7adada8b784f82ede2c`, identical to
  `artifacts/board_window0_wav.bin`, with cos `-0.002952041` / SNR `-0.486 dB`
  vs the GGUF chunked reference.  This was a compiler-lowering bug, not a
  board-only DDR race.
- `_quant_decode()` passed strided column views of `codes [72,16]` to
  `index_select`.  CRUN lowered those indices incorrectly: frame 0 and every
  8th frame selected the right code, while most others selected index 0.  The
  corruption started in `quant_decode` at frame 1 and propagated through the
  whole decoder.
- The fix transposes the code slab first and gathers from contiguous
  `[72]` row vectors (`ct[0:1, :].reshape(72)` for semantic and
  `ct[g+1:g+2, :].reshape(72)` for acoustic).  Isolated CRUN-vs-PYRUN after the
  fix: semantic embedding absmax error `0`, semantic/acoustic outputs within
  `0.0625` (rounding level), cos effectively `1.0`.
- Rebuilt model:
  `out_decoder_gguf/cache/decoder.3b9c207f26e2647f4ae9b5b35a2a7ce8f17de4cc61a7367c61ffbc70491670ba.compiled.axmodel`
  (11,067,413 bytes, md5 `e1621e034eb114d65da9d4b830198c69`, sha256
  `c20e843fbda96be6f19f46c96ed3b67ee7baea7f68e7ca84fe26c9738905c8a9`).
  The `decoder()` signature, input names/shapes/dtypes in `inputs.json`, and
  all state buffers are unchanged.
- CRUN two-window check (`check_crun_fixed.log`, saved outputs
  `artifacts/fixed_gguf_window{0,1}.bin`), vs GGUF chunked reference:
  window 0 cos `0.999827623`, SNR `34.528 dB`, absmax `0.03516`;
  window 1 cos `0.999906957`, SNR `37.140 dB`, absmax `0.02026`.
  Window 0 also matches PYRUN (`0.999828935` / `34.573 dB`) to rounding.
