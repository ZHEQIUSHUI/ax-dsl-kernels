# Qwen3 dense decode group on the prefill kernel's paged BF16 KV

## Scope

One AXModel call per decoded token for the whole Qwen3 dense text stack (28
layers inside one kernel, four NPU cores), reading and writing exactly the
prefill kernel's KV cache (`qwen3_kernel.py`, `ROWS=256`, `b0_kv`
`[28,16,2,256,8,128]` BF16).  Products: `asr_0.6b`, `asr_1.7b`, `tts_0.6b`,
`tts_1.7b`.  All four contain their MXFP8 output head (`lm8`): the ASR
151936-row text head or the TTS 3072-row codec head (GGUF `output.weight`).

The starting point is the frozen former group kernel
`ref/qwen3_asr/src/4acb7f6/q35_group.py` (28 attention layers + `lm8` head in
one call).  Its private `kc`/`vc` per-layer KV buffers are replaced by the
prefill kernel's single paged BF16 cache, ported from the 27B change
(`/data/junda/work/q27m/src`, `be6410c` vs its base): `KVX` (KV heads split
over the cores), `DEC_SPLIT` (the scaled queries exchanged with K/V through
shared OCM) and `pn = [cell, nblk, krow]`.  One kernel still covers all 28
layers; there are no per-layer kernels and no host ops between layers.

Board follow-up (E2): AX8860 can return stale DDR when one core reads a row
another core wrote earlier in the same call.  The current token's K/V therefore
travel from the projection cores through `SHARED_OCM`; attention scans only DDR
history (the current cell is removed from the scan mask inside the kernel) and
gets the current cell's score/value from that shared copy.  The DDR row is still
written for the next call.  The final-normalised row also reaches `out` and the
output head through a shared-OCM row rather than `hbuf`; only core 0 touches
`hbuf`, for its own PYRUN materialisation.  The ASR IO names, shapes and dtypes
are unchanged; E3 adds the TTS `logits` output and head pack members.

Removing the current cell can leave a scanned page with no visible history.
The score block therefore carries a trailing zero sentinel, and the block max
includes it: an all-masked block then contributes `exp(0 - 0) = 0`, rather than
treating its `-3e38` lanes as `exp(0) = 1` and contaminating the softmax.

Environment used for every build and check: `source /data/junda/work/axdsl_env.sh`
(npu-codebase `7913a0599`), `nice`; no board or GPU was used.

## Files

| file | role |
|---|---|
| `q35_group.py` | the group body: packed-layer members, paged KV write, split attention, MLP, final norm and output head; `prepare_group()` |
| `q35_step.py` | HF config / weight reader, M-RoPE cos/sin, layer argument names, `proj` / `slice_w` |
| `q35_layer_mc.py` | reused four-core layer helpers (RMSNorm, mask, shared-OCM reductions) |
| `q35_layer_pk.py` | packed runtime weights (`pw` fp8, `ps` E8M0, `pf` fp32, optional `pb` bf16) and the layout writer |
| `q35_mx_build.py` | MXFP8 quantisation / dequantisation and the MX `linear` swap |
| `build_group.py` | random weights of the real shapes, real IO, `RUN_MODE=CRUN` compile and `inputs.json` |
| `check_group.py` | PYRUN vs Transformers, same quantised -> dequantised weights |

`build_*.py` and `check_*.py` are separate.  The former group's extra modules
(`q35_chat_build.py`, `board_io.py`) are kept because `q35_mx_build.py` imports
them.

## Kernel IO (kernel order)

All four products share the same names and order; only the packed sizes and
the `logits` row count differ.  `inputs.json` next to each `.axmodel` is
the authority.

| # | name | dtype | shape | meaning |
|---:|---|---|---|---|
| 1 | `x` | bfloat16 | `[1,1,H]` | input hidden row (`H=1024` for the 0.6B products, `2048` for 1.7B) |
| 2 | `pw` | float8_e4m3fn | `[PW]` | all layers' MXFP8 weight values, flat pack |
| 3 | `ps` | float8_e8m0fnu (exported as UINT8) | `[PS]` | all layers' MXFP8 scales, flat pack |
| 4 | `pf` | float32 | `[PF]` | all layers' norms (layer norms, q/k norms), flat pack |
| 5 | `cs` | bfloat16 | `[2,64]` | RoPE/M-RoPE cos and sin for one token (`ROT/2 = 64` angles) |
| 6 | `pn` | int32 | `[3]` | `[cell, nblk, krow]` (see the KV contract) |
| 7 | `mask` | bfloat16 | `[16,256]` | additive causal mask over flat cells |
| 8 | `kv` | bfloat16 | `[L,16,2,256,8,128]` | the prefill kernel's whole paged KV pool (`L=28`) |
| 9 | `fnorm` | bfloat16 | `[H]` | ASR final RMSNorm weight; TTS keeps it for ABI compatibility but uses packed `LM.fnorm` |
| 10 | `hbuf` | bfloat16 | `[1,1,H]` | core-0 DDR scratch for materialising the last layer row; not a cross-core handoff |
| 11 | `out` | bfloat16 | `[1,1,H]` | final RMSNorm row (identical to the prefill kernel's next-row output) |
| 12 | `logits` | float32 | `[1,V]` | `out @ dequant(head)^T` in four row blocks; `V=151936` (ASR `lm_head`) or `3072` (TTS `codec_head`) |

Core 0 materialises the last layer row through `hbuf`, computes the final norm,
and publishes the result to all cores through shared OCM; core 0 stores `out`
and every core uses the shared row for its output-head slice.  `hbuf` is scratch:
core 0 overwrites it before its own read, and no other core reads it.

### Pack layout and GGUF sources

The build writes `<axmodel>.layout`; each line is
`<layer>.<member> <pack> <byte offset> <bytes> [blk rows cols bytes]`.  The
host converts each GGUF tensor below and copies it into the named pack at the
offset.  Layer `N` is the unpermuted Qwen layer number.  ASR strips the
`thinker.` prefix; TTS strips `talker.`; both map `model.layers.N` to `blk.N`.

| pack member | GGUF tensor | conversion |
|---|---|---|
| `ln1` | `blk.N.attn_norm.weight` | tensor, bf16 |
| `w_q` / `s_q` | `blk.N.attn_q.weight` | MXFP8 values / E8M0 scales |
| `w_k` / `s_k` | `blk.N.attn_k.weight` | MXFP8 values / E8M0 scales |
| `w_v` / `s_v` | `blk.N.attn_v.weight` | MXFP8 values / E8M0 scales |
| `q_norm` | `blk.N.attn_q_norm.weight` | tensor, fp32 in `pf` |
| `k_norm` | `blk.N.attn_k_norm.weight` | tensor, fp32 in `pf` |
| `w_o` / `s_o` | `blk.N.attn_output.weight` | MXFP8 values / E8M0 scales |
| `ln2` | `blk.N.ffn_norm.weight` | tensor, bf16 |
| `w_gate` / `s_gate` | `blk.N.ffn_gate.weight` | MXFP8 values / E8M0 scales |
| `w_up` / `s_up` | `blk.N.ffn_up.weight` | MXFP8 values / E8M0 scales |
| `w_down` / `s_down` | `blk.N.ffn_down.weight` | MXFP8 values / E8M0 scales |
| `fnorm` | ASR `thinker.model.norm.weight` | plain runtime input |
| `LM.fnorm` | TTS `talker.model.norm.weight` | fp32 in `pf`, E3 only |
| `LM.w_lm` / `LM.s_lm` | `output.weight` (ASR `thinker.lm_head.weight`; TTS `talker.codec_head.weight`, 3072 rows) | MXFP8, all four products |

Both heads use the former `lm8` group's format: MXFP8 values in `pw`, E8M0
scales in `ps`, split over the four cores by output row.  For ASR the tied
embedding was verified equal to `lm_head.weight` before choosing
`output.weight`.  For TTS, the GGUF converter maps `talker.codec_head` to
`output.weight`; this kernel emits exactly the 3072 codec logits, without
llama.cpp's `-inf` text-vocabulary prefix.

The remaining kernel inputs have no GGUF weight: `x` is the previous token's
embedding row (host), `cs` is computed from the position and `rope_theta`,
`pn` / `mask` are host bookkeeping for the page layout, `kv` is the existing
paged cache (runtime state, not a checkpoint tensor), `hbuf` is a scratch row,
and `out` / `logits` are kernel outputs.  The prefill kernel and this group
exchange `kv` byte-for-byte.  The former manifests split the old decode group
from a separate `lm_head` (`mul_mat_mx8_n151936_k*` for ASR,
`mul_mat_mx8_n3072_k*` for TTS); these rebuilt groups fold both heads in.

## KV / pn contract

`kv` is one runtime input, exactly the prefill kernel's `b0_kv`:
`[layers, pages, 2, BC, kv_heads, head_dim] = [28,16,2,256,8,128]` BF16, with
layer `j`'s view `[page, k/v, 256, 8, 128]`.  `BC=256`, `LMAX=4096`, so 16
pages cover the whole context.  K and V of one cell live at the same page and
row in the two `k/v` halves.

The host keeps the absolute cell (token position) and passes:

```
cell = absolute token position (0-based)
pn   = int32[3] = [cell, nblk, krow]
nblk = cell // 256 + 1                      # populated pages, including this cell's
krow = (cell // 256) * 2 * 256 + cell % 256  # flat row in the layer's [2*256, 8*128] view
```

The group writes K at `kv[j, cell//256, 0, cell%256]` and V at
`kv[j, cell//256, 1, cell%256]` for all eight KV heads at the same flat rows
`krow` and `krow + 256`, for the next call.  It does **not** use that DDR
read-back for the current attention: every core already has the whole current
K/V row from the `SHARED_OCM` exchange.  The kernel copies `mask` and replaces
the current cell's entry with `-3e38`, scans pages `0 .. nblk-1` without the
current cell, then merges the current score/value from the shared row into the
online softmax.  The host still supplies `mask` (`bf16 [16,256]`, 0 for flat
column `<= cell`, `-3e38` elsewhere, or `-inf`), including 0 at the current
cell.  A zero read-back remains only as a local store fence and is not an
attention input.  This is byte-for-byte the layout the prefill kernel writes and
reads; only the group's own two indices (`pn[1]`, `pn[2]`) describe it.

`cs` is the single-token RoPE table: `[cos; sin]`, each `[1,64]`, built from
the model's `rope_theta` and the token's position.  ASR and TTS use the
interleaved M-RoPE of `qwen3_kernel.py` (`mrope_section=[24,20,20]`); for the
pure text/codec TTS sequence the three axes are equal.  Embedding (plain RoPE)
does not need a decode group.

## Numerics

`check_group.py` quantises each checkpoint projection to MXFP8 exactly as the
build, dequantises it, and loads exactly those weights into both the group
kernel and a Transformers `Qwen3DecoderLayer` reference.  It runs the group by
default with PYRUN and with `--run-mode CRUN` through the AX8860 simulator.  It
then:

1. fills a 256-row page from the reference prefix and decodes two tokens
   (`pos 256, 257`) through the group's kernel;
2. compares `out` per row with the reference's final-norm row and each written
   K/V row with the reference KV row at the same cell (cosine per row);
3. writes two rows from an empty cache and repeats the comparison.

The group's RMSNorm uses FP32 squares and rounds the normalised value to BF16
before the weight multiply, like `Qwen3RMSNorm`; SiLU is evaluated in FP32 and
the residual/MLP reductions accumulate the four cores' FP32 partials before
one BF16 cast.  Q is scaled by `1/sqrt(128)` before the QK matmul (the prefill
kernel's folded-scale convention).  Variants with BF16-normalised SiLU or
post-matmul scaling were measured and were worse against the Transformers
reference.

## Results

The E2 tables below use the E2 shared-OCM implementation; the E3 table adds the
TTS codec head.  The reference is Transformers `Qwen3DecoderLayer` loaded with
the same MXFP8-quantised, dequantised weights; `cos mean` and `cos min` are
per-row cosines (one row per check, so mean == min).

### Full-depth PYRUN (28 layers)

Prefix cells `0..255`, then three decode steps at the page boundary.  Results:
`out/e2_final_<product>/result.json`.

| product | rows | out cos | logits cos | KV K | KV V |
|---|---|---:|---:|---:|---:|
| `asr_0.6b` | pos 256 | 0.9997822 | 0.9998314 | 0.9999788 | 0.9997865 |
| `asr_0.6b` | pos 257 | 0.9999136 | 0.9999933 | 0.9999722 | 0.9996953 |
| `asr_0.6b` | pos 258 | 0.9999586 | 0.9999931 | 0.9999795 | 0.9998108 |
| `asr_1.7b` | pos 256 | 0.9998704 | 0.9998702 | 0.9999821 | 0.9998655 |
| `asr_1.7b` | pos 257 | 0.9999835 | 0.9999974 | 0.9999800 | 0.9998825 |
| `asr_1.7b` | pos 258 | 0.9999818 | 0.9999956 | 0.9999733 | 0.9997804 |
| `tts_0.6b` | pos 256 | 0.9992010 | - | 0.9998769 | 0.9996538 |
| `tts_0.6b` | pos 257 | 0.9980428 | - | 0.9997188 | 0.9992329 |
| `tts_0.6b` | pos 258 | 0.9988034 | - | 0.9997857 | 0.9994017 |
| `tts_1.7b` | pos 256 | 0.9992406 | - | 0.9999692 | 0.9997134 |
| `tts_1.7b` | pos 257 | 0.9992893 | - | 0.9999656 | 0.9997916 |
| `tts_1.7b` | pos 258 | 0.9985573 | - | 0.9999570 | 0.9996311 |

### CRUN simulator (4 layers)

Same page-boundary case compiled and executed by the AX8860 simulator.  Results:
`out/e2_crun_<product>_l4/result.json`.

| product | rows | out cos | logits cos | KV K | KV V |
|---|---|---:|---:|---:|---:|
| `asr_0.6b` | pos 256 | 0.9999623 | 0.9999444 | 0.9999970 | 0.9999596 |
| `asr_0.6b` | pos 257 | 0.9999104 | 0.9999669 | 0.9999936 | 0.9999232 |
| `asr_0.6b` | pos 258 | 0.9999169 | 0.9999604 | 0.9999969 | 0.9998982 |
| `asr_1.7b` | pos 256 | 0.9999746 | 0.9999648 | 0.9999957 | 0.9999604 |
| `asr_1.7b` | pos 257 | 0.9999590 | 0.9999700 | 0.9999976 | 0.9999480 |
| `asr_1.7b` | pos 258 | 0.9999405 | 0.9999537 | 0.9999962 | 0.9999168 |
| `tts_0.6b` | pos 256 | 0.9999062 | - | 0.9999675 | 0.9999369 |
| `tts_0.6b` | pos 257 | 0.9998497 | - | 0.9999421 | 0.9998964 |
| `tts_0.6b` | pos 258 | 0.9998988 | - | 0.9999560 | 0.9999046 |
| `tts_1.7b` | pos 256 | 0.9999337 | - | 0.9999962 | 0.9999543 |
| `tts_1.7b` | pos 257 | 0.9999026 | - | 0.9999948 | 0.9999080 |
| `tts_1.7b` | pos 258 | 0.9999048 | - | 0.9999938 | 0.9999117 |

All CRUN rows are above the 0.9995 mean target.

### E3 TTS codec head (CRUN, 4 layers)

Same page-boundary case, with the in-kernel 3072-row codec head and the real
`talker.model.norm.weight` / GGUF `output_norm.weight`.  Results:
`out/e3_crun_tts_<product>_l4/result.json`.

| product | rows | out cos | logits cos | KV K | KV V |
|---|---|---:|---:|---:|---:|
| `tts_0.6b` | pos 256 | 0.9999062 | 0.9998996 | 0.9999675 | 0.9999369 |
| `tts_0.6b` | pos 257 | 0.9998497 | 0.9998472 | 0.9999421 | 0.9998964 |
| `tts_0.6b` | pos 258 | 0.9998988 | 0.9999112 | 0.9999560 | 0.9999046 |
| `tts_1.7b` | pos 256 | 0.9999337 | 0.9999319 | 0.9999962 | 0.9999543 |
| `tts_1.7b` | pos 257 | 0.9999026 | 0.9998696 | 0.9999948 | 0.9999080 |
| `tts_1.7b` | pos 258 | 0.9999048 | 0.9998577 | 0.9999938 | 0.9999117 |

### Empty-cache sentinel check

The E2 empty-history check used `asr_0.6b`, 4 layers,
`--prefix 0 --positions 2`:
pos 0 `out` is 1.0000001, KV K/V are 0.9999979, logits are 1.0000010; pos 1
`out` is 0.9998658 and logits are 0.9999721.  Results:
`out/e2_pos0/result.json`.

### Acceptance and pre-E2 baseline

`asr_0.6b` and `asr_1.7b` pass the 0.9995 full-depth target on all three rows
(`out`, `logits`, KV).  All four products pass at 4 layers in both PYRUN and
CRUN.

The pre-E2 decode kernel (direct same-call DDR row readback, two rows) measured:

| product | rows | out cos | logits cos | KV K | KV V |
|---|---|---:|---:|---:|---:|
| `asr_0.6b` | pos 256 / 257 | 0.9997550 / 0.9999472 | 0.9997280 / 0.9999929 | 0.9999811 / 0.9999745 | 0.9998251 / 0.9997851 |
| `asr_1.7b` | pos 256 / 257 | 0.9998994 / 0.9999908 | 0.9999342 / 0.9999983 | 0.9999805 / 0.9999800 | 0.9998423 / 0.9998975 |
| `tts_0.6b` | pos 256 / 257 | 0.9983856 / 0.9986184 | - | 0.9998377 / 0.9998348 | 0.9995612 / 0.9994718 |
| `tts_1.7b` | pos 256 / 257 | 0.9994513 / 0.9994653 | - | 0.9999645 / 0.9999701 | 0.9997253 / 0.9997912 |

The E2 full-depth TTS-1.7B rows remain below the 0.9995 target at
0.99924 / 0.99929 / 0.99856 (the third row is a new, longer-chain case).
TTS-0.6B is 0.99920 / 0.99804 / 0.99880.  Their KV rows are above 0.9992,
except TTS-0.6B pos-258 V at 0.9994017.  This matches the pre-E2 behaviour: the
TTS products already failed full-depth while all 4-layer TTS checks passed.

The TTS gap is not caused by the KV layout or this decode group.  The
project's 256-row prefill kernel (`/data/junda/work/qsmall/qwen3/NOTES.md`)
reports 4-layer TTS-0.6B 0.9997914 / 0.9994870 and TTS-1.7B 0.9998587 /
0.9995275 against the same reference; this group's E2 CRUN TTS-0.6B rows are
0.9999062 / 0.9998497 / 0.9998988 and TTS-1.7B rows are
0.9999337 / 0.9999026 / 0.9999048.  A one-layer TTS-0.6B check is 0.9999807,
so the full-depth loss is accumulated BF16 rounding of the reference's own
28-layer chain, amplified by the TTS talker weights, not a one-layer formula
error.  BF16-rounded RoPE was also measured at 4 layers and did not improve both
boundary rows, so the current FP32 rope products were kept.

## E4: TTS output_norm contract fix (board)

The E3 kernel's final TTS RMSNorm took its weight from the runtime `fnorm`
input, which the E2 host supplies as a unit vector (the E2 contract is: the
kernel returns the unit-normalised row and the host applies `output_norm` in
f32 afterwards).  The E3 `out` and `logits` therefore lacked the real
`talker.model.norm.weight` scale.

Board proof, identical HF-prefix cases (cells 0..15 filled, decode at 16/17/18)
on 10.126.35.209: the old E3 kernel run with the host's unit `fnorm` reproduces
the E2 raw row exactly, and is far from the E2 host-normalised row:

| product | old E3 unit vs E2 raw | old E3 unit vs E2 * output_norm |
|---|---|---:|
| `tts_0.6b` pos 16/17/18 | 1.0000002 | 0.9591 / 0.9698 / 0.9781 |
| `tts_1.7b` pos 16/17/18 | 1.0000001 / 1.0000002 | 0.8789 / 0.8825 / 0.8990 |

The old 1.7B unit-`fnorm` logits are 0.9341 / 0.9602 / 0.9578 against the HF
reference; the rebuilt kernel's are 0.99948 / 0.99963 / 0.99947.  The logits
were wrong under the board harness's unit-`fnorm` contract; a host-side double
norm of `out` alone would have left them wrong.  On the llama.cpp host the real
`output_norm.weight` is already bound as `fnorm`, so E3 and E4 are bit-identical
there and this contract fix is not the 1.7B non-stop cause.  The actual cause is
the E5 softmax-sentinel bug below.

Fix: `talker.model.norm.weight` is packed in the head pack as `LM.fnorm`, and
the TTS final RMSNorm reads that member instead of the runtime `fnorm`.  The
`fnorm` input stays in the ABI (the manifest and host IO are unchanged), but
TTS no longer depends on its value.  ASR and the no-head E2 mode are untouched.
The E3 manifest's `out` params must include `fnorm` -> the same tensor the E2
host uses for output_norm (GGUF `output_norm.weight`, HF
`talker.model.norm.weight`); the pack member is `LM.fnorm pf 258048 4096` for
0.6B and `LM.fnorm pf 487424 8192` for 1.7B (the `pf` pack grows to 262144 /
495616 bytes).

Rebuilt 0.6B/1.7B kernels, board, still given the host's unit `fnorm`, against
`E2 out * output_norm` in f32:

| product | pos 16 | pos 17 | pos 18 |
|---|---:|---:|---:|
| `tts_0.6b` | 0.9999987 | 0.9999987 | 0.9999987 |
| `tts_1.7b` | 0.9999987 | 0.9999989 | 0.9999989 |

28-layer CRUN (real weights, 16-row history, unit `fnorm`), E3 against the E2
group: 0.6B against the E2 group run in the same simulator
(`E2 out * output_norm` f32) is 0.9999987 / 0.9999989 / 0.9999987.  1.7B
against the board E2 row (`E2 out * output_norm` f32) is 0.9999987 /
0.9995689 / 0.9995537 for pos 16/17/18, all above 0.999.  Against the HF
reference: 1.7B `out` is 0.999667 / 0.999667 / 0.999603 and `logits` 0.999483 /
0.999628 / 0.999472; 0.6B is 0.997974 / 0.998310 / 0.996076 and `logits`
0.997753 / 0.998299 / 0.995960, identical to the E2 group's own HF gap
(0.997981 / 0.998309 / 0.996086), i.e. the known BF16-chain gap, not an E3
regression.

ASR rebuilds are byte-identical to the pre-E4 artifacts (`883c40c5...` and
`a5391422...`).  Board cases and scores are in `out/e4_board_tts_0.6b_v5`,
`out/e4_board_tts_1.7b_v5`; CRUN results in
`out/e4_crun_tts_{0.6b,1.7b}_l28_v5` and
`out/e4_crun_e2_tts_0.6b_l28_v5`.

## E5: TTS-1.7B softmax sentinel bug (root cause of non-stop)

The 1.7B-only failure is the per-block softmax maximum in
`q35_group.attention_split_qa`, not the E4 `fnorm` contract.  `m_s` has
`1 + BC + 1 = 258` columns: the carried maximum, 256 block scores, and one
trailing zero written only to give the reduction a source.  The E3/E4 group
reduced all 258 columns; the reference paged kernels
(`q35_layer_mc.attention_split*` and `paged_attention_common`) reduce only
`m_s[:, :1+BC]`.  TTS-1.7B scores are large and some heads have all-negative
blocks.  Including the zero clamps `m_next >= 0`, every
`exp(score - m_next)` underflows to zero, and the final `0/0` is `+inf`.
That poisons the remaining layers, `out`, and `logits`.  ASR and TTS-0.6B do
not hit the underflow.

Board evidence (10.126.35.209, TTS-1.7B, q3d data):

* One-layer group (layer 0) with q3d `x`: 1..15 visible K rows finite
  (2048/2048), 16 rows all `+inf` (0/2048 finite).  The first pre-fix
  divergence is therefore call 0, layer 0, at the 16th visible row; after the
  fix there is no divergence beyond BF16 noise (the replay is bit-identical).
* Old E4 kernel, q6 call 0 replayed 50 times: 50/50 runs had all 2048 `out`
  lanes non-finite (`out/e5_replay/old50_result.txt`).  The original q6 dump
  has all 60 calls non-finite: 122880/122880 `out` and 184320/184320 `logits`.
* Fixed one-layer kernel: K0 / K12 / K16 all 2048/2048 finite.

Fix: `SOFTMAX_MAX_257 = MODEL_KIND == "tts" and H == 2048`; TTS-1.7B uses
`max(m_s[:, :1+BC])`.  Every other product keeps the historical 258-column
reduction, so its graph and artifact bytes are unchanged.

60-call board replay (teacher-forced `x`/`cs`/`pn`/`mask`, KV resident and
chained from `kv0`) against production E2 (`d1ade29a...`): all 60 outputs are
bit-identical to BF16(`E2 out * fnorm`), every per-layer K and V row written at
`pn[2]` is bit-identical, and all 60 `out` and `logits` are finite.  Minimum
cosines: `out` 0.99999827, E2 replay vs q3d dump 0.99999988, K/V 0.99999988.
The E2 replay is bit-identical to all 60 dumped outputs.  `axrun` does not write
modified inputs back, so the replay uses `e5_chain_runner.cc`, which keeps the
KV in CMM across calls.

28-layer CRUN, prefix 16, positions 16/17/18:

| metric | pos 16 | pos 17 | pos 18 |
|---|---:|---:|---:|
| E4 `out` vs HF | 0.9994391 | 0.9993486 | 0.9996146 |
| E2 `out * fnorm` vs HF | 0.9994372 | 0.9993498 | 0.9996113 |
| E4 `out` vs E2 `out * fnorm` | 0.99999899 | 0.99999911 | 0.99999899 |
| E4 `K` row vs HF | 0.9999774 | 0.9999692 | 0.9999684 |
| E4 `V` row vs HF | 0.9998041 | 0.9996521 | 0.9998177 |
| E4 `logits` vs HF | 0.9993820 | 0.9992802 | 0.9994876 |

E4 `out` is bit-identical to BF16(`E2 out * fnorm`) at all three positions; the
K/V rows have the same scores in the E2 run.  Results are in
`out/e5_crun_tts_1.7b_l28/` and `out/e5_crun_e2_tts_1.7b_l28/`.

Rebuilt artifacts:

| product | md5 | bytes |
|---|---|---:|
| `asr_0.6b` | `883c40c58650681dca4f36336ac8362f` | 6322978 |
| `asr_1.7b` | `a5391422e030668df4b59157392c9a23` | 10030274 |
| `tts_0.6b` | `2d773a790c48037b83c54e9c695161c1` | 5718430 |
| `tts_1.7b` | `a4417bc478e1e146894324c3096edeba` | 8829342 |

ASR and TTS-0.6B are byte-identical to E4, so their E4 board/CRUN results still
apply.  The fixed TTS-1.7B is also on the board as
`/data/qiushui/e5/tts17_e5.axmodel`.  Replay captures are in
`out/e5_replay/chain/{e2_chain60,e4_chain60}` (board:
`/data/qiushui/e5/chain/`), score in `out/e5_replay/chain/score.json`; the
runner source is `e5_chain_runner.cc`.

## E6: unconditional 257-column softmax reduction

The E5 condition (`MODEL_KIND == "tts" and H == 2048`) is removed.  Every
product built from `q35_group.attention_split_qa` now takes
`max(m_s[:, :1+BC])`, excluding the zero sentinel column from the block
maximum.  The old 258-column reduction is latent in any group: an
all-very-negative score block gets `m_next = 0`, its probabilities underflow,
and the normalizer becomes `0/0`.  ASR-0.6B, ASR-1.7B and TTS-0.6B were finite
on these cases, but TTS-0.6B already triggered the bad reduction numerically:
its corrected graph differs from the old finite graph while both retain the
same pre-existing 0.6B BF16 gap.

Rebuilt artifacts (`output/build_<product>/qwen3_group28_attn_mx8_lm8_paged_c4096_npu4.axmodel`):

| product | md5 | bytes | previous md5 |
|---|---|---:|---|
| `asr_0.6b` | `d7ca229f479f2f99f8a61d37efe2577f` | 6322786 | `883c40c58650681dca4f36336ac8362f` |
| `asr_1.7b` | `f09563d5ee80f620e60affe997da913d` | 10030178 | `a5391422e030668df4b59157392c9a23` |
| `tts_0.6b` | `a9dc04505544291c175de9c6fb25f754` | 5718238 | `2d773a790c48037b83c54e9c695161c1` |
| `tts_1.7b` | `a4417bc478e1e146894324c3096edeba` | 8829342 | same |

All four `.layout` files are unchanged.  `board_io.py` reports no changes for
any shipped axmodel.  TTS-1.7B is byte-identical to E5.  There is no separate
embedding/other lm8 artifact under `output/`; `embedding_0.6b` and any other
`q35_group` user inherit the same unconditional reduction path.

28-layer CRUN, prefix 16, chained positions 16/17/18 (`out/e6_crun_<product>_l28/`):

| product | new `out` vs HF | new `logits` vs HF | previous build |
|---|---|---|---|
| `asr_0.6b` | 0.9999434 / 0.9998972 / 0.9999391 | 0.9999850 / 0.9999748 / 0.9999883 | finite; `out` >= 0.9999104 and `logits` >= 0.9999839 vs new |
| `asr_1.7b` | 0.9999861 / 0.9999570 / 0.9999602 | 0.9999953 / 0.9999828 / 0.9999900 | bit-identical to previous |
| `tts_0.6b` | 0.9976678 / 0.9980232 / 0.9969611 | 0.9970251 / 0.9982038 / 0.9961040 | finite; `out` >= 0.9977100, `logits` >= 0.9965805 vs old |
| `tts_1.7b` | 0.9994391 / 0.9993486 / 0.9996146 | 0.9993820 / 0.9992802 / 0.9994876 | bit-identical to E5 at all positions |

TTS-0.6B misses the 0.999 HF target here (0.99696 `out`, 0.99610 `logits`),
which is the same pre-existing full-depth BF16-chain gap recorded under Open
issues; it is finite and matches the old build to 0.9977100 / 0.9965805.  The
old finite TTS-0.6B graph was not bit-equivalent because the sentinel was
already changing underflowed blocks.  ASR and TTS-1.7B meet the target.

Board (10.126.35.209; staged under `/data/qiushui/e6/`):

* ASR-0.6B, 8 chained calls: finite; HF min `out` 0.9997721, `logits`
  0.9998333; versus old model `out` 0.9998811, `logits` 0.9999080.
* ASR-1.7B, 8 chained calls: finite; HF min `out` 0.9998716, `logits`
  0.9999064; old model bit-identical.
* TTS-0.6B, 8 chained calls: finite; HF min `out` 0.9978122, `logits`
  0.9971542; versus old model `out` 0.9994532, `logits` 0.9993623.  Same
  pre-existing HF gap as above; the old model has the identical HF minima, so
  the gap is not a regression from E6.
* TTS-1.7B, E5's exact 60-call chain: finite; `out` vs BF16(E2 out * fnorm)
  min 0.99999827, bit-identical on every call; per-layer K/V bit-identical.

Board K/V minima (not part of the requested out/logits gate): ASR-0.6B
0.9996699 / 0.9988992, ASR-1.7B 0.9998102 / 0.9994034, TTS-0.6B
0.9995159 / 0.9986238, TTS-1.7B 0.99999988 / 0.99999988 (K / V).

Scores are in `out/e6_board/{asr_0.6b,asr_1.7b,tts_0.6b}/score_board.json`,
`score_vs_old_board.json`, and `out/e6_board/tts_1.7b/score_vs_e2.json`.
Board model copies are `/data/qiushui/e6/asr_0.6b/model.axmodel`,
`asr_1.7b/model.axmodel`, `tts_0.6b/model.axmodel`, and
`/data/qiushui/e6/tts17_e6.axmodel`.

Sentinel-pattern grep (report only; no files changed):

* `/data/junda/work/q27m/src`: `q35_group.py:220`, `q35_layer_mc.py:283/341/592`,
  `q35_prefill.py:296`, `q35_step.py:255` all reduce `m_s[:, :1+BC]`; safe.
* `/data/junda/work/qsmall/mtp27`: `mtp_kernel.py:78/281` reduce
  `m_s[:, :1+BC]`; `verify_group.py` and `verify_group_m4.py` import the q27m
  bodies via `sys.path`; safe.
* `/data/junda/work/qsmall/sgl35`: attention comes from
  `sgl_kernel_axera.attention.flash_attn`; it writes a sentinel at
  `state_scores[:, :1+kv_len+1]` but reduces `state_scores[:, :1+kv_len]`
  (`/data/junda/work/npu-codebase/sgl-kernel-axera/python/sgl_kernel_axera/attention/flash_attn.py:120`); safe.

## Build

```
source /data/junda/work/axdsl_env.sh
RUN_MODE=CRUN nice python build_group.py output --model asr_0.6b --layers 28
RUN_MODE=CRUN nice python build_group.py output --model asr_1.7b --layers 28
RUN_MODE=CRUN nice python build_group.py output --model tts_0.6b --layers 28
RUN_MODE=CRUN nice python build_group.py output --model tts_1.7b --layers 28
```

Each run writes `output/build_<product>/<name>.axmodel`, its `.layout`, and
`inputs.json` with the names, shapes and dtypes in kernel order.  The same
script with `RUN_MODE=PYRUN` runs the graph without compiling.

Final E2 check commands (`nice`, after `axdsl_env.sh`):

```
python check_group.py --model asr_0.6b --layers 28 --prefix 256 --positions 3 --fresh-rows 0 --threads 8 --out out/e2_final_asr_0.6b
python check_group.py --model asr_1.7b --layers 28 --prefix 256 --positions 3 --fresh-rows 0 --threads 8 --out out/e2_final_asr_1.7b
python check_group.py --model tts_0.6b --layers 28 --prefix 256 --positions 3 --fresh-rows 0 --threads 8 --out out/e2_final_tts_0.6b
python check_group.py --model tts_1.7b --layers 28 --prefix 256 --positions 3 --fresh-rows 0 --threads 8 --out out/e2_final_tts_1.7b

# the same page-boundary steps through the AX8860 simulator (4 layers)
python check_group.py --model asr_0.6b --layers 4 --prefix 256 --positions 3 --fresh-rows 0 --threads 12 --run-mode CRUN --out out/e2_crun_asr_0.6b_l4
python check_group.py --model asr_1.7b --layers 4 --prefix 256 --positions 3 --fresh-rows 0 --threads 12 --run-mode CRUN --out out/e2_crun_asr_1.7b_l4
python check_group.py --model tts_0.6b --layers 4 --prefix 256 --positions 3 --fresh-rows 0 --threads 12 --run-mode CRUN --out out/e2_crun_tts_0.6b_l4
python check_group.py --model tts_1.7b --layers 4 --prefix 256 --positions 3 --fresh-rows 0 --threads 12 --run-mode CRUN --out out/e2_crun_tts_1.7b_l4

# E3: TTS with the in-kernel codec head
python check_group.py --model tts_0.6b --layers 4 --prefix 256 --positions 3 --fresh-rows 0 --threads 12 --run-mode CRUN --out out/e3_crun_tts_0.6b_l4
python check_group.py --model tts_1.7b --layers 4 --prefix 256 --positions 3 --fresh-rows 0 --threads 12 --run-mode CRUN --out out/e3_crun_tts_1.7b_l4
```

The final E2 checks use `--layers 4 --prefix 256 --positions 3` for CRUN and
`--layers 28 --prefix 256 --positions 3` for full-depth PYRUN.  Results are in
`out/e2_crun_<product>_l4/result.json` and
`out/e2_final_<product>/result.json`.

Compiled deliverables:

| product | axmodel | bytes |
|---|---|---:|
| `asr_0.6b` | `output/build_asr_0.6b/qwen3_group28_attn_mx8_lm8_paged_c4096_npu4.axmodel` | 6322978 |
| `asr_1.7b` | `output/build_asr_1.7b/qwen3_group28_attn_mx8_lm8_paged_c4096_npu4.axmodel` | 10030274 |
| `tts_0.6b` | `output/build_tts_0.6b/qwen3_group28_attn_mx8_lm8_paged_c4096_npu4.axmodel` | 5718430 |
| `tts_1.7b` | `output/build_tts_1.7b/qwen3_group28_attn_mx8_lm8_paged_c4096_npu4.axmodel` | 8829342 |

## Open issues

* The two TTS products' full-depth cosine at the page boundary is below the
  0.9995 target: TTS-1.7B is 0.99924 / 0.99929 / 0.99856 and TTS-0.6B is
  0.99920 / 0.99804 / 0.99880 for pos 256 / 257 / 258.  Their K/V rows are
  above 0.9992 except TTS-0.6B pos-258 V at 0.9994017.  Both TTS products pass
  at 4 layers in PYRUN and CRUN; a one-layer TTS-0.6B row is 0.9999807.  The
  full-depth gap is accumulated BF16 rounding against the Transformers
  reference, not a KV or layout mismatch; the project's prefill kernel reports
  the same TTS-0.6B behaviour at 4 layers.  See the results above.
* The in-kernel output head needs core 0 to materialise the last hidden row through
  the `hbuf` DDR scratch input: PYRUN recycles derived OCM values of the last
  layer's row when many head tiles are recorded.  Core 0 writes `hbuf` before
  its own read; no other core reads or writes it.  Its initial content does not
  matter.
* This group uses the prefill `b0_kv` layout, but the prefill kernel is driven
  by `indptr/indices/last_page_len/cache_location` while the group takes
  `pn = [cell, nblk, krow]`.  The host must keep the two views consistent.
* The host path for a whole-model group must use the kernel's `out` directly:
  it is already final-normalised.  Applying `output_norm` to it again would
  double-normalise the hidden row (the logits would stay correct because the
  kernel head uses the same row).

## E7: one copy of the row-split weights (2026-11-03)

`BANKS=1` makes a decode group read `w_q/k/v`, `w_gate/up` and their E8M0 scales
from the prefill kernel's banks: layer `j` takes block `j // 4` of
`k_b<j % 4>_<name>`, which is the prefill input `b<j % 4>_<name>`.  The host
binds each `k_<bank>` input to that prefill bank buffer.  `w_o`, `w_down`,
their scales, the norms and the output head stay in the packs (the same
column-split split as 27B step 1).

Pack bytes before -> after (MiB): ASR-0.6B 586.38 -> 297.63, ASR-1.7B 1692.49
-> 768.49, TTS-0.6B 436.47 -> 147.72, TTS-1.7B 1392.66 -> 468.66.  Removed
duplication is 288.75 MiB (0.6B) / 924 MiB (1.7B); the remaining column-split
`w_o` / `w_down` + scales are 144.375 / 462 MiB.  The TTS code predictor
(`a.gen.code.*`) and codec are separate weights, not copies of these layers.

Host checks: every `k_b*` input equals its prefill `b*` input's shape and bytes;
all 140 row-split layer members (28 x 5 weights/scales) are bit-identical to
host quantization; `check_banks.py` passes page-boundary positions
0,1,254,255,256,257,258,511,512,4095.

CRUN: ASR-0.6B and ASR-1.7B pass full 28 layers.  TTS-0.6B and TTS-1.7B pass
the standard 4-layer CRUN check.  The full 28-layer TTS cosine floor is the
Transformers reference chain, not the bank path: the same-input `BANKS=0`
TTS-0.6B run gives exactly the same per-position cosines and difference
statistics as the banked run (out 0.9991911 / 0.9982210 / 0.9988307).  Details
and the TTS-1.7B confirmation are in the top-level `NOTES.md` of the smallbank
work directory.

Current artifacts:

| product | axmodel md5 | bytes |
|---|---|---:|
| ASR-0.6B | `44b8e1fb35532fd03fcca5c014fccff2` | 6443178 |
| ASR-1.7B | `4a12f8cb768b74ddb70900c4f784f1af` | 10049610 |
| TTS-0.6B | `1a1215ab0595e7ae87876884041082be` | 5842276 |
| TTS-1.7B | `e684b513067a86fafe239f4ff74be83f` | 8971846 |
