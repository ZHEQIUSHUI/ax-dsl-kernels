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
| 9 | `fnorm` | bfloat16 | `[H]` | final RMSNorm weight (`output_norm.weight`) |
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
| `fnorm` | `output_norm.weight` (ASR `thinker.model.norm.weight`, TTS `talker.model.norm.weight`) | plain runtime input |
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
| `tts_0.6b` | `output/build_tts_0.6b/qwen3_group28_attn_mx8_lm8_paged_c4096_npu4.axmodel` | 5717950 |
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
* No board run was performed after E3; only the TTS CRUN simulator numerics and
  the two rebuilt AXModels.
