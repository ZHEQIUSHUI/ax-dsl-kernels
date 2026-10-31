# Qwen3-ASR audio encoder as one kernel

`/data/junda/work/qsmall/asrenc`, environment `source /data/junda/work/sgl_env.sh`
(npu-codebase `cad0db3d6` + sgl-kernel-axera).  All long builds/checks were run
under `nice`.

## What the kernel replaces

The frozen adapter (`ref/llama/qwen3_asr/qwen3_asr.cpp`) currently does, per call:

```
host mel im2col -> audio.fe1.n<n> -> audio.fe2.n<n> -> host conv3 im2col
-> audio.fe3.n<n> -> host conv_out row gather -> audio.conv_out.n<n>
-> host + position_embd -> L x audio.layer<NF> -> audio.proj<NF>
```

with `n = 1,2,4,8` chunks of 100 mel frames (the adapter rounds the chunk count
up to one of those), `NF = 64` for `n <= 4` and `128` for `n = 8`, and
`L = 18` (0.6B) or `24` (1.7B).  Each call is one `n_window_infer = 800`
inference window, i.e. at most 104 valid encoder rows, so the window mask is the
single block `[0, 13n)` plus masked padding rows.

The delivered kernel is **one AXModel per (model, n)** that takes

```
mel [n*100, 128] bf16  ->  out [P, O] fp32,  P = 64 (n<=4) / 128 (n=8)
```

and internally performs the on-chip im2col, conv2d1/2/3 + GELU, conv_out GEMM,
sinusoidal positions, all `L` encoder layers, the padding/window mask, final
LayerNorm, and projector.  The first `13*n` output rows are valid; the rest are
padding rows (the adapter copies only `13*n` rows).  Four NPU cores are used.

## Design

- **Front-end.**  The mel is copied once to an OCM bf16 vector with one zero
  sentinel row.  Constant `int32` gather tables implement each 3x3/stride-2
  convolution on chip.  Core `c` owns four of the 16 conv3 frequency rows
  (`FQ = 16/4`), so:
  - conv1: local band 19 h-rows x 50 time columns x 16 taps;
  - conv2: local band 9 h-rows x 25 time columns x 9 taps x 480 channels;
  - conv3: 4 frequency rows x 13 time columns x 9 taps x 480 channels.
  The 3-D map is exchanged through `SHARED_OCM` once (pub/sub), then each core
  gathers the rows it owns for conv_out.  The gather, transpose to
  `[row, 480, 16]`, `K = 16*480 = 7680` GEMM, position add and `x0` are in the
  kernel.  No host tensor is produced between any of these stages.
- **Rows.**  `P = 64` for `n <= 4`, `128` for `n = 8`; each of the four cores
  owns `P/4` rows.  All row-index tables (conv1/2/3 and conv_out) are built at
  compile time for each `n` and core; no runtime `n` arithmetic remains.
- **Layers.**  `ax.dsl.range(0, 0, L, 1)` is a runtime loop over the layer
  weight banks `[L, ...]`; the loop index selects bias/weight rows through
  `sgl_kernel_axera.models.bank_access.load_tile`.  One AXModel therefore
  contains all 18/24 layers, not one call per layer.  Per layer:
  LayerNorm(eps 1e-5) -> q/k/v -> per-head (64-dim) scores `* 1/sqrt(64)` ->
  mask -> softmax -> PV -> out-proj -> residual -> LayerNorm -> fc1/GELU(erf) ->
  fc2 -> residual.  K/V are exchanged through `SHARED_OCM`; two pub/sub barriers
  protect the write and copy-out phases.  Scores and residuals are fp32; GEMM
  inputs are bf16 because the AX8860 `AxMatMul` has no fp32 form.  The attention
  output, LayerNorm outputs and residual stream are kept fp32, and only the GEMM
  boundaries are bf16 (this is the compilable variant of the fp32 reference
  path).  The quantized K/V exchange is bf16, as in the frozen `q3a_layer.py`.
- **Mask / window.**  `mask[13n:] = -3e38`, zero otherwise.  For the supported
  `n <= 8` there is exactly one inference window, so no cross-window mask is
  needed.  Padded rows have no valid conv_out row (their gather is the zero
  sentinel), are hidden as keys, and their queries never reach a valid output
  row.  The position is added to every row, but only the first `13n` are used.
- **Projector.**  Final LayerNorm, `proj1` (H), GELU(erf), `proj2` (O), fp32
  output `[P, O]`.  There is no conv_out bias in the current checkpoint/GGUF
  (the adapter makes it optional), so it is not an ABI input.

## Variants

| model | H | F | heads | L | O | n | P | valid rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-0.6B | 896 | 3584 | 14 | 18 | 1024 | 1/2/4/8 | 64/64/64/128 | 13/26/52/104 |
| Qwen3-ASR-1.7B | 1024 | 4096 | 16 | 24 | 2048 | 1/2/4/8 | 64/64/64/128 | 13/26/52/104 |

All eight AXModels are in `output/axmodels/` (sizes below), and `inputs.json`
lists the exact kernel-order ABI for each (`mel`, 25 weight inputs, `out`).

| model | n=1 | n=2 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| 0.6B | 957,230 B | 1,346,496 B | 2,141,476 B | 3,855,655 B |
| 1.7B | 1,000,590 B | 1,403,040 B | 2,161,668 B | 3,966,599 B |

The sizes grow only with the front-end/row tables: the 18/24 layers are a
runtime loop in one graph, not unrolled weight data.

## Reuse

- `sgl_kernel_axera.models.bank_access.load_tile`: runtime `[L, ...]` bank row
  tiles (same idiom as `embedding/emb_kernel.py`).
- `axnn.dsl.kernels.utils.linear.linear`: tiled bf16 GEMMs for the conv front-end,
  conv_out and projector.
- `ref/qwen3_asr/src/127cc3a/q3a_layer.py`: LayerNorm eps, per-head attention,
  `SHARED_OCM` K/V handoff and the two barriers, GELU erf.
- `ref/qwen3_asr/src/127cc3a/q3a_fe.py`: 3x3/stride-2 tap/row layouts and the
  per-core output-channel split; `q3a_proj.py`: fused post-LN/projector;
  `q3a_audio.py` / `q35_vit.py`: pre/attn/post layer structure.
- `ref/llama/qwen3_asr/qwen3_asr.cpp`: call sequence, `13*n` rows, padding mask,
  `n_window_infer=800`, position `r % 13`, adapter tensor names.
- `embedding/emb_kernel.py`: runtime `ax.dsl.range` over weight banks.

`ref/` and `embedding/` were read only.

## Checks

Reference: the official `qwen-asr` wheel's
`core/transformers_backend/modeling_qwen3_asr.py`, loaded from the real
checkpoint (bf16 tensors widened to fp32), with the current transformers
`check_model_inputs` decorator stubbed.  Audio: the 15 s real-speech
`audio/asr_en.wav` (the official Qwen3-ASR example asset,
`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav`;
48 kHz, 7.5 s, tiled to the requested 8 s for n=8, resampled to 16 kHz),
`WhisperFeatureExtractor` (n_fft 400, hop 160, 128 mels).  The kernel input is bf16 by contract, so the
reference is fed the **same bf16-quantized mel** (`check_asr.py` default;
`--fp32-mel` gives the stricter unquantized-input comparison).  Weights are the
same checkpoint bf16 values in both paths.

`check_asr.py --tag both --n 1 --n 8 --layers 1,4,all` (PYRUN, bf16 mel for
both sides):

| model | n | depth | cos mean | cos min |
|---|---:|---:|---:|---:|
| 0.6B | 1 | 1 | 0.99999583 | 0.99999254 |
| 0.6B | 1 | 4 | 0.99999416 | 0.99998702 |
| 0.6B | 1 | 18 | 0.99998711 | 0.99997596 |
| 0.6B | 8 | 1 | 0.99999710 | 0.99999489 |
| 0.6B | 8 | 4 | 0.99999578 | 0.99999227 |
| 0.6B | 8 | 18 | 0.99999202 | 0.99997759 |
| 1.7B | 1 | 1 | 0.99999658 | 0.99999440 |
| 1.7B | 1 | 4 | 0.99999131 | 0.99998051 |
| 1.7B | 1 | 24 | 0.99998720 | 0.99996206 |
| 1.7B | 8 | 1 | 0.99999757 | 0.99999629 |
| 1.7B | 8 | 4 | 0.99999655 | 0.99998840 |
| 1.7B | 8 | 24 | 0.99999131 | 0.99986455 |

All full-depth variants clear mean >= 0.9995 and min >= 0.999.  Raw JSON:
`output/check_asr.json` (all four reports have `"quant_mel": true`).

For the stricter view where the reference gets the **unquantized fp32 mel**
while the kernel still gets bf16 (`--fp32-mel`, raw JSON
`output/check_asr_fp32mel.json`), the full-depth numbers are:

| model | n | depth | cos mean | cos min |
|---|---:|---:|---:|---:|
| 0.6B | 1 | 18 | 0.99998161 | 0.99997010 |
| 0.6B | 8 | 18 | 0.99999141 | 0.99996554 |
| 1.7B | 1 | 24 | 0.99983170 | 0.99901310 |
| 1.7B | 8 | 24 | 0.99999014 | 0.99983183 |

All still clear the acceptance thresholds; the 1.7B n=1 minimum is the tightest
because the bf16 input quantization is not applied to that reference.

`check_random.py` (independent random bf16 weights, real shapes) independently
validated the im2col indexing, PE, mask, layer banks and projector.  Final
full-depth random-weight cosines:

| model | n | L | cos mean | cos min |
|---|---:|---:|---:|---:|
| 0.6B | 1 | 18 | 0.99999130 | 0.99998827 |
| 0.6B | 8 | 18 | 0.99999374 | 0.99999203 |
| 1.7B | 1 | 24 | 0.99999197 | 0.99999031 |
| 1.7B | 8 | 24 | 0.99999317 | 0.99999185 |

## Input -> GGUF sources

The adapter's GGUF tensor names (as read by `qwen3_asr.cpp`) and the HF
checkpoint tensors are:

| kernel input | GGUF tensor | HF source | layout / conversion |
|---|---|---|---|
| `mel` | `inp_raw` (runtime graph input) | log-mel | `[n*100,128]` bf16, time-major |
| `w1`, `b1` | `a.conv2d.1.weight`, `.bias` | `conv2d1.weight`, `.bias` | `[480,16]`: 9 taps `kh*3+kw`, 7 zero pads |
| `w2`, `b2` | `a.conv2d.2.weight`, `.bias` | `conv2d2.weight`, `.bias` | `[480,4320]`: K index `(kh*3+kw)*480+ic` |
| `w3`, `b3` | `a.conv2d.3.weight`, `.bias` | `conv2d3.weight`, `.bias` | same `[480,4320]` tap-major layout |
| `conv_out_w` | `a.conv_out.weight` | `conv_out.weight` | `[H,7680]`, K index `oc*16+oh`; no bias in this checkpoint |
| `ln1w`, `ln1b` | `a.blk.N.ln1.weight`, `.bias` | `layers.N.self_attn_layer_norm.*` | bank `[L,H]` |
| `qkvw`, `qkvb` | `a.blk.N.attn_qkv.weight`, `.bias` | `layers.N.self_attn.{q,k,v}_proj.*` | `[L,3H,H]` / `[L,3H]`, q\|k\|v joined |
| `ow`, `ob` | `a.blk.N.attn_out.weight`, `.bias` | `layers.N.self_attn.out_proj.*` | `[L,H,H]` / `[L,H]` |
| `ln2w`, `ln2b` | `a.blk.N.ln2.weight`, `.bias` | `layers.N.final_layer_norm.*` | `[L,H]` |
| `upw`, `upb` | `a.blk.N.ffn_up.weight`, `.bias` | `layers.N.fc1.*` | `[L,F,H]` / `[L,F]` |
| `downw`, `downb` | `a.blk.N.ffn_down.weight`, `.bias` | `layers.N.fc2.*` | `[L,H,F]` / `[L,H]` |
| `postw`, `postb` | `a.post_ln.weight`, `.bias` | `ln_post.*` | `[H]` |
| `p1w`, `p1b` | `mm.a.mlp.1.weight`, `.bias` | `proj1.*` | `[H,H]` / `[H]` |
| `p2w`, `p2b` | `mm.a.mlp.2.weight`, `.bias` | `proj2.*` | `[O,H]` / `[O]` |
| (positions) | `a.position_embd.weight` (optional) | generated `SinusoidsPositionEmbedding` | row `r % 13`; generated in-kernel |
| `out` | projector rows | - | `[P,O]` fp32, first `13n` valid |

Important layout difference from the old adapter: the fused conv3 consumes the
same tap-major `[OC][KH][KW][IC]` rows as conv2 (`(kh*3+kw)*480+ic`), whereas the
old fe3 host im2col produced `[OC][IC][KH][KW]` (the adapter's `taps=false`
path).  A host/runtime that feeds this kernel must produce the tap-major layout
for **both** w2 and w3.  `conv_out_w`, q/k/v, and all matrix banks are already
row-major bf16 and can be passed as the adapter's bf16 copies.

## Build / check

```bash
source /data/junda/work/sgl_env.sh
nice python3 build_asr.py --tags 0.6b,1.7b   # random weights, real shapes, CRUN compile
nice python3 check_asr.py --tag both --n 1 --n 8 --layers 1,4,all
nice python3 check_random.py                 # N=1/4/8, L=1/4/all
```

`build_asr.py` waits for `<cache>/*.compiled.axmodel` to become stable and stops
the simulator child, as requested.  `check_*.py` are separate from `build_*.py`.
Logs: `output/build_*.log`, `output/check_*.log`.

## Open issues

- The 1.7B download completed, but the full shard 1 hash does not match the
  ModelScope value (`f3bb49cd...` vs `a4cd1f1a...`); shard 2 matches.  All 397
  audio-tower tensors live in the first 635,035,192 bytes of shard 1, and that
  prefix was re-downloaded fresh: it is byte-identical to the local prefix
  (sha256 `a6c55019...`), so the real weights used by the checks are verified.
  The corrupted remainder is the LLM tail, outside this task.  (`asr_ref.py`
  still has a partial-safetensors reader, useful for interrupted downloads.)
- If the reference is given the **unquantized fp32 mel** while the kernel gets
  bf16 mel, 1.7B n=1 loses margin (full depth mean 0.99983170, min 0.99901310);
  this is the kernel's required bf16 input quantization, not an internal
  indexing error.  The default check matches the inputs (`--quant-mel`); the
  `--fp32-mel` table above gives the stricter view.  Both pass.
- n=2 and n=4 are compiled and shape-checked with random weights; real-speech
  accuracy was measured for n=1 and n=8 as requested.
- On the board the adapter still needs to be changed to pass this kernel's ABI;
  the frozen `qwen3_asr.cpp` still calls the old fe/layer/proj kernels and does
  the host im2col/PE work.  This task delivered the fused AXModels, not an
  adapter edit.
- The current checkpoint has no `conv_out.bias` and no positional-embedding
  tensor; the kernel therefore has no conv_out-bias input and generates the
  sinusoids itself.  A future GGUF with `a.conv_out.bias` would need an extra
  ABI input.
- `AxMatMul` on this toolchain supports bf16 inputs only (fp32 and mixed-input
  forms fail `OpBuildException`).  The kernel therefore uses bf16 GEMM
  boundaries, fp32 LayerNorm/softmax/attention-output/residual arithmetic.
