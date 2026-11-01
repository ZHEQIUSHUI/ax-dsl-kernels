# Qwen3-TTS code predictor: one runtime-top-k frame kernel

## Result

One graph per model now serves every `top_k` in 1..50.  There is no separate
`cp.frame.k1` / `cp.frame.k50` graph and no host-side top-k selection.

Artifacts:

| model | axmodel | bytes | sha256 |
|---|---:|---:|---|
| 0.6B | `out_new/0.6b_k50_clean/qwen3tts_cp_frame5_mx8_h1024_krt_c16_npu4.axmodel` | 2,486,201 | `d8e4e6a57b7de73caaac8c590bb5cfd2df92223214627b8eead1eb0660026978` |
| 1.7B | `out_new/1.7b_k50_clean/qwen3tts_cp_frame5_mx8_h2048_krt_c16_npu4.axmodel` | 2,512,199 | `c6470edfc59b1a4d545db7696f14c4ec15e5ee475d79064070480ddf7adec4a3` |

Each has a `.layout` beside it.  The `k50` in the output-directory names is
historical; both files accept runtime `top_k`.  `inputs.json` describes both
ABIs in kernel order.

Environment used:

- PYRUN and PyTorch checks: `source /data/junda/work/sgl_env.sh`.
- CRUN/axmodel builds and simulator cycles: `source /data/junda/work/axdsl_env.sh`
  (npu-codebase `7913a059`, build context toolchain `8.0` / `7913a059`).

## Design

`cp_kernel.py` emits a kernel whose wrapper is still under 0.5 KB:

- Position 0 (the `e0` seed) is static and writes the first KV row for all five
  layers.
- Positions 1..8 and 9..15 are two `ax.dsl.range` loops.  The split keeps each
  `AxDynamicLd` head source inside the toolchain's 64 MiB SDMA window.
- All five layers execute inside each runtime-loop body.  We did not add a
  runtime layer loop: the profile shows the layer matrices are not duplicated
  overhead; the removed cost is the per-position static launch/barrier and
  load-chunk duplication.
- The 16-position K/V is loop-carried on chip and stored to the resident
  `kc0..kc4` / `vc0..vc4` outputs once at the end.  Codes and logits are also
  accumulated on chip and stored once.
- The 15 codebook heads are two contiguous runtime inputs (`heads_lo` rows
  0..8*V, `heads_hi` rows 8*V..15*V) because one 15-table `AxDynamicLd` source
  crosses the 64 MiB limit.  The head lookup is tiled by `HEAD_TILE=64`; the
  embedding table is selected by the sampled code.
- Sampling always asks hardware top-k for 50.  A runtime `top_k` masks the tail
  before softmax and uses a runtime `index_select` to swap candidates 0/1 only
  when `top_k > 1`.  Thus k=1 and k=50 reproduce ggml's two old cases exactly;
  intermediate k values use the same unordered top-50 prefix and ggml's swap.

The layer arithmetic is reused verbatim from the frozen `4acb7f6` sources:
`q3tts_cp_step.py`, `q35_group.py`, `q35_layer_mc.py`, `q35_layer_pk.py`, and
`q35_mx_build.py`.  `cp_kernel.py` only replaces the scheduling (position loop,
on-chip accumulation, runtime sampler) and keeps their MXFP8 pack layout and
attention/MLP math.

## ABI and host changes

Inputs, in kernel order (`inputs.json` is authoritative):

| input | 0.6B | 1.7B | dtype |
|---|---|---|---|
| `e0` | `[1,1,1024]` | `[1,1,2048]` | bfloat16 |
| `e1` | `[1,1,1024]` | `[1,1,2048]` | bfloat16 |
| `pw` | `[78643200]` | `[78643200]` | float8_e4m3fn |
| `ps` | `[2457600]` | `[2457600]` | float8_e8m0fnu |
| `pf` | `[12544]` | `[13568]` | float32 |
| `pb` | - | `[2097152]` | bfloat16 |
| `heads_lo` | `[16384,1024]` | `[16384,1024]` | bfloat16 |
| `heads_hi` | `[14336,1024]` | `[14336,1024]` | bfloat16 |
| `embd` | `[30720,1024]` | `[30720,2048]` | bfloat16 |
| `cs16` | `[32,64]` | `[32,64]` | bfloat16 |
| `pn16` | `[32]` | `[32]` | int32 |
| `mask16` | `[16,16]` | `[16,16]` | bfloat16 |
| `rnd` | `[1,15]` | `[1,15]` | float32 |
| `top_k` | `[1]` | `[1]` | int32 |

Outputs are unchanged from the frame ABI: `codes [4,16] int32`, `lgs [15,2048]
float32`, and `kc<j>` / `vc<j>` `[16,1024] float16` for j=0..4.

Exact host changes in `ref/llama/qwen3_tts/qwen3_tts.cpp`:

1. Replace `man.get("cp.frame.k" + std::to_string(fr.top_k))` with one role,
   for example `cp.frame.krt`; the binding no longer depends on top-k.
2. Add an `int32[1]` CMM input `top_k` and update it/flush it in `frame_run`
   every frame.  Host should reject values outside 1..50.
3. Replace the single `heads` binding with two views into the same contiguous
   BF16 head scratch: `heads_lo` is the first `8*V*H*2` bytes, `heads_hi` the
   remaining `7*V*H*2` bytes.  `embd` remains one input.
4. Keep `e0`, `e1`, packs, `cs16`, `pn16`, `mask16`, `rnd`, all outputs, and
   the resident KV binding unchanged.  `codes[0]` is still supplied by the host.
5. Delete the old k1/k50 kernel binding cache keyed by top-k; there is only one
   graph cache entry per model.  Top-k may change every frame without rebinding.

## Weight/input sources

The production conversion is the one in the frozen manifests (`params.cp` and
`params.cp_out`) and in `q3tts_cp_step.build_packs`: each BF16 weight below is
MXFP8-E4M3 quantized in groups of 32 with E8M0 scales, using the existing
column-strip blocking.  The code predictor's GGUF names and Hugging Face
checkpoint names are:

| kernel input | GGUF source | ModelScope checkpoint source | conversion |
|---|---|---|---|
| `e0` | graph input `inp_raw` | talker hidden state | host F32 -> BF16 |
| `e1` | `a.gen.code.out_embd.weight`, row `inp_code0` | `talker.model.codec_embedding.weight[code0]` | BF16 row |
| `pw` / `ps` | `a.gen.code.blk.{L}.attn_q.weight`, `attn_k`, `attn_v`, `attn_out`, `ffn_gate`, `ffn_up`, `ffn_down` | `talker.code_predictor.model.layers.{L}.self_attn.{q,k,v,o}_proj.weight`; `mlp.{gate,up,down}_proj.weight` | MXFP8 codes / E8M0 scales |
| `pf` (`L*.ln1`) | `a.gen.code.blk.{L}.ln1.weight` | `...layers.{L}.input_layernorm.weight` | F32 flat pack |
| `pf` (`L*.q_norm`, `k_norm`) | `a.gen.code.blk.{L}.attn_q_norm.weight`, `attn_k_norm.weight` | `...self_attn.{q,k}_norm.weight` | F32 flat pack |
| `pf` (`L*.ln2`) | `a.gen.code.blk.{L}.ln2.weight` | `...layers.{L}.post_attention_layernorm.weight` | F32 flat pack |
| `pf` (`LP.norm`) | `a.gen.code.output_norm.weight` | `talker.code_predictor.model.norm.weight` | F32 flat pack |
| `pb` (1.7B) | `a.gen.code.proj_in.weight` | `talker.code_predictor.small_to_mtp_projection.weight` | BF16 flat pack |
| `pf` (`LP.b_pin`, 1.7B) | `a.gen.code.proj_in.bias` | `talker.code_predictor.small_to_mtp_projection.bias` | F32 flat pack |
| `heads_lo` | first 8 tables of `a.gen.code.head.weight` | `talker.code_predictor.lm_head.{0..7}.weight` | BF16 |
| `heads_hi` | remaining 7 tables of `a.gen.code.head.weight` | `talker.code_predictor.lm_head.{8..14}.weight` | BF16 |
| `embd` | `a.gen.code.embd.weight` | `talker.code_predictor.model.codec_embedding.{0..14}.weight` | BF16 |
| `cs16` | derived, RoPE theta from graph | - | cos/sin pairs for positions 0..15 |
| `pn16` | derived | - | `(pos, pos/BC+1)` |
| `mask16` | derived | - | causal mask, LMAX=BC=16 |
| `rnd` | `inp_rand_0..14` | - | F32 |
| `top_k` | new host input | - | int32 |

The frozen manifests' `cp.frame.k1` / `.k50` inputs use `heads`, `pf` etc.;
the pack bytes and all member offsets are unchanged.  Only the contiguous head
input is exposed as the two views above.

## Checks

Commands used:

```sh
# PYRUN (sgl_env)
python check_cp.py --impl new --model 0.6b --top-k 1 --frame 1 --real \
  --out out_check/real_0.6b_k1_f1_pyrun.pt --run-mode PYRUN

# CRUN (axdsl_env)
RUN_MODE=CRUN python build_cp.py out_new/0.6b_k50_clean --model 0.6b --top-k 50
python check_cp.py --impl old --model 0.6b --top-k 50 --frame 0 \
  --out out_check/old_0.6b_k50_f0_crun.pt --run-mode CRUN
```

`build_cp.py` uses deterministic random MXFP8 weights at the real shapes;
weights are runtime inputs, so the compiled graph is independent of the random
values.  `make_inputs.py` generated `inputs.json` in one fresh interpreter per
model.

Random-weight PYRUN (all five layers, all 15 positions), kernel vs the PyTorch
reference on the same dequantized packs:

| case | logit cosine mean/min | KV cosine mean/min |
|---|---|---|
| 0.6B, k=1, frame 0 | 0.9999915 / 0.9999905 | 0.9999936 / 0.9999902 |
| 0.6B, k=50, frame 0 | 0.9999915 / 0.9999907 | 0.9999935 / 0.9999896 |
| 1.7B, k=1, frame 0 | 0.9999918 / 0.9999908 | 0.9999935 / 0.9999903 |
| 1.7B, k=50, frame 0 | 0.9999921 / 0.9999906 | 0.9999935 / 0.9999900 |

Additional runtime-k smoke checks on 0.6B (`k=2,25,49`) all produced
`codes_vs_own=True` and logit/KV cosine above 0.99999.  The 0.6B new PYRUN also
passed frames 1 and 2 (documented in `check_f1_*` / `check_f2_*` logs).

Real ModelScope weights, PYRUN, same dequantized packs, `e1` is code-0's row,
`e0` is the frame's deterministic random talker-width row, and the reference is
teacher-forced with the kernel's emitted codes:

| case | logits mean / min | KV mean / min | result |
|---|---|---|---|
| 0.6B f1 k=1 | 0.9998898 / 0.9996726 | 0.9999101 / 0.9995167 | pass |
| 0.6B f1 k=50 | 0.9999424 / 0.9997113 | 0.9999334 / 0.9996361 | pass |
| 1.7B f0 k=1 | 0.9999661 / 0.9998269 | 0.9999462 / 0.9999099 | pass |
| 1.7B f0 k=50 | 0.9999554 / 0.9998269 | 0.9999312 / 0.9998761 | pass |

The 0.6B frame-0 greedy check is the only marginal case: logits `0.9997019 /
0.9978978`, KV `0.9999498 / 0.9998000`.  This is an input-conditioned
ill-conditioned logit row, not a mapping bug: the same MXFP8-dequantized torch
reference versus the original BF16 checkpoint has cosine `0.5547` on that row.
The implementation matches its quantized reference; frame 1 is the passing
full-depth case.  Per-layer KV checks (the first layers included) are in the
`kv_cos` arrays and logs.

CRUN old vs new, identical inputs, all four cases:

| model/top-k | codes | logit cosine | KV cosine mean/min | old cycles | new cycles |
|---|---|---|---:|---:|---:|
| 0.6B k=1 | equal | 1.0 | 1.0 / 1.0 | 7,546,196 | 5,970,960 |
| 0.6B k=50 | equal | 1.0 | 1.0 / 1.0 | 7,537,261 | 5,970,960 |
| 1.7B k=1 | equal | 1.0 | 1.0 / 1.0 | 8,145,654 | 6,304,367 |
| 1.7B k=50 | equal | 1.0 | 1.0 / 1.0 | 8,102,355 | 6,304,367 |

The new graph is 20.8% fewer simulator cycles for 0.6B and 22.2% fewer for
1.7B at top-50; k=1 is 20.9% and 22.6% fewer.  If the board ratio follows the
simulator, the measured board baselines (9.1 / 9.7 ms) would become about
7.2 / 7.5 ms; that is a projection, not a board measurement.

## Profile

The reduction is scheduling/launch overhead, not less arithmetic.  On the 0.6B
trace, summed `AxMatMul` time is essentially unchanged (`67261.7` old vs
`67293.1` new trace units), while `BARRIER` drops `3138.2 -> 68.2` and
`LOAD-CHUNK` events drop `11145.4 -> 9538.5`.  Runtime loads add `AxDynamicLd`
(`64.5 -> 3009.6`), still a net win.  The 1.7B trace has the same shape
(`AxMatMul 69839.1 -> 69881.5`, `BARRIER 3198.2 -> 73.1`).

Static graph size also falls sharply:

| model | jobs old -> new | mcode bytes old -> new |
|---|---:|---:|
| 0.6B k=50 | 190,445 -> 34,718 | 19,592,352 -> 2,469,248 |
| 1.7B k=50 | 192,296 -> 34,996 | 19,576,576 -> 2,494,912 |

Use `simulator_cycles` in `build_context.json`; `estimated_cycles` is
misleading here (`~16.7M`) because it does not model the runtime-loop overlap.

## Open issues

- DDR read bytes regress: 0.6B `1,601,150,080 -> 4,270,598,656`, 1.7B
  `1,903,223,936 -> 4,283,212,800`.  The runtime position loop prevents the
  compiler from retaining static weight loads across positions.  This is left
  explicit because simulator cycles improve notwithstanding it; board DDR
  behavior should be measured before claiming the projected 7.2/7.5 ms.
- The frozen old kernel cannot run in PYRUN: its in-place KV outputs are never
  materialized on that path (`KeyError: kc0`).  All old/new comparisons are
  therefore CRUN only, with resident KV prefilled on the host side.
- PYRUN codes vs the independent torch sampler can differ on a CDF boundary;
  the kernel's own sampler is checked separately and matches exactly.  Old and
  new CRUN are bit-identical where required (including k=50).
- `top_k` values other than 1/50 have PYRUN smoke coverage but are not
  separately CRUN-compared; the graph is identical for all k.
- No board run was performed.  The simulator improves, but the DDR regression
  above is the remaining board risk.
