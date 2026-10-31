# Qwen3 dense text kernels

## Scope and products

`qwen3_kernel.py` is one config-driven implementation of the Qwen3 dense text
decoder.  It emits one whole-depth prefill AXModel (256 rows) and one
whole-depth decode AXModel (one row).  Five products map to three shape/RoPE
variants:

| product | hidden | layers | MLP | position encoding | checkpoint root |
|---|---:|---:|---:|---|---|
| `embedding_0.6b` | 1024 | 28 | 3072 | RoPE | `model.layers.N`, `model.norm.weight` |
| `asr_0.6b` | 1024 | 28 | 3072 | interleaved M-RoPE | `thinker.model.layers.N`, `thinker.model.norm.weight` |
| `asr_1.7b` | 2048 | 28 | 6144 | interleaved M-RoPE | same ASR prefix |
| `tts_0.6b` | 1024 | 28 | 3072 | interleaved M-RoPE | `talker.model.layers.N`, `talker.model.norm.weight` |
| `tts_1.7b` | 2048 | 28 | 6144 | interleaved M-RoPE | same talker prefix |

All variants have 16 Q heads, 8 KV heads, head dimension 128, RoPE theta
1,000,000, RMSNorm epsilon 1e-6, and 28 layers.  M-RoPE is
`mrope_section=[24,20,20]` with `interleaved=true` for ASR and TTS.  The
context is 4096 for every product, so the cache has 16 pages of 256 tokens.

### Differences from vanilla Qwen3

* Qwen3-Embedding uses ordinary one-axis RoPE (`position_ids` shape `[T]`).
* ASR and TTS use interleaved M-RoPE with `mrope_section=[24,20,20]`
  (`position_ids` shape `[3,T]`).  Their head dimension is still 128, with 16
  Q heads and 8 KV heads, and their text stacks are otherwise vanilla Qwen3
  dense decoders with Q/K RMSNorm and SwiGLU.
* TTS's 0.6B talker has `hidden_size=1024` and `text_hidden_size=2048`; the
  1.7B talker has both at 2048.  The 2048-wide text embedding is projected to
  the talker width by `talker.text_projection` outside this kernel.  The
  official `qwen-tts` 0.1.1 `get_rope_index` assigns the same position to all
  three M-RoPE axes for a pure text/codec sequence; the kernel also accepts
  three distinct axes, which is covered by `--varied-mrope`.
* TTS's output vocabulary is the 3072-entry codec vocabulary and its output
  head is `talker.codec_head`; ASR's is `thinker.lm_head`; both are outside.

The two ASR products read `thinker_config.text_config`; the final text norm is
`thinker.model.norm.weight`, and `thinker.lm_head.weight` is outside this
kernel.  The two TTS products read `talker_config`; the text stack is the
talker stack.  The 0.6B and 1.7B talkers have `text_hidden_size=2048` in both
models, but their talker `hidden_size` is 1024 and 2048 respectively.  The
outside `talker.text_projection` maps the 2048-wide text embedding to the
talker width; `talker.model.text_embedding`, `talker.model.codec_embedding`
(3072 rows), and `talker.codec_head` stay outside.  The official `qwen-tts`
0.1.1 modeling code confirms that the talker's `get_rope_index` gives the same
position on all three M-RoPE axes for this pure text/codec embedding sequence;
the kernel nevertheless accepts distinct `[3,T]` positions.

## Kernel design

`forward_qwen3_blocks` is the single forward.  The builder records it twice,
with `ROWS=256` and `ROWS=1`, using the same runtime weight banks, the same
final RMSNorm, and the same BF16 paged KV cache:

* `b0_kv` is one runtime input holding the whole
  `[28,16,2,256,8,128]` BF16 cache.  Its per-layer views have the layout
  `[page, k/v, 256, 8 KV heads, 128]`.
* Four phase banks (`b0..b3`) cover layers `phase + 4*block`.  A runtime
  `ax.dsl.range` loops over seven blocks and, inside each block, runs the four
  phase layers in order.  Every attention/MLP weight bank has leading shape
  `[7,...]`; all projection weights are runtime MXFP8/E8M0 inputs.
* The attention weights are `w_norm_attn`, `w_q/w_scale_q`, `w_k/w_scale_k`,
  `w_v/w_scale_v`, `w_norm_q`, `w_norm_k`, `w_o/w_scale_o` and their scale2 /
  zero-point metadata.  MLP weights are `w_norm`, `w1/w_scale1`,
  `w2/w_scale2`, `w3/w_scale3` and metadata.
* `w_norm_q` already contains `q_norm.weight / sqrt(128)`, matching the pilot
  kernel's folded attention scale.
* Prefill uses SGL `flash_attn_with_kvcache`; decode uses SGL
  `decode_paged_kv_kernel`.  SGL's prefill primitive supports query lengths 128
  and 256 only, so a one-row prefill call is not buildable.  The decode branch
  is therefore selected inside the same whole-depth Python forward rather than
  by a separate per-layer graph, and it reads/writes the identical paged cache.
* The final RMSNorm is inside the graph.  Output is `[rows,hidden]`; for
  prefill rows beyond `valid_tokens` are zeroed.  Decode emits its one row.
  The output head (Embedding pooling, ASR lm_head, TTS codec head) is outside.

`prepare()` returns kernels keyed `1` (decode) and `256` (prefill).  Compiled
graphs are under `out/build_<product>/prefill1` and `prefill256`.

## Reuse

The whole-depth runtime loop, bank layout, fidelity hook, MXFP8 conversion and
check style follow `/data/junda/work/qsmall/embedding`'s finished pilot
(`emb_kernel.py`, `check_emb.py`, `build_emb.py`); the pilot itself was not
edited.  Attention, paged KV, MLP and bank helpers are reused from the
sgl-kernel-axera checkout selected by `source /data/junda/work/sgl_env.sh`
(npu-codebase `cad0db3d6`): `flash_attn_with_kvcache`,
`decode_paged_kv_kernel`, `mlp_kernel`, and
`block_arguments`/`block_stages`/`block_layer`.  The manifests and adapters in
`/data/junda/work/qsmall/ref/{qwen3_embedding,qwen3_asr,qwen3_tts}` and
`ref/llama/qwen3/qwen3.cpp` define the GGUF contract.

The reported builds and checks use `/data/junda/work/sgl_env.sh`
(npu-codebase `cad0db3d6`, sgl-kernel-axera).  No file outside this directory
was modified.

## Runtime inputs and GGUF sources

The ONNX input names and order are recorded in `inputs.json`; the compiled
graph is the authority.  Runtime tensors under four phase suffixes have these
sources (all layer numbers are the unpermuted Qwen layer number):

| kernel input | source for the ggml-axera adapter |
|---|---|
| `x` | input hidden rows: `token_embd.weight` embedding output (Embedding/ASR), or the talker's text-projection + `token_embd` codec embedding sum (TTS) |
| `final_norm` | `output_norm.weight` |
| `b0_kv` | paged KV pool, no GGUF weight |
| `b*_attn_cos`, `b*_attn_sin` | RoPE/M-RoPE tables computed from the model's `rope_theta` and position ids; no GGUF weight |
| `b*_attn_w_norm_attn` | `blk.N.attn_norm.weight` |
| `b*_attn_w_q`, `b*_attn_w_scale_q` | `blk.N.attn_q.weight`, MXFP8 values and E8M0 scales |
| `b*_attn_w_k`, `b*_attn_w_scale_k` | `blk.N.attn_k.weight`, MXFP8 values and E8M0 scales |
| `b*_attn_w_v`, `b*_attn_w_scale_v` | `blk.N.attn_v.weight`, MXFP8 values and E8M0 scales |
| `b*_attn_w_norm_q` | `blk.N.attn_q_norm.weight / sqrt(128)` |
| `b*_attn_w_norm_k` | `blk.N.attn_k_norm.weight` |
| `b*_attn_w_o`, `b*_attn_w_scale_o` | `blk.N.attn_output.weight`, MXFP8 values and E8M0 scales |
| `b*_mlp_w_norm` | `blk.N.ffn_norm.weight` |
| `b*_mlp_w1`, `b*_mlp_w_scale1` | `blk.N.ffn_gate.weight`, MXFP8 values and E8M0 scales |
| `b*_mlp_w2`, `b*_mlp_w_scale2` | `blk.N.ffn_up.weight`, MXFP8 values and E8M0 scales |
| `b*_mlp_w3`, `b*_mlp_w_scale3` | `blk.N.ffn_down.weight`, MXFP8 values and E8M0 scales |
| `b*_*_w_scale2_*` | constant `1.0` scale2 metadata, not a GGUF tensor |
| `b*_*_w_zeropoint_*`, `b*_*_w_zp_scale*` | constant zero, not a GGUF tensor |
| `b*_mlp_input_permutation` | constant zero, not a GGUF tensor |
| `position_ids` | runtime positions; `[T]` for Embedding, `[3,T]` for ASR/TTS |
| `indptr`, `indices`, `last_page_len`, `cache_location` | runtime paging state; no GGUF weight |
| `output`, `valid_tokens` | runtime output and prefill row count; no GGUF weight |

The ASR converter strips the `thinker.` prefix and maps
`model.layers.N` to the generic `blk.N` names above; its final norm becomes
`output_norm.weight`.  The TTS converter strips `talker.` for the talker text
stack and likewise maps `model.layers.N` to `blk.N`; its `talker.codec_head`
becomes the GGUF `output.weight`, which is outside this kernel.  The TTS
converter folds text embeddings/projection and concatenates the codec
embedding into `token_embd.weight`, also outside this kernel.

## Checks

`check_qwen3.py` quantizes each checkpoint projection to MXFP8, dequantizes it,
and loads exactly those dequantized weights into both the PYRUN kernel and a
Transformers Qwen3 decoder reference.  It checks a 256-row prefill, a second
256-row paged-history call, decode T=1 against row 0 of a 512-token reference,
and the K/V pages against the reference cache.  `--varied-mrope` exercises the
three distinct M-RoPE axes; the real TTS path uses equal axes.

`build_qwen3.py` builds random weights of the real shapes for CRUN and can also
smoke both PYRUN graphs.  `compile_variant.sh` runs CRUN and stops after the
prefill and decode compiled AXModels exist.  `collect_inputs.py` extracts the
actual ONNX input names, shapes, and dtypes in kernel order into `inputs.json`.

Numerical results are appended below when the long PYRUN jobs finish.

## Open issues

* SGL's prefill attention accepts query lengths 128/256 only; decode therefore
  uses SGL's decode primitive inside the same whole-depth forward.  It is not a
  separate layer-group graph and does not have a different KV layout.
* Prefill writes KV for all 256 slots.  `valid_tokens` masks the returned rows;
  it does not remove padding slots from a partial page.  The checked/history
  path uses full 256-token pages.  A host that submits a partial tail should
  either pad to a full page or avoid later treating the padding slots as valid.
* The output head is intentionally outside.  No board-side compiled-model run
  was performed here; only PYRUN numerics and CRUN compilation were checked.
