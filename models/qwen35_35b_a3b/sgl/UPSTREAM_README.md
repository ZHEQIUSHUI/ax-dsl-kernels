<h2 align="left">
SGLang Kernels AXERA
</h2>

## Introduction

SGLang Kernels for Axera NPU

## Software and hardware

Supported Hardware Models: Axera 8860 Series Products
Platform: aarch64/x86
Supporting Software

## Quick Start

### Qwen target model kernels

The Qwen3.8-27B fused forward, its MLP, and the Qwen3.5-35B-A3B fused decode
use four cores. Launch these kernels with `[4, 4]`. Shared DSL utilities retain
their general grid interfaces.

Qwen3.5 MoE prefills above 256 tokens visit experts in ID order. Each expert
processes its own active token tiles; empty experts load no projection weights.
All four cores cooperate on each expert, sharing activations while keeping
their Gate/Up and Down weight shards in private OCM for the entire expert.
Gate/Up uses one MatMul with native parallel IC512 jobs and a private BF16
NMC destination. Down uses M64/OC256 tiles with BF16 output.
The routing probability scales the BF16 SwiGLU activation before Down,
reducing the probability-multiply pass to the narrower intermediate channels.
A dynamic Range visits the sorted nonempty experts with an unroll factor of one.
It prefetches only the next expert's immutable routes and weight shards; DDR-backed large
prefills also prefetch that expert's first input tile. The final expert consumes
the cache without another lookahead load. Each core retains its output shard
in private OCM throughout the expert traversal.
Token preparation shares the same selection and padding-mask logic for direct
loads and prefetched tiles. Loop state contains the changing expert data and
output accumulator; immutable input tensors are captured by the loop callbacks.
Preparation writes normalization and routing-map initialization directly to
their destination tensors.
The prefill router keeps each core's immutable weight and scale shard in
private OCM across preparation tiles. Router GEMM keeps the full 256-row tile
and uses native IC512 partials with FP32 NMC accumulation across four CONV
units per core. Channel scaling still produces BF16 logits before TopK.
Shared-expert preparation also retains its immutable projection operands across
tiles and uses M64 Gate/Up tiles with FP32 accumulation before the BF16 cast.
The single-output shared-expert gate uses TENG multiplication and FP32 reduction
on each core's own normalized rows before the full input tile is published.
Its dot result is rounded to BF16 before sigmoid.
Loop indices and structural constants are created outside the callbacks;
each Range owns a separate index tensor. The BF16 NMC destination is still
cleared for every token tile using a reusable scalar zero.

Single-token NVFP4 MoE decode stages up to four selected experts per private
weight buffer. The production top-8 path uses two sets, allowing the second
group's Gate/Up and Down reads to overlap the first group's computation.
Dynamic expert traversals reuse fixed buffer identities instead of carrying
new weight tensors through Range. A static odd tail owns a separate set to
preserve earlier readers. The fast path requires one intermediate-channel
wave and at most 4 MiB of buffered weights and group scales per core.
Routing contributions retain the original two-expert FP32 reduction order.
Down weights and group scales gather contiguous output-row blocks together.
The production path uses 64 rows per index, giving 32768 packed weight
elements and 2048 scale elements per gather row while retaining the same
expert selection, buffer layout, and MatMul dimensions.

Attention projections and MoE output reductions use BF16 partials. GDN
recurrent state, normalization reductions, and router projections retain FP32.
The Qwen3.8 MLP uses BF16 gate/up results. For shared output, FP4 down-projection
prefill uses OC256 partials and BF16 NMC moves into shared OCM. Four
cores rotate through the output shards, publishing each completed round
before the next contributor. DDR output reduces the published partials in
each owner's private BF16 shard.
FP4 prefill uses IC768 tiles; down-projection decode uses IC512 tiles.
Single-token dense MLP stages complete Gate/Up scale rows per OC tile and
the owned Down IC scale shard across output rows. The latter is bounded to
2 MiB per core; the 27B shape reads C272 instead of C32/C16 DDR scale waves.
Prefill retains its per-wave scale reads and the existing compact compute tiles.
Single-token GDN NVFP4 input projections stage complete weight rows when
there are at most two IC waves and the weight/scale payload fits 3 MiB.
Computation retains the original IC-wave boundaries and BF16 accumulation.
GDN decode output projections load all output rows of an owned IC scale
shard together before slicing the OC rounds and IC512 arithmetic waves in
OCM. The production DDR scale gather has 64 channels and stages 128 KiB
per core. The complete scale slab is bounded to 256 KiB, and each original
group-scale byte is loaded once.
Multi-token group-16 GDN projections double-buffer complete OC tiles when
each tile's weight and scale payload fits 512 KiB. Wider projections retain
their IC pipeline and cache up to 1 MiB of complete scale rows when at least
four IC waves share them. Arithmetic still uses the original BF16 IC waves.
GDN prefix values are converted to head-major layout before exponentiation,
so the prefix and its exponential share one layout conversion.
Prefill output projections with a 1--2 MiB scale matrix read complete rows
cooperatively into shared OCM; each row is transferred once for all IC owners.
Completed model activations and KV rows use BF16. GQA prefill Q/K/V projections
use IC1024 tiles. Shared K/V projections are published
once and read directly by their paired query cores. These shared projection
buffers do not change paged KV addressing or the runtime cache layout.

Shared-KV decode writes RoPE query rows and gated attention results directly
into their shared channel shards. Gathered query rows are snapshotted into
private OCM before entering the history-page Range: paired cores can execute
different numbers of page iterations. The snapshot separates shared-input
reads from those independently sized loops.
History V pages are gathered at full head width and partitioned on chip for
the existing PV compute tiles. The four-KV-head decode path also shares
complete output-projection scale rows, bounded to 2 MiB, across IC owners.

### Compile and Run

### Installation

### Test

Execute sgl_kernel_axera test scripts, for example

```bash
python3 tests/python/sgl_kernel_axera/test_hello_world.py
```

### GDN partial prefill

`GDNKernel` accepts a BF16 `[1, 256, hidden]` input with a runtime
`valid_tokens` tensor: INT32 `[1]`, containing a value from 1 through 256.
Only the first `valid_tokens` output rows are defined. Recurrent state and all
three convolution histories stop at that real sequence end. Padding may contain
nonzero activations; it does not advance either kind of state.

The length is a runtime input, so different tail lengths share one kernel
signature, including the full block with `valid_tokens=256`. Decode omits it.
The convenience `gdn()` wrapper supplies 256 when called with a complete block.
Qwen3.8 uses one fixed-shape fused prefill graph for both full blocks and tails.
Qwen3.6 passes the same runtime length to its independent GDN launches. Both
paths also pass a runtime length of 256 for full blocks, keeping the prefill ABI
identical across valid lengths. Framework token counts,
paged KV lengths and the selected logits row continue to describe real tokens.

Text serving pads token IDs on the host before H2D and reserves positions in
256-row buckets. Metadata consumes a runtime INT32 `[1]` token count, generates
per-block valid lengths, and masks padded positions. The MoE path clears padded
activations with a fixed-shape kernel and a runtime count. Neither operation
specializes on the actual tail length. Caller-supplied fixed-bucket token IDs
must already have zero padding.
AXERA ChunkCache retains an INT32 view of the active request's slot-table row;
appending slots preserves the prefix, and retraction clears it before row reuse.
It does not copy or cast a variable-length prefix after each prefill.

Ordinary AXERA generation with one usable request-pool slot relays its sampled
token tensor directly to the next forward, without a FutureMap scatter/gather
pair. Larger request pools retain indexed token relay. The sampler owns two
output buffers; the scheduler consumes each round's result and `copy_done`
before that slot is reused. Forward-stream FIFO ordering protects the input
read. Result D2H keeps its independent stream and lifetime tracking.

Attention history page counts are runtime inputs; an empty history uses a
zero-trip Range that preserves the initial softmax state. Startup initializes
model-owned resources without dummy forwards or server warmup requests. Kernels compile
on demand from real requests. Qwen3.8 retains one decode graph and one 256-row
prefill graph; runtime lengths do not select new compilation signatures.
Different model weights, KV capacity, dtype, core count, or physical buckets
still select different signatures. Allocation and request-lifecycle kernels
have their own finite signatures, independent of the prefill/decode graphs. Multimodal position preparation is
outside this text-serving contract.

From the repository root:

```bash
source script/npu_dev
PYTHONPATH="$PWD/sgl-kernel-axera/python:$PYTHONPATH" RUN_MODE=PYRUN \
  pytest -q sgl-kernel-axera/tests/test_gdn_attn.py -k runtime_tail
```
