# Qwen3.5-9B whole-model kernels

Status: prefill, the 16-layer decode groups (including the full 32-layer chain), and the 3072-patch ViT pass their host
CRUN checks. The ViT passes the released-kernel comparison on the same weights and input. Board testing is pending
because the replacement board has not been reachable.

The text path has one prefill kernel for the whole 32-layer model (a runtime loop over the 8 blocks and their
`b0..b3` weight banks) and one decode-group kernel family (16 layers per call, two calls per token). The decode group
reads its row-split projection weights from the prefill banks (`k_<bank>` inputs), so the backbone has one CMM copy.
The layer kernels in `kernels/` are IO descriptors only; the adapter uses them to bind one layer's parameters.

## Kernel files

| role | file |
|---|---|
| prefill | `qwen35_9b_prefill256_mx4_kvc_c16384_npu4.axmodel` |
| decode.group | `qwen35_9b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.axmodel` |
| lm_head | `mul_mat_mx4_n248320_k4096_npu4.axmodel` |
| vit | `qwen35_9b_vit3072_npu4.axmodel` |
| layer.gdn (IO only) | `qwen35_gdn_mx4_npu1.axmodel` |
| layer.attn (IO only) | `qwen35_attn_mx4_kvc_c16384_npu1.axmodel` |

`kernels.md5` has the md5 of every file. `qwen35_9b.manifest.json` is the complete manifest: prefill `167 in / 1 out`,
decode.group `61 in / 53 out`, lm_head `3 in / 1 out`, and ViT `31 in / 2 out`.

## Build

Prefill (the checker builds, runs CRUN against the HF reference with the same MXFP4 dequantized weights, and writes
the kernel):

```sh
source /data/junda/work/sgl_env.sh
cd models/qwen35_9b/whole/prefill
MODEL=/data/junda/work/models/Qwen3.5-9B GG=/data/junda/work/gguf/qwen3.5-9b-Q4_K_M.gguf \
HF=/data/junda/work/models/Qwen3.5-9B FULL=1 RUN_MODE=CRUN PRODUCE=1 \
KNAME=qwen35_9b_prefill256_mx4_kvc_c16384_npu4 nice -n 10 python check.py
```

Decode group (BANKS=1 makes the row-split projections read the prefill banks):

```sh
source /data/junda/work/axdsl_env.sh
export PYTHONPATH=$PWD/ax-dsl-kernels/models/qwen35_9b/whole/src
QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 BANKS=1 \
MODEL=/data/junda/k4090/cfg/Qwen3.5-9B WFMT4=1 KVX=1 DEC_SPLIT=1 LMAX=16384 \
STEPS=1 G=16 FIRST=0 DUMP=build/9b/dec_g16/o \
nice python ax-dsl-kernels/models/qwen35_9b/whole/src/q35_group.py build/9b/dec_g16
```

ViT (the output is the production `KNAME`):

```sh
source /data/junda/work/axdsl_env.sh
cd models/qwen35_9b/whole/vit
MODEL=/data/junda/work/models/Qwen3.5-9B N=3072 GH=48 GW=64 QB=512 BC=256 \
IMAGE=bus3072.png REF_BF16=1 VIT_ATTN_MODE=online VIT_ATTN_FP32=2 \
RUN_MODE=CRUN DUMP=build/9b/vit3072_prod nice -n 15 python vit_full.py
```

Manifest:

```sh
python tools/manifest/mkmanifest.py qwen35_whole build/9b/stage_io/io.json build/9b/stage_io 0 qwen35_9b \
  > build/9b/stage_io/manifest.json
```

## Checks

- Prefill CRUN vs the HF reference with the same quantized weights: per-row cos mean `0.99989`, min `0.99955` over all
  256 rows and all 32 layers. PASS.
- Decode group 0 with the real MXFP4 checkpoint weights at step 0: output cos `0.999978`, worst GDN state cos
  `0.999915`, worst K-cache cos `0.999994`. PASS.
- Full 32-layer decode chain with real weights: group 1 starts from group 0's hidden state but zero recurrent/KV state,
  as required at step 0. Final output cos `0.999711`, worst GDN state cos `0.999640`, worst K-cache cos `0.999887`.
  PASS.
- ViT release equivalence on the exact same `bus256.png` input and BF16-rounded weights: released output rows
  `0.9997616 / 0.9980317` (mean / min), ours `0.9996377 / 0.9978574`; minimum-cos delta `0.000174`, within the
  `0.0005` acceptance rule. The saved references are bit-identical. PASS.
- Host index-range checks are in `checks/index_ranges.py` (positions, pages, KV cells) and passed on the planned real
  inputs (`prompt4k.txt`: 4000 tokens / 16 pages / 160 cells; `prompt_greedy.txt`: 5 tokens / 1 page / 5 cells).
- Board: no run on the replacement `.120` board yet; it was unreachable (`No route to host`). The earlier `.63` board
  returned `0x8006008a` but is known broken and is not used.
