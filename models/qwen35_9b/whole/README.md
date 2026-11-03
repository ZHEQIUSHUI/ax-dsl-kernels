# Qwen3.5-9B whole-model kernels (WIP)

Status: the prefill and text decode pieces build and pass their host checks; the ViT does not yet pass the acceptance
gate, and the board rejected the prefill with `RunSync 0x8006008a` (work stopped per the task rule).

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
| layer.gdn (IO only) | `qwen35_gdn_mx4_npu1.axmodel` |
| layer.attn (IO only) | `qwen35_attn_mx4_kvc_c16384_npu1.axmodel` |

`kernels.md5` has the md5 of every file. `qwen35_9b_text.manifest.json` is the text-only manifest used for the board
attempt (the ViT role is intentionally absent until its kernel is accepted).

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

ViT (attempted, not accepted): `vit/vit_full.py` with `MODEL=/data/junda/work/models/Qwen3.5-9B N=3072 QB=512 BC=256
IMAGE=bus3072.png REF_BF16=1 VIT_ATTN_MODE=online VIT_ATTN_FP32=2 RUN_MODE=CRUN`. The 256-patch full-depth run (same
source at the time) failed the output gate: mean 0.9997982, min 0.9988794.

Manifest:

```sh
python tools/manifest/mkmanifest.py qwen35_whole build/9b/stage_io/io.json build/9b/stage_io 0 qwen35_9b \
  > build/9b/stage_io/manifest.json
```

## Checks

- Prefill CRUN vs the HF reference with the same quantized weights: per-row cos mean `0.99989`, min `0.99955` over all
  256 rows and all 32 layers. PASS.
- Decode group, step 0 (`STEPS=1`, so only the first 16-layer group): output cos `0.999758`, worst GDN state cos
  `0.999529`, worst K-cache cos `0.999772`. It has not been run over the full 32-layer chain.
- Host index-range checks are in `checks/index_ranges.py` (positions, pages, KV cells) and passed on the planned real
  inputs.
- Board: model load and CMM bank setup completed, then `prefill` returned `0x8006008a` (`RunSync` timeout). No board
  benchmark or release comparison was completed.
