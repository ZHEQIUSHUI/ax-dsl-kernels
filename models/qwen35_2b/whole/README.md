# Qwen3.5-2B whole-model kernels

Status: the whole 24-layer prefill and the two 12-layer decode groups pass their host CRUN checks. The 3072-patch ViT
compiles and its manifest IO is exact, but its generic `.999` row-min gate fails on the real `bus3072.png` image. The
existing separate per-layer ViT kernels fail the same gate, and the whole kernel's merged output is slightly better;
board testing is pending because `.120` is unreachable.

The adapter is the shared `qwen35_whole` host path selected by the manifest's `"model"`. The text path has one prefill
kernel for all 24 layers (a runtime loop over the six blocks and the `b0..b3` banks) and one 12-layer decode-group
kernel family, two calls per token. The decode group reads its row-split projection weights from the prefill banks
(`k_<bank>` inputs), so the backbone has one CMM copy. The layer kernels below are IO descriptors only; the adapter uses
them to bind one layer's parameters.

## Kernel files

| role | file |
|---|---|
| prefill | `qwen35_2b_prefill256_mx4_kvc_c16384_npu4.axmodel` |
| decode.group | `qwen35_2b_group12_gggagggaggga_mx4_kvc_c16384_npu4.axmodel` |
| lm_head | `mul_mat_mx4_n248320_k2048_npu4.axmodel` |
| vit | `qwen35_2b_vit3072_npu4.axmodel` |
| layer.gdn (IO only) | `qwen35_2b_gdn_mx4_npu1.axmodel` |
| layer.attn (IO only) | `qwen35_2b_attn_mx4_kvc_c16384_npu1.axmodel` |

The decode group's FP4 metadata is normalized with `board_io.py`; its manifest IO is byte-for-byte the kernel's board IO.

## Build

Prefill (real MXFP4 weights checked against the HF reference, then the production kernel is written):

```sh
source /data/junda/work/sgl_env.sh
cd models/qwen35_2b/whole/prefill
MODEL=/data/junda/hf/Qwen3.5-2B GG=/data/junda/hf/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf \
HF=/data/junda/hf/Qwen3.5-2B FULL=1 RUN_MODE=CRUN PRODUCE=1 WFMT4=1 \
KNAME=qwen35_2b_prefill256_mx4_kvc_c16384_npu4 nice -n 10 python check.py
```

Decode groups (`BANKS=1`; `FIRST=0` and `FIRST=12` build the two groups):

```sh
source /data/junda/work/axdsl_env.sh
export PYTHONPATH=$PWD/ax-dsl-kernels/models/qwen35_2b/whole/src
QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 BANKS=1 \
MODEL=/data/junda/k4090/cfg/Qwen3.5-2B WFMT4=1 KVX=0 DEC_SPLIT=1 LMAX=16384 \
STEPS=1 G=12 FIRST=0 DUMP=build/2b/dec_g12_g0 \
nice python ax-dsl-kernels/models/qwen35_2b/whole/src/q35_group.py build/2b/dec_g12_g0
```

ViT:

```sh
source /data/junda/work/axdsl_env.sh
cd models/qwen35_2b/whole/vit
MODEL=/data/junda/hf/Qwen3.5-2B N=3072 GH=48 GW=64 QB=512 BC=256 \
IMAGE=bus3072.png REF_BF16=1 VIT_ATTN_MODE=online VIT_ATTN_FP32=0 \
RUN_MODE=CRUN DUMP=build/2b/vit3072 KNAME=qwen35_2b_vit3072 nice -n 15 python vit_full.py
```

## Checks

- Prefill CRUN vs the HF reference with the same MXFP4 weights: per-row cos mean `0.999881`, min `0.999437` over all
  256 rows and 24 layers. PASS.
- Decode groups 0 and 1 with the real checkpoint weights, and the full 24-layer chain: output `1.000000` (bitwise),
  worst GDN state `1.000000`, worst K-cache `1.000000`. PASS.
- Host index ranges: 24 layers / 6 blocks / 64 pages / context 16384; ViT 3072 patches / 6 query tiles / 12 key blocks.
  PASS.
- ViT full-depth 3072-patch CRUN on `build/9b/bus3072.png`: `x` rows mean/min `0.998106 / 0.160082` and output rows
  `0.992441 / 0.403248`; 13 x rows and 16 output rows are below `0.9`. FAIL against the generic `.999` gate.
  The separate per-layer kernels fail the same way on the real image (`bus256.png`: layer-15 rows
  `0.999353 / 0.945638`, merged output rows `0.997783 / 0.920831`), while the whole kernel's merged output is
  `0.998151 / 0.943694`. The failure starts with the model's layer-10/23 massive activations (reference row 211 norm
  jumps from `18.5` to `161.9`, then to `5212.6` at layer 23), which makes BF16 attention rows hypersensitive. Row 58
  (norm `8.49`) is the worst chained row at layer 15; the large row 211 itself stays at cos `0.999997`, so a tiny key-row
  error is amplified by softmax. Three precision variants did not improve the baseline (`0.997832 / 0.728709`):
  exact-HF residual rounding `0.997586 / 0.689126`, output-accumulator FP32 `0.998269 / 0.679624`, and FP32 softmax
  `0.997927 / 0.686096`. This is the same release limitation seen in the AXERA 9B 3072 ViT, not a whole-kernel
  composition bug. Logs: `build/2b/vit3072.log`, `build/2b/vit_layers_img256_chain_rows.log`,
  `build/2b/vit_dbg_layers16.log`, `build/2b/vit_img256_l16_round.log`, `build/2b/vit_img256_l16_f32sm.log`.
- Board: pending, `.120` unreachable (`No route to host`). There is no local AX8860 2B release package, so the board
  release comparison is unavailable until one is provided.
