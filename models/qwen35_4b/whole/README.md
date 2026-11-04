# Qwen3.5-4B whole-model kernels

Status: PASS on board `10.126.35.209`. The text path, 16-layer decode groups and output head pass their host CRUN
checks with the real MXFP4 weights, and every board gate passes. The full-depth ViT produces the board-passing
kernel; its generic row-min CRUN gate fails as it does for the 9B/2B full-depth ViTs, with cos reported below. NPU
zh PPL is `25.5670 +/- 3.66917` against host CPU `25.1465 +/- 3.65724` on the same GGUF, **+1.67%**, inside the 5%
gate. Peak CMM is 3,655,290,880 B (3.405 GiB), steady CMM 3,655,299,072 B (3.405 GiB), host VmHWM 1,046,280 KB at
load and 1,063,012 KB steady.

The text path has one 256-row prefill kernel for all 32 layers (a runtime loop over the eight `b0..b3` weight banks)
and one 16-layer decode-group kernel called twice per token. The decode group reads every projection, including
`w_down`, `w_out` and `w_o`, from the prefill banks (`BANKS=1 KSTRIDE=1`), so the backbone has one CMM copy. The
layer kernels in `kernels/` are IO descriptors only; the adapter uses them to bind one layer's parameters. Context is
16384 only; MTP, int8 KV and longer contexts are out of scope for this package.

## Kernel files

| role | file | bytes | md5 |
|---|---|---:|---|
| prefill | `qwen35_4b_prefill256_mx4_kvc_c16384_npu4.axmodel` | 4,716,020 | `8fc639ed0bcee2b706673a0422d20b02` |
| decode.group | `qwen35_4b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.axmodel` | 11,919,246 | `1841f206fcf7e483b43d9cd3ac908aee` |
| lm_head | `mul_mat_mx4_n248320_k2560_npu4.axmodel` | 3,698,573 | `8db9606dba2adf81e50ea82d77f8be4a` |
| vit | `qwen35_4b_vit3072_npu4.axmodel` | 65,076,332 | `ebb353722bded6e62e36413198b6e498` |
| layer.gdn (IO only) | `qwen35_4b_gdn_mx4_npu1.axmodel` | 734,863 | `f08e0af60ae36b61cb69c427dbe210b8` |
| layer.attn (IO only) | `qwen35_4b_attn_mx4_kvc_c16384_npu1.axmodel` | 750,805 | `102ae803be92ce3053e08ee51cfc02d8` |

The decode layout is `qwen35_4b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.layout` (2,674 B, md5
`f2bc41f3a10decfe3456a183bad6fa52`). `kernels.md5` lists all seven files plus the manifest.
`qwen35_4b.manifest.json` (md5 `f649c5bd854f4cd9d437743f8041033b`) has prefill `167 in / 1 out`, decode.group
`77 in / 53 out`, lm_head `3 in / 1 out`, ViT `31 in / 2 out`, and the two layer IO descriptors. The decode group's
FP4 metadata is normalized with `board_io.py`; its manifest IO is byte-for-byte the board IO.

## Build

Prefill (the checker runs CRUN against the HF reference with MXFP4-dequantized GGUF weights and writes the kernel):

```sh
source /data/junda/work/sgl_env.sh
cd models/qwen35_4b/whole/prefill
MODEL=/data/junda/hf/Qwen3.5-4B GG=/data/junda/work/gguf/qwen3.5-4b-mxfp4.gguf \
HF=/data/junda/hf/Qwen3.5-4B FULL=1 RUN_MODE=CRUN PRODUCE=1 \
KNAME=qwen35_4b_prefill256_mx4_kvc_c16384_npu4 nice -n 10 python check.py
```

Decode groups (`BANKS=1 KSTRIDE=1` makes every projection read the prefill banks):

```sh
source /data/junda/work/axdsl_env.sh
export PYTHONPATH=$PWD/ax-dsl-kernels/models/qwen35_4b/whole/src
COMMON="QW_RANDOM=0 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 BANKS=1 KSTRIDE=1 \
MODEL=/data/junda/hf/Qwen3.5-4B WFMT4=1 KVX=1 DEC_SPLIT=1 LMAX=16384 STEPS=1 G=16 KPFX=qwen35_4b"
env $COMMON FIRST=0 DUMP=build/4b/dec_actual0/o SAVE_PT=$PWD/build/4b/dec_actual0.pt \
  nice -n 10 python ax-dsl-kernels/models/qwen35_4b/whole/src/q35_group.py build/4b/dec_actual0
env $COMMON FIRST=16 INIT_STATE=zero INIT_PT=$PWD/build/4b/dec_actual0.pt \
  DUMP=build/4b/dec_actual1/o SAVE_PT=$PWD/build/4b/dec_actual1.pt \
  nice -n 10 python ax-dsl-kernels/models/qwen35_4b/whole/src/q35_group.py build/4b/dec_actual1
```

Output head (the tied `token_embd.weight`, 248320 x 2560):

```sh
source /data/junda/work/axdsl_env.sh
export PYTHONPATH=/data/junda/work/llama.cpp/gguf-py:$PYTHONPATH
cd models/qwen35_4b/whole/lm4
GG=/data/junda/work/gguf/qwen3.5-4b-mxfp4.gguf LM_K=2560 OUT=data nice -n 10 python gen_data.py
LM_C=248320 LM_K=2560 CORES=4 nice -n 10 python build_lm4.py 1
```

ViT (3072 patches, the whole 24-layer encoder and merger in one call):

```sh
source /data/junda/work/axdsl_env.sh
cd models/qwen35_4b/whole/vit
MODEL=/data/junda/hf/Qwen3.5-4B N=3072 GH=48 GW=64 QB=512 BC=256 \
IMAGE=/data/junda/work/qsmall/whole4b/build/4b/assets/bus3072.png \
REF_BF16=1 VIT_ATTN_MODE=online VIT_ATTN_FP32=0 RUN_MODE=CRUN \
DUMP=/data/junda/work/qsmall/whole4b/build/4b/vit3072_real nice -n 10 python vit_full.py
```

Manifest (after collecting every kernel's board IO into `io.json`):

```sh
python tools/manifest/mkmanifest.py qwen35_whole build/4b/stage_io/io.json build/4b/stage_io qwen35_4b \
  > build/4b/stage_io/manifest.json
```

## Checks

- Prefill CRUN vs the HF reference with MXFP4 projection weights: per-row cos mean `0.999851`, min `0.999035` over
  all 256 rows and 32 layers. Rows 0/1/255 are `0.9997 / 0.9995 / 0.9999`. PASS. Log
  `build/4b/logs/prefill_real_crun.log`.
- Decode group 0 with real weights at step 0: output cos `0.999988`, worst GDN state `0.999983`, worst K-cache
  `0.999996`. PASS. Log `build/4b/logs/dec_actual0_save.log`.
- Full 32-layer decode chain with real weights: group 1 starts from group 0's hidden state with zero recurrent/KV
  state. Output cos `0.999923`, worst GDN state `0.999777`, worst K-cache `0.999887`. PASS. Log
  `build/4b/logs/dec_actual1.log`.
- Output head rows 0-3 vs torch: `1.000001907 / 1.000002384 / 1.000004768 / 1.000002265`. PASS. Logs
  `build/4b/logs/lm_gen_data.log`, `build/4b/logs/lm_build.log`.
- ViT 3072 real-weight CRUN: x after layer 23 mean `0.995667`, min `0.712629` at row 142; output mean `0.997883`, min
  `0.691175`. The generic `.999` row-min gate fails (the same full-depth behaviour as the 9B/2B ViTs); the required
  board image-description gate passes with this kernel. Log `build/4b/logs/vit3072_real_crun.log`.
- Host index ranges: 32 layers / 8 blocks / 64 pages / context 16384; ViT 3072 patches / 6 query tiles / 12 key
  blocks / 4x6 row tiles / 768 merged rows. PASS. Log `build/4b/index_ranges.log`.

## Board

All NPU work used `/data/qiushui/lockrun.sh whole4b` on `10.126.35.209`; the final six-kernel run is tag `4bfinal`.
No `RunSync`, `0x800`, `NPU error` or kernel-failure line appeared. Thermal zones after the run were 54.8-57.1 C.

| gate | result |
|---|---|
| greedy text (`-n 64`) | rc 0, md5 `4760137c7a9b43e223374586631dd736`; coherent reasoning prefix |
| greedy text (`-n 256`, supplementary) | rc 0, md5 `0662e94c7e9fc15781e9f420a67aed7f`; reaches “Paris” |
| zh PPL (512, 2 chunks) | NPU **25.5670 +/- 3.66917**; host CPU **25.1465 +/- 3.65724**, **+1.67%**, PASS |
| 4k prefill, page timer | **32.605 ms per 256-row page**; 16 pages `521.7 ms`, all-page wrapper `549.9 ms` |
| 4k prefill, 4008-token total | about **7.68k tok/s** excluding the one-time 5.62 s bank conversion |
| load / decode | 9.07 s / **60.60 tok/s** (16.50 ms/token, 127 runs) |
| image (bus.jpg, 768 max tokens, n=128) | rc 0, 25 s, md5 `563d7a39ea1fef3bed5d0b819bc3ce5c`; coherent bus description |
| ViT | **204.465 ms (one call)**; patch CPU 5.718 ms, tables 14.956 ms, weight load 429.889 ms |
| load peak CMM | **3,655,290,880 B (3.405 GiB)** |
| steady CMM | **3,655,299,072 B (3.405 GiB)** |
| host VmHWM at load / steady | **1,046,280 KB** / 1,063,012 KB |
| host RSS at load | anon 264,812 KB, file 766,836 KB |
| host RSS steady | anon 375,600 KB, file 687,412 KB |
| server load | 21 s |

The profiled `lm_head` is 1.981 ms/call; the `output head` 1643 ms/call is the one-time output-weight conversion in
`bind`, not a per-token projection. Similarly, the 10.2 s first-prompt prompt-eval time includes the 5.6 s prefill-bank
conversion; the `prefill (page)` timer is the steady prefill number.

The first board attempt used a static `dist/4b/bin` that embedded an older AX engine. It stopped during context
creation with `0x80060086` (`NPU_DEV_STATUS_MEM_ERROR`) before any inference. The kernel itself was valid: the board
`iojson` tool and a direct `AX_SYS_Init / AX_ENGINE_Init / CreateContext` diagnostic all returned 0. Replacing the
static tree with the dynamic 9B release build at the same llama.cpp commit (`dc4a61d`) made the board use the working
`/opt/lib` engine and all gates passed. The superseded static binaries are not part of this package.
