# Qwen3.5-9B whole-model kernels

Status: PASS on board `10.126.35.209` with the native mixed MXFP4 GGUF. The whole 32-layer text path, two 16-layer
decode groups, the output head and the 3072-patch ViT run as whole-model kernels. Three kernel sets cover 16k / 128k /
256k. The 128k set passed needle-in-a-haystack at 32k / 64k / 120k and `-c 4096` PPL 5.0061; the 256k set passed the
same needles and PPL 5.0139. CMM after load is 5.838 GiB / 9.478 GiB / 13.604 GiB. No `RunSync`, `0x800` or NPU error
appeared.

The adapter is the shared `qwen35_whole` host path selected by the manifest's `"model"`. The text path loops over the
eight 4-layer blocks in a prefill kernel and reads its row-split projection weights from the prefill banks; the decode
group reads the same banks (`k_<bank>` inputs), so the backbone has one CMM copy. The layer kernels below are IO
descriptors only; the adapter uses them to bind one layer's parameters.

## Context lengths

One prefill call covers `BLOCKS` 4-layer blocks. Its KV input is
`BLOCKS * (CTX / 256) * 2 * 256 * num_key_value_heads * head_dim * 2` bytes. Qwen3.5-9B has 4 KV heads and head dim 256,
so one block costs 512 MiB at 128k and 1 GiB at 256k. The runtime's IO limit is 2 GiB, and `BLOCKS` must divide the
eight blocks:

| set | CTX | BLOCKS | calls | KV input per call | prefill kernel | decode group |
|---|---:|---:|---:|---:|---|---|
| 16k | 16384 | 8 | 1 | 512 MiB | `qwen35_9b_prefill256_mx4_kvc_c16384_npu4` | `qwen35_9b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4` |
| 128k | 131072 | 2 | 4 | 1 GiB | `qwen35_9b_prefill256_mx4_kvc_c131072_b2_npu4` | `qwen35_9b_group16_gggagggagggaggga_mx4_kvc_c131072_npu4` |
| 256k | 262144 | 1 | 8 | 1 GiB | `qwen35_9b_prefill256_mx4_kvc_c262144_b1_npu4` | `qwen35_9b_group16_gggagggagggaggga_mx4_kvc_c262144_npu4` |

Two blocks at 256k would be exactly 2 GiB, above the `< 2 GiB` kernel-input limit, so the 256k set uses one block per
call. Only prefill and decode group change between sets; `lm_head`, ViT, and the two IO descriptors are shared. The
three decode groups use the same `.layout`.

## Kernel files

| role | 16k | 128k | 256k |
|---|---|---|---|
| prefill | `qwen35_9b_prefill256_mx4_kvc_c16384_npu4.axmodel` | `qwen35_9b_prefill256_mx4_kvc_c131072_b2_npu4.axmodel` | `qwen35_9b_prefill256_mx4_kvc_c262144_b1_npu4.axmodel` |
| decode.group | `qwen35_9b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.axmodel` | `..._c131072_npu4.axmodel` | `..._c262144_npu4.axmodel` |
| lm_head | `mul_mat_mx4_n248320_k4096_npu4.axmodel` | same | same |
| vit | `qwen35_9b_vit3072_npu4.axmodel` | same | same |
| layer.gdn (IO only) | `qwen35_gdn_mx4_npu1.axmodel` | same | same |
| layer.attn (IO only) | `qwen35_attn_mx4_kvc_c16384_npu1.axmodel` | same | same |

`kernels.md5` lists every file of the three sets. The manifests are `qwen35_9b.manifest.json` (16k),
`qwen35_9b_c131072.manifest.json` and `qwen35_9b_c262144.manifest.json`; each has the board's IO table and its own
`context`. Prefill is `167 in / 1 out`, decode.group `61 in / 53 out`, lm_head `3 in / 1 out`, ViT `31 in / 2 out`.

## Build

Prefill with 128k, two blocks per call (the checker runs the full 32 layers on CRUN against the HF reference, then
writes the production kernel):

```sh
source /data/junda/work/sgl_env.sh
cd models/qwen35_9b/whole/prefill
MODEL=/data/junda/work/models/Qwen3.5-9B GG=/data/junda/work/gguf/qwen3.5-9b-mxfp4-mix.gguf \
HF=/data/junda/work/models/Qwen3.5-9B CTX=131072 BLOCKS=2 FULL=1 RUN_MODE=CRUN PRODUCE=1 \
KNAME=qwen35_9b_prefill256_mx4_kvc_c131072_b2_npu4 \
OUT=/data/junda/work/qsmall/long9b/build/9b/pro_c131072_b2 nice -n 10 python3 check_chain.py
```

For 256k use `CTX=262144 BLOCKS=1 KNAME=qwen35_9b_prefill256_mx4_kvc_c262144_b1_npu4`. The chain checker builds the
first call's kernel, binds each call to its own bank slice and phase-major state pool, and feeds the previous call's
output as the next call's `x`.

Decode groups (`BANKS=1`; `LMAX` is the set's context, `FIRST=0` / `FIRST=16` build the two groups):

```sh
source /data/junda/work/axdsl_env.sh
export PYTHONPATH=$PWD/ax-dsl-kernels/models/qwen35_9b/whole/src
QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 BANKS=1 \
MODEL=/data/junda/k4090/cfg/Qwen3.5-9B WFMT4=1 KVX=1 DEC_SPLIT=1 LMAX=131072 \
STEPS=1 G=16 FIRST=0 DUMP=build/9b/dec_g16_g0 nice python ax-dsl-kernels/models/qwen35_9b/whole/src/q35_group.py build/9b/dec_g16_g0
```

ViT is unchanged from the 16k release; see `vit/vit_full.py`.

## Checks

- 16k prefill with the native MXFP4 weights: per-row cos mean `0.999890`, min `0.999146` over all 256 rows and 32
  layers. PASS.
- 128k prefill chain (`BLOCKS=2`, four calls): mean `0.999894`, min `0.999134` (gate `0.999`). PASS.
- 256k prefill chain (`BLOCKS=1`, eight calls): mean `0.999894`, min `0.999347` (gate `0.999`). PASS.
- Decode group 0 with the real MXFP4 weights: output cos `0.999978`, worst GDN state `0.999915`, worst K-cache
  `0.999994`. Full 32-layer chain: output `0.999711`, worst GDN `0.999640`, worst K-cache `0.999887`. PASS.
- Host index ranges on the real 32k / 64k / 120k prompts: 30,917 / 61,177 / 113,463 tokens, all within 16k, 128k and
  256k page / KV bounds. KV input is 1,073,741,824 B per call at both 128k and 256k. PASS.
- ViT full-depth 3072-patch CRUN keeps the 16k release's known generic-gate limitation; board image output is coherent
  and byte-identical between the released and new host.

## Board (`10.126.35.209`, native MXFP4)

All NPU work used `/data/qiushui/lockrun.sh long9b`. Long tests used `llama-completion -c <ctx> -ub 4096 -b 4096 -n 24
--temp 0` on the 27B needle material (code `7391-ALPHA-BRAVO` at 30% of `book_en.txt`) and `llama-perplexity -c 4096
--chunks 4` on the same book.

| gate | 16k | 128k | 256k |
|---|---:|---:|---:|
| load / CMM after load | 30 s / 6,268,108,800 B | 33 s / 10,177,429,504 B | 35 s / 14,606,860,288 B |
| load peak CMM | 6,277,849,088 B | 10,177,429,504 B | 14,737,485,824 B |
| 4k prefill page | 57.114 ms (16 pages, 913.8 ms) | - | - |
| short-context decode | 42.71 tok/s | - | - |
| 32k prompt prefill / decode | - | 2434.11 tok/s / 30.49 tok/s | 2457.50 tok/s / 30.33 tok/s |
| 64k prompt prefill / decode | - | 2027.78 tok/s / 23.56 tok/s | 1989.71 tok/s / 23.34 tok/s |
| 120k prompt prefill / decode | - | 1528.87 tok/s / 17.88 tok/s | 1481.37 tok/s / 17.66 tok/s |
| needle found (32k / 64k / 120k) | - | 3 / 3 | 3 / 3 |
| English-book PPL, `-c 4096` | - | 5.0061 +/- 0.12592 | 5.0139 +/- 0.12610 |

The 16k gates are the release's: greedy md5 `a7785d21b97952f6e180618317c8c836`, zh PPL `20.8258 +/- 2.88086` against
host CPU `20.5512 +/- 2.89327` (+1.34%), and image md5 `4b8ad6e3a712219b00f65917be7ad658`. Host VmHWM is
1,374,128 KB / 1,380,656 KB / 1,386,600 KB. The lazy host buffer for llama.cpp's own caches lowers 16k CMM from the
released 7.17 GB to 6.27 GB. Thermal zones stayed at 50-69 C in the long runs; no protection pause was needed for 9B.

## Manifest

On the board, from the staged kernel directory:

```sh
/data/qiushui/tools/iojson *.axmodel > io.json 2> io.err
```

Then, with the board's `io.json` and the decode `.layout` directory:

```sh
CTX=131072 python3 tools/manifest/mkmanifest.py qwen35_whole io.json <layout dir> qwen35_9b > manifest.json
```

`CTX` also selects the 262144 set. The manifest names only the bounded prefill and context decode group; the board's
loader reads the whole file list from it.
