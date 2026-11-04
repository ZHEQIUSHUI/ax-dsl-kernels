# Qwen3.5-2B whole-model kernels

Status: PASS on board `10.126.35.209` with the native mixed MXFP4 GGUF. The whole 24-layer text path, two 12-layer
decode groups, the output head and the 3072-patch ViT run as whole-model kernels. Three kernel sets cover 16k / 128k /
256k. The 128k set passed needle-in-a-haystack at 32k / 64k / 120k and `-c 4096` PPL 17.8910; the 256k set passed the
same needles and PPL 17.8937. CMM after load is 1.531 GiB / 3.118 GiB / 4.743 GiB. No `RunSync`, `0x800` or NPU error
appeared.

The adapter is the shared `qwen35_whole` host path selected by the manifest's `"model"`. The text path loops over the
six 4-layer blocks in a prefill kernel and reads its row-split projection weights from the prefill banks; the decode
group reads the same banks (`k_<bank>` inputs), so the backbone has one CMM copy. The layer kernels below are IO
descriptors only; the adapter uses them to bind one layer's parameters.

## Context lengths

One prefill call covers `BLOCKS` 4-layer blocks. Its KV input is
`BLOCKS * (CTX / 256) * 2 * 256 * num_key_value_heads * head_dim * 2` bytes. Qwen3.5-2B has 2 KV heads and head dim 256,
so one block costs 256 MiB at 128k and 512 MiB at 256k. The runtime's IO limit is 2 GiB, and `BLOCKS` must divide the
six blocks:

| set | CTX | BLOCKS | calls | KV input per call | prefill kernel | decode group |
|---|---:|---:|---:|---:|---|---|
| 16k | 16384 | 6 | 1 | 192 MiB | `qwen35_2b_prefill256_mx4_kvc_c16384_npu4` | `qwen35_2b_group12_gggagggaggga_mx4_kvc_c16384_npu4` |
| 128k | 131072 | 6 | 1 | 1.5 GiB | `qwen35_2b_prefill256_mx4_kvc_c131072_b6_npu4` | `qwen35_2b_group12_gggagggaggga_mx4_kvc_c131072_npu4` |
| 256k | 262144 | 3 | 2 | 1.5 GiB | `qwen35_2b_prefill256_mx4_kvc_c262144_b3_npu4` | `qwen35_2b_group12_gggagggaggga_mx4_kvc_c262144_npu4` |

Only prefill and decode group change between sets; `lm_head`, ViT, and the two IO descriptors are shared. The three
decode groups use the same `.layout`.

## Kernel files

| role | 16k | 128k | 256k |
|---|---|---|---|
| prefill | `qwen35_2b_prefill256_mx4_kvc_c16384_npu4.axmodel` | `qwen35_2b_prefill256_mx4_kvc_c131072_b6_npu4.axmodel` | `qwen35_2b_prefill256_mx4_kvc_c262144_b3_npu4.axmodel` |
| decode.group | `qwen35_2b_group12_gggagggaggga_mx4_kvc_c16384_npu4.axmodel` | `..._c131072_npu4.axmodel` | `..._c262144_npu4.axmodel` |
| lm_head | `mul_mat_mx4_n248320_k2048_npu4.axmodel` | same | same |
| vit | `qwen35_2b_vit3072_npu4.axmodel` | same | same |
| layer.gdn (IO only) | `qwen35_2b_gdn_mx4_npu1.axmodel` | same | same |
| layer.attn (IO only) | `qwen35_2b_attn_mx4_kvc_c16384_npu1.axmodel` | same | same |

`kernels.md5` lists every file of the three sets. The manifests are `qwen35_2b.manifest.json` (16k),
`qwen35_2b_c131072.manifest.json` and `qwen35_2b_c262144.manifest.json`; each has the board's IO table and its own
`context`. Prefill is `167 in / 1 out`, decode.group `61 in / 40 out`, lm_head `3 in / 1 out`, ViT `31 in / 2 out`.

## Build

Prefill with 128k, six blocks per call (the checker runs the full 24 layers on CRUN against the HF reference, then
writes the production kernel):

```sh
source /data/junda/work/sgl_env.sh
cd models/qwen35_2b/whole/prefill
MODEL=/data/junda/hf/Qwen3.5-2B GG=/data/junda/work/gguf/qwen3.5-2b-mxfp4-mix.gguf \
HF=/data/junda/hf/Qwen3.5-2B CTX=131072 BLOCKS=6 FULL=1 RUN_MODE=CRUN PRODUCE=1 \
KNAME=qwen35_2b_prefill256_mx4_kvc_c131072_b6_npu4 \
OUT=/data/junda/work/qsmall/long9b/build/2b/pro_c131072_b6 nice -n 10 python3 check_chain.py
```

For 256k use `CTX=262144 BLOCKS=3 COS_MIN=0.998 KNAME=qwen35_2b_prefill256_mx4_kvc_c262144_b3_npu4`. The chain checker
builds the first call's kernel, binds each call to its own bank slice and phase-major state pool, and feeds the previous
call's output as the next call's `x`.

Decode groups (`BANKS=1`; `LMAX` is the set's context, `FIRST=0` / `FIRST=12` build the two groups):

```sh
source /data/junda/work/axdsl_env.sh
export PYTHONPATH=$PWD/ax-dsl-kernels/models/qwen35_2b/whole/src
QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 BANKS=1 \
MODEL=/data/junda/k4090/cfg/Qwen3.5-2B WFMT4=1 KVX=0 DEC_SPLIT=1 LMAX=131072 \
STEPS=1 G=12 FIRST=0 DUMP=build/2b/dec_g12_g0 nice python ax-dsl-kernels/models/qwen35_2b/whole/src/q35_group.py build/2b/dec_g12_g0
```

ViT is unchanged from the 16k release; see `vit/vit_full.py`.

## Checks

- 16k prefill with the native MXFP4 weights: per-row cos mean `0.999881`, min `0.999437` over all 256 rows and 24
  layers. PASS.
- 128k prefill chain (`BLOCKS=6`, one call): mean `0.999887`, min `0.998669` (gate `0.998`). PASS.
- 256k prefill chain (`BLOCKS=3`, two calls): mean `0.999885`, min `0.998928` (gate `0.998`). PASS.
- Decode groups with the real MXFP4 weights: output `1.000000` (bitwise), worst GDN state `1.000000`, worst K-cache
  `1.000000`. PASS.
- Host index ranges on the real 32k / 64k / 120k prompts: 30,917 / 61,177 / 113,463 tokens, all within 16k, 128k and
  256k page / KV bounds. KV input is 1,610,612,736 B at both 128k and 256k. PASS.
- ViT full-depth 3072-patch CRUN keeps the 16k release's known generic-gate limitation; board image output is coherent
  and byte-identical between the released and new host. The separate layer ViTs fail the same row-min gate; this is a
  property of the 2B image encoder, not of the whole-model composition.

## Board (`10.126.35.209`, native MXFP4)

All NPU work used `/data/qiushui/lockrun.sh long9b`. Long tests used `llama-completion -c <ctx> -ub 4096 -b 4096 -n 24
--temp 0` on the 27B needle material (code `7391-ALPHA-BRAVO` at 30% of `book_en.txt`) and `llama-perplexity -c 4096
--chunks 4` on the same book.

| gate | 16k | 128k | 256k |
|---|---:|---:|---:|
| load / CMM after load | 18 s / 1,644,498,944 B | 23 s / 3,347,619,840 B | 22 s / 5,092,712,448 B |
| load peak CMM | 1,644,494,848 B | 3,347,619,840 B | 5,092,708,352 B |
| 32k prompt prefill / decode | - | 5751.84 tok/s / 70.65 tok/s | 5687.16 tok/s / 71.06 tok/s |
| 64k prompt prefill / decode | - | 4398.38 tok/s / 52.75 tok/s | 4335.07 tok/s / 49.82 tok/s |
| 120k prompt prefill / decode | - | 3030.14 tok/s / 37.67 tok/s | 3007.56 tok/s / 37.82 tok/s |
| needle found (32k / 64k / 120k) | - | 3 / 3 | 3 / 3 |
| English-book PPL, `-c 4096` | - | 17.8910 +/- 0.53341 | 17.8937 +/- 0.53355 |

The 16k gates are the release's: greedy md5 `b70ae7f89700ea695760612faa37b4e4`, zh PPL `38.8794 +/- 5.87153` against
host CPU `38.0032 +/- 5.83504` (+2.31%), and image md5 `385d9524c8849e281aee19957c4336ac`. Host VmHWM is
765,032 KB / 773,236 KB / 778,376 KB. The lazy host buffer for llama.cpp's own caches lowers 16k CMM from the released
2.05 GB to 1.64 GB. Thermal zones stayed at 53-66 C in the long runs; no protection pause was needed for 2B.

## Manifest

On the board, from the staged kernel directory:

```sh
/data/qiushui/tools/iojson *.axmodel > io.json 2> io.err
```

Then, with the board's `io.json` and the decode `.layout` directory:

```sh
CTX=131072 python3 tools/manifest/mkmanifest.py qwen35_whole io.json <layout dir> qwen35_2b > manifest.json
```

`CTX` also selects the 262144 set. The manifest names only the bounded prefill and context decode group; the board's
loader reads the whole file list from it.
