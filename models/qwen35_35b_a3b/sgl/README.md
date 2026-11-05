# Qwen3.5/3.6-35B-A3B NVFP4 kernels (sgl-kernel-axera)

These are the other team's fused Qwen3.5/3.6-35B-A3B kernels, rebuilt at the real
35B shape and used from llama.cpp by the `qwen35_moe_sgl` adapter.

- upstream: `ZHEQIUSHUI/sgl-kernel-axera`, branch `fix/sglang_axera_opt`,
  commit `866053298cb072c5eb39222582b3ff3563a6dbc4`.
- model: 40 layers (30 GDN + 10 GQA), hidden 2048, 256 experts top-8,
  context 16384 (64 KV pages x 256 cells), 4 NPU cores.
- weights: NVFP4 (E2M1 pairs + E4M3 scale per 16 + one FP32 tensor scale) for
  every projection and routed expert. Norms/conv taps are BF16. One copy of
  the 189 runtime banks (18.03 GiB) is shared by decode and prefill.

## Artifacts

`kernels.md5` lists md5, size and role. The `.axmodel` files themselves are
not in this repository.

| role | file | bytes | compile |
|---|---|---:|---:|
| decode (all 40 layers, 10 block loop) | `sgl35_decode.axmodel` | 1,372,258 | 193 s |
| prefill.gdn (256 rows) | `sgl35_gdn.axmodel` | 1,002,250 | 65 s |
| prefill.attn (256 rows, M-RoPE) | `sgl35_flash.axmodel` | 441,842 | 25 s |
| prefill.moe (fixed 4096 rows) | `sgl35_moe4096.axmodel` | 594,422 | 45 s |
| prefill.padding (4096 rows) | `sgl35_padding.axmodel` | 33,090 | 5 s |
| lm_head (release) | `mul_mat_mx4_n248320_k2048_npu4.axmodel` | 1,331,441 | - |
| vit (release, whole encoder) | `qwen35_35b_a3b_vit3072_npu4.axmodel` | 92,225,540 | - |

`io.json` is the board `iojson` dump of all seven kernels (name, bytes, dtype,
shape). `scripts/make_manifest.py` turns it into the adapter's `manifest.json`.

## Build

The build environment is the npu-sgl venv plus the npu-sgl36 worktree:

```sh
export PATH=/data/junda/work/npu-sgl/.venv/bin:$PATH
source /data/junda/work/npu-sgl36/script/npu_dev
export PYTHONPATH=$PWD/python:$PWD/tests:/data/junda/work/npu-sgl36/axnn
```

### Decode + weights

```sh
# Quantize the BF16 HF checkpoint, dump the 189 banks, compile the 40-layer model.
PROCS=24 python scripts/build_decode.py /data/junda/hf/Qwen3.5-35B-A3B out_sgl_pack_v2 64
# -> out_sgl_pack_v2/{<bank>.bin, manifest.json, sgl35_decode.axmodel}
```

`NLAYERS=4|8` builds the shorter packs used for the CRUN blockwise checks;
`COMPILE=0` dumps banks only. Quantize+dump takes 565 s with 24 workers,
compile 193 s. The v2 pack stores `o_norm_w` as BF16 (the first pack's FP32
version is not compatible).

### Prefill

```sh
RUN_MODE=CRUN python scripts/build_prefill.py gdn     out_sgl_prefill
RUN_MODE=CRUN python scripts/build_prefill.py flash   out_sgl_prefill
RUN_MODE=CRUN python scripts/build_prefill.py moe     out_sgl_prefill
RUN_MODE=CRUN python scripts/build_prefill.py padding out_sgl_prefill
```

Each command writes `out_sgl_prefill/sgl35_<role>.axmodel`. The kernels are
compiled at `CTX=16384` and `ROWS=4096` (the MoE workspace); `SIM` unset stops
the simulator once the artifact exists.

### Manifest

```sh
iojson <kernel-dir> > io.json
python scripts/make_manifest.py io.json <kernel-dir> [release/kernels/manifest.json]
```

The last argument supplies the release `lm_head` and `vit` entries; it defaults
to the 35B release package used on this board.

## Checks

```sh
# 4/8-layer CRUN decode vs the BF16 HF reference (NLAYERS=4/8 packs)
RUN_MODE=CRUN python scripts/ref_decode.py out_sgl_pack_n8 ref_n8
RUN_MODE=CRUN python scripts/check_decode_crun.py out_sgl_pack_n8 ref_n8 10

# static ABI check: adapter bank bindings vs the pack and io.json
python scripts/check_adapter_abi.py io.json out_sgl_pack_v2/manifest.json \
    <kernel-dir>/manifest.json
```

Measured CRUN cos (vs BF16): 4 layers mean 0.99489 / min 0.99155; 8 layers,
10 tokens, mean 0.99378 / min 0.99036. The 8-layer check catches the 27B
failure mode (every block reading block 0's weights).

Quality: CPU zh PPL (2 chunks of 512) of the BF16 GGUF rewritten with
NVFP4-dequantized tensors is 16.8576 vs Q8_0 16.4566 (+2.44%). The board's NPU
PPL with these banks is 16.8898.

## Runtime IO (non-bank entries)

Decode: `x`/`attention` bf16 [1,2048], `position_ids` i32 [3,1],
`indptr` i32 [2], `indices` i32 [64], `last_page_len` i32 [1],
`cache_location` i32 [2], output bf16 [1,2048].

Prefill GDN: `x` bf16 [256,2048], `position_ids` absent (recurrent state),
`valid_tokens` i32 [1], state/conv banks, 5 outputs.

Prefill attention: `x` bf16 [256,2048], `position_ids` i32 [3,256],
`paged_kv_cache` bf16 [64,2,256,2,256], `indptr`/`indices`/`cache_location`,
merged `cos_sin_cache` f32 [1,16384,64].

Prefill MoE: `x` bf16 [4096,2048], 24 inputs, one [4096,2048] output;
`w_up` aliases `w_gate`. Padding: `activation` bf16 [4096,2048] and
`token_count` i32 [1], in place.

The full tables are in `io.json`.
