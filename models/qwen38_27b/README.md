# Qwen3.8-27B (AX8860, 4 cores)

The kernels of the llama.cpp ggml-axera adapter `qwen38_27b` (`ggml/src/ggml-axera/models/qwen38_27b`). Weights are
runtime inputs: the adapter converts them from the GGUF at load (MXFP4 GGUF rows as they are; other types, e.g. a
community Q4_K_M / Q8_0, quantized to MXFP4).

| file | role | source |
|---|---|---|
| `qwen38_27b_prefill256_mx4_kvc_c16384_npu4` | `prefill` | `prefill/build27.py`: sgl-kernel-axera `models/qwen38_27b/forward.py` (npu-codebase cad0db3d6) at the 27B shapes, MXFP4 weights. The whole model in one call per 256 cells: a runtime loop over the 16 blocks (3 delta-net + 1 attention layers, dense MLP), weights in banks `[16, ...]` indexed by the block, valid rows at runtime |
| `qwen38_27b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4_ks` (+ `.layout`) | `decode.group` | `decode/q35_group.py` with `BANKS=1 KSTRIDE=1` (ax-dsl-kernels ab3e90c + the prefill kernel's state layouts): 16 layers per call, 4 calls per token. One copy of the weights: every projection weight is read from the prefill kernel's banks (inputs `k_<bank name>`: the group's 4 blocks of each bank; the row-split ones by rows, the column-split ones — `w_down` / `w_out` / `w_o` — as each core's column strip of the row-major bank), the packs hold only the small parameters (norms, conv, A / dt, scales of the strips: 1.8 MB) |
| `qwen35_group16_verify4_m4_gggagggagggaggga_mx4_kvc_c16384_npu4_ks` | `decode.verify` | `verify/verify_group_m4.py` (on `decode/q35_group.py`, the same banks and packs): the decode group for 4 rows per call (speculative decoding: a draft of up to 3 tokens after the last cell), with the recurrent / conv state after each row written to per-row snapshots (`ssnap`, `cqsnap`, `cksnap`, `cvsnap`); `ntok` valid rows. The attention rows run one by one; the GDN recurrence head by head over the rows |
| `qwen35_mtp_t1_mx4_kvc_c16384_npu4`, `qwen35_mtp_m8_...`, `qwen35_mtp_prefill256_...` | `mtp.t1`, `mtp.m8`, `mtp.prompt` | `mtp/mtp_kernel.py` (`mtp/build_mtp.py`, `mtp/build_mtp_m8.py`): the model's MTP block (`blk.64.nextn.*`: eh_proj over [norm(embedding), norm(h)], one GDN layer, final norm) for 1 row (a draft step), up to 8 consecutive rows (the rows accepted by a verification) and 256-row pages (the prompt); its own paged KV; weights in packs |
| `mul_mat_mx4_n65536_k5120_npu4` (+ `draft_vocab.i32`) | `lm_head.draft` | `head/build_lmd.py`: the output head over a 65536-token draft vocabulary (`head/count.py`, `head/mkvocab.py`: the most frequent tokens of wiki en / zh and code; the host gathers their rows of the MXFP4 head) — a draft step's logits in 1.1 ms instead of 4 |
| `mul_mat_mx4_m4_n248320_k5120_npu4` | `lm_head` | `head/build_lm4.py`: the MXFP4 output head for up to 4 rows per call (`m` valid rows): a verification's 4 logits rows in one call |
| `qwen38_27b_vit3072_npu4` | `vit` | `vit/vit_full.py` (on `vit/q35_vit.py`'s parts): the whole vision encoder in one call, the 27 layers unrolled, then the post norm and the 2x2 merger; up to 3072 patches (the patch count at runtime: `nblk` key blocks of 256 + a key mask); bf16 weights stacked `[27, ...]`. Head-parallel q / k / v and attention per core, the post-attention rows in rounds of 4 tiles; cores exchange rows through the shared OCM only. Tiles QB 512 / BC 256 (board: 275 ms; QB 128: 290 ms, QB 1024: 295 ms) |

State layouts shared by prefill and decode (the adapter's CMM pools): KV cache paged `[16 layers][64 pages][K, V][256][4 x 256]`
bf16; delta-net state `[48 layers][48 heads][128][128]` f32; conv state q / k `[48][3][2048]`, v `[48][3][6144]` bf16,
oldest row first; value heads in HF (grouped) order. The decode group writes the new K / V row given by the host
(`pn = cell, blocks, page * 512 + offset`).

Build: `NPU_DECODE=<npu-codebase@7913a0599> NPU_PREFILL=<npu-codebase@cad0db3d6> HF_27B=<HF Qwen3.8-27B> ./build.sh <out>` builds
the decode group, the prefill and the ViT kernels (+ layout). The verify, MTP and head kernels are built by the scripts in
`verify/`, `mtp/` and `head/` as they were run (npu-codebase 7913a0599; they name their data and source directories on the
build host: set `sys.path` / the paths at the top of each to this directory's `decode/` and to a dump of the real weights
made by `verify/prep_verify.py` / `mtp/prep_mtp.py` from the GGUF). `manifest.json`: the kernels' IO tables read on the board
(`iojson`) + the pack members from the decode `.layout` (the `k_*` inputs are bound by the host to the prefill banks of the
same name: `b<phase>_<name>`, block j // 4 of layer j).

Checks: `verify/check_verify_m4_crun.py` runs the verify kernel on CRUN with the real weights for T = 2 .. 4 against a
torch reference (output / KV / state and conv snapshots: min cos 0.99989); `mtp/check_mtp.py`, `mtp/check_mtp_m8.py` the MTP
kernels (cos > 0.9999); `prefill/check27.py` runs block 0 of the prefill kernel (PYRUN) with weights converted from the GGUF as the
adapter does against transformers' layers (cos 0.99995 with the GGUF's weights); `decode/q35_group.py` compares the
group with the single-core layer kernels in the simulator; `vit/vit_full.py` with `RUN_MODE=PYRUN` (N=256, VIT_L=27) runs the ViT kernel on the HF weights against transformers' vision encoder (layer 26 cos 0.99982, merged output 0.99976).

Context: one kernel set for up to 262144 cells (`kernels.md5`; the adapter reads the context from `manifest.json`) with
an 8-bit KV cache: K / V rows stored as e4m3 x a constant scale (`KV8=2`, `_kvf8`: the K page is read by the attention as
a native MX matmul with a constant E8M0 scale, V is widened to bf16; `decode/q35_group.py`, `verify/`, and the prefill
through `prefill/sgl_kv8_kvf8.patch` on sgl-kernel-axera at `prefill/sgl_base_commit.txt`), so a KV page is half the
bytes of the bf16 one and the attention of a long context reads half as much. The host sizes the KV pools by llama.cpp's
`-c` (`GGML_AXERA_KV_PAGES` / the `cache_k_l` allocation), so the NPU memory follows the context given, not the kernel's
maximum; the prefill kernel is one block per call (`BLOCKS=1`, `_b1`: the KV input under the runtime's 2 GiB IO limit, the
host calls it 16 times per page). The verify kernel (`verify/verify_group_m4.py`) runs the 4 rows' attention in one pass
over the KV (one scan per layer instead of four). Builds: `g16ks_c256k_kvf8.sh` (decode), `verify/` with `KV8=2
LMAX=262144`, `prefill/b27_c256k_kvf8.sh`. CRUN: decode output cos 0.99972 / K cache 0.99939 against the bf16 reference;
verify min cos 0.99947 at 256k. Earlier sets (bf16 KV at 16k / 128k / 256k) are in the history of `kernels.md5`.

Board (2026-11-05, .184, SDK V0.5.1, the fp8 256k set): greedy outputs identical to the bf16 sets; CMM after load 15.7 GB
at -c 4096 (+ the KV of the context given), load 65 s, host 0.85 GB; MTP 22 - 23.5 tok/s on short answers, 20.0 on a
400-token answer (verify 74.6 ms); needle-in-a-haystack 64k found (prefill 803 tok/s, plain decode 8.9 tok/s, MTP 17.9),
120k found (614 tok/s, plain 7.7, MTP 13.6); PPL -c 4096 4.2856 (bf16 256k set 4.289).

Board (2026-11-03, SDK V0.5.1): one copy of the weights: CMM after load 30.8 -> 18.2 GB (with the MTP kernels), load 62 s;
decode 13.4 - 13.5 tok/s as before; MTP (`--spec-type draft-mtp --spec-draft-n-max 3`, greedy): 22 - 24 tok/s on short
answers, 20 tok/s on a 400-token answer (2.0 - 2.4 tokens accepted per round; a verification round ~100 ms: verify 76.6,
lm_head 4.0, 3 drafts 3 x 3.5 + 3 x 1.1, host 8).

Board (.184, SDK V0.5.1, 2026-10-30): short completion identical to the delivered path; 3908-token prompt 3.3-3.6 s
(delivered 4.76 s), decode 13.7-14.1 tok/s (delivered 13.4). Image (bus.jpg, 3072 patches): ViT 276 ms in one call + 5 ms cpu (20 kernels: 393 ms), the same description. The prefill kernel's results vary slightly between runs
(NMC accumulation order on the board), within the quality tolerance.

Reproducibility: the AXDSL compiler is deterministic for a fixed source and toolchain; `kernels.md5` holds the files of
the build above.
