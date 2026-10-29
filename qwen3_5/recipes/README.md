# Qwen3.8-27B GDN prefill kernels (qwen35_gdn_prefill<P>_mx4_pv_pk_npu4, P = 128 / 64 / 16 / 4)

Built with `build_gdn_prefill_27b.sh` (q35_layer_pk.py, WFMT=mx8 WFMT4=1, PAD=1): every MXFP4 projection in output tiles
of 256 rows over input tiles of 1280 channels whose bf16 products are summed in fp32 (PRJ_ROWS=256 PRJ_IC=1280
PRJ_ACC=part).

Not PRJ_NMC_OC / PRJ_NMC_IC (MatMul NMC accumulation of the input tiles): on the AX8860 board its results change from
run to run (same inputs, same handle or a fresh process; 60 % of the outputs differ), so the completions of the earlier
27B package were not reproducible. With PRJ_IC=1280: the same outputs on every run and on both boards; 1.702 ms per
128-row call against 1.694 with NMC (prompt of 3908 tokens: 2183 vs 2176 ms).
