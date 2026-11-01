# GGUF bank layout

Generated from `pack_gguf.py` / `artifacts/gguf_banks/manifest.json`.  The bank
order, shapes and offsets are unchanged from `weight_layout.py`; the output is a
raw little-endian BF16 concatenation.

## Reading the GGUF tensor

Names in the table omit the common `a.gen.wav.` prefix.  `GGUFReader` exposes
the tensor data as an ndarray whose shape is `ne[]` reversed
(so the last axis is `ne0`, the fastest axis).  For a scalar source value at ggml
indices `(i0, i1, ...)`, where `i0` is the `ne0` axis:

```
value = data[i0 + ne0 * (i1 + ne1 * (...))]
```

`pack_gguf.read_tensor()` converts that value to FP32, then `pack_gguf.bf()`
rounds it to BF16 (round-to-nearest-even).  A source tensor that is already BF16
is therefore copied bit-for-bit; F16/F32 sources are rounded to BF16.  The table
below reports `ne[]` in ggml order (`ne0` first) exactly as stored in the file.

All names below are under `a.gen.wav.`.

## Exact transforms

All bank indices below are row-major.  `IC`/`OC`/`K` are input channels, output
channels and kernel taps; `S` is the ConvTranspose stride and `J = K/S` (2 for
DAC, 1 for the upsamplers).  `G[...]` is the logical GGUF array with axes in
reverse `ne[]` order (for example a 2-D matrix `G[o,i]` is `value(i0=i, i1=o)`)
after the BF16 conversion above.  The `src` formulas use the flat ggml data
offset directly.

- **Dense / biases / norms** (`copy`): source `ne = [IC, OC]` and
  `bank[o, i] = G[o, i]` (the 2-D tensor is already in row-major `[OC, IC]`
  order; its bytes are copied, not transposed).  1-D tensors are
  `bank[c] = G[c]`.
- **Codebooks** (`copy`, already folded): `q_sem_embed[o,i] =
  G[o,i]`, where `G = embedding_sum / cluster_usage`; `q_ac_embed[g,o,i] =
  G[g,o,i]`.  The kernel no longer divides by usage.  `q_sem_usage` and
  `q_ac_usage` are constants `1.0` so the existing bank byte layout is preserved.
- **SnakeBeta** (`copy`, already folded): `bank[c] = G[c]`, where
  `G_alpha = exp(alpha_raw)` and `G_beta = 1/(exp(beta_raw)+1e-9)`.  The kernel
  uses these values directly (no host `log`, no kernel `exp`/`1/x`).
- **Conv1d** (`conv`): source `ne = [K, IC, OC]`,
  `bank[o, k*IC + i] = G[o, i, k]`.  In C++:
  `src = k + K*(i + IC*o)`, `dst = o*(K*IC) + k*IC + i`.
- **ConvTranspose1d** (`trans`): source `ne = [K, OC, IC]`, split into `S`
  phases and `J` taps: `bank[p, o, j*IC + i] = G[i, o, p + j*S]`.  In C++:
  `src = (p+j*S) + K*(o + OC*i)`, `dst = (p*OC + o)*(J*IC) + j*IC + i`.
- **Depthwise conv** (`dw`): source `ne = [K, 1, C]`,
  `bank[k, t, c] = G[c, 0, k]` for every `t` in `0..T-1`.  In C++:
  `src = k + K*(0 + 1*c)`, `dst = (k*T + t)*C + c`.

For stacked transformer banks, `bank[l, ...]` comes from
`a.gen.wav.tfm.blk.l.<suffix>` and each layer uses the same `copy` rule above.

## Bank tensors

| bank | offset (BF16 elems) | tensor | shape | GGUF tensor | ggml type | ne[] | mapping |
|---|---:|---|---|---|---|---|---|
| `quant_w` | 0 | `q_sem_embed` | `[2048, 256]` | `quant.first.codebook.weight` | BF16 | `[256, 2048]` | codebook[o,i] = gguf[o,i] |
| `quant_w` | 524288 | `q_sem_usage` | `[2048]` | — | — | — | constant 1.0 (unused) |
| `quant_w` | 526336 | `q_sem_out` | `[512, 256]` | `quant.first.out_proj.weight` | BF16 | `[256, 512]` | bank[o,i] = gguf[o,i] |
| `quant_w` | 657408 | `q_ac_embed` | `[15, 2048, 256]` | `quant.rest.codebook.weight` | BF16 | `[256, 2048, 15]` | bank[g,o,i] = gguf[g,o,i] |
| `quant_w` | 8521728 | `q_ac_usage` | `[15, 2048]` | — | — | — | constant 1.0 (unused) |
| `quant_w` | 8552448 | `q_ac_out` | `[512, 256]` | `quant.rest.out_proj.weight` | BF16 | `[256, 512]` | bank[o,i] = gguf[o,i] |
| `quant_w` | 8683520 | `pre_conv_w` | `[1024, 1536]` | `pre_conv.weight` | F16 | `[3, 512, 1024]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `quant_w` | 10256384 | `pre_conv_b` | `[1024]` | `pre_conv.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `quant_w` | 10257408 | `tfm_in_w` | `[512, 1024]` | `tfm.in_proj.weight` | BF16 | `[1024, 512]` | bank[o,i] = gguf[o,i] |
| `quant_w` | 10781696 | `tfm_in_b` | `[512]` | `tfm.in_proj.bias` | F32 | `[512]` | bank[c] = gguf[c] |
| `quant_w` | 10782208 | `tfm_out_w` | `[1024, 512]` | `tfm.out_proj.weight` | BF16 | `[512, 1024]` | bank[o,i] = gguf[o,i] |
| `quant_w` | 11306496 | `tfm_out_b` | `[1024]` | `tfm.out_proj.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `quant_w` | 11307520 | `tfm_norm` | `[512]` | `tfm.output_norm.weight` | F32 | `[512]` | bank[c] = gguf[c] |
| `quant_w` | 11308032 | `tfm_ln1` | `[8, 512]` | `tfm.blk.0..7.ln1.weight` | F32 | `[512]` | bank[l,...] = gguf.blk[l].ln1.weight |
| `quant_w` | 11312128 | `tfm_q` | `[8, 1024, 512]` | `tfm.blk.0..7.attn_q.weight` | BF16 | `[512, 1024]` | bank[l,...] = gguf.blk[l].attn_q.weight |
| `quant_w` | 15506432 | `tfm_k` | `[8, 1024, 512]` | `tfm.blk.0..7.attn_k.weight` | BF16 | `[512, 1024]` | bank[l,...] = gguf.blk[l].attn_k.weight |
| `quant_w` | 19700736 | `tfm_v` | `[8, 1024, 512]` | `tfm.blk.0..7.attn_v.weight` | BF16 | `[512, 1024]` | bank[l,...] = gguf.blk[l].attn_v.weight |
| `quant_w` | 23895040 | `tfm_o` | `[8, 512, 1024]` | `tfm.blk.0..7.attn_out.weight` | BF16 | `[1024, 512]` | bank[l,...] = gguf.blk[l].attn_out.weight |
| `quant_w` | 28089344 | `tfm_attn_scale` | `[8, 512]` | `tfm.blk.0..7.ls1.weight` | F32 | `[512]` | bank[l,...] = gguf.blk[l].ls1.weight |
| `quant_w` | 28093440 | `tfm_ln2` | `[8, 512]` | `tfm.blk.0..7.ln2.weight` | F32 | `[512]` | bank[l,...] = gguf.blk[l].ln2.weight |
| `quant_w` | 28097536 | `tfm_gate` | `[8, 1024, 512]` | `tfm.blk.0..7.ffn_gate.weight` | BF16 | `[512, 1024]` | bank[l,...] = gguf.blk[l].ffn_gate.weight |
| `quant_w` | 32291840 | `tfm_up` | `[8, 1024, 512]` | `tfm.blk.0..7.ffn_up.weight` | BF16 | `[512, 1024]` | bank[l,...] = gguf.blk[l].ffn_up.weight |
| `quant_w` | 36486144 | `tfm_down` | `[8, 512, 1024]` | `tfm.blk.0..7.ffn_down.weight` | BF16 | `[1024, 512]` | bank[l,...] = gguf.blk[l].ffn_down.weight |
| `quant_w` | 40680448 | `tfm_ffn_scale` | `[8, 512]` | `tfm.blk.0..7.ls2.weight` | F32 | `[512]` | bank[l,...] = gguf.blk[l].ls2.weight |
| `up_w` | 0 | `up0_tc_w` | `[2, 1024, 1024]` | `up.blk.0.conv.weight` | F32 | `[2, 1024, 1024]` | trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S] |
| `up_w` | 2097152 | `up0_tc_b` | `[1024]` | `up.blk.0.conv.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 2098176 | `up0_dw_taps` | `[7, 144, 1024]` | `up.blk.0.dwconv.weight` | F16 | `[7, 1, 1024]` | dw: bank[k,t,c] = gguf[c,0,k] (same for all t) |
| `up_w` | 3130368 | `up0_dw_b` | `[1024]` | `up.blk.0.dwconv.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 3131392 | `up0_norm_w` | `[1024]` | `up.blk.0.norm.weight` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 3132416 | `up0_norm_b` | `[1024]` | `up.blk.0.norm.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 3133440 | `up0_pw1_w` | `[4096, 1024]` | `up.blk.0.pw1.weight` | BF16 | `[1024, 4096]` | bank[o,i] = gguf[o,i] |
| `up_w` | 7327744 | `up0_pw1_b` | `[4096]` | `up.blk.0.pw1.bias` | F32 | `[4096]` | bank[c] = gguf[c] |
| `up_w` | 7331840 | `up0_pw2_w` | `[1024, 4096]` | `up.blk.0.pw2.weight` | BF16 | `[4096, 1024]` | bank[o,i] = gguf[o,i] |
| `up_w` | 11526144 | `up0_pw2_b` | `[1024]` | `up.blk.0.pw2.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 11527168 | `up0_gamma` | `[1024]` | `up.blk.0.gamma` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 11528192 | `up1_tc_w` | `[2, 1024, 1024]` | `up.blk.1.conv.weight` | F32 | `[2, 1024, 1024]` | trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S] |
| `up_w` | 13625344 | `up1_tc_b` | `[1024]` | `up.blk.1.conv.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 13626368 | `up1_dw_taps` | `[7, 288, 1024]` | `up.blk.1.dwconv.weight` | F16 | `[7, 1, 1024]` | dw: bank[k,t,c] = gguf[c,0,k] (same for all t) |
| `up_w` | 15690752 | `up1_dw_b` | `[1024]` | `up.blk.1.dwconv.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 15691776 | `up1_norm_w` | `[1024]` | `up.blk.1.norm.weight` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 15692800 | `up1_norm_b` | `[1024]` | `up.blk.1.norm.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 15693824 | `up1_pw1_w` | `[4096, 1024]` | `up.blk.1.pw1.weight` | BF16 | `[1024, 4096]` | bank[o,i] = gguf[o,i] |
| `up_w` | 19888128 | `up1_pw1_b` | `[4096]` | `up.blk.1.pw1.bias` | F32 | `[4096]` | bank[c] = gguf[c] |
| `up_w` | 19892224 | `up1_pw2_w` | `[1024, 4096]` | `up.blk.1.pw2.weight` | BF16 | `[4096, 1024]` | bank[o,i] = gguf[o,i] |
| `up_w` | 24086528 | `up1_pw2_b` | `[1024]` | `up.blk.1.pw2.bias` | F32 | `[1024]` | bank[c] = gguf[c] |
| `up_w` | 24087552 | `up1_gamma` | `[1024]` | `up.blk.1.gamma` | F32 | `[1024]` | bank[c] = gguf[c] |
| `dac_w` | 0 | `dac_entry_w` | `[1536, 7168]` | `dac.entry.weight` | F16 | `[7, 1024, 1536]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 11010048 | `dac_entry_b` | `[1536]` | `dac.entry.bias` | F32 | `[1536]` | bank[c] = gguf[c] |
| `dac_w` | 11011584 | `dac0_snake_a` | `[1536]` | `dac.blk.0.snake.alpha` | F32 | `[1536]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 11013120 | `dac0_snake_b` | `[1536]` | `dac.blk.0.snake.beta` | F32 | `[1536]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 11014656 | `dac0_tc_w` | `[8, 768, 3072]` | `dac.blk.0.conv.weight` | F32 | `[16, 768, 1536]` | trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S] |
| `dac_w` | 29889024 | `dac0_tc_b` | `[768]` | `dac.blk.0.conv.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 29889792 | `dac0_r0_a1` | `[768]` | `dac.blk.0.res.0.act1.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 29890560 | `dac0_r0_b1` | `[768]` | `dac.blk.0.res.0.act1.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 29891328 | `dac0_r0_c1w` | `[768, 5376]` | `dac.blk.0.res.0.conv1.weight` | F16 | `[7, 768, 768]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 34020096 | `dac0_r0_c1b` | `[768]` | `dac.blk.0.res.0.conv1.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 34020864 | `dac0_r0_a2` | `[768]` | `dac.blk.0.res.0.act2.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 34021632 | `dac0_r0_b2` | `[768]` | `dac.blk.0.res.0.act2.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 34022400 | `dac0_r0_c2w` | `[768, 768]` | `dac.blk.0.res.0.conv2.weight` | F16 | `[1, 768, 768]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 34612224 | `dac0_r0_c2b` | `[768]` | `dac.blk.0.res.0.conv2.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 34612992 | `dac0_r1_a1` | `[768]` | `dac.blk.0.res.1.act1.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 34613760 | `dac0_r1_b1` | `[768]` | `dac.blk.0.res.1.act1.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 34614528 | `dac0_r1_c1w` | `[768, 5376]` | `dac.blk.0.res.1.conv1.weight` | F16 | `[7, 768, 768]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 38743296 | `dac0_r1_c1b` | `[768]` | `dac.blk.0.res.1.conv1.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 38744064 | `dac0_r1_a2` | `[768]` | `dac.blk.0.res.1.act2.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 38744832 | `dac0_r1_b2` | `[768]` | `dac.blk.0.res.1.act2.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 38745600 | `dac0_r1_c2w` | `[768, 768]` | `dac.blk.0.res.1.conv2.weight` | F16 | `[1, 768, 768]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 39335424 | `dac0_r1_c2b` | `[768]` | `dac.blk.0.res.1.conv2.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 39336192 | `dac0_r2_a1` | `[768]` | `dac.blk.0.res.2.act1.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 39336960 | `dac0_r2_b1` | `[768]` | `dac.blk.0.res.2.act1.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 39337728 | `dac0_r2_c1w` | `[768, 5376]` | `dac.blk.0.res.2.conv1.weight` | F16 | `[7, 768, 768]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 43466496 | `dac0_r2_c1b` | `[768]` | `dac.blk.0.res.2.conv1.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 43467264 | `dac0_r2_a2` | `[768]` | `dac.blk.0.res.2.act2.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 43468032 | `dac0_r2_b2` | `[768]` | `dac.blk.0.res.2.act2.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 43468800 | `dac0_r2_c2w` | `[768, 768]` | `dac.blk.0.res.2.conv2.weight` | F16 | `[1, 768, 768]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 44058624 | `dac0_r2_c2b` | `[768]` | `dac.blk.0.res.2.conv2.bias` | F32 | `[768]` | bank[c] = gguf[c] |
| `dac_w` | 44059392 | `dac1_snake_a` | `[768]` | `dac.blk.1.snake.alpha` | F32 | `[768]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 44060160 | `dac1_snake_b` | `[768]` | `dac.blk.1.snake.beta` | F32 | `[768]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 44060928 | `dac1_tc_w` | `[5, 384, 1536]` | `dac.blk.1.conv.weight` | F32 | `[10, 384, 768]` | trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S] |
| `dac_w` | 47010048 | `dac1_tc_b` | `[384]` | `dac.blk.1.conv.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 47010432 | `dac1_r0_a1` | `[384]` | `dac.blk.1.res.0.act1.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 47010816 | `dac1_r0_b1` | `[384]` | `dac.blk.1.res.0.act1.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 47011200 | `dac1_r0_c1w` | `[384, 2688]` | `dac.blk.1.res.0.conv1.weight` | F16 | `[7, 384, 384]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 48043392 | `dac1_r0_c1b` | `[384]` | `dac.blk.1.res.0.conv1.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 48043776 | `dac1_r0_a2` | `[384]` | `dac.blk.1.res.0.act2.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 48044160 | `dac1_r0_b2` | `[384]` | `dac.blk.1.res.0.act2.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 48044544 | `dac1_r0_c2w` | `[384, 384]` | `dac.blk.1.res.0.conv2.weight` | F16 | `[1, 384, 384]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 48192000 | `dac1_r0_c2b` | `[384]` | `dac.blk.1.res.0.conv2.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 48192384 | `dac1_r1_a1` | `[384]` | `dac.blk.1.res.1.act1.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 48192768 | `dac1_r1_b1` | `[384]` | `dac.blk.1.res.1.act1.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 48193152 | `dac1_r1_c1w` | `[384, 2688]` | `dac.blk.1.res.1.conv1.weight` | F16 | `[7, 384, 384]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 49225344 | `dac1_r1_c1b` | `[384]` | `dac.blk.1.res.1.conv1.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 49225728 | `dac1_r1_a2` | `[384]` | `dac.blk.1.res.1.act2.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 49226112 | `dac1_r1_b2` | `[384]` | `dac.blk.1.res.1.act2.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 49226496 | `dac1_r1_c2w` | `[384, 384]` | `dac.blk.1.res.1.conv2.weight` | F16 | `[1, 384, 384]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 49373952 | `dac1_r1_c2b` | `[384]` | `dac.blk.1.res.1.conv2.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 49374336 | `dac1_r2_a1` | `[384]` | `dac.blk.1.res.2.act1.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 49374720 | `dac1_r2_b1` | `[384]` | `dac.blk.1.res.2.act1.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 49375104 | `dac1_r2_c1w` | `[384, 2688]` | `dac.blk.1.res.2.conv1.weight` | F16 | `[7, 384, 384]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 50407296 | `dac1_r2_c1b` | `[384]` | `dac.blk.1.res.2.conv1.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 50407680 | `dac1_r2_a2` | `[384]` | `dac.blk.1.res.2.act2.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 50408064 | `dac1_r2_b2` | `[384]` | `dac.blk.1.res.2.act2.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 50408448 | `dac1_r2_c2w` | `[384, 384]` | `dac.blk.1.res.2.conv2.weight` | F16 | `[1, 384, 384]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 50555904 | `dac1_r2_c2b` | `[384]` | `dac.blk.1.res.2.conv2.bias` | F32 | `[384]` | bank[c] = gguf[c] |
| `dac_w` | 50556288 | `dac2_snake_a` | `[384]` | `dac.blk.2.snake.alpha` | F32 | `[384]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 50556672 | `dac2_snake_b` | `[384]` | `dac.blk.2.snake.beta` | F32 | `[384]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 50557056 | `dac2_tc_w` | `[4, 192, 768]` | `dac.blk.2.conv.weight` | F32 | `[8, 192, 384]` | trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S] |
| `dac_w` | 51146880 | `dac2_tc_b` | `[192]` | `dac.blk.2.conv.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 51147072 | `dac2_r0_a1` | `[192]` | `dac.blk.2.res.0.act1.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 51147264 | `dac2_r0_b1` | `[192]` | `dac.blk.2.res.0.act1.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 51147456 | `dac2_r0_c1w` | `[192, 1344]` | `dac.blk.2.res.0.conv1.weight` | F16 | `[7, 192, 192]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 51405504 | `dac2_r0_c1b` | `[192]` | `dac.blk.2.res.0.conv1.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 51405696 | `dac2_r0_a2` | `[192]` | `dac.blk.2.res.0.act2.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 51405888 | `dac2_r0_b2` | `[192]` | `dac.blk.2.res.0.act2.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 51406080 | `dac2_r0_c2w` | `[192, 192]` | `dac.blk.2.res.0.conv2.weight` | F16 | `[1, 192, 192]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 51442944 | `dac2_r0_c2b` | `[192]` | `dac.blk.2.res.0.conv2.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 51443136 | `dac2_r1_a1` | `[192]` | `dac.blk.2.res.1.act1.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 51443328 | `dac2_r1_b1` | `[192]` | `dac.blk.2.res.1.act1.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 51443520 | `dac2_r1_c1w` | `[192, 1344]` | `dac.blk.2.res.1.conv1.weight` | F16 | `[7, 192, 192]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 51701568 | `dac2_r1_c1b` | `[192]` | `dac.blk.2.res.1.conv1.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 51701760 | `dac2_r1_a2` | `[192]` | `dac.blk.2.res.1.act2.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 51701952 | `dac2_r1_b2` | `[192]` | `dac.blk.2.res.1.act2.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 51702144 | `dac2_r1_c2w` | `[192, 192]` | `dac.blk.2.res.1.conv2.weight` | F16 | `[1, 192, 192]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 51739008 | `dac2_r1_c2b` | `[192]` | `dac.blk.2.res.1.conv2.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 51739200 | `dac2_r2_a1` | `[192]` | `dac.blk.2.res.2.act1.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 51739392 | `dac2_r2_b1` | `[192]` | `dac.blk.2.res.2.act1.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 51739584 | `dac2_r2_c1w` | `[192, 1344]` | `dac.blk.2.res.2.conv1.weight` | F16 | `[7, 192, 192]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 51997632 | `dac2_r2_c1b` | `[192]` | `dac.blk.2.res.2.conv1.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 51997824 | `dac2_r2_a2` | `[192]` | `dac.blk.2.res.2.act2.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 51998016 | `dac2_r2_b2` | `[192]` | `dac.blk.2.res.2.act2.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 51998208 | `dac2_r2_c2w` | `[192, 192]` | `dac.blk.2.res.2.conv2.weight` | F16 | `[1, 192, 192]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52035072 | `dac2_r2_c2b` | `[192]` | `dac.blk.2.res.2.conv2.bias` | F32 | `[192]` | bank[c] = gguf[c] |
| `dac_w` | 52035264 | `dac3_snake_a` | `[192]` | `dac.blk.3.snake.alpha` | F32 | `[192]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52035456 | `dac3_snake_b` | `[192]` | `dac.blk.3.snake.beta` | F32 | `[192]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52035648 | `dac3_tc_w` | `[3, 96, 384]` | `dac.blk.3.conv.weight` | F32 | `[6, 96, 192]` | trans: bank[p,o,j*IC+i] = gguf[i,o,p+j*S] |
| `dac_w` | 52146240 | `dac3_tc_b` | `[96]` | `dac.blk.3.conv.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52146336 | `dac3_r0_a1` | `[96]` | `dac.blk.3.res.0.act1.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52146432 | `dac3_r0_b1` | `[96]` | `dac.blk.3.res.0.act1.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52146528 | `dac3_r0_c1w` | `[96, 672]` | `dac.blk.3.res.0.conv1.weight` | F16 | `[7, 96, 96]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52211040 | `dac3_r0_c1b` | `[96]` | `dac.blk.3.res.0.conv1.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52211136 | `dac3_r0_a2` | `[96]` | `dac.blk.3.res.0.act2.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52211232 | `dac3_r0_b2` | `[96]` | `dac.blk.3.res.0.act2.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52211328 | `dac3_r0_c2w` | `[96, 96]` | `dac.blk.3.res.0.conv2.weight` | F16 | `[1, 96, 96]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52220544 | `dac3_r0_c2b` | `[96]` | `dac.blk.3.res.0.conv2.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52220640 | `dac3_r1_a1` | `[96]` | `dac.blk.3.res.1.act1.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52220736 | `dac3_r1_b1` | `[96]` | `dac.blk.3.res.1.act1.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52220832 | `dac3_r1_c1w` | `[96, 672]` | `dac.blk.3.res.1.conv1.weight` | F16 | `[7, 96, 96]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52285344 | `dac3_r1_c1b` | `[96]` | `dac.blk.3.res.1.conv1.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52285440 | `dac3_r1_a2` | `[96]` | `dac.blk.3.res.1.act2.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52285536 | `dac3_r1_b2` | `[96]` | `dac.blk.3.res.1.act2.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52285632 | `dac3_r1_c2w` | `[96, 96]` | `dac.blk.3.res.1.conv2.weight` | F16 | `[1, 96, 96]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52294848 | `dac3_r1_c2b` | `[96]` | `dac.blk.3.res.1.conv2.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52294944 | `dac3_r2_a1` | `[96]` | `dac.blk.3.res.2.act1.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52295040 | `dac3_r2_b1` | `[96]` | `dac.blk.3.res.2.act1.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52295136 | `dac3_r2_c1w` | `[96, 672]` | `dac.blk.3.res.2.conv1.weight` | F16 | `[7, 96, 96]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52359648 | `dac3_r2_c1b` | `[96]` | `dac.blk.3.res.2.conv1.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52359744 | `dac3_r2_a2` | `[96]` | `dac.blk.3.res.2.act2.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52359840 | `dac3_r2_b2` | `[96]` | `dac.blk.3.res.2.act2.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52359936 | `dac3_r2_c2w` | `[96, 96]` | `dac.blk.3.res.2.conv2.weight` | F16 | `[1, 96, 96]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52369152 | `dac3_r2_c2b` | `[96]` | `dac.blk.3.res.2.conv2.bias` | F32 | `[96]` | bank[c] = gguf[c] |
| `dac_w` | 52369248 | `post_snake_a` | `[96]` | `dac.post_snake.alpha` | F32 | `[96]` | bank[c] = gguf[c] (already exp) |
| `dac_w` | 52369344 | `post_snake_b` | `[96]` | `dac.post_snake.beta` | F32 | `[96]` | bank[c] = gguf[c] (already reciprocal) |
| `dac_w` | 52369440 | `post_conv_w` | `[1, 672]` | `dac.post_conv.weight` | F16 | `[7, 96, 1]` | conv: bank[o,k*IC+i] = gguf[o,i,k] |
| `dac_w` | 52370112 | `post_conv_b` | `[1]` | `dac.post_conv.bias` | F32 | `[1]` | bank[c] = gguf[c] |

## Bank files

The three files are concatenations in the row order above (no padding):

| file | bytes | md5 |
|---|---:|---|
| `quant_w.bin` | 81369088 | `490681368dfb0592842a92da6fe446ec` |
| `up_w.bin` | 48177152 | `b5b7dc2ef8cb0045f3860d66add7242a` |
| `dac_w.bin` | 104740226 | `e5981fc8b5d47395b423b8b38d99e9c1` |

The GGUF tensors `a.gen.wav.quant.*.in_proj.weight` belong to the encoder and
are not part of the decoder banks.
