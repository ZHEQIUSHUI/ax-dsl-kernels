# ax-dsl-kernels

用 AXDSL（axnn 的 kernel DSL）为大模型写的 NPU kernel，以及配套的验证和板端工具。目标芯片 AX8860。

kernel 的权重都是**运行时输入**，所以同一种层只要一个 axmodel，所有层共用，权重可以直接来自 Hugging Face
safetensors 或 GGUF。llama.cpp 的 [ggml-axera 后端](https://github.com/ZHEQIUSHUI/llama.cpp)按名字加载这里生成的 axmodel。

## 目录

| 目录 | 内容 |
|---|---|
| `qwen3_5/` | Qwen3.5（GDN + 门控注意力）：kernel、数据导出、和 transformers / llama.cpp 的对比 |
| `qwen3_5/board/` | 板端程序：24 层逐层比对（`q35_board`）、板上完整生成和 PPL 打分（`q35_chat`） |
| `ggml/` | 给 ggml-axera 生成 kernel：从 GGUF 读出权重形状，每种形状一个 GEMV axmodel |
| `bench/` | 单项测试：GEMV（常量 / 运行时权重、MX 格式）、多层合并、decode attention、KV 写入、一层 decode |
| `tools/` | 通用板端工具：按名字喂输入的执行器、常驻 KV 写入验证、多核并行测试、多核 kernel 压测 |

### 新增模型

- 一个模型架构一个顶层目录，名字取 Hugging Face config 里的 `model_type`，比如 `qwen3_5/`、`qwen3/`、`minicpm/`。
  同一架构的不同参数规模（0.8B / 2B / 9B）共用一套 kernel，只是形状不同。
- 目录内的组织参照 `qwen3_5/`：kernel 和模拟器验证放在目录下，板端程序放在 `board/`。
- 多个模型都要用的代码（MX 量化、模拟器常驻 buffer 补丁、板端 CMM 封装等），等第二个模型用到时再抽到 `common/`。
- `ggml/`、`bench/`、`tools/` 和具体模型无关，留在顶层。

## 环境

- **主机**：x86-64 + axnn（npu-codebase 的 Python 环境）。所有脚本默认跑 NPU 模拟器（CRUN），编出的 axmodel
  在 `<DUMP>/.../cache/*.compiled.axmodel`。
- **模型路径**：用环境变量 `MODEL` 指定 Hugging Face 模型目录，对比 llama.cpp 时还要 `GGUF`、`LLAMA_DEBUG`。
- **板端程序**：`make BSP=<bsp>/arm64_glibc CROSS=<toolchain>/bin/aarch64-none-linux-gnu-`。BSP 库需要
  glibc 2.38，所以要 Arm GNU 13.2 工具链。板上运行时 `export LD_LIBRARY_PATH=/opt/lib`。

## Qwen3.5-0.8B

两个单核 kernel：
- **GDN 层**：RMSNorm → qkv/z/b/a 投影 → 带状态的 conv1d + SiLU → gated delta rule（fp32 状态）→ 门控
  RMSNorm → out_proj → MLP。
- **门控注意力层**：q（含输出门）/k/v 投影 → q/k RMSNorm → 部分 RoPE → KV 原地写入常驻 cache →
  GQA attention（运行时块数和 mask）→ sigmoid 门 → o_proj → MLP。

宿主负责 embedding 查表和 RoPE 的 cos/sin。lm_head 是单独的 kernel，按词表分块调用。

GDN 的 value 头多于 key 头时（9B：32 / 16），kernel 按 GGUF 的顺序取头：llama.cpp 转换时把 value 头从 HF 的分组顺序重排成平铺顺序（value 头 j 用 key 头 j % NK），`VORDER=tiled`（默认）；和 HF 比对的脚本加载 HF 权重时做同样的重排。

| 脚本 | 用途 |
|---|---|
| `q35_step.py` | 模拟器上逐 token 跑完整 decode，逐层和 transformers 比 |
| `q35_vs_llamacpp.py` | 逐位置和 llama.cpp CPU 比 logits |
| `q35_board_prep.py` → `board/q35_board` → `q35_board_cmp.py` | 真机 24 层逐层输出，和模拟器、transformers 比 |
| `q35_chat_build.py` / `q35_mx_build.py` → `board/q35_chat` | 板上完整生成（bf16 / MXFP8），`max_new=0` 时做 PPL 打分 |
| `q35_tok.py`、`q35_chat_check.py`、`q35_score.py` | prompt 编码、生成结果和 transformers 比对、PPL 对照 |
| `ggml_template_build.py` | ggml-axera 模板用的 decode 层 kernel（bf16 / MXFP8，KVL=cell 时按上下文 `_kvc_c<LMAX>` 命名） |
| `q35_prefill.py` | prefill kernel：每次 P 个连续 token，模拟器上和逐 token decode 对比 |
| `q35_batch.py` | batched decode kernel：B 个序列各一个 token（GDN、attention、lm_head），模拟器上和逐序列 decode 对比 |
| `q35_flash.py` | 长上下文的拆分 attention（flash decoding）：head / 按段的部分 attention / tail，模拟器上跨 2 段和融合 kernel 对比；`CORES=4` 时段 kernel 按块分给 4 个核（`_x4`，`nblk` 为每核 [start, stop)） |
| `q35_vit.py` | 视觉编码器（ViT）：`vit_embed` / `vit_pre` / `vit_attn` / `vit_post` / `vit_merge`，按 patch 数 N 分档（`_<N>_h<hidden>`，各尺寸共用代码：0.8B 768、2B/4B 1024、9B 及以上 1152），模拟器上逐层和 transformers 比 |
| `q35_hf_layer.py` | 任意尺寸的一个 GDN 层和一个注意力层的 decode / prefill kernel 直接和 transformers 的 `Qwen3_5DecoderLayer` 比 |
| `q35_mc.py` | 多核层链（实验性）：`gdn_mc` / `mlp_mc` / `attn_head_mc` / `attn_tail_mc`（P = 1 decode、64、128 prefill），按头或输出通道切到 4 个核、核间不同步，残差以 base + 每核 fp32 部分和在调用间传递；tail 在 NPU 上合并 NP 个段结果；另有 4 核 lm_head GEMV `mul_mat_bf16_n31040_k1024_x4` |
| `q3a_audio.py` | Qwen3-ASR 音频编码器的 transformer 层：复用 `q35_vit.py` 的 pre / attn / post（无 RoPE，宿主给 cos 1 / sin 0；GELU erf；LayerNorm eps 1e-5），`qwen3a_{pre,attn,post}<N>_h<H>`，模拟器上和 torch 比 |
| `q3a_conv.py` | Qwen3-ASR 卷积前端的矩阵乘（ggml 的 im2col 行 × 卷积核）：`qwen3a_conv_o<OC>_k<K>_r<R>`，conv2d.2 / conv2d.3 两层 480 通道，占 CPU 上编码器的大部分时间 |

### Qwen3（稠密）

Qwen3 只有注意力层（无输出门、全宽 RoPE、RMSNorm 权重原样），和 Qwen3.5 的门控注意力层是同一套 kernel，所以不另开目录：
`q35_step.py` 读到没有 `text_config` / `layer_types` 的 config（或 `text_config` 里是 qwen3，如 Qwen3-ASR）时去掉门和 GDN 层，
`ggml_template_build.py`、`q35_prefill.py`、`q35_flash.py`、`q35_batch.py` 照常用，产物名前缀 `qwen3_`（`q35_step.KP`）。
覆盖 Qwen3、Qwen3-VL / Qwen3-ASR 的文本模型、Qwen3-TTS 的 talker、Qwen3-Embedding / Reranker；形状相同的模型共用 kernel。
`HEAD_C` 要整除词表（151936 = 8 × 18992；TTS talker 的 codec 词表 3072）。

实测（2026-10-17，单核、bf16，不含 prefill kernel）：
- **正确性**：真机和模拟器逐层余弦相似度 1.000000；生成的中英文文本和 transformers 贪心结果逐 token 相同。
- **速度**：每 token 36.3 ms（24 层 25.4 ms + lm_head 9.6 ms + argmax 1.3 ms），约 27.5 token/s。
- **MXFP8**：模拟器周期比 bf16 少约 43%，还没上板实测。

## ggml-axera 的 kernel 约定

`ggml/gguf_gemv_kernels.py model.gguf out_dir` 为 GGUF 里每种 BF16 二维权重形状生成：

```
mul_mat_bf16_n<N>_k<K>.axmodel   输入 x [1, K] bf16、w [N, K] bf16；输出 out [1, N] f32
```

后端从 `GGML_AXERA_KERNEL_DIR` 按这个名字查找；找不到的形状回退到 CPU。

## 已知限制（当前驱动）

- 多核 kernel 内的 `sync_all` 会导致驱动崩溃，不要在共享板上跑。
- 多个任务不能同时在不同的核上执行（其余任务返回 `0x8006008a`），所以目前只用单核 kernel。
- 无同步的多核 kernel（`program_id` 切分，各核只写自己的输出）单独压测稳定（`tools/mc_stress.cc`、`tools/mc_alt.cc`：单个、交替、链式依赖、单多核混用），但在 llama.cpp 里串成层链时偶发 NaN / 结果漂移 / `0x8006008a` 超时，原因未明，所以 ggml-axera 默认不用（`GGML_AXERA_CORES=4` 打开）。有别的进程同时用 NPU 时多核调用必然超时。
- 多核 kernel 的输入越界（如段 kernel 的 `nblk` 超出范围）会让多核一直失败到重启，单核不受影响。
- c32768 的多核段 kernel 在板上核 1–3 结果错误（模拟器正确），多核段长用 16384。
- affinity 掩码每个核占 2 位：`0x3`、`0xc`、`0x30`、`0xc0`。
- 模拟器跑 kernel 前会把输出 buffer 清零。原地更新的常驻 KV cache 要先预填，见 `q35_step.py` 里的 RESIDENT 补丁。
