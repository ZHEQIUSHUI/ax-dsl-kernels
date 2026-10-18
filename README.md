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
| `tools/` | 通用板端工具：按名字喂输入的执行器、常驻 KV 写入验证、多核并行测试 |

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
| `q35_flash.py` | 长上下文的拆分 attention（flash decoding）：head / 按段的部分 attention / tail，模拟器上跨 2 段和融合 kernel 对比 |

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
- affinity 掩码每个核占 2 位：`0x3`、`0xc`、`0x30`、`0xc0`。
- 模拟器跑 kernel 前会把输出 buffer 清零。原地更新的常驻 KV cache 要先预填，见 `q35_step.py` 里的 RESIDENT 补丁。
