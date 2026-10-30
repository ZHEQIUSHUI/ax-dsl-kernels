# Qwen3.5 ViT kernels on 4 cores (q35_vit.py CORES=4)

`build_vit_4core_27b.sh` / `build_vit_4core_35b.sh`: qwen35_vit_{embed,pre,attn,post,merge}<N>_h1152_npu4 for N = 256 /
1024 / 2048 / 3072. embed / pre / attn / post are the same kernels for 9B / 27B / 35B-A3B (the weights are inputs);
merge projects to the LLM's width (27B 5120, 35B-A3B 2048), so each model's kernel directory takes its own build's.
Checked against transformers (cos >= 0.99998 at every N); on the board bus.jpg at 3072 patches: 336 ms.
