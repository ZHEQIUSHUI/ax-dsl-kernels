"""Qwen3-ASR audio encoder layers on the NPU: q35_vit.py's pre / attn / post kernels on the audio tower's shapes (a
pre-LN transformer like the vision one, without RoPE: the host passes cos = 1, sin = 0; GELU erf instead of tanh;
LayerNorm eps 1e-5). The conv front-end, the post norm and the projector stay on the CPU (ggml graph).

  qwen3a_pre<N>_h<H>   LayerNorm1, q / k / v projections (+ bias), q pre-scaled
  qwen3a_attn<N>_h<H>  bidirectional attention over the N rows (key mask for the padding)
  qwen3a_post<N>_h<H>  x + out_proj, + fc2(gelu(fc1(LayerNorm2)))

Check on the simulator: layer LAYER (default 0) of the checkpoint on random inputs of n real rows (NREAL, default
N - 37) against torch. Usage: MODEL=<Qwen3-ASR -hf dir> N=512 python q3a_audio.py [kernel_dir]
"""
import glob
import json
import os
import shutil
import sys
import tempfile

import torch
from loguru import logger
from safetensors import safe_open

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3-ASR Hugging Face model directory")
AC = json.load(open(f"{MODEL}/config.json"))["audio_config"]
H, NH, F = AC["d_model"], AC["encoder_attention_heads"], AC["encoder_ffn_dim"]
# q35_vit takes its shapes from a vision_config: one with the audio tower's (patch / merge sizes unused here)
_cfg = tempfile.mkdtemp(prefix="q3a_cfg_")
json.dump({"vision_config": {"hidden_size": H, "num_heads": NH, "intermediate_size": F, "out_hidden_size": AC["output_dim"],
                             "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2}},
          open(f"{_cfg}/config.json", "w"))
os.environ["MODEL"] = _cfg
os.environ.setdefault("DUMP", "out_q3a_audio")

import axnn.dsl as ax  # noqa: E402
import q35_vit as v  # noqa: E402

v.EPS = 1e-5  # the audio tower's LayerNorm
v.gelu_tanh = lambda x, output_dtype=ax.bf16: ax.dsl.gelu(x, output_dtype=output_dtype)  # GELU erf
N, HD, BC, QB = v.N, v.HD, v.BC, v.QB


def layer_weights(i):
    p = f"model.audio_tower.layers.{i}."
    with safe_open(f"{MODEL}/model.safetensors", "pt") as f:
        g = lambda n: f.get_tensor(p + n).float()  # noqa: E731
        return dict(ln1_w=g("self_attn_layer_norm.weight"), ln1_b=g("self_attn_layer_norm.bias"),
                    w_qkv=torch.cat([g(f"self_attn.{x}_proj.weight") for x in "qkv"]),
                    b_qkv=torch.cat([g(f"self_attn.{x}_proj.bias") for x in "qkv"]),
                    w_o=g("self_attn.out_proj.weight"), b_o=g("self_attn.out_proj.bias"),
                    ln2_w=g("final_layer_norm.weight"), ln2_b=g("final_layer_norm.bias"),
                    w_up=g("fc1.weight"), b_up=g("fc1.bias"), w_down=g("fc2.weight"), b_down=g("fc2.bias"))


def reference(x, w):
    """the encoder layer in fp32 (pre-LN, bidirectional attention over the n rows, GELU erf)"""
    n = x.shape[0]
    ln = lambda t, a, b: torch.nn.functional.layer_norm(t, (H,), a, b, 1e-5)  # noqa: E731
    qkv = ln(x, w["ln1_w"], w["ln1_b"]) @ w["w_qkv"].T + w["b_qkv"]
    q, k, vv = (t.reshape(n, NH, HD) for t in qkv.split(H, -1))
    att = torch.softmax(torch.einsum("qhd,khd->hqk", q * HD**-0.5, k), -1)
    x1 = x + torch.einsum("hqk,khd->qhd", att, vv).reshape(n, H) @ w["w_o"].T + w["b_o"]
    u = torch.nn.functional.gelu(ln(x1, w["ln2_w"], w["ln2_b"]) @ w["w_up"].T + w["b_up"])
    return x1 + u @ w["w_down"].T + w["b_down"]


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(v.BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    torch.manual_seed(0)
    n = int(os.environ.get("NREAL", N - 37))
    w = layer_weights(int(os.environ.get("LAYER", 0)))
    x = torch.randn((n, H)) * 0.5
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    pad = lambda t: torch.cat([t, torch.zeros((N - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)])  # noqa: E731
    built = []

    def run(fn, sub, args, order):
        k = v.kern(fn, sub)
        k[1](*[args[a] for a in order])
        built.append((sub, fn.__name__))

    kmask = torch.zeros((N,))
    kmask[n:] = -3e38
    p = dict(x=bf(pad(x)), ln1_w=bf(w["ln1_w"]), ln1_b=bf(w["ln1_b"]), w_qkv=bf(w["w_qkv"]), b_qkv=bf(w["b_qkv"]),
             cos=torch.ones((N, HD // 2), dtype=torch.bfloat16), sin=torch.zeros((N, HD // 2), dtype=torch.bfloat16),
             q_out=torch.zeros((N, NH, HD), dtype=torch.bfloat16), k_out=torch.zeros((N, H), dtype=torch.bfloat16),
             v_out=torch.zeros((N, H), dtype=torch.bfloat16))
    run(v.vit_pre, f"pre{N}", p, v.PRE_ORDER)
    t = dict(qs=p["q_out"], kc=p["k_out"], vc=p["v_out"], nblk=torch.tensor([-(-n // BC)], dtype=torch.int32),
             kmask=bf(kmask.reshape(N // BC, 1, BC).expand(N // BC, QB, BC)), out=torch.zeros((N, H), dtype=torch.bfloat16))
    run(v.vit_attn, f"attn{N}", t, v.ATTN_ORDER)
    o = dict(x=p["x"], attn=t["out"], w_o=bf(w["w_o"]), b_o=bf(w["b_o"]), ln2_w=bf(w["ln2_w"]), ln2_b=bf(w["ln2_b"]),
             w_up=bf(w["w_up"]), b_up=bf(w["b_up"]), w_down=bf(w["w_down"]), b_down=bf(w["b_down"]),
             out=torch.zeros((N, H), dtype=torch.bfloat16))
    run(v.vit_post, f"post{N}", o, v.POST_ORDER)
    ref = reference(x, w)
    logger.info(f"audio layer N={N} H={H} (n={n}): out vs torch cos {v.cos(o['out'][:n].float(), ref):.6f}")

    if out_dir:
        for sub, fn in built:
            (path,) = glob.glob(f"{v.BUILD}/{sub}/cache/{fn}.*.compiled.axmodel")
            dst = f"{out_dir}/qwen3a_{fn[4:]}{N}_h{H}.axmodel"  # vit_pre -> qwen3a_pre<N>_h<H>
            shutil.copy(path, dst)
            logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
