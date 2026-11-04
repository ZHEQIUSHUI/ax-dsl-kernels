"""Chain the per-call prefill kernel (BLOCKS blocks per call) over all layers on CRUN.

The kernel is compiled from the first call's layers, then each call is bound to its own real MXFP4 weights and its
own phase-major state pool; a call's x is the previous call's output. The final hidden state is compared against the
same transformers reference as check.py (per-row cos mean >= COS_MEAN, min >= COS_MIN; the defaults are the 16k
release's 0.9995 / 0.999). The min is relaxed for the native-MXFP4 references: the kernel's bf16 arithmetic loses
~0.0012 at the first rows over 24 - 32 layers (per 4-layer chunk the min is >= 0.99946), see NOTES.md.

MODEL=<hf dir> GG=<gguf> HF=<hf dir> CTX=<cells> BLOCKS=<blocks per call> OUT=<dir> [COS_MEAN=] [COS_MIN=]
python check_chain.py
"""
import json
import glob
import os
from pathlib import Path
import shutil
import sys

import torch
import axnn.dsl as ax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check as C  # noqa: E402  (reads MODEL / GG / HF / CTX / FULL from the environment)

from sgl_kernel_axera.attention.gdn import runtime_weights, split_conv_state  # noqa: E402
from sgl_kernel_axera.models.qwen38_27b.forward import prepare  # noqa: E402
from sgl_kernel_axera.models.weights import block_arguments  # noqa: E402

NL = C.NL
BLOCKS = int(os.environ.get("BLOCKS", C.C["num_hidden_layers"] // 4))
assert 0 < BLOCKS <= NL // 4 and (NL // 4) % BLOCKS == 0
NCALL = NL // (4 * BLOCKS)
T, H, CTX = C.T, C.H, C.CTX
OUT = Path(os.getenv("OUT", str(C.ROOT / "build" / "2b" / f"check_chain_c{CTX}_b{BLOCKS}")))
COS_MEAN, COS_MIN = float(os.environ.get("COS_MEAN", "0.9995")), float(os.environ.get("COS_MIN", "0.999"))
OUT.mkdir(parents=True, exist_ok=True)


def make_layers(first, n):
    layers = []
    for L in range(first, first + n):
        if L % 4 == 3:
            layers.append(C.gqa_w(L))
            continue
        w, state, conv = C.gdn_w(L)
        w = runtime_weights(w)
        w["w_pre"] = None
        for name, value in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"),
                               (torch.zeros_like(state), *split_conv_state(torch.zeros_like(conv), C.KH * C.KD, C.VH * C.VD)),
                               strict=True):
            w[name + "_in"] = w[name + "_out"] = value.contiguous()
        layers.append((None, w, C.mlp_w(L)))
    return layers


def pool_call(layers):
    """the call's delta-net states in the kernel's phase-major order: phase p, block b -> p * BLOCKS + b"""
    order = [4 * b + p for p in range(3) for b in range(BLOCKS)]
    for name in ("state", "q_conv_state", "k_conv_state", "v_conv_state"):
        pool = torch.stack([layers[il][1][name + "_in"] for il in order])
        for index, il in enumerate(order):
            layers[il][1][name + "_in"] = layers[il][1][name + "_out"] = pool[index]
    return layers


def hf_prefixes(x, stops):
    """hidden states after the first n layers, n in stops (DIAG=1: localize a chain mismatch)"""
    import json
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
    cfg = AutoConfig.from_pretrained(C.HF)
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "eager"
    idx = json.load(open(f"{C.HF}/model.safetensors.index.json"))["weight_map"]
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    pos = torch.arange(T)[None]
    h = x.float().reshape(1, T, H)
    out = {}
    for i in range(max(stops)):
        layer = Qwen3_5DecoderLayer(cfg, i).float().eval()
        pre = [p for p in (f"model.language_model.layers.{i}.", f"model.layers.{i}.")
               if any(k.startswith(p) for k in idx)][0]
        sd = {}
        for k, f in idx.items():
            if k.startswith(pre):
                with safe_open(f"{C.HF}/{f}", "pt") as fh:
                    sd[k[len(pre):]] = fh.get_tensor(k).float()
        C.gguf_into_hf(sd, i)
        layer.load_state_dict(sd, strict=True)
        mask = torch.full((1, 1, T, T), float("-inf")).triu(1) if layer.layer_type == "full_attention" else None
        with torch.no_grad():
            o = layer(h, position_embeddings=rot(h, pos), attention_mask=mask, position_ids=pos)
            h = o[0] if isinstance(o, tuple) else o
        if i + 1 in stops:
            out[i + 1] = h.reshape(T, H).clone()
            print(f"hf prefix {i + 1} done", flush=True)
    return out


def main():
    torch.set_num_threads(32)
    ax.init(chip_num=1)
    torch.manual_seed(0)
    ids = torch.randint(0, 200000, (T,))
    x = C.f32("token_embd.weight")[ids].bfloat16()
    ref_path = C.ROOT / "build" / "2b" / f"check_ref_{H}_{NL}.pt"
    ref = torch.load(ref_path) if ref_path.exists() else C.hf_block(x)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, ref_path)
    stops = {4 * BLOCKS * (k + 1) for k in range(NCALL)}
    pref = hf_prefixes(x, stops) if os.environ.get("DIAG") == "1" else {}

    calls = [pool_call(make_layers(k * 4 * BLOCKS, 4 * BLOCKS)) for k in range(NCALL)]
    kernels, weights0 = prepare(calls[0], mrope_section=tuple(C.C["rope_parameters"]["mrope_section"]),
                                cache_path=OUT, run_mode="CRUN")
    const = {"layout": weights0["layout"], "mrope_section": weights0["mrope_section"]}
    names0 = set(weights0) - set(const)
    y = x.clone()
    meta = (torch.arange(T, dtype=torch.int32).expand(3, -1).contiguous(), torch.tensor([0, 0], dtype=torch.int32),
            torch.arange(CTX // 256, dtype=torch.int32), torch.tensor([T], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int32))
    vt = torch.tensor([T], dtype=torch.int32)
    for k in range(NCALL):
        if k == 0:
            wk = weights0
        else:
            layout, tensors = block_arguments(calls[k], "mlp")
            assert set(tensors) == names0, (k, sorted(set(tensors) ^ names0))
            wk = {**const, **tensors}
        print(f"call {k}/{NCALL}: {4 * BLOCKS} layers, kernel building/run", flush=True)
        kernels[T][4, 4](y, *meta, y, valid_tokens=vt, **wk)
        assert bool(torch.isfinite(y).all())
        if os.environ.get("DIAG") == "1":
            pc = torch.nn.functional.cosine_similarity(y.float(), pref[4 * BLOCKS * (k + 1)].float(), dim=-1)
            print(f"DIAG call {k}: prefix {4 * BLOCKS * (k + 1)} cos mean {pc.mean():.6f} min {pc.min():.6f}", flush=True)
            torch.save(y, OUT / f"call{k}.pt")
    cos = torch.nn.functional.cosine_similarity(y.float(), ref.float(), dim=-1)
    print(f"ACCEPT chain cos mean {cos.mean():.6f} min {cos.min():.6f} "
          f"(gates {COS_MEAN} / {COS_MIN}: {'PASS' if cos.mean() >= COS_MEAN and cos.min() >= COS_MIN else 'FAIL'})",
          flush=True)
    torch.save(y, OUT / "y.pt")
    assert cos.mean() >= COS_MEAN and cos.min() >= COS_MIN, (float(cos.mean()), float(cos.min()))
    if os.environ.get("PRODUCE") == "1":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        import board_io
        compiled = sorted(glob.glob(str(OUT / "forward_qwen38_blocks.*.compiled.axmodel")), key=os.path.getmtime)
        assert compiled, f"no compiled axmodel under {OUT}"
        dst = OUT / f"{os.environ['KNAME']}.axmodel"
        shutil.copyfile(compiled[-1], dst)
        print("board io:", board_io.fix(str(dst)), flush=True)
        print("produced", dst, dst.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
