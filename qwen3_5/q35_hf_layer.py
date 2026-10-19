"""Check the decode and prefill layer kernels of one GDN layer and one attention layer against transformers'
Qwen3_5DecoderLayer, for any Qwen3.5 size: empty state and KV cache, positions 0 .. P - 1, the layer's real weights
(GGUF value-head order, VORDER=tiled). The decode kernel runs the first NDEC tokens one by one, the prefill kernel
all P at once.

Usage: MODEL=<hf dir> PCHUNK=16 NDEC=2 GDN_LAYER=0 ATTN_LAYER=3 python q35_hf_layer.py
"""
import os

os.environ.setdefault("KVL", "cell")

import torch
from loguru import logger

import axnn.dsl as ax
import q35_prefill as pf
import q35_step as q

P = pf.P
NDEC = int(os.environ.get("NDEC", 2))
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_hf_layer"))


def hf_layer(W, i):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(q.MODEL).text_config
    cfg._attn_implementation = "eager"
    layer = Qwen3_5DecoderLayer(cfg, i).float().eval()
    pre = f"model.language_model.layers.{i}."
    sd = {k[len(pre) :]: W.raw(k).float() for k in W.where if k.startswith(pre)}
    layer.load_state_dict(sd, strict=True)
    return layer, Qwen3_5TextRotaryEmbedding(cfg)


def hf_out(layer, rot, x):
    pos = torch.arange(P)[None]
    pe = rot(x, pos)
    mask = torch.full((1, 1, P, P), float("-inf")).triu(1) if layer.layer_type == "full_attention" else None
    with torch.no_grad():
        return layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pos)


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def main():
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    torch.manual_seed(0)
    xs = W("embed_tokens.weight")[torch.randint(0, 200000, (P,))].reshape(1, P, q.H).contiguous()  # bf16
    zeros = lambda *sh, dt=torch.bfloat16: torch.zeros(sh, dtype=dt)  # noqa: E731

    if os.environ.get("ONLY") != "attn":
        il = int(os.environ.get("GDN_LAYER", 0))
        layer, rot = hf_layer(W, il)
        ref = hf_out(layer, rot, xs.float())
        a = q.gdn_args(W, il)
        st0, cv0 = torch.zeros((q.NV, q.DK, q.DV)), zeros(1, q.CONV - 1, 2 * q.KD + q.VD)
        k_dec, k_pre = kern(q.gdn_layer, "gdn_dec"), kern(pf.gdn_layer_prefill, f"gdn_pre{P}")
        st, cv = st0.clone(), cv0.clone()
        for t in range(NDEC):
            case = dict(a, x=xs[:, t : t + 1].contiguous(), state_in=st, conv_in=cv, out=zeros(1, 1, q.H),
                        state_out=torch.zeros_like(st), conv_out=torch.zeros_like(cv))
            k_dec[1](*[case[n] for n in q.GDN_ORDER])
            st, cv = case["state_out"], case["conv_out"]
            logger.info(f"gdn layer {il} decode token {t}: cos vs HF {pf.cos(case['out'][:, 0], ref[:, t]):.6f}")
        case = dict(a, x=xs, state_in=st0.clone(), conv_in=cv0.clone(), out=zeros(1, P, q.H),
                    state_out=torch.zeros_like(st0), conv_out=torch.zeros_like(cv0))
        k_pre[1](*[case[n] for n in pf.GDN_ORDER])
        worst = min(pf.cos(case["out"][:, t], ref[:, t]) for t in range(P))
        logger.info(f"gdn layer {il} prefill P={P}: min cos vs HF {worst:.6f}, all {pf.cos(case['out'], ref):.6f}")

    if os.environ.get("ONLY") != "gdn":
        il = int(os.environ.get("ATTN_LAYER", 3))
        layer, rot = hf_layer(W, il)
        ref = hf_out(layer, rot, xs.float())
        a = q.attn_args(W, il)
        kc0, vc0 = zeros(q.LMAX, q.HKV * q.D, dt=torch.half), zeros(q.LMAX, q.HKV * q.D, dt=torch.half)
        k_dec, k_pre = kern(q.attn_layer, "attn_dec"), kern(pf.attn_layer_prefill, f"attn_pre{P}")
        kc, vc = kc0.clone(), vc0.clone()
        for t in range(NDEC):
            c, s = q.rope_cos_sin(t)
            case = dict(a, x=xs[:, t : t + 1].contiguous(), cos=c, sin=s, kc=kc, vc=vc, pos=torch.tensor([t], dtype=torch.int32),
                        nblk=torch.tensor([t // q.BC + 1], dtype=torch.int32), mask=pf.decode_mask(t), out=zeros(1, 1, q.H))
            q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
            k_dec[1](*[case[n] for n in q.ATTN_ORDER])
            q.RESIDENT.clear()
            kc, vc = case["kc"], case["vc"]
            logger.info(f"attn layer {il} decode token {t}: cos vs HF {pf.cos(case['out'][:, 0], ref[:, t]):.6f}")
        cs = [q.rope_cos_sin(t) for t in range(P)]
        case = dict(a, x=xs, cos=torch.cat([c for c, _ in cs]), sin=torch.cat([s for _, s in cs]), kc=kc0.clone(),
                    vc=vc0.clone(), pos=torch.tensor([0], dtype=torch.int32),
                    nblk=torch.tensor([(P - 1) // q.BC + 1], dtype=torch.int32), mask=pf.prefill_mask(0), out=zeros(1, P, q.H))
        q.RESIDENT.update(kc=kc0.clone(), vc=vc0.clone())
        k_pre[1](*[case[n] for n in pf.ATTN_ORDER])
        q.RESIDENT.clear()
        worst = min(pf.cos(case["out"][:, t], ref[:, t]) for t in range(P))
        logger.info(f"attn layer {il} prefill P={P}: min cos vs HF {worst:.6f}, all {pf.cos(case['out'], ref):.6f}")


if __name__ == "__main__":
    main()
