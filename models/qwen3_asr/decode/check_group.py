"""PYRUN / CRUN numerical check of the one-call Qwen3 dense decode group.

The group kernel's MXFP8 weights are quantized from the checkpoint, then
dequantized and loaded into both the PYRUN kernel and a Transformers Qwen3
decoder reference.  The paged BF16 KV page is filled from the reference prefix
(256 rows), the group decodes the next token, and its output / KV row are
compared with the reference at the same position.  ``--run-mode CRUN`` compiles
and simulates the same cases instead of using the Python reference execution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MODEL_ROOTS = {
    "embedding_0.6b": Path("/data/junda/work/qsmall/embedding/hf/Qwen3-0.6B"),
    "asr_0.6b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-ASR-0.6B"),
    "asr_1.7b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-ASR-1.7B"),
    "tts_0.6b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-TTS-12Hz-0.6B-Base"),
    "tts_1.7b": Path("/data/junda/work/qsmall/qwen3/hf/Qwen3-TTS-12Hz-1.7B-Base"),
}


def _preparse():
    for i, a in enumerate(sys.argv):
        if a == "--model":
            return sys.argv[i + 1]
    return "asr_0.6b"


_NAME = _preparse()
os.environ.setdefault("MODEL", str(MODEL_ROOTS[_NAME]))
os.environ.setdefault("LMAX", "4096")
os.environ.setdefault("BC", "256")
os.environ.setdefault("WFMT", "mx8")
os.environ.setdefault("CORES", "4")
os.environ.setdefault("BLOCK", "1")
os.environ.setdefault("DEC_SPLIT", "1")

import axnn.dsl as ax  # noqa: E402
import q35_group as Q  # noqa: E402
import q35_step as q  # noqa: E402
import q35_layer_mc as L  # noqa: E402
import q35_mx_build as mb  # noqa: E402


class Shards:
    def __init__(self, root: Path):
        self.root = root
        index = root / "model.safetensors.index.json"
        if index.is_file():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            from safetensors import safe_open
            with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
                self.map = {name: "model.safetensors" for name in handle.keys()}
        self.handles = {}

    def get(self, name):
        from safetensors import safe_open
        shard = self.map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.root / shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name)

    def has(self, name):
        return name in self.map

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)


def layer_prefix(name, index):
    if name.startswith("asr_"):
        return f"thinker.model.layers.{index}."
    if name.startswith("tts_"):
        return f"talker.model.layers.{index}."
    return f"model.layers.{index}."


def layer_input_norm(shards, name, index):
    return shards.get(layer_prefix(name, index) + "input_layernorm.weight")


def hf_config(name, layer_count):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    cfg = dict(vocab_size=151936, hidden_size=q.H, intermediate_size=q.F, num_hidden_layers=layer_count,
               num_attention_heads=q.HQ, num_key_value_heads=q.HKV, head_dim=q.D, hidden_act="silu",
               max_position_embeddings=q.LMAX, initializer_range=0.02, rms_norm_eps=q.EPS, use_cache=True,
               tie_word_embeddings=False, rope_theta=q.THETA, attention_bias=False, attention_dropout=0.0)
    config = Qwen3Config(**cfg)
    config._attn_implementation = "eager"
    return config


def dequant_layer(shards, name, index):
    """quantize the layer's projections exactly as the group does, then dequantize."""
    raw = q.attn_args(q.Weights(str(shards.root)), index)
    mx, deq = mb.mx_args(raw, mb.ATTN_W)
    prefix = layer_prefix(name, index)
    out = {
        "input_layernorm.weight": shards.get(prefix + "input_layernorm.weight"),
        "post_attention_layernorm.weight": shards.get(prefix + "post_attention_layernorm.weight"),
        "self_attn.q_norm.weight": shards.get(prefix + "self_attn.q_norm.weight"),
        "self_attn.k_norm.weight": shards.get(prefix + "self_attn.k_norm.weight"),
    }
    for short, hf in (("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                      ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight")):
        out[hf] = deq[{"q": "w_q", "k": "w_k", "v": "w_v", "o": "w_o"}[short]]
    for n, hf in (("w_gate", "mlp.gate_proj.weight"), ("w_up", "mlp.up_proj.weight"),
                  ("w_down", "mlp.down_proj.weight")):
        out[hf] = deq[n]
    return out


def make_reference_layers(shards, name, n, config):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    layers = []
    states = []
    for i in range(n):
        st = dequant_layer(shards, name, i)
        layer = Qwen3DecoderLayer(config, i).to(torch.bfloat16).eval()
        layer.load_state_dict(st, strict=True)
        layers.append(layer)
        states.append(st)
    return layers, states


def positions(name, start, rows, varied):
    base = np.arange(start, start + rows, dtype=np.int32)
    if not q.HAS_MROPE:
        return torch.from_numpy(base)
    if varied:
        h = start // 2 + np.arange(rows, dtype=np.int32) // 2
        w = start // 4 + np.arange(rows, dtype=np.int32) // 4
    else:
        h, w = base, base
    return torch.from_numpy(np.stack((base, h, w), axis=0))


def position_embeddings(config, pos):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    if not q.HAS_MROPE:
        return Qwen3RotaryEmbedding(config)(torch.zeros((1, 1, q.H), dtype=torch.bfloat16), pos[None])
    half = q.D // 2
    inv = 1.0 / (q.THETA ** (torch.arange(0, q.D, 2, dtype=torch.float32) / q.D))
    freqs = inv[None, :, None] * pos.float()[:, None, :]
    freqs = freqs.transpose(1, 2)
    mixed = freqs[0].clone()
    for axis, off in ((1, 1), (2, 2)):
        length = q.MROPE_SECTION[axis] * 3
        mixed[..., off:length:3] = freqs[axis, ..., off:length:3]
    emb = torch.cat((mixed, mixed), dim=-1)
    return emb.cos().to(torch.bfloat16)[None], emb.sin().to(torch.bfloat16)[None]


def run_hf_layers(hf_layers, hidden, pos, config, collect_kv=False):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    tokens = hidden.shape[0]
    mask = torch.full((1, 1, tokens, tokens), float("-inf"), dtype=torch.bfloat16).triu(1)
    rope = position_embeddings(config, pos)
    current = hidden
    keys, values = [], []
    with torch.no_grad():
        for layer in hf_layers:
            if collect_kv:
                att = layer.self_attn
                normed = layer.input_layernorm(current)
                k = att.k_proj(normed).view(1, tokens, q.HKV, q.D)
                k = att.k_norm(k).transpose(1, 2)
                v = att.v_proj(normed).view(1, tokens, q.HKV, q.D).transpose(1, 2)
                _, k = apply_rotary_pos_emb(k, k, *rope)
                keys.append(k.transpose(1, 2).reshape(tokens, q.HKV, q.D))
                values.append(v.transpose(1, 2).reshape(tokens, q.HKV, q.D))
            current = layer(current[None], attention_mask=mask, position_ids=pos[None], position_embeddings=rope)[0]
    return current, keys, values


def final_norm(hidden, weight):
    x = hidden.float()
    denom = torch.rsqrt(x.square().mean(-1, keepdim=True) + q.EPS)
    return ((x * denom).to(hidden.dtype) * weight)


def stats(actual, expected):
    a, b = actual.float(), expected.float()
    cos = F.cosine_similarity(a, b, dim=-1)
    return dict(cos_mean=cos.mean().item(), cos_min=cos.min().item(),
                abs_mean=(a - b).abs().mean().item(), abs_max=(a - b).abs().max().item())


def print_stats(label, st):
    print(f"{label}: cos mean={st['cos_mean']:.7f} min={st['cos_min']:.7f} "
          f"abs mean={st['abs_mean']:.6g} max={st['abs_max']:.6g}", flush=True)


def make_inputs(shards, name, rows, generator):
    if name.startswith("tts_"):
        text = shards.get("talker.model.text_embedding.weight")
        text_ids = torch.randint(0, text.shape[0], (rows,), generator=generator)
        th = text[text_ids].float()
        fc1 = shards.get("talker.text_projection.linear_fc1.weight").float()
        b1 = shards.get("talker.text_projection.linear_fc1.bias").float()
        fc2 = shards.get("talker.text_projection.linear_fc2.weight").float()
        b2 = shards.get("talker.text_projection.linear_fc2.bias").float()
        p = F.linear(F.silu(F.linear(th, fc1, b1)), fc2, b2)
        codec = shards.get("talker.model.codec_embedding.weight")
        ids = torch.randint(0, codec.shape[0], (rows,), generator=generator)
        return (p + codec[ids].float()).to(torch.bfloat16)
    candidates = ["model.embed_tokens.weight", "thinker.model.embed_tokens.weight",
                  "thinker.model.language_model.embed_tokens.weight", "talker.model.embed_tokens.weight"]
    ename = next((n for n in candidates if shards.has(n)), None)
    emb = shards.get(ename)
    ids = torch.randint(0, emb.shape[0], (rows,), generator=generator)
    return emb[ids].to(torch.bfloat16)


def paged_store(kv, pos, k, v):
    page, row = divmod(pos, q.BC)
    kv[:, page, 0, row] = k.to(torch.bfloat16)
    kv[:, page, 1, row] = v.to(torch.bfloat16)


def mask_for(pos):
    col = torch.arange(q.LMAX).reshape(q.LMAX // q.BC, q.BC)
    return torch.where(col <= pos, 0.0, -3e38).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="asr_0.6b")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prefix", type=int, default=256)
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--varied-mrope", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fresh-rows", type=int, default=4)
    ap.add_argument("--skip-fresh", action="store_true")
    ap.add_argument("--run-mode", default=os.environ.get("RUN_MODE", "PYRUN"), choices=("PYRUN", "CRUN"))
    args = ap.parse_args()
    if args.layers % 4:
        raise ValueError("layers must be a multiple of four")
    torch.set_num_threads(args.threads)
    root = MODEL_ROOTS[args.model]
    layer_count = args.layers
    lmhead = args.model.startswith(("asr_", "tts_"))
    out_dir = Path(args.out or HERE / "out" / f"check_{args.model}_l{layer_count}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ax.init(chip_num=1, core_num=L.CORES)
    shards = Shards(root)
    config = hf_config(args.model, layer_count)

    W = q.Weights(str(root))
    gen = torch.Generator().manual_seed(20261031)
    total = args.prefix + args.positions
    hidden = make_inputs(shards, args.model, total, gen)

    # reference through the full causal sequence
    hf_layers, _ = make_reference_layers(shards, args.model, layer_count, config)
    pos_all = positions(args.model, 0, total, args.varied_mrope)
    print(f"reference {layer_count} layers, {total} rows", flush=True)
    ref_hidden, ref_keys, ref_values = run_hf_layers(hf_layers, hidden, pos_all, config, collect_kv=True)
    norm_name = {"asr_": "thinker.model.norm.weight", "tts_": "talker.model.norm.weight"}.get(
        args.model[:4], "model.norm.weight")
    fnorm = shards.get(norm_name)
    ref_norm = final_norm(ref_hidden, fnorm)
    ref_k = torch.stack(ref_keys, 0)      # [L, T, HKV, D]
    ref_v = torch.stack(ref_values, 0)

    # reference lm head
    # group kernel (the head's reference uses the same quantized -> dequantized weights)
    print(f"build group {args.run_mode}", flush=True)
    kernel, packs, lm, meta = Q.prepare_group(W, out_dir, layers=layer_count, lmhead=lmhead,
                                              run_mode=args.run_mode)
    lm_ref = None
    if lmhead:
        head_w = mb.dequant(lm["w_lm"], lm["s_lm"]).float()
        lm_ref = ref_norm.float() @ head_w.T
    import importlib
    sys.path.insert(0, meta["build"])
    importlib.invalidate_caches()
    fn = importlib.import_module("group_kernel").group_kernel
    names = L.order(fn)
    print("inputs", names, flush=True)
    logits = torch.empty((1, lm["w_lm"].shape[0]), dtype=torch.float32) if lmhead else None

    def group_step(pos, x, kv):
        cs = torch.cat(q.rope_cos_sin(pos_all[:, pos] if pos_all.ndim == 2 else pos_all[pos]))
        pn = torch.tensor([pos, pos // q.BC + 1, (pos // q.BC) * 2 * q.BC + pos % q.BC], dtype=torch.int32)
        out = torch.empty((1, 1, q.H), dtype=torch.bfloat16)
        case = dict(x=x.reshape(1, 1, q.H), cs=cs, pn=pn, mask=mask_for(pos), kv=kv, fnorm=fnorm,
                    hbuf=torch.zeros((1, 1, q.H), dtype=torch.bfloat16), out=out, **packs)
        if lmhead:
            case["logits"] = logits
        kernel[4](*[case[n] for n in names])
        return out, (logits.clone() if lmhead else None)

    result = {"model": args.model, "layers": layer_count, "prefix": args.prefix,
              "positions": args.positions, "varied_mrope": args.varied_mrope, "run_mode": args.run_mode}

    # AP: prefix page from the reference KV, then decode each target token (page boundary 256)
    kv = torch.zeros((layer_count, q.LMAX // q.BC, 2, q.BC, q.HKV, q.D), dtype=torch.bfloat16)
    for p in range(args.prefix):
        paged_store(kv, p, ref_k[:, p], ref_v[:, p])
    print(f"decode after {args.prefix}-row prefix page", flush=True)
    outs, logs = [], []
    for i in range(args.positions):
        pos = args.prefix + i
        o, lg = group_step(pos, hidden[pos], kv)
        outs.append(o.reshape(q.H))
        logs.append(lg)
        sk = stats(o.reshape(-1), ref_norm[pos].reshape(-1))
        print_stats(f"output pos {pos}", sk)
        result[f"out_pos{pos}"] = sk
        # the row the group wrote at pos: compare with the reference's K / V row
        page, row = divmod(pos, q.BC)
        for kind, idx, ref in (("K", 0, ref_k), ("V", 1, ref_v)):
            st = stats(kv[:, page, idx, row].reshape(-1), ref[:, pos].reshape(-1))
            print_stats(f"kv {kind} row {pos}", st)
            result[f"kv_{kind}_pos{pos}"] = st
        if lmhead:
            kst = stats(lg.reshape(-1), lm_ref[pos].reshape(-1))
            print_stats(f"logits pos {pos}", kst)
            result[f"logits_pos{pos}"] = kst

    # B: the first few rows written by the group itself from an empty cache
    kv2 = torch.zeros_like(kv)
    rows = 0 if args.skip_fresh else min(args.fresh_rows, args.prefix)
    if rows:
        print("group writes rows from empty cache", flush=True)
    for pos in range(rows):
        o, lg = group_step(pos, hidden[pos], kv2)
        st = stats(o.reshape(-1), ref_norm[pos].reshape(-1))
        print_stats(f"fresh output pos {pos}", st)
        result[f"fresh_out_pos{pos}"] = st
        page, row = divmod(pos, q.BC)
        for kind, idx, ref in (("K", 0, ref_k), ("V", 1, ref_v)):
            st = stats(kv2[:, page, idx, row].reshape(-1), ref[:, pos].reshape(-1))
            print_stats(f"fresh kv {kind} row {pos}", st)
            result[f"fresh_kv_{kind}_pos{pos}"] = st

    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_dir / 'result.json'}", flush=True)
    shards.close()


if __name__ == "__main__":
    main()
