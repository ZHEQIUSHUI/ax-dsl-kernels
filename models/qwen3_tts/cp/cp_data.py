"""Inputs at the real code-predictor shapes, from random or TTS weights."""

from __future__ import annotations

import os
import shutil


ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
}


def configure(model: str, real: bool = False):
    cfg = os.path.join(ROOT, "cfg", model)
    if not os.path.exists(os.path.join(cfg, "config.json")):
        raise SystemExit(f"missing {cfg}/config.json")
    if real:
        src = os.path.join(ROOT, "hf", MODELS[model], "model.safetensors")
        if not os.path.exists(src):
            raise SystemExit(f"missing {src}")
        model_dir = os.path.join(ROOT, "hf", f"real_{model}")
        os.makedirs(model_dir, exist_ok=True)
        shutil.copyfile(os.path.join(cfg, "config.json"), os.path.join(model_dir, "config.json"))
        link = os.path.join(model_dir, "model.safetensors")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.relpath(src, model_dir), link)
        os.environ["MODEL"] = model_dir
        os.environ["QW_RANDOM"] = "0"
        os.environ["CP_REAL_WEIGHTS"] = "1"
    else:
        os.environ["MODEL"] = cfg
        os.environ["QW_RANDOM"] = "1"
        os.environ.pop("CP_REAL_WEIGHTS", None)
    os.environ["HT"] = "1024" if model == "0.6b" else "2048"
    os.environ.setdefault("LMAX", "16")
    os.environ.setdefault("BC", "16")
    os.environ.setdefault("CORES", "4")
    return cfg


def _real_weights(S):
    """Map q35's layer names onto the full Qwen3-TTS checkpoint."""
    import q35_step as q

    base_cls = q.Weights
    if getattr(base_cls, "_tts_real", False):
        return

    class RealWeights:
        _tts_real = True

        def __init__(self, path):
            self.base = base_cls(path)

        def __call__(self, name):
            if name.startswith(("layers.", "norm.")):
                name = "talker.code_predictor.model." + name
            return self.base(name)

    q.Weights = RealWeights


def _put_pack(pk, Qg, key, value):
    """Overwrite one packed member (used for real projection / final norm)."""
    pack, offset, shape = Qg.SPEC[key]
    flat = value.reshape(-1)
    pk[pack][offset : offset + flat.numel()] = flat


def make_case(S, frame: int = 0):
    """Same construction and RNG order as q3tts_cp_frame.main()."""
    import torch

    NP, V, H, HT, CORES = 16, S.V, S.H, S.HT, S.CORES
    real = os.environ.get("CP_REAL_WEIGHTS") == "1"
    if real:
        _real_weights(S)
    pk, sizes, es, raw, extra = S.build_packs()
    if real:
        from safetensors import safe_open

        from q35_group import SPEC

        path = os.path.join(os.environ["MODEL"], "model.safetensors")
        with safe_open(path, "pt") as w:
            if S.PROJ:
                extra["w_pin"] = w.get_tensor("talker.code_predictor.small_to_mtp_projection.weight").bfloat16().contiguous()
                extra["b_pin"] = w.get_tensor("talker.code_predictor.small_to_mtp_projection.bias").float().contiguous()
                _put_pack(pk, S.Qg, "LP.w_pin", extra["w_pin"])
                _put_pack(pk, S.Qg, "LP.b_pin", extra["b_pin"])
            extra["norm"] = w.get_tensor("talker.code_predictor.model.norm.weight").float().contiguous()
            _put_pack(pk, S.Qg, "LP.norm", extra["norm"])
            heads = torch.stack(
                [w.get_tensor(f"talker.code_predictor.lm_head.{i}.weight").bfloat16() for i in range(NP - 1)]
            )
            embd = torch.stack(
                [w.get_tensor(f"talker.code_predictor.model.codec_embedding.{i}.weight").bfloat16() for i in range(NP - 1)]
            )
    else:
        heads = (torch.randn((NP - 1, V, H)) * H**-0.5 * 4).bfloat16()
        embd = (torch.randn((NP - 1, V, HT)) * 0.5).bfloat16()
    cs16, pn16, mask16 = [], [], []
    for pos in range(NP):
        cs_, sn_ = S.L.rope_rows(pos)
        cs16.append(torch.cat([cs_, sn_]))
        pn16.append(torch.tensor([pos, pos // S.L.BC + 1], dtype=torch.int32))
        mrow = torch.full((S.L.LMAX,), -3e38)
        mrow[: pos + 1] = 0
        mask16.append(mrow.reshape(S.L.LMAX // S.L.BC, S.L.BC))
    torch.manual_seed(1234 + frame)
    e0 = (torch.randn((1, 1, HT)) * 0.5).bfloat16()
    if real:
        from safetensors import safe_open

        with safe_open(os.path.join(os.environ["MODEL"], "model.safetensors"), "pt") as w:
            keys = set(w.keys())
            name = "talker.model.codec_embedding.weight"
            if name not in keys and "talker.codec_embedding.weight" in keys:
                name = "talker.codec_embedding.weight"
            e1 = w.get_tensor(name)[0:1].bfloat16().reshape(1, 1, HT)
    else:
        e1 = (torch.randn((1, 1, HT)) * 0.5).bfloat16()
    rnd = torch.rand((1, NP - 1))
    case = dict(
        pk,
        e0=e0,
        e1=e1,
        heads=heads.reshape(-1, H),
        heads_lo=heads[:8].reshape(-1, H),
        heads_hi=heads[8:].reshape(-1, H),
        embd=embd.reshape(-1, HT),
        cs16=torch.cat(cs16).bfloat16(),
        pn16=torch.cat(pn16),
        mask16=torch.cat(mask16).bfloat16(),
        rnd=rnd,
        codes=torch.zeros((CORES, NP), dtype=torch.int32),
        lgs=torch.zeros((NP - 1, V)),
    )
    kv = [[torch.zeros((S.L.LMAX, S.L.HKV * S.L.D), dtype=torch.half) for _ in range(2)] for _ in range(S.NL)]
    for j in range(S.NL):
        case[f"kc{j}"], case[f"vc{j}"] = kv[j]
    return case, pk, sizes, es, raw, extra

