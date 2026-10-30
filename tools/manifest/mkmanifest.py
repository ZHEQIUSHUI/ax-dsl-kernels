#!/usr/bin/env python3
"""manifest.json of a model's kernel directory: the contract between ax-dsl-kernels and ggml-axera.

mkmanifest.py <model: qwen35_dense|qwen35_moe> <io.json> <layout dir> > manifest.json
mkmanifest.py qwen3 <io.json> <layout dir> <context> [lm_head kernel] [audio=<encoder width>] [tts=1] > manifest.json

- params: per layer kind (gdn / attn), each layer parameter the kernels take: its GGUF tensor (within blk.<i>.), the
  conversion from it and its kernel form (bytes, dtype, shape). Taken from the layer kernels' IO.
- params "out" (whole-model group kernels with the output head, members LM.*): the final norm and the output
  projection (output.weight, or the tied token_embd.weight).
- kernels: by role, the npu4 kernel of the delivered path: file, IO table and, for packed kernels, the packs and
  their members (parameter, layer, pack, offset, bytes, column blocks).
Only the kernels listed here are loaded; a missing file is an error at load.
"""
import json, os, re, sys

CONTEXT = 16384

GGUF = {  # kernel parameter -> GGUF tensor of the layer
    "ln1": "attn_norm.weight", "ln2": "post_attention_norm.weight", "w_qkv": "attn_qkv.weight",
    "w_z": "attn_gate.weight", "w_b": "ssm_beta.weight", "w_a": "ssm_alpha.weight", "dt_bias": "ssm_dt.bias",
    "norm_w": "ssm_norm.weight", "w_out": "ssm_out.weight", "w_gate": "ffn_gate.weight", "w_up": "ffn_up.weight",
    "w_down": "ffn_down.weight", "w_q": "attn_q.weight", "w_k": "attn_k.weight", "w_v": "attn_v.weight",
    "q_norm": "attn_q_norm.weight", "k_norm": "attn_k_norm.weight", "w_o": "attn_output.weight",
    "a_log": "ssm_a", "conv_w": "ssm_conv1d.weight",
}
GGUF_MOE = {  # MoE layers: the dense MLP is the shared expert; the router and the shared expert's gate
    "w_gate": "ffn_gate_shexp.weight", "w_up": "ffn_up_shexp.weight", "w_down": "ffn_down_shexp.weight",
    "w_router": "ffn_gate_inp.weight", "w_shg": "ffn_gate_inp_shexp.weight",
}
GGUF_QWEN3 = {"ln2": "ffn_norm.weight"}  # Qwen3: the MLP's input norm
# kernel IO that is not a parameter: what the runtime binds
RUNTIME = {"x", "out", "pos", "nblk", "mask", "cos", "sin", "cs", "pn", "state_in", "conv_in", "state_out",
           "conv_out", "kc", "vc", "h2_out", "logits_out", "rv", "cix", "topk_v", "topk_i"}

ROLES = {
    "qwen35_moe": {  # Qwen3.5-35B-A3B, MXFP8 layer weights, MXFP4 routed experts
        "layer.gdn": "qwen35moe_gdn_mx8_npu4",
        "layer.attn": "qwen35moe_attn_mx8_kvc_c16384_npu4",
        "decode.group": "qwen35moe_group8_gggaggga_mx8_kvc_c16384_npu4",
        "lm_head": "mul_mat_mx4_n248320_k2048_npu4",
        "prefill256.gdn": "qwen35moe_gdn_prefill256_mx8_tk_pk_npu4",
        "prefill256.attn": "qwen35moe_attn_prefill256_tk_kvc_c16384_pk_npu4",
        "prefill128.gdn": "qwen35moe_gdn_prefill128_mx8_pk_npu4",
        "prefill128.attn": "qwen35moe_attn_prefill128_kvc_c16384_pk_npu4",
        **{f"prefill{p}.gdn": f"qwen35moe_gdn_prefill{p}_mx8_npu4" for p in (64, 16, 4)},
        **{f"prefill{p}.attn": f"qwen35moe_attn_prefill{p}_kvc_c16384_npu4" for p in (64, 16, 4)},
        **{f"moe.slots{s}_g{g}": f"qwen35moe_slots{s}_g{g}_px_mxfp4_bo_npu4"
           for s, g in ((64, 64), (32, 64), (16, 64), (32, 128), (16, 128), (8, 128))},
        **{f"moe.combine_t{t}": f"qwen35moe_combine8_t{t}_r32768_npu4" for t in (512, 1024, 2048)},
    },
}


def qwen35_dense_roles(io):
    """Qwen3.5 dense models (9B, Qwen3.8-27B), MXFP4 layer weights: the decode group (the whole model with the output
    head when the directory has it, else the most layers), the four prefill chunk sizes, the MXFP4 output head"""
    groups = [f[:-8] for f in io if re.fullmatch(r"qwen35_group\d+_[ga]+_mx4(_lm4)?_kvc_c16384_npu4\.axmodel", f)]
    if not groups:
        sys.exit("no qwen35 decode group")
    g = sorted(groups, key=lambda g: ("_lm4_" not in g, -int(re.search(r"group(\d+)_", g).group(1))))[0]
    H = elements(next(e[3] for e in io[g + ".axmodel"]["in"] if e[0] == "x"))
    return {
        "layer.gdn": "qwen35_gdn_mx4_npu1",  # IO only: the parameters of a GDN layer
        "layer.attn": "qwen35_attn_mx4_kvc_c16384_npu1",  # IO only: the parameters of an attention layer
        "decode.group": g,
        "lm_head": f"mul_mat_mx4_n248320_k{H}_npu4",
        **{f"prefill{p}.gdn": f"qwen35_gdn_prefill{p}_mx4_pv_pk_npu4" for p in (128, 64, 16, 4)},
        **{f"prefill{p}.attn": f"qwen35_attn_prefill{p}_mx4_pk_kvc_c16384_npu4" for p in (128, 64, 16, 4)},
    }


def qwen3_roles(io, ctx, lm):
    """Qwen3 text decoders (Embedding, ASR text model, TTS talker), MXFP8: the whole-model decode group (the one with
    the output head when the directory has it), the four prefill chunk sizes, the output head GEMV if given"""
    groups = sorted(f[:-8] for f in io if re.fullmatch(rf"qwen3_group\d+_a+_mx8(_lm8)?_kvc_c{ctx}_npu4\.axmodel", f))
    if not groups:
        sys.exit(f"no qwen3 decode group of context {ctx}")
    r = {
        "layer.attn": f"qwen3_attn_mx8_kvc_c{ctx}_npu1",  # IO only: the parameters of an attention layer
        "decode.group": sorted(groups, key=lambda g: "_lm8_" not in g)[0],
        **{f"prefill{p}.attn": f"qwen3_attn_prefill{p}_mx8_pk_kvc_c{ctx}_npu4" for p in (128, 64, 16, 4)},
    }
    if lm:
        r["lm_head"] = lm
    return r


def audio_roles(H):
    """the Qwen3-ASR audio encoder (q3a_fe.py, q3a_layer.py, q3a_proj.py) of width H: per call of n chunks of 100 mel
    frames (n = 1 / 2 / 4 / 8) the front-end's three convolutions and conv_out, the layer and projector kernels of 64
    and 128 rows"""
    r = {}
    for n in (1, 2, 4, 8):
        r[f"audio.fe1.n{n}"] = f"qwen3a_fe_o480_k16_r{3200 * n}_npu4"
        r[f"audio.fe2.n{n}"] = f"qwen3a_fe2_o480_n{n}_npu4"
        r[f"audio.fe3.n{n}"] = f"qwen3a_fe_o480_k4320_r{208 * n}_npu4"
        r[f"audio.conv_out.n{n}"] = f"mm_bf16_m{13 * n}_n{H}_k7680_npu4"
    for nf in (64, 128):
        r[f"audio.layer{nf}"] = f"qwen3a_layer{nf}_h{H}_npu4"
        r[f"audio.proj{nf}"] = f"qwen3a_proj{nf}_h{H}_npu4"
    return r


def tts_roles(io):
    """the Qwen3-TTS generation graphs: the code predictor's frame kernels (q3tts_cp_frame.py, by top-k), and every
    4-core conv1d / GEMM kernel of the directory by its shape (the codec decoder's, the speaker encoder's)"""
    r = {}
    for f in io:
        b = f[:-8]
        m = re.fullmatch(r"qwen3tts_cp_frame\d+_mx8_h\d+_k(\d+)_c\d+_npu4", b)
        if m:
            r[f"cp.frame.k{m.group(1)}"] = b
        elif re.fullmatch(r"(conv1d_bf16|mm_bf16)_.*_npu4", b):
            r["gen." + b[:-5]] = b
    return r


# the code predictor's parameters in the frame kernels' packs: layer members L<j>.<p> (tensors a.gen.code.blk.<j>.*),
# LP.<p> the input projection and the final norm
CP_GGUF = {"ln1": "ln1.weight", "ln2": "ln2.weight", "q_norm": "attn_q_norm.weight", "k_norm": "attn_k_norm.weight",
           "q": "attn_q.weight", "k": "attn_k.weight", "v": "attn_v.weight", "o": "attn_out.weight",
           "gate": "ffn_gate.weight", "up": "ffn_up.weight", "down": "ffn_down.weight"}
CP_OUT = {"LP.w_pin": "a.gen.code.proj_in.weight", "LP.b_pin": "a.gen.code.proj_in.bias",
          "LP.norm": "a.gen.code.output_norm.weight"}


def cp_params(man):
    for role, k in man["kernels"].items():
        if not role.startswith("cp."):
            continue
        for m in k["members"]:
            p = m["param"]
            if p.startswith("LP."):
                kind, g = "cp_out", CP_OUT[p]
            else:
                kind, g = "cp", "a.gen.code.blk.{L}." + CP_GGUF[p[2:] if p[:2] in ("w_", "s_") else p]
            conv = ("mxfp8" if p[:2] == "w_" else "mxfp8_scales") if p[:2] in ("w_", "s_") and kind == "cp" else \
                   "bf16" if m["pack"] == "pb" else "tensor"
            d = {"gguf": g, "convert": conv, "bytes": m["bytes"], "dtype": 0, "shape": [m["bytes"]]}
            old = man["params"].setdefault(kind, {}).setdefault(p, d)
            assert old == d, (role, p, old, d)
            m["kind"] = kind


# the vision encoder (9B / 27B / 35B-A3B: hidden 1152), q35_vit.py CORES=4: the patch counts whose five kernels are
# all in the directory's IO tables
VIT_KINDS = ("embed", "pre", "attn", "post", "merge")
VIT_N = (256, 512, 1024, 1536, 2048, 2560, 3072)


def vit_roles(io):
    r = {}
    for n in VIT_N:
        names = {k: f"qwen35_vit_{k}{n}_h1152_npu4" for k in VIT_KINDS}
        if all(f + ".axmodel" in io for f in names.values()):
            r.update({f"vit.{k}{n}": f for k, f in names.items()})
    return r


def io_entries(io):
    return [{"name": n, "bytes": b, "dtype": t, "shape": s} for n, b, t, s in io]


def elements(shape):
    e = 1
    for d in shape:
        e *= d
    return e


def qwen38_27b_roles(io):
    """Qwen3.8-27B: the whole-model prefill kernel (16 blocks in a runtime loop, 256 cells per call), the 16-layer decode
    group on its state layouts (value heads in HF order), the MXFP4 output head"""
    return {
        "layer.gdn": "qwen35_gdn_mx4_npu1",  # IO only: the parameters of a GDN layer
        "layer.attn": "qwen35_attn_mx4_kvc_c16384_npu1",  # IO only: the parameters of an attention layer
        "decode.group": "qwen38_27b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4",
        "prefill": "qwen38_27b_prefill256_mx4_kvc_c16384_npu4",
        "lm_head": "mul_mat_mx4_n248320_k5120_npu4",
    }


def vperm(ins, n):
    """the value-head reorder of a delta-net parameter (GGUF tiled -> HF grouped) for the decode group of qwen38_27b:
    rows of the v part of qkv / conv (after q and k), rows of z / b / a, the 48 values of a_log / dt_bias, columns of out"""
    VH, KH = 48, 16
    base = n[2:] if n.startswith(("w_", "s_")) else n
    if base in ("qkv", "conv_w"):
        rows = ins[n][2][0]
        v = ins["w_z"][2][0]  # value rows: those of z
        return f"rows:{rows - v}:{VH}:{KH}"
    if base in ("z", "b", "a", "a_log", "dt_bias"):
        return f"rows:0:{VH}:{KH}"
    if base == "out":
        return f"cols:{VH}:{KH}"
    return None


def params(model, kernel):
    """the parameters of a layer kernel: GGUF tensor, conversion, kernel form"""
    ins = {n: (b, t, s) for n, b, t, s in kernel["in"]}
    out = {}
    for n, (b, t, s) in ins.items():
        if n in RUNTIME:
            continue
        if n.startswith("s_") and "w_" + n[2:] in ins:
            w = "w_" + n[2:]
            src, conv = w, None  # scales of w_<x>: their format follows the elements'
        else:
            src, conv = n, None
        g = {"qwen35_moe": GGUF_MOE, "qwen3": GGUF_QWEN3}.get(model, {}).get(src) or GGUF.get(src)
        if g is None:
            sys.exit(f"{n}: no GGUF tensor")
        if n == "a_log":
            conv = "log_neg"  # GGUF ssm_a = -exp(A_log)
        elif n == "conv_w":
            conv = "bf16_taps_reversed"  # kernel tap 0 = the current token
        elif ("s_" + n[2:]) in ins and n.startswith("w_"):
            conv = "mxfp4" if 2 * b == elements(s) else "mxfp8"
        elif n.startswith("s_") and src != n:
            wb, _, ws = ins[src]
            conv = ("mxfp4" if 2 * wb == elements(ws) else "mxfp8") + "_scales"
        else:
            conv = "tensor"  # as it is (bf16 / f32), a bf16 copy of a quantized / f16 tensor
        out[n] = {"gguf": g, "convert": conv, "bytes": b, "dtype": t, "shape": s}
        vp = vperm(ins, n) if model == "qwen38_27b" and "w_qkv" in ins else None
        if vp:
            out[n]["vperm"] = vp
    return out


def layout(path):
    packs, members = {}, []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        name, pack, off, nbytes = f[0], f[1], int(f[2]), int(f[3])
        if pack == "-":
            packs[name] = nbytes
            continue
        m = {"name": name, "pack": pack, "offset": off, "bytes": nbytes}
        lm = re.match(r"L(\d+)\.(.+)", name)
        if lm:
            m["layer"], m["param"] = int(lm.group(1)), lm.group(2)
        else:
            m["param"] = name
        rest = f[4:]
        if rest[:1] == ["blk"]:  # column strips: rows cols strip width
            m["strips"] = [int(x) for x in rest[1:4]]
        elif len(rest) >= 3:  # column blocks: rows, bytes per row, byte ranges a:b of a row
            m["blocks"] = {"rows": int(rest[0]), "row_bytes": int(rest[1]),
                           "ranges": [[int(a), int(b)] for a, b in (r.split(":") for r in rest[2].split(","))]}
        members.append(m)
    return packs, members


def main():
    model, io_path, lay_dir = sys.argv[1:4]
    io = json.load(open(io_path))
    ctx = int(sys.argv[4]) if model == "qwen3" else CONTEXT
    opt = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[5:] if "=" in a}
    pos = [a for a in sys.argv[5:] if "=" not in a]
    roles = ({**qwen3_roles(io, ctx, pos[0] if pos else None), **(audio_roles(int(opt["audio"])) if "audio" in opt else {}),
              **(tts_roles(io) if opt.get("tts") == "1" else {})}
             if model == "qwen3" else
             qwen35_dense_roles(io) if model == "qwen35_dense" else qwen38_27b_roles(io) if model == "qwen38_27b" else
             ROLES[model])
    man = {"format": 1, "model": model, "context": ctx, "cores": 4, "params": {}, "kernels": {}}
    for role, base in {**roles, **vit_roles(io)}.items():
        f = base + ".axmodel"
        if f not in io:
            sys.exit(f"{role}: {f} not in {io_path}")
        if role.startswith("layer."):  # parameter source only: not a kernel of the path
            man["params"][role[6:]] = params(model, io[f])
            continue
        k = {"file": f, "in": io_entries(io[f]["in"]), "out": io_entries(io[f]["out"])}
        lp = os.path.join(lay_dir, base + ".layout")
        if os.path.exists(lp):
            k["packs"], k["members"] = layout(lp)
            for e in k["in"]:  # packs without members (a layout lists only those it fills): empty, of the IO's size
                if re.fullmatch(r"p[wsfb4](_\d+)?", e["name"]) and e["name"] not in k["packs"]:
                    k["packs"][e["name"]] = e["bytes"]
        man["kernels"][role] = k
    if model == "qwen35_moe":  # routed experts: GGUF MXFP4 repacked in place as planes (all codes, then all scales)
        g = man["kernels"]["moe.slots32_g64"]["in"]
        shape = {e["name"]: e for e in g}
        man["params"]["experts"] = {
            # gate and up experts as one gate-up weight per expert (gate rows, then up rows)
            "e_gw": {"gguf": "ffn_gate_exps.weight,ffn_up_exps.weight", "convert": "mxfp4_planar_codes"},
            "e_gs": {"gguf": "ffn_gate_exps.weight,ffn_up_exps.weight", "convert": "mxfp4_planar_scales"},
            "e_dw": {"gguf": "ffn_down_exps.weight", "convert": "mxfp4_planar_codes"},
            "e_ds": {"gguf": "ffn_down_exps.weight", "convert": "mxfp4_planar_scales"},
        }
        for n, v in man["params"]["experts"].items():
            v.update(bytes=shape[n]["bytes"], dtype=shape[n]["dtype"], shape=shape[n]["shape"])
    # bf16 copies of a layer kind's parameters for kernels that take them unquantized (35B attention prefill): the
    # kind "<kind>_bf16", described from such a kernel's IO
    bf16_src = {"qwen35_moe": {"attn": "prefill64.attn"}}.get(model, {})
    for kind, role in bf16_src.items():
        ins = {e["name"]: e for e in man["kernels"][role]["in"]}
        bk = {}
        for n, p in man["params"][kind].items():
            e = ins.get(n)
            if e and e["bytes"] != p["bytes"]:
                assert e["bytes"] == 2 * elements(e["shape"]), n
                bk[n] = {"gguf": p["gguf"], "convert": "bf16", "bytes": e["bytes"], "dtype": e["dtype"], "shape": e["shape"]}
        man["params"][kind + "_bf16"] = bk
    cp_params(man)
    # the output head's members of a whole-model group (LM.norm, LM.w_lm, LM.s_lm): parameters of kind "out"
    for role, k in man["kernels"].items():
        lmm = {m["param"]: m for m in k.get("members", []) if m["param"].startswith("LM.")}
        if not lmm:
            continue
        H = elements(next(e["shape"] for e in k["in"] if e["name"] == "x"))
        V = elements(next(e["shape"] for e in k["out"] if e["name"] == "logits"))
        fmt = {V * H: "mxfp8", V * H // 2: "mxfp4"}[lmm["LM.w_lm"]["bytes"]]
        assert lmm["LM.norm"]["bytes"] == 4 * H and lmm["LM.s_lm"]["bytes"] == V * H // 32
        out = man["params"].setdefault("out", {})
        out["LM.norm"] = {"gguf": "output_norm.weight", "convert": "tensor", "bytes": 4 * H, "dtype": 3, "shape": [H]}
        for n, c, b in (("LM.w_lm", fmt, lmm["LM.w_lm"]["bytes"]), ("LM.s_lm", fmt + "_scales", V * H // 32)):
            out[n] = {"gguf": "output.weight|token_embd.weight", "convert": c, "bytes": b, "dtype": 11 if n == "LM.w_lm" else 1,
                      "shape": [b]}
        for m in lmm.values():
            m["kind"] = "out"
    # the parameter kind of each member and each parameter input: the layer kind of the kernel (group members: their
    # layer's), its _bf16 variant where the bytes are those of the bf16 copy
    def kind_of(role, layer):
        if role == "decode.group":
            pat = re.search(r"group\d+_([ga]+)_", man["kernels"][role]["file"]).group(1)
            return "gdn" if pat[layer] == "g" else "attn"
        return role.rsplit(".", 1)[1] if role.endswith((".gdn", ".attn")) else None
    def resolve(role, layer, name, nbytes):
        k = kind_of(role, layer)
        if name.startswith("e_") and model == "qwen35_moe":
            k = "experts"
        if k is None or name not in man["params"].get(k, {}):
            return None
        if man["params"][k][name]["bytes"] == nbytes:
            return k
        if name in man["params"].get(k + "_bf16", {}) and man["params"][k + "_bf16"][name]["bytes"] == nbytes:
            return k + "_bf16"
        sys.exit(f"{role}: {name} ({nbytes} bytes) matches no parameter of {k}")
    for role, k in man["kernels"].items():
        for m in k.get("members", []):
            if m.get("kind") in ("out", "cp", "cp_out"):
                continue
            m["kind"] = resolve(role, m.get("layer", 0), m["param"], m["bytes"])
            if m["kind"] is None:
                sys.exit(f"{role}: member {m['name']} is not a layer parameter")
        for e in k["in"]:
            base = re.sub(r"\d+$", "", e["name"]) if role == "decode.group" else e["name"]
            kd = resolve(role, 0, base, e["bytes"]) if base.startswith("e_") or role != "decode.group" else None
            if kd:
                e["param"], e["kind"] = base, kd
    # every packed member is a known parameter of its layer kind
    known = set().union(*(set(man["params"].get(k, {})) for k in ("gdn", "attn", "out", "cp", "cp_out")))
    for role, k in man["kernels"].items():
        for m in k.get("members", []):
            if m["param"] not in known:
                sys.exit(f"{role}: member {m['name']} is not a layer parameter")
    json.dump(man, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
