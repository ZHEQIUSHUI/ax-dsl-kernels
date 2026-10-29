#!/usr/bin/env python3
"""manifest.json of a model's kernel directory: the contract between ax-dsl-kernels and ggml-axera.

mkmanifest.py <model: qwen35_dense|qwen35_moe> <io.json> <layout dir> > manifest.json

- params: per layer kind (gdn / attn), each layer parameter the kernels take: its GGUF tensor (within blk.<i>.), the
  conversion from it and its kernel form (bytes, dtype, shape). Taken from the layer kernels' IO.
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
# kernel IO that is not a parameter: what the runtime binds
RUNTIME = {"x", "out", "pos", "nblk", "mask", "cos", "sin", "cs", "pn", "state_in", "conv_in", "state_out",
           "conv_out", "kc", "vc", "h2_out", "logits_out", "rv", "cix", "topk_v", "topk_i"}

ROLES = {
    "qwen35_dense": {  # Qwen3.8-27B, MXFP4 layer weights
        "layer.gdn": "qwen35_gdn_mx4_npu1",  # IO only: the parameters of a GDN layer
        "layer.attn": "qwen35_attn_mx4_kvc_c16384_npu1",  # IO only: the parameters of an attention layer
        "decode.group": "qwen35_group16_gggagggagggaggga_mx4_kvc_c16384_npu4",
        "lm_head": "mul_mat_mx4_n248320_k5120_npu4",
        **{f"prefill{p}.gdn": f"qwen35_gdn_prefill{p}_mx4_pv_pk_npu4" for p in (128, 64, 16, 4)},
        **{f"prefill{p}.attn": f"qwen35_attn_prefill{p}_mx4_pk_kvc_c16384_npu4" for p in (128, 64, 16, 4)},
    },
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


def io_entries(io):
    return [{"name": n, "bytes": b, "dtype": t, "shape": s} for n, b, t, s in io]


def elements(shape):
    e = 1
    for d in shape:
        e *= d
    return e


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
        g = (GGUF_MOE.get(src) if model == "qwen35_moe" else None) or GGUF.get(src)
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
    man = {"format": 1, "model": model, "context": CONTEXT, "cores": 4, "params": {}, "kernels": {}}
    for role, base in ROLES[model].items():
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
        man["kernels"][role] = k
    if model == "qwen35_moe":  # routed experts: GGUF MXFP4 repacked in place as planes (codes, then scales)
        g = man["kernels"]["moe.slots32_g64"]["in"]
        shape = {e["name"]: e for e in g}
        man["params"]["experts"] = {
            "e_gw": {"gguf": "ffn_gate_up_exps.weight", "convert": "mxfp4_planar_codes"},
            "e_gs": {"gguf": "ffn_gate_up_exps.weight", "convert": "mxfp4_planar_scales"},
            "e_dw": {"gguf": "ffn_down_exps.weight", "convert": "mxfp4_planar_codes"},
            "e_ds": {"gguf": "ffn_down_exps.weight", "convert": "mxfp4_planar_scales"},
        }
        for n, v in man["params"]["experts"].items():
            v.update(bytes=shape[n]["bytes"], dtype=shape[n]["dtype"], shape=shape[n]["shape"])
    # every packed member is a known parameter of its layer kind
    known = set(man["params"]["gdn"]) | set(man["params"]["attn"])
    for role, k in man["kernels"].items():
        for m in k.get("members", []):
            if m["param"] not in known:
                sys.exit(f"{role}: member {m['name']} is not a layer parameter")
    json.dump(man, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
