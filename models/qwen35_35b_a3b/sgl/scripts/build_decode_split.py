"""(build_decode.py with decode_split: one KV input per attention layer, the blocks unrolled; DUMP=0 skips the bank files)
Pack Qwen3.5-35B-A3B HF bf16 weights into the current 866053298c fused decode IO banks.

Writes one <bank>.bin per kernel input, manifest.json (shape/dtype/bytes/state),
then compiles the real-shape decode model. Usage:
python pack_sgl35_current.py <hf_dir> <out_dir> [pages]
(NLAYERS=<multiple of 4>: the first layers only; COMPILE=0: dump only)
"""
import os, sys, glob, json, shutil, time
from pathlib import Path
from multiprocessing import Pool
import torch
from safetensors import safe_open
import axnn.dsl as ax
from axnn.dsl import builder_ as B
from sgl_kernel_axera.models.qwen36_35b_a3b.decode_split import prepare_split as prepare
from sgl_kernel_axera.attention.gdn import runtime_weights
from sgl_kernel_axera.quantization import quantize_nvfp4

class Compiled(Exception):
    pass
def no_sim(self, *a, **k):
    raise Compiled()
for cls in [c for c in vars(B).values() if isinstance(c, type) and hasattr(c, "simulate_for_1_chip")]:
    setattr(cls, "simulate_for_1_chip", no_sim)

HF, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PAGES = int(sys.argv[3]) if len(sys.argv) > 3 else 64
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(HF / "config.json"))
cfg = cfg.get("text_config", cfg)
H, NL = cfg["hidden_size"], cfg["num_hidden_layers"]
NL = int(os.environ.get("NLAYERS", NL))
KH, VH, KD, VD = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
HEADS, KVH, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
E, TOPK, FI = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
rope = cfg.get("rope_parameters") or {"rope_theta": cfg["rope_theta"], "partial_rotary_factor": cfg["partial_rotary_factor"],
                                       "mrope_section": cfg["rope_scaling"]["mrope_section"]}
WMAP = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
P = "model.language_model.layers."

def get(name):
    with safe_open(HF / WMAP[name], "pt") as f:
        return f.get_tensor(name)

U8 = {torch.float4_e2m1fn_x2: "f4", torch.float8_e4m3fn: "f8"}
def to_np(t):
    if t.dtype in U8:
        return (U8[t.dtype], t.view(torch.uint8).numpy())
    if t.dtype == torch.bfloat16:
        return ("bf16", t.view(torch.int16).numpy())
    return ("pt", t.numpy())
def from_np(v):
    kind, a = v
    t = torch.from_numpy(a)
    return {"f4": lambda: t.view(torch.float4_e2m1fn_x2), "f8": lambda: t.view(torch.float8_e4m3fn),
            "bf16": lambda: t.view(torch.bfloat16), "pt": lambda: t}[kind]()

def nv(w):
    return quantize_nvfp4(w.bfloat16())

def plus1(w):
    return (w.float() + 1).bfloat16()

def experts(w, parts):
    """[E, R, K] bf16 -> packed [E, R, K/2], scales [E, R, K/16], scale2 [E, parts] (each part of R its own)"""
    e, r, k = w.shape
    q = torch.empty((e, r, k // 2), dtype=torch.uint8)
    s = torch.empty((e, r, k // 16), dtype=torch.float8_e4m3fn)
    s2 = torch.empty((e, parts))
    rp = r // parts
    for i in range(e):
        for j in range(parts):
            a, b, c = nv(w[i, j * rp:(j + 1) * rp])
            q[i, j * rp:(j + 1) * rp] = a.view(torch.uint8)
            s[i, j * rp:(j + 1) * rp] = b
            s2[i, j] = c[0]
    return q.view(torch.float4_e2m1fn_x2), s, s2 if parts > 1 else s2.reshape(e)

def load_layer(i):
    torch.set_num_threads(2)
    p = P + str(i) + "."
    m = p + "mlp."
    moe = dict(input_norm_w=plus1(get(p + "post_attention_layernorm.weight")), add_residual=True,
               shared_expert_gate=get(m + "shared_expert_gate.weight").T.contiguous(),
               w_route=get(m + "gate.weight"), w_scale_route=torch.ones((E, 1)), w_scale2_route=torch.ones(1))
    gu = experts(get(m + "experts.gate_up_proj"), 2)
    dn = experts(get(m + "experts.down_proj"), 1)
    for name, wt in (("gate", gu), ("down", dn)):
        for prefix, v in zip(("w_", "w_scale_", "w_scale2_"), wt):
            moe[prefix + name] = v
    for name in ("gate", "up", "down"):
        for prefix, v in zip(("w_", "w_scale_", "w_scale2_"), nv(get(m + f"shared_expert.{name}_proj.weight"))):
            moe[prefix + "shared_expert_mlp_" + name] = v
    if i % 4 < 3:
        a = p + "linear_attn."
        kw, vw = KH * KD, VH * VD
        qkv = get(a + "in_proj_qkv.weight")
        conv = get(a + "conv1d.weight").squeeze(1)
        w = {"input_norm_w": plus1(get(p + "input_layernorm.weight"))}
        src = dict(zip("qkv", qkv.split([kw, kw, vw])))
        src.update(z=get(a + "in_proj_z.weight"), a=get(a + "in_proj_a.weight"), b=get(a + "in_proj_b.weight"), o=get(a + "out_proj.weight"))
        for n, t in src.items():
            w[f"w_{n}"], w[f"w_{n}_scale"], w[f"w_{n}_scale2"] = nv(t)
        for n, t in zip("qkv", conv.split([kw, kw, vw])):
            w[f"w_{n}_conv"] = t.contiguous()
        w.update(a_log=get(a + "A_log").float(), dt_bias=get(a + "dt_bias"),
                 o_norm_w=get(a + "norm.weight").bfloat16())
        w = runtime_weights(w)
        w["w_pre"] = None
    else:
        a = p + "self_attn."
        w = {}
        for n in "qkvo":
            for prefix, v in zip(("w_", "w_scale_", "w_scale2_"), nv(get(a + n + "_proj.weight"))):
                w[prefix + n] = v
            w["w_zeropoint_" + n] = torch.zeros(1)
        w.update(w_norm_attn=plus1(get(p + "input_layernorm.weight")),
                 w_norm_q=(get(a + "q_norm.weight").float() + 1) * HD ** -0.5,
                 w_norm_k=plus1(get(a + "k_norm.weight")))
    return i, {k: (to_np(v) if torch.is_tensor(v) else v) for k, v in w.items()}, \
        {k: (to_np(v) if torch.is_tensor(v) else v) for k, v in moe.items()}

t0 = time.time()
with Pool(int(os.environ.get("PROCS", 20))) as pool:
    got = sorted(pool.map(load_layer, range(NL)))
print("quantized", time.time() - t0, flush=True)

rot = int(HD * rope["partial_rotary_factor"])
CTX = PAGES * 256
ang = torch.arange(CTX)[:, None] / (rope["rope_theta"] ** (torch.arange(0, rot, 2).float() / rot))[None, :]
cos_sin = torch.cat((ang.cos(), ang.sin()), dim=-1)
layers = []
for i, w, moe in got:
    w = {k: (from_np(v) if isinstance(v, tuple) else v) for k, v in w.items()}
    moe = {k: (from_np(v) if isinstance(v, tuple) else v) for k, v in moe.items()}
    moe.update(top_k=TOPK, num_experts=E, norm_topk_prob=cfg.get("norm_topk_prob", True), fused_gate_up=True)
    moe["w_up"], moe["w_scale_up"], moe["w_scale2_up"] = moe["w_gate"], moe["w_scale_gate"], moe["w_scale2_gate"]
    if i % 4 < 3:
        layers.append((None, w, moe))
    else:
        w.update(cos=None, sin=None, cos_sin_cache=cos_sin, partial_rotary_factor=rope["partial_rotary_factor"])
        layers.append((torch.zeros(PAGES, 2, 256, KVH, HD, dtype=torch.bfloat16), w, moe))
lin = [a for kv, a, _ in layers if kv is None]
pools = (torch.zeros(len(lin), 1, VH, KD, VD), torch.zeros(len(lin), 1, 3, KH * KD, dtype=torch.bfloat16),
         torch.zeros(len(lin), 1, 3, KH * KD, dtype=torch.bfloat16), torch.zeros(len(lin), 1, 3, VH * VD, dtype=torch.bfloat16))
for i, a in enumerate(lin):
    for name, pool in zip(("state", "q_conv_state", "k_conv_state", "v_conv_state"), pools):
        a[name + "_in"] = a[name + "_out"] = pool[i]
kvp = torch.stack([kv for kv, _, _ in layers if kv is not None])
j = 0
for i in range(len(layers)):
    if layers[i][0] is not None:
        layers[i] = (kvp[j], layers[i][1], layers[i][2]); j += 1
kernel, weights = prepare(layers, mrope_section=tuple(rope["mrope_section"]), cache_path=OUT / "cache")
print("prepared", time.time() - t0, flush=True)

pool_ptrs = {t.untyped_storage().data_ptr() for t in (*pools, kvp)}
man = {}
for k, v in weights.items():
    if not torch.is_tensor(v):
        continue
    state = v.untyped_storage().data_ptr() in pool_ptrs
    raw = v.contiguous().view(torch.uint8).reshape(-1) if v.dtype != torch.uint8 else v.contiguous().reshape(-1)
    man[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": raw.numel(), "state": state}
    if not state and os.environ.get("DUMP", "1") != "0":
        raw.numpy().tofile(OUT / f"{k}.bin")
json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
with open(OUT / "manifest.txt", "w") as f:  # for the backend: name bytes state
    for k, m in man.items():
        f.write("%s %d %d\n" % (k, m["bytes"], m["state"]))
print("dumped", len(man), sum(m["bytes"] for m in man.values()) / 2**30, "GiB", time.time() - t0, flush=True)
if os.environ.get("COMPILE", "1") == "0":
    sys.exit(0)

x = torch.zeros(1, H, dtype=torch.bfloat16); att = torch.zeros_like(x)
prefix = CTX - 1
md = (torch.tensor([prefix], dtype=torch.int32).repeat(3, 1), torch.tensor([0, PAGES], dtype=torch.int32),
      torch.arange(PAGES, dtype=torch.int32), torch.tensor([256], dtype=torch.int32),
      torch.tensor([PAGES - 1, 255], dtype=torch.int32))
try:
    kernel[4, 4](x, att, *md, x, **weights)
except Compiled:
    pass
print("compiled", time.time() - t0, flush=True)
f = glob.glob(str(OUT / "cache" / "**" / "*.compiled.axmodel"), recursive=True)
assert len(f) == 1, f
shutil.copy(f[0], OUT / os.environ.get("NAME", "sgl35_decode.axmodel"))
print(f[0], os.path.getsize(f[0]))
