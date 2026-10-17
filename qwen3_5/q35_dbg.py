"""Debug layer 3 (first full-attention layer): feed HF layer-2 outputs, check KV cache rows and RoPE."""
import os, torch
os.environ.setdefault("DUMP", "out_q35_dbg")
import q35_step as q
import axnn.dsl as ax
from transformers import AutoModelForCausalLM

os.makedirs(q.DUMP, exist_ok=True); os.chdir(q.DUMP)
ax.init(chip_num=1, core_num=1)
r = torch.load("../out_q35_step/result.pt")
toks = r["tokens"][:4]
model = AutoModelForCausalLM.from_pretrained(q.MODEL, dtype=torch.bfloat16).eval()
L3 = model.model.layers[3] if hasattr(model.model, "layers") else model.model.language_model.layers[3]
cap = {}
L3.self_attn.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("k_proj", o))
with torch.no_grad():
    o = model(torch.tensor([toks]), output_hidden_states=True)
x_in = o.hidden_states[3][0]  # layer-3 input
ref_out = o.hidden_states[4][0]
# HF 的 k（q_norm/k_norm + rope 后）
sa = L3.self_attn
with torch.no_grad():
    h = L3.input_layernorm(x_in[None])
    k = sa.k_norm(sa.k_proj(h).view(1, len(toks), q.HKV, q.D)).transpose(1, 2)
    pos_ids = torch.arange(len(toks))[None]
    rot = model.model.rotary_emb if hasattr(model.model, "rotary_emb") else model.model.language_model.rotary_emb
    cos, sin = rot(h, pos_ids)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    _, k_ref = apply_rotary_pos_emb(k, k, cos, sin)  # [1, HKV, T, D]
print("hf cos shape", cos.shape, "my cos pos1", q.rope_cos_sin(1)[0][0, :4], "hf", cos[0, 1, :4])

W = q.Weights(q.MODEL)
a = q.attn_args(W, 3)
kern = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{q.DUMP}/attn")(q.attn_layer)
kc = torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16)
vc = torch.zeros_like(kc)
cc = lambda x, y: torch.nn.functional.cosine_similarity(x.reshape(-1).double(), y.reshape(-1).double(), dim=0).item()
for pos in range(len(toks)):
    cs, sn = q.rope_cos_sin(pos)
    m = torch.full((q.LMAX,), -3e38); m[: pos + 1] = 0
    out = torch.empty((1, 1, q.H), dtype=torch.bfloat16)
    case = dict(a, x=x_in[pos].reshape(1, 1, q.H).clone(), cos=cs, sin=sn, kc=kc, vc=vc,
                pos=torch.tensor([pos], dtype=torch.int32), nblk=torch.tensor([pos // q.BC + 1], dtype=torch.int32),
                mask=m.reshape(q.LMAX // q.BC, q.BC).bfloat16(), out=out)
    q.RESIDENT.update(kc=kc.clone(), vc=vc.clone())
    kern[1](*[case[n] for n in q.ATTN_ORDER])
    print(f"pos {pos}: out cos {cc(out, ref_out[pos]):.6f}; k rows cos:",
          [round(cc(kc[:, t], k_ref[0, :, t]), 5) for t in range(pos + 1)],
          "row0 norm", kc[:, 0].float().norm().item())
