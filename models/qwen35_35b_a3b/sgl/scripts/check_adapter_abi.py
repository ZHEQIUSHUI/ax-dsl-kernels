"""Static check: the qwen35_moe_sgl adapter's bank slicing vs manifest io.json + the pack."""
import json, sys

io = json.load(open(sys.argv[1]))
pack = json.load(open(sys.argv[2]))          # out_sgl_pack_v2/manifest.json
man = json.load(open(sys.argv[3]))           # board manifest.json
NB = 10
state = {"state_in", "q_conv_state_in", "k_conv_state_in", "v_conv_state_in"}
bad = []

def need(name, nbytes, where):
    p = pack.get(name)
    if p is None:
        bad.append(f"{where}: no bank {name}"); return None
    if p["bytes"] % NB:
        bad.append(f"{where}: {name} {p['bytes']} not / {NB}"); return None
    row = p["bytes"] // (3 * NB if "state_in" in name and name.startswith("b0_attn_") else NB)
    return row

# decode: every input is a bank (except the seven runtime ones) with the kernel's bytes
for name, by, dt, sh in io["sgl35_decode.axmodel"]["in"]:
    if name in ("x","attention","position_ids","indptr","indices","last_page_len","cache_location"):
        continue
    p = pack.get(name)
    if p is None or p["bytes"] != by:
        bad.append(f"decode {name}: pack {p and p['bytes']} vs {by}")

for role, phase in (("prefill.gdn", "b0_attn_"), ("prefill.attn", "b3_attn_")):
    for name, by, dt, sh in io[{"prefill.gdn":"sgl35_gdn.axmodel","prefill.attn":"sgl35_flash.axmodel"}[role]]["in"]:
        if name in ("x","position_ids","indptr","indices","cache_location","cos_sin_cache","valid_tokens"):
            continue
        if name.startswith("w_zeropoint_"):
            continue
        if name == "paged_kv_cache":
            row = need("b3_kv", by, role)
            if row is not None and row != by: bad.append(f"{role} paged_kv_cache row {row} vs {by}")
            continue
        bank = "b0_attn_" + name if name in state else phase + {"x_norm_w":"w_norm_attn","q_norm_w":"w_norm_q","k_norm_w":"w_norm_k"}.get(name, name)
        row = need(bank, by, role)
        if row is None: continue
        if "scale2" in name:
            if row < by: bad.append(f"{role} {name}: bank row {row} < {by}")
        elif row != by:
            bad.append(f"{role} {name}: bank {bank} row {row} vs {by}")

for name, by, dt, sh in io["sgl35_moe4096.axmodel"]["in"]:
    if name == "x": continue
    b = name
    if name in ("w_up","w_scale_up"): b = name.replace("w_up","w_gate").replace("w_scale_up","w_scale_gate")
    if name == "w_scale2_up": b = "w_scale2_gate"
    row = need("b0_moe_" + b, by, "moe")
    if row is None: continue
    if ("scale2" in name) or name == "w_scale2_route":
        if row < by: bad.append(f"moe {name}: bank row {row} < {by}")
    elif row != by:
        bad.append(f"moe {name}: bank row {row} vs {by}")

for name, by, dt, sh in io["sgl35_padding.axmodel"]["in"]:
    if name not in ("activation", "token_count"):
        bad.append(f"padding: unexpected input {name}")

# roles the adapter requires
for r in ("decode","prefill.gdn","prefill.attn","prefill.moe","prefill.padding","lm_head"):
    if r not in man["kernels"]:
        bad.append(f"manifest: no role {r}")
print("\n".join(bad) if bad else "OK: all adapter bank bindings match the pack and the kernels' IO")
sys.exit(1 if bad else 0)
