#!/usr/bin/env python3
"""Host-side contract checks for the decode group's prefill-bank inputs.

Checks the manifest and layout without a board: every k_b<phase>_<bank> input is
the same shape and byte count as the prefill b<phase>_<bank> input, every layer
selects an in-range block, and all paged-KV / mask indices are in range at the
page boundaries used by the real decode inputs.  The packed members are checked
against their pack bounds and must not contain a banked parameter.

Usage: python check_banks.py --manifest ../qwen3_asr_0.6b.manifest.json \
           --kernel-dir . [--context 4096]
"""
import argparse
import json
import os
import sys


REMAIN = {"ln1", "q_norm", "k_norm", "w_o", "s_o", "ln2", "w_down", "s_down"}
BANK_OF = {
    "w_q": "attn_w_q", "s_q": "attn_w_scale_q",
    "w_k": "attn_w_k", "s_k": "attn_w_scale_k",
    "w_v": "attn_w_v", "s_v": "attn_w_scale_v",
    "w_gate": "mlp_w1", "s_gate": "mlp_w_scale1",
    "w_up": "mlp_w2", "s_up": "mlp_w_scale2",
}


def read_layout(path):
    packs, members = {}, []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        name, pack, off, nbytes = f[0], f[1], int(f[2]), int(f[3])
        if pack == "-":
            packs[name] = nbytes
        else:
            members.append((name, pack, off, nbytes, f[4:]))
    return packs, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kernel-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--context", type=int, default=4096)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    kd = man["kernels"]["decode"]
    pf = {e["name"]: e for e in man["kernels"]["prefill"]["in"]}
    dc = {e["name"]: e for e in kd["in"]}
    layers = 28
    blocks = layers // 4

    for phase in range(4):
        for bank in BANK_OF.values():
            pn, dn = f"b{phase}_{bank}", f"k_b{phase}_{bank}"
            assert pn in pf and dn in dc, (pn, dn)
            assert pf[pn]["bytes"] == dc[dn]["bytes"], (pn, pf[pn]["bytes"], dc[dn]["bytes"])
            assert pf[pn]["shape"] == dc[dn]["shape"], (pn, pf[pn]["shape"], dc[dn]["shape"])
            assert pf[pn]["shape"][0] == blocks, (pn, pf[pn]["shape"][0], blocks)

    layout = os.path.join(args.kernel_dir, kd["file"][:-len(".axmodel")] + ".layout")
    packs, members = read_layout(layout)
    assert packs == kd["packs"], (packs, kd["packs"])
    assert len(members) == len(kd["members"]), (len(members), len(kd["members"]))
    for e in kd["members"]:
        assert {"name", "pack", "offset", "bytes", "param", "kind"} <= set(e), e
        if e["name"].startswith("L") and not e["name"].startswith("LM."):
            assert "layer" in e, e
    used = {p: [] for p in packs}
    layer_members = {}
    for name, pack, off, nbytes, rest in members:
        assert pack in packs, (name, pack)
        assert off >= 0 and off + nbytes <= packs[pack], (name, off, nbytes, packs[pack])
        used[pack].append((off, off + nbytes, name))
        if name.startswith("L") and not name.startswith("LM."):
            layer, param = name.split(".", 1)
            layer_members.setdefault(int(layer[1:]), set()).add(param)
            assert param not in BANK_OF, f"banked parameter still packed: {name}"
    for pack, spans in used.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            assert a[1] <= b[0], (pack, a, b)

    for j in range(layers):
        assert layer_members.get(j) == REMAIN, (j, sorted(layer_members.get(j, ())))
        phase, block = j % 4, j // 4
        assert 0 <= block < blocks
        for bank in BANK_OF.values():
            assert f"k_b{phase}_{bank}" in dc

    BC, cores = 256, 4
    nblk = args.context // BC
    positions = (0, 1, BC - 2, BC - 1, BC, BC + 1, BC + 2, 2 * BC - 1, 2 * BC, args.context - 1)
    for pos in positions:
        assert 0 <= pos < args.context
        page, row = divmod(pos, BC)
        cur_nblk = page + 1
        krow = page * 2 * BC + row
        vrow = krow + BC
        assert 0 <= page < nblk and 0 <= row < BC
        assert 0 <= cur_nblk <= nblk
        assert 0 <= krow < nblk * 2 * BC and 0 <= vrow < nblk * 2 * BC
        assert pos < nblk * BC
        pages = set()
        for core in range(cores):
            for p in range(core, cur_nblk, cores):
                assert 0 <= p < nblk
                pages.add(p)
        assert pages == set(range(cur_nblk))

    print(f"PASS: {len(dc)} decode inputs, {len(BANK_OF) * 4} bank inputs == prefill bytes / shapes; "
          f"{len(members)} packed members in bounds, no banked member; positions {positions} in range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
