#!/usr/bin/env python3
"""Replace a Qwen3 manifest's decode IO / packs / members from a compiled group.

The decode group's k_b<phase>_<bank> inputs must be byte-for-byte the prefill
kernel's b<phase>_<bank> inputs.  This checks that contract while updating the
manifest, so a stale prefill bank cannot silently be paired with a new decode
kernel.

Usage: update_decode_banks.py manifest.json decode.axmodel decode.layout
"""
import json
import os
import sys

import onnx

import mkmanifest
from onnx_io import entry


def main():
    manifest_path, kernel_path, layout_path = sys.argv[1:4]
    man = json.load(open(manifest_path))
    dec = man["kernels"]["decode"]
    assert os.path.basename(kernel_path) == dec["file"], (kernel_path, dec["file"])

    # mkmanifest.layout() describes the bytes; the adapter also needs the parameter kind of every member.  That
    # is a property of the member name and survives a decode-kernel rebuild.
    kinds = {m["name"]: m["kind"] for m in dec.get("members", [])}
    model = onnx.load(kernel_path, load_external_data=False)
    dec["in"] = mkmanifest.io_entries([entry(v) for v in model.graph.input])
    dec["out"] = mkmanifest.io_entries([entry(v) for v in model.graph.output])
    dec["packs"], dec["members"] = mkmanifest.layout(layout_path)
    for m in dec["members"]:
        assert m["name"] in kinds, f"decode layout member {m['name']} has no previous kind"
        m["kind"] = kinds[m["name"]]

    prefill = {e["name"]: e for e in man["kernels"]["prefill"]["in"]}
    banks = 0
    for e in dec["in"]:
        if not e["name"].startswith("k_b"):
            continue
        p = prefill.get(e["name"][2:])
        assert p is not None, f"decode bank {e['name']} has no prefill input"
        assert (e["bytes"], e["shape"], e["dtype"]) == (p["bytes"], p["shape"], p["dtype"]), (e, p)
        banks += 1
    assert banks == 40, banks

    with open(manifest_path, "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(f"{manifest_path}: decode {len(dec['in'])} inputs, {len(dec['members'])} members, {banks} bank inputs")


if __name__ == "__main__":
    main()
