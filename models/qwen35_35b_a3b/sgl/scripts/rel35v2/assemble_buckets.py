"""Assemble a context-bucketed decode release from a 256k base manifest.

The bucket kernels differ from the base kernel only in the declared page count of
the per-layer KV inputs, the page table and the RoPE prefix; every weight bank is
identical and the adapter binds all variants to one set of banks and KV pools.
"""
import argparse
import json
import os
import shutil

KV_PAGE_BYTES = 2 * 256 * 2 * 256 * 2  # K/V x 256 cells x 2 heads x 256 dims, bf16


def bucket_io(base_in, pages):
    out = []
    for x in base_in:
        x = dict(x)
        if x["name"] == "indices":
            x["shape"] = [pages]
            x["bytes"] = pages * 4
        elif x["name"] == "b3_attn_cos_sin_cache":
            x["shape"] = [1, pages * 256, 64]
            x["bytes"] = pages * 256 * 64 * 4
        elif x["name"].startswith("b3_kv"):
            x["shape"] = [pages, 2, 256, 2, 256]
            x["bytes"] = pages * KV_PAGE_BYTES
        out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("base_manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--bucket", action="append", default=[], metavar="ROLE:PAGES:FILE")
    a = ap.parse_args()

    man = json.load(open(a.base_manifest))
    dec = man["kernels"]["decode"]
    os.makedirs(a.out_dir, exist_ok=True)

    for spec in a.bucket:
        role, pages, name = spec.split(":")
        pages = int(pages)
        k = json.loads(json.dumps(dec))
        k["file"] = name
        k["in"] = bucket_io(dec["in"], pages)
        man["kernels"][role] = k

    json.dump(man, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)

    for role, k in man["kernels"].items():
        src = os.path.join(a.base_dir, k["file"])
        dst = os.path.join(a.out_dir, k["file"])
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    print("wrote", a.out_dir, "kernels:", list(man["kernels"]))


if __name__ == "__main__":
    main()
