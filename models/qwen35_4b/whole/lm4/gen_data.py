"""Load Qwen3.5-4B's tied token_embd.weight and make MXFP4 lm_head build data."""
import hashlib
import json
import os
import time

import torch

from mx4_common import load_output_weight, quantize_mxfp4, torch_logits

GGUF = os.environ.get("GG", "/data/junda/work/gguf/qwen3.5-4b-mxfp4.gguf")
OUT = os.path.abspath(os.environ.get("OUT", "data"))
K = int(os.environ.get("LM_K", "2560"))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("loading token_embd.weight ...", flush=True)
    w = load_output_weight(GGUF, "token_embd.weight")
    assert w.shape[1] == K, w.shape
    print("weight", tuple(w.shape), w.dtype, f"{time.time() - t0:.1f}s", flush=True)
    print("quantizing MXFP4 ...", flush=True)
    pk, sc = quantize_mxfp4(w)
    del w
    print("packed", tuple(pk.shape), tuple(sc.shape), f"{time.time() - t0:.1f}s", flush=True)
    torch.manual_seed(1234)
    x = torch.randn(4, K).bfloat16()
    x[2] += 7.0
    x[3] -= 7.0
    ref = torch_logits(x, pk, sc)
    print("ref", tuple(ref.shape), f"{time.time() - t0:.1f}s", flush=True)
    pk.numpy().tofile(os.path.join(OUT, "w.bin"))
    sc.numpy().tofile(os.path.join(OUT, "s.bin"))
    x.view(torch.uint16).numpy().tofile(os.path.join(OUT, "x.bin"))
    torch.save(ref, os.path.join(OUT, "ref_torch.pt"))
    meta = {
        "gguf": GGUF,
        "tensor": "token_embd.weight",
        "N": int(pk.shape[0]),
        "K": int(sc.shape[1] * 32),
        "w_bytes": int(pk.numel()),
        "s_bytes": int(sc.numel()),
        "w_md5": md5(os.path.join(OUT, "w.bin")),
        "s_md5": md5(os.path.join(OUT, "s.bin")),
        "x_md5": md5(os.path.join(OUT, "x.bin")),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
