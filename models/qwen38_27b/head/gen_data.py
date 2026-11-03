"""Load the real Qwen3.8-27B output.weight (Q8_0), pack MXFP4 as the host does, and make test data."""
import hashlib, json, os, time
import torch
from mx4_common import load_output_weight, quantize_mxfp4, torch_logits

GGUF = "/data/junda/gguf/qwen3.8-27b-mxfp4.gguf"
OUT = os.path.abspath("data")
os.makedirs(OUT, exist_ok=True)


def md5(path, limit=None):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("loading output.weight ...", flush=True)
    w = load_output_weight(GGUF)
    print("weight", tuple(w.shape), w.dtype, f"{time.time()-t0:.1f}s", flush=True)
    print("quantizing MXFP4 ...", flush=True)
    pk, sc = quantize_mxfp4(w)
    del w
    print("packed", tuple(pk.shape), tuple(sc.shape), f"{time.time()-t0:.1f}s", flush=True)
    torch.manual_seed(1234)
    x = (torch.randn(4, 5120) * 1.0).bfloat16()
    x[2] += 7.0  # distinct rows for the untouched-row check
    x[3] -= 7.0
    ref = torch_logits(x, pk, sc)
    print("ref", tuple(ref.shape), f"{time.time()-t0:.1f}s", flush=True)
    pk.numpy().tofile(os.path.join(OUT, "w.bin"))
    sc.numpy().tofile(os.path.join(OUT, "s.bin"))
    x.view(torch.uint16).numpy().tofile(os.path.join(OUT, "x.bin"))
    torch.save(ref, os.path.join(OUT, "ref_torch.pt"))
    meta = {
        "gguf": GGUF, "tensor": "output.weight", "N": int(pk.shape[0]), "K": int(sc.shape[1] * 32),
        "w_bytes": int(pk.numel()), "s_bytes": int(sc.numel()),
        "w_md5": md5(os.path.join(OUT, "w.bin")), "s_md5": md5(os.path.join(OUT, "s.bin")),
        "x_md5": md5(os.path.join(OUT, "x.bin")),
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2), f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
