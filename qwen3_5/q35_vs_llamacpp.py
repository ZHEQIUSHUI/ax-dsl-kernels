"""Compare NPU-simulated Qwen3.5-0.8B logits (q35_step.py result.pt) with llama.cpp CPU (bf16 GGUF) per position.

llama-debug saves the last-position logits of a prompt, so it is run once per token prefix; the saved token ids
are checked against the NPU token ids so both sides see exactly the same input.
"""
import os, subprocess, sys
import numpy as np, torch
from transformers import AutoTokenizer

RES = sys.argv[1]
GGUF = os.environ.get("GGUF") or sys.exit("set GGUF to the bf16 GGUF of the model")
BIN = os.environ.get("LLAMA_DEBUG") or sys.exit("set LLAMA_DEBUG to a CPU build of llama-debug")
MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")
r = torch.load(RES)
tok = AutoTokenizer.from_pretrained(MODEL)
toks, npu, ref = r["tokens"], r["npu_logits"], r["ref_logits"]
out = os.path.join(os.path.dirname(RES), "llamacpp"); os.makedirs(out, exist_ok=True)
cs = lambda a, b: torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0).item()
rows, agree_npu, agree_hf = [], 0, 0
for t in range(npu.shape[0]):
    text = tok.decode(toks[: t + 1])
    d = f"{out}/p{t}"
    subprocess.run([BIN, "-m", GGUF, "-p", text, "--save-logits", "--logits-output-dir", d, "-t", "16", "-c", "512"],
                   check=True, capture_output=True)
    base = [f for f in os.listdir(d) if f.endswith("-tokens.bin")][0].replace("-tokens.bin", "")
    ids = np.fromfile(f"{d}/{base}-tokens.bin", dtype=np.int32).tolist()
    if ids != toks[: t + 1]:
        print(f"pos {t}: token mismatch {ids} vs {toks[:t+1]}, skipped"); continue
    lc = torch.from_numpy(np.fromfile(f"{d}/{base}.bin", dtype=np.float32))
    a_n, a_h = int(lc.argmax() == npu[t].argmax()), int(lc.argmax() == ref[t].argmax())
    agree_npu += a_n; agree_hf += a_h
    rows.append((t, cs(lc, npu[t]), cs(lc, ref[t]), cs(npu[t], ref[t])))
    print(f"pos {t:2d} cos(llama.cpp,NPU)={rows[-1][1]:.6f} cos(llama.cpp,HF)={rows[-1][2]:.6f} "
          f"cos(NPU,HF)={rows[-1][3]:.6f} top1 llama.cpp={tok.decode([int(lc.argmax())])!r} NPU={tok.decode([int(npu[t].argmax())])!r}")
n = len(rows)
print(f"RESULT positions={n} min_cos(llama.cpp,NPU)={min(x[1] for x in rows):.6f} "
      f"min_cos(llama.cpp,HF)={min(x[2] for x in rows):.6f} min_cos(NPU,HF)={min(x[3] for x in rows):.6f} "
      f"top1 agree llama.cpp-NPU {agree_npu}/{n} llama.cpp-HF {agree_hf}/{n}")
