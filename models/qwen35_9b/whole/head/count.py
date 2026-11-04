"""Token frequencies of corpora (Qwen3.8 tokenizer) -> counts_<name>.npy; then the draft vocabularies' coverage of the
target's own outputs (tree27 trace responses)."""
import glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

D = "/data/junda/work/vocab"
tok = Tokenizer.from_file(f"{D}/tokenizer.json")
V = 248320


def count(name, texts, limit_chars):
    out = f"{D}/counts_{name}.npy"
    if os.path.exists(out):
        return np.load(out)
    c = np.zeros(V, np.int64)
    batch, n = [], 0
    for t in texts:
        if not t or not t.strip():
            continue
        batch.append(t)
        n += len(t)
        if len(batch) == 512:
            for e in tok.encode_batch(batch, add_special_tokens=False):
                np.add.at(c, np.array(e.ids, np.int64), 1)
            batch = []
        if n > limit_chars:
            break
    for e in tok.encode_batch(batch, add_special_tokens=False):
        np.add.at(c, np.array(e.ids, np.int64), 1)
    np.save(out, c)
    print(name, n, "chars", c.sum(), "tokens", (c > 0).sum(), "distinct", flush=True)
    return c


def parquet_texts(f):
    for b in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=2048):
        yield from b.column(0).to_pylist()


def code_texts():
    for f in glob.glob("/data/junda/work/qs_mtpref/**/*", recursive=True):
        if f.endswith((".cpp", ".h", ".c", ".py", ".md", ".cu", ".js", ".ts", ".sh", ".json")) and os.path.isfile(f):
            try:
                yield open(f, encoding="utf-8", errors="ignore").read()
            except Exception:
                pass


cs = {"en": count("en", parquet_texts(f"{D}/wt2.parquet"), 150e6),
      "code": count("code", code_texts(), 80e6)}
def zh_texts():
    for line in open(f"{D}/zhw.jsonl", encoding="utf-8", errors="ignore"):
        try:
            yield json.loads(line)["completion"]
        except Exception:
            pass


cs["zh"] = count("zh", zh_texts(), 150e6)
# ranking: the corpora's frequencies normalized, summed
score = sum(c / c.sum() for c in cs.values())
order = np.argsort(-score, kind="stable")
np.save(f"{D}/order.npy", order)
# evaluation: the target's generated tokens
ev = {}
for f in sorted(glob.glob("/data/junda/work/qsmall/tree27/trace/parta_*.json")):
    d = json.load(open(f))
    if isinstance(d, dict) and "token_ids" in d:
        ev[d["name"]] = np.array(d["token_ids"])
for N in (65536, 81920, 98304, 131072):
    sub = np.zeros(V, bool)
    sub[order[:N]] = True
    sub[:256] = True  # (bytes)
    row = " ".join(f"{k}={sub[v].mean() * 100:.1f}" for k, v in ev.items())
    allv = np.concatenate(list(ev.values()))
    print(f"N={N}: all {sub[allv].mean() * 100:.2f}%  {row}", flush=True)
