"""draft_vocab_<N>.i32: the N draft-vocabulary token ids (ascending): the byte tokens, the added (special) tokens, then
the most frequent (order.npy) up to N."""
import json, sys
import numpy as np
D = "/data/junda/work/vocab"
order = np.load(f"{D}/order.npy")
tj = json.load(open(f"{D}/tokenizer.json"))
added = [t["id"] for t in tj.get("added_tokens", [])]
for N in map(int, sys.argv[1:]):
    s = set(range(256)) | set(added)
    for t in order:
        if len(s) >= N:
            break
        s.add(int(t))
    ids = np.array(sorted(s), np.int32)
    assert len(ids) == N and ids.max() < 248320, (len(ids), ids.max())
    ids.tofile(f"{D}/draft_vocab_{N}.i32")
    print(N, "added", len(added), "max id", ids.max())
