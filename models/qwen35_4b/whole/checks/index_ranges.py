#!/usr/bin/env python3
"""Host-side range checks of every index the whole-model path gathers with.

python index_ranges.py <hf config dir>

Checks the adapter's page arithmetic (prefill positions / page ids / KV rows), the decode cell -> K/V row mapping at
both ends of the context, the mask block count, the decode bank slices against the prefill's [blocks, ...] banks, and
the ViT's patch-tile / key-block / mask-block / merger-row arithmetic at the board input's 3072 patches and smaller.
"""
import json
import math
import sys

PG = 256
CTX = 16384


def check_real_inputs(hf, paths):
    """the board's actual prompts: tokenize them and check the positions / pages / cache-location rows the host
    will feed at these lengths (prefill, the decode after the prompt, and a turn restarting inside a page)"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    for p in paths:
        ids = tok(open(p).read(), add_special_tokens=False)["input_ids"]
        for off, n in [(0, len(ids))] + [(s, min(512, len(ids) - s)) for s in range(0, len(ids), 512)]:
            assert n > 0 and off + n <= len(ids)
            assert off + n <= CTX
            for r in range(n):
                cell = off + r
                page, row = divmod(cell, PG)
                assert 0 <= cell < CTX and 0 <= page < CTX // PG
                assert 0 <= page * 2 * PG + row < (CTX // PG) * 2 * PG
            pages = math.ceil(n / PG)
            assert pages <= CTX // PG
            assert all(0 <= q < CTX // PG for q in range(pages))
            last = n - (pages - 1) * PG
            assert 1 <= last <= PG
            cell = off + n  # the first decode after this prefill
            if cell < CTX:
                page, row = divmod(cell, PG)
                assert math.ceil((cell + 1) / PG) <= CTX // PG
                assert 0 <= page * 2 * PG + row < (CTX // PG) * 2 * PG
        # a next turn starting a few cells inside a page: the adapter restarts at the page's first cell
        if len(ids) > 8 and len(ids) - 5 < CTX:
            c0 = len(ids) - 5
            s0, k = c0 - c0 % PG, c0 % PG
            assert 0 <= s0 <= c0 < CTX and 0 < k < PG
        print(f"ok: {p}: {len(ids)} tokens, {math.ceil(len(ids) / PG)} pages, last page {len(ids) - (math.ceil(len(ids) / PG) - 1) * PG} cells")


def main():
    cfg = json.load(open(f"{sys.argv[1]}/config.json"))
    c = cfg["text_config"]
    layers = c["num_hidden_layers"]
    blocks = layers // 4
    assert layers % 4 == 0 and CTX % PG == 0
    # prefill: a batch at cell 0 and one that restarts inside its page; every M-RoPE coordinate is < the context
    for c0, n in ((0, 4096), (0, CTX), (3 * PG, 5 * PG), (CTX - PG, PG)):
        assert c0 % PG == 0 and c0 + n <= CTX, (c0, n)
        for r in range(n):
            cell = c0 + r
            page = cell // PG
            assert 0 <= cell < CTX
            assert 0 <= page < CTX // PG
            for pos in (cell, cell, cell):  # the three M-RoPE coordinates are token positions
                assert 0 <= pos < CTX
    # decode: pn = cell, blocks, page * 2 * PG + offset; the KV cache is [pages][2][256][kv heads][head dim]
    for cell in (0, 1, PG - 1, PG, 2 * PG - 1, 4095, 4096, CTX - 1):
        page, off = divmod(cell, PG)
        nblk = math.ceil((cell + 1) / PG)
        row = page * 2 * PG + off
        assert 0 <= cell < CTX and 0 < nblk <= CTX // PG and 0 <= row < (CTX // PG) * 2 * PG
    # a 16-layer group needs 4 prefill blocks; its two calls cover the 8 blocks exactly
    G = 16
    assert layers % G == 0 and G % 4 == 0
    for gi in range(layers // G):
        first = gi * (G // 4)
        assert 0 <= first and first + G // 4 <= blocks
    print(f"ok: {layers} layers, {blocks} blocks, {CTX // PG} pages, context {CTX}")
    # ViT: N=3072 patch slots, QB=512 query rows, BC=256 key rows, MT=128 rows per projection tile,
    # MERGE=2 -> 4 rows per merged output.
    v = cfg["vision_config"]
    N, QB, BC, MT, MERGE, CORES = 3072, 512, 256, 128, v["spatial_merge_size"], 4
    RC = N // CORES
    assert N % QB == 0 and N % BC == 0 and N % (MERGE * MERGE) == 0 and RC % MT == 0
    assert MT % (MERGE * MERGE) == 0
    for q0 in range(0, N, QB):
        assert q0 + QB <= N and q0 % BC == 0
    for c in range(CORES):  # each core's row tiles, in MT-row rounds
        for r in range(RC // MT):
            row0, row1 = c * RC + r * MT, c * RC + (r + 1) * MT
            assert 0 <= row0 < row1 <= N and row0 % (MERGE * MERGE) == 0 and row1 % (MERGE * MERGE) == 0
    for n in (MERGE * MERGE, BC - MERGE * MERGE, BC, BC + MERGE * MERGE, 1024, 2048, N):
        nblk = math.ceil(n / BC)
        assert 0 < nblk <= N // BC
        for b in range(nblk):
            k0, k1 = b * BC, (b + 1) * BC
            assert 0 <= k0 < k1 <= N  # kv dynamic_ld(kc / vc, b * BC, [BC])
            assert 0 <= b < N // BC  # mask dynamic_ld(kmask, b, [QB, BC])
        assert n % (MERGE * MERGE) == 0  # the host pads an image patch count to a whole merged tile
        assert (n // (MERGE * MERGE)) <= N // (MERGE * MERGE)
    print(f"ok: ViT {N} patches, {N // QB} query tiles, {N // BC} key blocks, {CORES}x{RC // MT} row tiles, "
          f"{N // (MERGE * MERGE)} merged rows")
    if len(sys.argv) > 2:
        import os
        check_real_inputs(os.environ.get("MODEL", sys.argv[1]), sys.argv[2:])


if __name__ == "__main__":
    main()
