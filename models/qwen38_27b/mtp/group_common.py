"""Packed layer-weight construction shared by the 27B verify group and the MTP kernels.

The packing and layout code is deliberately the q27m/be6410c code path: the same
member order, 64-byte alignment, MXFP4 E2M1/E8M0 format and column-strip blocking.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import q35_group as G
import q35_layer_mc as L
import q35_layer_pk as K
import q35_mx_build as mb
import q35_step as q


def make_layer_args(W, kinds, first: int = 0):
    """Random/real bf16 layer args plus their blocked MXFP4 twins (q35_group.main)."""
    args, raw = [], []
    for j, kind in enumerate(kinds):
        conv = (lambda a, names: K.mx4_args(a, names)) if K.MX4 else (lambda a, names: mb.mx_args(a, names)[0])
        mk = lambda: (conv(q.gdn_args(W, first + j), mb.GDN_W) if kind == "gdn"
                      else conv(q.attn_args(W, first + j), mb.ATTN_W))
        q.VORDER = "tiled"
        raw.append(mk())
        q.VORDER = "grouped"
        a = mk()
        q.VORDER = "tiled"
        for n in K.BW:
            if n in a:
                G.FULL[(j, n)] = a[n].shape[1] * (2 if a[n].dtype == torch.float4_e2m1fn_x2 else 1)
        K.FULL_COLS[kind] = {n: G.FULL[(j, n)] for n in K.BW if n in a}
        args.append(K.block_args(a))
    return args, raw


def _base(p):
    return p.split("_")[0]


class GroupBundle:
    def __init__(self, args, raw, kinds, packs, spec, sizes, pdt, full, lines):
        self.args = args
        self.raw = raw
        self.kinds = kinds
        self.packs = packs
        self.spec = spec
        self.sizes = sizes
        self.pdt = pdt
        self.full = full
        self.lines = lines

    def activate(self):
        G.SPEC.clear()
        G.SPEC.update(self.spec)
        G.FULL.clear()
        G.FULL.update(self.full)
        K.BLK.clear()

    def members(self, j):
        self.activate()
        return G.members(self.packs, j)

    def write_layout(self, path):
        with open(path, "w") as f:
            for p in self.pdt:
                f.write(f"{p} - 0 {self.sizes[p] * torch.tensor([], dtype=self.pdt[p]).element_size()}\n")
            f.write("\n".join(self.lines) + "\n")


def make_group_bundle(W, kinds, first: int = 0):
    args, raw = make_layer_args(W, kinds, first)
    g = len(kinds)
    order = {"gdn": L.order(G.GL), "attn": L.order(G.AL)}
    items = [(f"L{j}.{n}", j, args[j][n]) for j, kind in enumerate(kinds) for n in order[kind]
             if torch.is_tensor(args[j].get(n)) and n != "x"]
    spec = {}
    used = {}
    for S in [s for s in range(1, g + 1) if g % s == 0]:
        spec.clear()
        used = {(f"{p}_{s}" if s else p): 0 for s in range(S) for p in K.PACKS}
        for key, j, tns in items:
            sidx = j // (g // S)
            p = K.pack_of(tns) + (f"_{sidx}" if sidx else "")
            es = tns.element_size()
            off = (used[p] * es + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es
            spec[key] = (p, off, tuple(tns.shape))
            used[p] = off + tns.numel()
        if max(used[p] * torch.tensor([], dtype=K.PACKS[_base(p)]).element_size() for p in used) < G.PACKMAX:
            break
    pdt = {p: K.PACKS[_base(p)] for p in used if p in K.PACKS or used[p]}
    es = {p: torch.tensor([], dtype=dt).element_size() for p, dt in pdt.items()}
    sizes = {p: max(1, (used[p] * es[p] + K.ALIGN - 1) // K.ALIGN * K.ALIGN // es[p]) for p in pdt}
    pk = {p: torch.zeros(sizes[p], dtype=torch.uint8 if es[p] == 1 else dt) for p, dt in pdt.items()}
    for key, (p, off, shape) in spec.items():
        n = key.split(".", 1)[1]
        tns = args[int(key.split(".")[0][1:])][n].reshape(-1)
        tns = tns.view(torch.uint8) if es[p] == 1 else tns
        pk[p][off : off + tns.numel()] = tns
    pk = {p: (pk[p].view(dt) if es[p] == 1 else pk[p]) for p, dt in pdt.items()}

    lines = []
    for key, (p, off, shape) in spec.items():
        j = int(key.split(".")[0][1:])
        n = key.split(".", 1)[1]
        numel = int(np.prod(shape))
        w = n if n.startswith("w_") else "w_" + n[2:]
        blk = ""
        if w in K.BW and (j, w) in G.FULL and not n.startswith("t_"):
            gg = 1 if n == w else K.SG
            cols, bw = G.FULL[(j, w)] // gg, K.BW[w] // gg
            if _base(p) == "p4":
                cols, bw = cols // 2, bw // 2
            blk = f" blk {numel // cols} {cols} {bw}"
        lines.append(f"{key} {p} {off * es[p]} {numel * es[p]}{blk}")
    bundle = GroupBundle(args, raw, kinds, pk, dict(spec), sizes, pdt, dict(G.FULL), lines)
    bundle.activate()
    return bundle


def compare_layout(path: str, lines):
    """Return (identical, first difference) against an existing .layout's member lines."""
    old = []
    for line in open(path):
        if line.startswith(("p4", "pw", "ps", "pf", "pb", "pe")):
            continue
        old.append(line.rstrip("\n"))
    return old == lines, next(((a, b) for a, b in zip(old, lines) if a != b), None)


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

