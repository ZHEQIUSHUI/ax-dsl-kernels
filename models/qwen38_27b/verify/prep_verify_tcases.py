#!/usr/bin/env python3
"""Regenerate the T=1..4 verify x/cs/pn/mask inputs (packs unchanged)."""
import os, sys
from pathlib import Path
sys.path.insert(0, "/data/junda/work/q27m/src_ks")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k, v in dict(KVL="cell", BC="256", LMAX="16384", ROT="64", G="16", FIRST="0").items():
    os.environ.setdefault(k, v)
import torch
import q35_layer_mc as L
import q35_step as q
from board_common import dump_json, write_bin

POS0=int(os.environ.get("POS0","254")); SEED=int(os.environ.get("SEED","3")); T=int(os.environ.get("T","4"))
ROOT=Path(os.environ.get("ROOT","board/verify")).resolve()
NBLK,BC=L.NBLK,L.BC
torch.manual_seed(SEED)
xs=(torch.randn((1,T,L.H),dtype=torch.float32)*0.02).bfloat16()
cs_all=[]; pn_all=[]; mask_all=[]
for t in range(T):
    cell=POS0+t
    c,s=q.rope_cos_sin(cell)
    cs_all.append(torch.cat([c,s]).reshape(2,L.ROT//2))
    pn_all.append(torch.tensor([cell,cell//BC+1,(cell//BC)*2*BC+cell%BC],dtype=torch.int32))
    mask_all.append(L.mask_rows(cell).reshape(NBLK,BC))
cs_all=torch.stack(cs_all); pn_all=torch.stack(pn_all); mask_all=torch.stack(mask_all)
for tt in range(1,T+1):
    x=xs.clone(); x[:,tt:]=0
    # rows >= tt advance normally; x is zero so they cannot disturb the valid prefix
    d=ROOT/f"T{tt}"/"in"; d.mkdir(parents=True,exist_ok=True)
    sizes={n:write_bin(d/f"{n}.bin",t) for n,t in (("x",x),("cs",cs_all),("pn",pn_all),("mask",mask_all),("ntok",torch.tensor([tt],dtype=torch.int32)))}
    j=ROOT/f"T{tt}"/"io.json"
    import json
    old=json.load(open(j)) if j.exists() else {}
    old.update({"bytes":sizes,"pos0":POS0,"ntok":tt,"written_rows":[POS0+t for t in range(tt)]})
    dump_json(j,old)
    print("wrote",d)
