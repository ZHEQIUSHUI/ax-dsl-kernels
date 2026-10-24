"""Bitwise checks of the GDN prefill kernels with a valid-row count (PAD, q35_step.with_pad): one four-core packed
GDN prefill call (PCHUNK = P) from a zero state on layer 0's weights, NVROWS tokens (rows of L.tokens(W, 0)) and the
padding rows filled per FILL (next: the following tokens' rows, last: copies of the last token's, zero); saves the
tokens' outputs, the state and the conv state to <out>.pt. PV=0: the plain kernel (NVROWS = P).
Usage: PCHUNK=<P> NVROWS=<n> FILL=<next|last|zero> PV=<1|0> ... python pv_bit.py <out.pt>
       python pv_bit.py --cmp a.pt b.pt  (bitwise: out rows of both, state, conv)"""
import os
import sys

import torch

if sys.argv[1] == "--cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    n = min(a["out"].shape[1], b["out"].shape[1])
    res = {k: bool(torch.equal(a[k], b[k])) for k in ("state", "conv")}
    res["out"] = bool(torch.equal(a["out"][:, :n], b["out"][:, :n]))
    print(f"PVCMP {os.path.basename(sys.argv[2])} vs {os.path.basename(sys.argv[3])}: rows {n} " +
          " ".join(f"{k} {'SAME' if v else 'DIFF'}" for k, v in res.items()))
    sys.exit(0)

import q35_layer_pk as K
from q35_layer_pk import L, mb, q

PV = os.environ.get("PV", "1") == "1"
n = int(os.environ.get("NVROWS", L.P))
fill = os.environ.get("FILL", "next")


def main():
    W = q.Weights(q.MODEL)
    single = L.refs()[0]
    names = L.order(single)
    a = K.mx4_args(q.gdn_args(W, 0), mb.GDN_W) if K.MX4 else mb.mx_args(q.gdn_args(W, 0), mb.GDN_W)[0]
    if K.MX4:
        K.FULL_COLS["gdn"] = {k: a[k].shape[1] * (2 if a[k].dtype == torch.float4_e2m1fn_x2 else 1) for k in K.BW if k in a}
    else:
        K.FULL_COLS["gdn"] = {k: a[k].shape[1] for k in K.BW if k in a}
    ab = K.block_args(a)
    pk = K.packed(ab, "gdn", K.layout("gdn", ab, names))
    fn = K.gdn_layer_mx4_pk if K.MX4 else K.gdn_layer_mx_pk
    if PV:
        fn = q.with_pad(fn)
    x = L.tokens(W, 0)
    if fill == "last":
        x[:, n:] = x[:, n - 1 : n]
    elif fill == "zero":
        x[:, n:] = 0
    st = torch.zeros((L.NV, L.DK, L.DV))
    cv = torch.zeros((1, L.CONV - 1, 2 * L.KD + L.VD), dtype=torch.bfloat16)
    case = dict(pk, x=x, out=torch.zeros((1, L.P, L.H), dtype=torch.bfloat16), state_in=st, conv_in=cv,
                state_out=torch.zeros_like(st), conv_out=torch.zeros_like(cv))
    if PV:
        case.update(zip(("rv", "cix"), q.pad_inputs(L.P, n)))
    k = L.kern(fn, f"bit_{fn.__name__}")
    L.run(k, L.CORES, case, L.order(fn))
    torch.save({"out": case["out"][:, :n].clone(), "state": case["state_out"].clone(), "conv": case["conv_out"].clone()},
               sys.argv[1])


if __name__ == "__main__":
    main()
