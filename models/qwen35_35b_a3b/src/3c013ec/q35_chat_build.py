"""Build everything the on-board Qwen3.5-0.8B generator (q35_chat) needs, besides the per-layer weights.

- attn_layer recompiled for a longer context (LMAX/BC from env, default 1024/64); gdn_layer is reused as is.
- lm_head kernel: final RMSNorm + one vocab chunk of the tied embedding (VC rows), called V/VC times per token.
  Checked on the simulator against torch for chunk 0.
- Data (into <out>): embed.bin [V, H] bf16 (embedding lookup and lm_head weight), final_norm.bin fp32 (1 + w),
  vocab.bin (per id: u32 length + raw bytes), meta.txt, eos.txt, and the axmodels (lm_head_npu1.axmodel,
  attn_npu1.axmodel; gdn_npu1.axmodel is q35_step's).

Usage: LMAX=1024 BC=64 python q35_chat_build.py board_chat   (layer weights: link L*/ and layers.txt from board_q35)
"""
import glob
import os
import shutil
import sys

os.environ.setdefault("LMAX", "1024")
os.environ.setdefault("BC", "64")

import torch
from loguru import logger
from transformers import AutoTokenizer

import axnn.dsl as ax
from axnn.dsl.kernels.utils import rmsnorm
from axnn.dsl.kernels.utils.linear import linear
import board_io
import q35_step as q

VC = int(os.environ.get("VC", 24832))  # 248320 = 10 x 24832，每块 97 个 256 行的权重 tile
OUT = None  # set from the command line in main()
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_chat"))


def lm_head(x, norm, w, logits):
    h = rmsnorm(ax.dsl.load(x), ax.dsl.load(norm).reshape(1, 1, q.H), eps=q.EPS)
    lg = linear(h, w, None, "lm_head", output_dtype=ax.bf16, oc_tile_size=256)
    ax.dsl.store(lg, r=logits)
    return logits


def bytes_to_unicode():
    """GPT-2 byte-level BPE byte -> printable unicode map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def axmodel(name):
    (path,) = glob.glob(f"{BUILD}/cache/{name}.*.compiled.axmodel")
    return path


def main():
    global OUT
    OUT = os.path.abspath(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    os.chdir(BUILD)
    ax.init(chip_num=1, core_num=1)
    W = q.Weights(q.MODEL)
    emb = W("embed_tokens.weight")
    V = emb.shape[0]
    assert V % VC == 0, (V, VC)
    norm = q.one_plus(W("norm.weight"))

    # lm_head：模拟器跑第 0 块，对比 torch
    x = emb[1000:1001].reshape(1, 1, q.H).contiguous()
    k_head = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/lm_head")(lm_head)
    logits = torch.zeros((1, 1, VC), dtype=torch.bfloat16)
    k_head[1](x, norm, emb[:VC].contiguous(), logits)
    xf = x.float().reshape(1, q.H)
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + q.EPS) * norm) @ emb[:VC].float().T
    logger.info(f"lm_head chunk0 cos={cos(logits.float(), ref):.6f} argmax npu={int(logits.argmax())} "
                f"ref={int(ref.argmax())}")

    # 注意力层按新的 LMAX/BC 重新编译：用第 3 层权重在 pos=0 跑一次
    a = q.attn_args(W, 3)
    c, s = q.rope_cos_sin(0)
    mask = torch.full((q.LMAX,), -3e38, dtype=torch.bfloat16)
    mask[0] = 0
    case = dict(a, x=x, cos=c, sin=s,
                kc=torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16),
                vc=torch.zeros((q.HKV, q.LMAX, q.D), dtype=torch.bfloat16),
                pos=torch.tensor([0], dtype=torch.int32), nblk=torch.tensor([1], dtype=torch.int32),
                mask=mask.reshape(q.LMAX // q.BC, q.BC), out=torch.zeros((1, 1, q.H), dtype=torch.bfloat16))
    k_attn = ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/attn")(q.attn_layer)
    k_attn[1](*[case[n] for n in q.ATTN_ORDER])
    logger.info(f"attn_layer LMAX={q.LMAX} BC={q.BC} compiled, out finite={bool(case['out'].float().isfinite().all())}")

    shutil.copy(axmodel("lm_head"), board_io.kernel_file(OUT, "lm_head"))
    shutil.copy(axmodel("attn_layer"), board_io.kernel_file(OUT, "attn"))

    emb.contiguous().view(torch.int16).numpy().tofile(f"{OUT}/embed.bin")
    norm.numpy().tofile(f"{OUT}/final_norm.bin")

    # 词表解码表：byte-level BPE 的 token 串还原成原始字节，特殊 token 保留原文
    tok = AutoTokenizer.from_pretrained(q.MODEL)
    byte_decoder = {u: b for b, u in bytes_to_unicode().items()}
    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
    with open(f"{OUT}/vocab.bin", "wb") as f:
        n_tok = len(tok)
        names = tok.convert_ids_to_tokens(list(range(n_tok)))
        for i in range(V):
            t = names[i] if i < n_tok else None
            if t is None:
                b = b""
            elif i in special:
                b = t.encode()
            else:
                try:
                    b = bytes(byte_decoder[ch] for ch in t)
                except KeyError:
                    b = tok.decode([i]).encode()
            f.write(len(b).to_bytes(4, "little") + b)
    eos = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")})
    open(f"{OUT}/eos.txt", "w").write(" ".join(map(str, eos)) + "\n")
    # meta: LMAX BC H V VC ROT THETA
    open(f"{OUT}/meta.txt", "w").write(f"{q.LMAX} {q.BC} {q.H} {V} {VC} {q.ROT} {q.THETA}\n")
    logger.info(f"wrote {OUT}: V={V} VC={VC} eos={eos}")


if __name__ == "__main__":
    main()
