"""Compare the on-board Qwen3.5-0.8B decode (q35_board → layer_out.bin [T, L, H] bf16) with the simulator run and the
transformers reference stored in q35_step.py's result.pt; logits are computed on the host (final norm + tied embedding).

Usage: python q35_board_cmp.py out_q35_step2/result.pt board_q35/layer_out.bin
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "18"))


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def main():
    res = torch.load(sys.argv[1])
    sim_hs, ref_hs, ref_logits, tokens = res["npu_hs"], res["ref_hs"], res["ref_logits"], res["tokens"]
    L, T, H = sim_hs.shape
    raw = np.fromfile(sys.argv[2], dtype=np.uint16).reshape(T, L, H)
    board = torch.from_numpy(raw.astype(np.int32) << 16).view(torch.float32).permute(1, 0, 2)  # [L, T, H]

    f = safe_open(f"{MODEL}/model.safetensors-00001-of-00001.safetensors", "pt")
    embed = f.get_tensor("model.language_model.embed_tokens.weight").float()
    norm = 1 + f.get_tensor("model.language_model.norm.weight").float()

    def final_norm(h):
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * norm

    # HF hidden_states 的最后一项已过 final norm，比较前给板端输出也加上
    board_vs_ref = board.clone()
    board_vs_ref[-1] = final_norm(board[-1])
    layers = json.load(open(f"{MODEL}/config.json"))["text_config"]["layer_types"]
    worst_sim = worst_ref = 1.0
    for i in range(L):
        cs = min(cos(board[i, t], sim_hs[i, t]) for t in range(T))
        cr = min(cos(board_vs_ref[i, t], ref_hs[i, t]) for t in range(T))
        worst_sim, worst_ref = min(worst_sim, cs), min(worst_ref, cr)
        print(f"layer {i:2d} {layers[i]:17s} board~sim {cs:.6f}  board~HF {cr:.6f}")

    logits = final_norm(board[-1]) @ embed.T
    sim_logits = res["npu_logits"]
    lc_ref = min(cos(logits[t], ref_logits[t]) for t in range(T))
    lc_sim = min(cos(logits[t], sim_logits[t]) for t in range(T))
    top_ref = sum(int(logits[t].argmax() == ref_logits[t].argmax()) for t in range(T))
    top_sim = sum(int(logits[t].argmax() == sim_logits[t].argmax()) for t in range(T))
    tok = AutoTokenizer.from_pretrained(MODEL)
    gen = [int(logits[t].argmax()) for t in range(PROMPT_LEN - 1, T)]
    print(f"board text: {tok.decode(gen)!r}")
    print(f"HF    text: {tok.decode([int(ref_logits[t].argmax()) for t in range(PROMPT_LEN - 1, T)])!r}")
    print(f"RESULT layer_min_cos board~sim={worst_sim:.6f} board~HF={worst_ref:.6f} "
          f"logits_min_cos board~HF={lc_ref:.6f} board~sim={lc_sim:.6f} top1 board~HF={top_ref}/{T} board~sim={top_sim}/{T}")


if __name__ == "__main__":
    main()
