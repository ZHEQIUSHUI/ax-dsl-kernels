"""Qwen3.5 vision encoder (ViT) kernels, single-core, every weight a runtime input: one set of axmodels per model
size and patch count N serves all layers. The shapes come from the Hugging Face vision_config (0.8B: 12 layers of 768
with 12 heads of 64; 2B / 4B: 24 x 1024, 16 x 64; 9B / 27B / 35B-A3B: 27 x 1152, 16 x 72).

  vit_embed<N>   x = px @ w_patch.T + b_patch + pos (px: the patches' 3 x 16 x 16 pixels, the two identical temporal
                 frames folded into w_patch = w0 + w1; pos: the learned position table interpolated by the host)
  vit_pre<N>     h = LayerNorm1(x); q, k, v = h @ w_qkv.T + b_qkv; RoPE on q and k (cos / sin of each patch's row and
                 column, from the host), q pre-scaled -> q [N, NH, HD], k / v [N, NH * HD]
  vit_attn<N>    softmax(q k^T + key mask) v per head, bidirectional: query blocks of QB rows unrolled, key blocks of
                 BC cells in a runtime loop (nblk blocks; kmask [N / BC, QB, BC], the key mask repeated for the QB
                 rows by the host, hides the padded patches of the last one)
  vit_post<N>    x1 = x + attn @ w_o.T + b_o; out = x1 + fc2(gelu_tanh(fc1(LayerNorm2(x1))))
  vit_merge<N>   LayerNorm (post_ln) per patch, 2 x 2 neighbours concatenated, fc0 -> GELU -> fc2 -> [N / 4, OUT] f32

The host orders the patches in merge order (each 2 x 2 group consecutive, as the HF processor and llama.cpp do) and
pads N to the kernel's patch count (whole key blocks); padded rows are computed and ignored.

CORES=<n> (default 1): every kernel on n cores without synchronization, none reading another core's results:
embed / pre / post / merge split by rows (N / CORES patches per core), the attention by heads (NH / CORES heads per
core over all the patches, whose q / k / v the pre kernel's call before wrote).

Written as qwen35_vit_<kernel><N>_h<H>_npu<CORES>.axmodel. Check on the simulator against transformers' Qwen3_5VisionModel,
layer by layer. Usage:
  MODEL=<hf dir> N=256 python q35_vit.py [kernel_dir]
"""
import glob
import json
import math
import os
import shutil
import sys

import ml_dtypes
import numpy as np
import torch
from loguru import logger
from safetensors import safe_open

import axnn.dsl as ax
from axnn.dsl.kernels.utils.activation import gelu_tanh
from axnn.dsl.kernels.utils.linear import linear
from axnn.dsl.kernels.utils.rope import rope
import board_io

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the Qwen3.5 Hugging Face model directory")
VC = json.load(open(f"{MODEL}/config.json"))["vision_config"]
H, NH, F, OUT = VC["hidden_size"], VC["num_heads"], VC["intermediate_size"], VC["out_hidden_size"]
HD = H // NH
PS, TPS, MERGE = VC["patch_size"], VC["temporal_patch_size"], VC["spatial_merge_size"]
CP = 3 * PS * PS  # pixels per patch (one temporal frame)
EPS = 1e-6
N = int(os.environ.get("N", 256))  # patches per call, a multiple of BC and MT
# attention blocks (simulator, N = 1024 / 2048): 512 query rows per block beat 128 / 256 / 1024 (0.8B -41 %, 9B -56 % with
# the key blocks below); key blocks of 256 cells where a block of K or V stays within 384 KB (H <= 768: a 9B block of
# 256 x 1152 does not build), else 128
BC = int(os.environ.get("BC", 256 if H <= 768 else 128))  # key block of the attention
QB = int(os.environ.get("QB", min(512, N)))  # query rows per block of the attention
CORES = int(os.environ.get("CORES", 1))
MT = int(os.environ.get("MT", min(128, N // CORES)))  # rows per tile of the projections
BUILD = os.path.abspath(os.environ.get("DUMP", "out_q35_vit"))
RC = N // CORES  # rows per core (embed, pre, post, merge)
NHC = NH // CORES  # heads per core (attention)
assert N % BC == 0 and N % QB == 0 and RC % MT == 0 and MT % (MERGE * MERGE) == 0 and NH % CORES == 0


def core_rows():
    """the row tiles of this core: slices of MT rows within its RC rows"""
    c = ax.program_id(0)
    return [slice(c * RC + t * MT, c * RC + (t + 1) * MT) for t in range(RC // MT)]


def oc_tile(cols):
    """Output-channel tile so one bf16 weight tile stays around 512 KB of OCM."""
    return max(16, min(256, (512 * 1024) // (cols * 2)))


def proj(x, w, b, name, output_dtype=ax.bf16):
    """x @ w.T + b (bias added in fp32)"""
    y = linear(x, w, None, name, output_dtype=ax.fp32, oc_tile_size=oc_tile(w.shape[1]))
    return ax.dsl.add(y, ax.dsl.load(b).reshape(1, 1, w.shape[0]), output_dtype=output_dtype)


def layernorm(x, w, b):
    """LayerNorm over the last dim in fp32, bf16 out"""
    n = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=x.shape[-1])
    mean = ax.dsl.div(ax.dsl.sum(x, dim=-1, keepdim=True, output_dtype=ax.fp32), n)
    xc = ax.dsl.sub(x, mean, output_dtype=ax.fp32)
    var = ax.dsl.div(ax.dsl.sum(ax.dsl.mul(xc, xc, output_dtype=ax.fp32), dim=-1, keepdim=True, output_dtype=ax.fp32), n)
    den = ax.dsl.sqrt(ax.dsl.add(var, ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=EPS)))
    y = ax.dsl.mul(ax.dsl.div(xc, den, output_dtype=ax.fp32), ax.dsl.load(w).reshape(1, 1, x.shape[-1]), output_dtype=ax.fp32)
    return ax.dsl.add(y, ax.dsl.load(b).reshape(1, 1, x.shape[-1]), output_dtype=ax.bf16)


def vit_embed(px, w_patch, b_patch, pos, out):
    for rows in core_rows():
        x = proj(ax.dsl.load(px[rows]).reshape(1, MT, CP), w_patch, b_patch, "vit_patch", output_dtype=ax.fp32)
        ax.dsl.store(ax.dsl.add(x, ax.dsl.load(pos[rows]).reshape(1, MT, H), output_dtype=ax.bf16).reshape(MT, H), r=out[rows])
    return out


def vit_pre(x, ln1_w, ln1_b, w_qkv, b_qkv, cos, sin, q_out, k_out, v_out):
    scale = ax.dsl.full(shape=(1,), dtype=ax.fp32, fill_value=HD**-0.5)
    for rows in core_rows():
        h = layernorm(ax.dsl.load(x[rows]).reshape(1, MT, H), ln1_w, ln1_b)
        cs, sn = ax.dsl.load(cos[rows]), ax.dsl.load(sin[rows])  # [MT, HD / 2]
        # w_qkv rows [q | k | v], each head-major; one projection each, so every result is contiguous
        part = lambda i, nm: proj(h, w_qkv[i * H : (i + 1) * H], b_qkv[i * H : (i + 1) * H], nm)  # noqa: E731
        qh = rope(part(0, "vit_q").reshape(MT, NH, HD), cs, sn)
        kh = rope(part(1, "vit_k").reshape(MT, NH, HD), cs, sn)
        ax.dsl.store(ax.dsl.mul(qh, scale, output_dtype=ax.bf16), r=q_out[rows])
        ax.dsl.store(kh, r=k_out[rows].reshape(MT, NH, HD))  # (OCM rows of HD = 72 are padded: no reshape)
        ax.dsl.store(part(2, "vit_v").reshape(MT, H), r=v_out[rows])
    return q_out, k_out, v_out


def vit_attn(qs, kc, vc, nblk, kmask, out):
    """bidirectional attention, online softmax over key blocks (as q35_flash.attn_fd_seg, one query block of QB rows
    and all heads per loop, both K and V blocks loaded once for all heads)"""
    kv_block_shape = ax.dsl.from_numpy(np.array([BC], dtype=np.uint32))
    mask_block_shape = ax.dsl.from_numpy(np.array([1], dtype=np.uint32))
    block_stride = ax.dsl.from_numpy(np.array([BC], dtype=np.int32))
    minus_1 = ax.dsl.from_numpy(np.full(shape=(1,), dtype=ml_dtypes.bfloat16, fill_value=-1))
    range_step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="range_step")
    range_start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="range_start")
    range_stop = ax.dsl.load(nblk)
    zero_col = ax.dsl.zeros(shape=(QB, 1), dtype=ax.bf16)
    h0 = ax.program_id(0) * NHC  # this core's heads h0 .. h0 + NHC - 1

    def body(index, kc, vc, kmask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, *heads):
        t0 = ax.dsl.cast(ax.dsl.mul(index, block_stride), dtype=ax.uint32)
        kb = ax.dsl.dynamic_ld(kc, t0, kv_block_shape, dim=(0,), output_shape=(BC, H))
        vb = ax.dsl.dynamic_ld(vc, t0, kv_block_shape, dim=(0,), output_shape=(BC, H))
        mq = ax.dsl.dynamic_ld(kmask, ax.dsl.cast(index, dtype=ax.uint32), mask_block_shape, dim=(0,),
                               output_shape=(1, QB, BC)).reshape(QB, BC)  # the key mask, one row per query row
        for j in range(NHC):
            hh = h0 + j
            qh, mi, li, oi = heads[4 * j : 4 * j + 4]
            kj = kb[:, hh * HD : (hh + 1) * HD]
            vj = vb[:, hh * HD : (hh + 1) * HD]
            m_s = ax.dsl.create_tensor(shape=(QB, 1 + BC + 1), dtype=ax.bf16)
            ax.dsl.move(r=m_s[:, :1], x=mi)
            ax.dsl.move(r=m_s[:, -1:], x=zero_col)
            sj = ax.dsl.matmul(qh, kj, transposed=True, output_dtype=ax.bf16)
            ax.dsl.add(sj, mq, output_dtype=ax.bf16, r=m_s[:, 1 : 1 + BC])
            m_next = ax.dsl.max(m_s[:, : 1 + BC], dim=-1, keepdim=True)
            m_s = ax.dsl.sub(m_s, m_next, r=m_s)
            pj = ax.dsl.exp(m_s[:, 1 : 1 + BC])
            p32 = ax.dsl.exp(m_s[:, 1 : 1 + BC], output_dtype=ax.fp32)
            lij = ax.dsl.sum(p32, dim=-1, output_dtype=ax.fp32)
            ratio = ax.dsl.exp(m_s[:, :1], output_dtype=ax.fp32).reshape(QB)
            ratio_li = ax.dsl.mul(li, ratio, output_dtype=ax.fp32)
            ax.dsl.add(ratio_li, lij, r=li, output_dtype=ax.fp32)
            oij = pj @ vj
            ratio_oi = ax.dsl.mul(ratio.reshape(QB, 1), oi, output_dtype=ax.fp32)
            ax.dsl.add(ratio_oi, oij, r=oi, output_dtype=ax.bf16)
            ax.dsl.mul(m_s[:, -1:], minus_1, r=mi)
        return (kc, vc, kmask, block_stride, kv_block_shape, mask_block_shape, zero_col, minus_1, *heads)

    for b in range(N // QB):
        rows = slice(b * QB, (b + 1) * QB)
        qa = ax.dsl.load(qs[rows])  # [QB, NH, HD]
        heads = []
        for j in range(NHC):
            hh = h0 + j
            qh = ax.dsl.create_tensor((QB, HD), ax.bf16, name=f"vit_q{b}_{j}")
            ax.dsl.move(x=qa[:, hh : hh + 1, :].reshape(QB, HD), r=qh)
            heads += [qh, ax.dsl.full(shape=(QB, 1), dtype=ax.bf16, fill_value=-3e38),
                      ax.dsl.zeros(shape=(QB,), dtype=ax.fp32), ax.dsl.zeros(shape=(QB, HD), dtype=ax.bf16)]
        index = ax.dsl.create_tensor((1,), ax.int32, name=f"vit_index{b}")
        ax.dsl.move(x=range_start, r=index)
        res = ax.dsl.range(index, range_start, range_stop, range_step, body, kc, vc, kmask, block_stride,
                           kv_block_shape, mask_block_shape, zero_col, minus_1, *heads)
        heads = res[8:]
        o = ax.dsl.create_tensor((QB, NHC * HD), ax.bf16, name=f"vit_o{b}")
        for j in range(NHC):
            _, _, li, oi = heads[4 * j : 4 * j + 4]
            ax.dsl.div(oi, li.reshape(QB, 1), r=o[:, j * HD : (j + 1) * HD])
        ax.dsl.store(o, r=out[rows, h0 * HD : (h0 + NHC) * HD])
    return out


def vit_post(x, attn, w_o, b_o, ln2_w, ln2_b, w_up, b_up, w_down, b_down, out):
    for rows in core_rows():
        xt = ax.dsl.load(x[rows]).reshape(1, MT, H)
        a = proj(ax.dsl.load(attn[rows]).reshape(1, MT, H), w_o, b_o, "vit_o", output_dtype=ax.fp32)
        x1 = ax.dsl.add(xt, a, output_dtype=ax.bf16)
        u = gelu_tanh(proj(layernorm(x1, ln2_w, ln2_b), w_up, b_up, "vit_up", output_dtype=ax.fp32), output_dtype=ax.bf16)
        d = proj(u, w_down, b_down, "vit_down", output_dtype=ax.fp32)
        ax.dsl.store(ax.dsl.add(x1, d, output_dtype=ax.bf16).reshape(MT, H), r=out[rows])
    return out


def vit_merge(x, ln_w, ln_b, w0, b0, w2, b2, out):
    G = MERGE * MERGE
    for rows in core_rows():
        h = layernorm(ax.dsl.load(x[rows]).reshape(1, MT, H), ln_w, ln_b).reshape(1, MT // G, G * H)
        u = ax.dsl.gelu(proj(h, w0, b0, "vit_mm0", output_dtype=ax.fp32), output_dtype=ax.bf16)
        y = proj(u, w2, b2, "vit_mm2", output_dtype=ax.fp32)
        ax.dsl.store(y.reshape(MT // G, OUT), r=out[rows.start // G : rows.stop // G])
    return out


EMBED_ORDER = ["px", "w_patch", "b_patch", "pos", "out"]
PRE_ORDER = ["x", "ln1_w", "ln1_b", "w_qkv", "b_qkv", "cos", "sin", "q_out", "k_out", "v_out"]
ATTN_ORDER = ["qs", "kc", "vc", "nblk", "kmask", "out"]
POST_ORDER = ["x", "attn", "w_o", "b_o", "ln2_w", "ln2_b", "w_up", "b_up", "w_down", "b_down", "out"]
MERGE_ORDER = ["x", "ln_w", "ln_b", "w0", "b0", "w2", "b2", "out"]


def load_visual():
    """transformers' vision tower with the checkpoint's model.visual.* weights (fp32)"""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = AutoConfig.from_pretrained(MODEL).vision_config
    cfg._attn_implementation = "eager"
    vm = Qwen3_5VisionModel(cfg).float().eval()
    sd = {}
    for f in glob.glob(f"{MODEL}/*.safetensors"):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual.") :]] = fh.get_tensor(k).float()
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return vm


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1).double(), b.reshape(-1).double(), dim=0).item()


def kern(fn, sub):
    os.makedirs(f"{BUILD}/{sub}", exist_ok=True)
    os.chdir(f"{BUILD}/{sub}")
    return ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True, dump_path=f"{BUILD}/{sub}")(fn)


def attn_random():
    """VIT_RANDOM=1: the attention kernel alone on random q / k / v (only the config's sizes, no weights), vs torch;
    written to the kernel directory if one is given"""
    torch.manual_seed(0)
    n = int(os.environ.get("NREAL", N))
    qf = (torch.randn((N, NH, HD)) * HD**-0.5).bfloat16()
    kf, vf = torch.randn((N, H)).bfloat16(), torch.randn((N, H)).bfloat16()
    kmask = torch.zeros((N,))
    kmask[n:] = -3e38
    t = dict(qs=qf, kc=kf, vc=vf, nblk=torch.tensor([-(-n // BC)], dtype=torch.int32),
             kmask=kmask.reshape(N // BC, 1, BC).expand(N // BC, QB, BC).bfloat16().contiguous(),
             out=torch.zeros((N, H), dtype=torch.bfloat16))
    kern(vit_attn, f"attn{N}")[CORES](*[t[a] for a in ATTN_ORDER])
    q32, k32, v32 = qf[:n].float(), kf[:n].float().view(n, NH, HD), vf[:n].float().view(n, NH, HD)
    att = torch.einsum("hqk,khd->qhd", torch.softmax(torch.einsum("qhd,khd->hqk", q32, k32), -1), v32).reshape(n, H)
    logger.info(f"vit attn N={N} H={H} QB={QB} BC={BC} (n={n}, random): cos vs torch {cos(t['out'][:n], att):.6f}")
    if len(sys.argv) > 1:  # the kernel alone, e.g. a new attention block size for existing other kernels
        (path,) = glob.glob(f"{BUILD}/attn{N}/cache/vit_attn.*.compiled.axmodel")
        dst = board_io.kernel_file(os.path.abspath(sys.argv[1]), f"qwen35_vit_attn{N}_h{H}", CORES)
        shutil.copy(path, dst)
        logger.info(f"wrote {dst}")


def main():
    out_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(BUILD, exist_ok=True)
    ax.init(chip_num=1, core_num=CORES)
    if os.environ.get("VIT_RANDOM") == "1":
        return attn_random()
    torch.manual_seed(0)
    vm = load_visual()
    gh, gw = int(os.environ.get("GH", 16)), int(os.environ.get("GW", N // int(os.environ.get("GH", 16))))
    n = gh * gw  # real patches (<= N)
    assert n <= N and gh % MERGE == 0 and gw % MERGE == 0
    grid = torch.tensor([[1, gh, gw]])
    pixels = torch.rand((n, 3 * TPS * PS * PS)) * 2 - 1  # normalized pixels, the two temporal frames ...
    pixels = pixels.view(n, 3, TPS, PS * PS)
    pixels[:, :, 1] = pixels[:, :, 0]  # ... identical, as for an image
    pixels = pixels.reshape(n, -1)
    rec = {}
    hooks = [vm.blocks[i].register_forward_hook(lambda m, a, o, i=i: rec.__setitem__(i, o.detach())) for i in range(len(vm.blocks))]
    hooks.append(vm.blocks[0].register_forward_pre_hook(lambda m, a, kw: rec.__setitem__("x0", a[0].detach()), with_kwargs=True))
    hooks.append(vm.blocks[0].register_forward_pre_hook(lambda m, a, kw: rec.__setitem__("pe", kw["position_embeddings"]), with_kwargs=True))
    with torch.no_grad():
        ref = vm(pixels, grid_thw=grid).pooler_output  # [n / 4, OUT]
    for h in hooks:
        h.remove()
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    pad = lambda t: torch.cat([t, torch.zeros((N - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)]) if t.shape[0] < N else t  # noqa: E731
    built = []

    def run(fn, sub, args, order):
        k = kern(fn, sub)
        k[CORES](*[args[a] for a in order])
        built.append((sub, fn.__name__))

    # embed: px (one frame) against w0 + w1
    w = vm.patch_embed.proj.weight.detach()  # [H, 3, TPS, PS, PS]
    w_patch = bf(w.sum(2).reshape(H, CP))
    px = bf(pad(pixels.view(n, 3, TPS, PS * PS)[:, :, 0].reshape(n, CP)))
    pos = rec["x0"] - torch.nn.functional.linear(pixels.view(n, 3, TPS, PS * PS)[:, :, 0].reshape(n, CP), w.sum(2).reshape(H, CP),
                                                 vm.patch_embed.proj.bias.detach())  # the interpolated position table
    a = dict(px=px, w_patch=w_patch, b_patch=bf(vm.patch_embed.proj.bias.detach()), pos=bf(pad(pos)), out=torch.zeros((N, H), dtype=torch.bfloat16))
    run(vit_embed, f"embed{N}", a, EMBED_ORDER)
    logger.info(f"vit_embed N={N}: x0 cos {cos(a['out'][:n], rec['x0']):.6f}")

    cs, sn = rec["pe"]  # [n, HD], halves equal
    cos_t, sin_t = bf(pad(cs[:, : HD // 2])), bf(pad(sn[:, : HD // 2]))
    kmask = torch.zeros((N,), dtype=torch.float32)
    kmask[n:] = -3e38
    kmask = bf(kmask.reshape(N // BC, 1, BC).expand(N // BC, QB, BC))
    nblk = torch.tensor([-(-n // BC)], dtype=torch.int32)
    # LAYERS=all CHAIN=1: every layer on the previous kernel output (from the embed kernel's), else on HF's input
    lv = os.environ.get("LAYERS", "0")
    layers = list(range(len(vm.blocks))) if lv == "all" else [int(i) for i in lv.split(",")]
    chain = os.environ.get("CHAIN") == "1"
    x_prev = a["out"]
    for i in layers:
        blk = vm.blocks[i]
        x_in = x_prev if chain else bf(pad(rec["x0"] if i == 0 else rec[i - 1]))
        g = lambda t: bf(t.detach())  # noqa: E731
        p = dict(x=x_in, ln1_w=g(blk.norm1.weight), ln1_b=g(blk.norm1.bias), w_qkv=g(blk.attn.qkv.weight), b_qkv=g(blk.attn.qkv.bias),
                 cos=cos_t, sin=sin_t, q_out=torch.zeros((N, NH, HD), dtype=torch.bfloat16),
                 k_out=torch.zeros((N, H), dtype=torch.bfloat16), v_out=torch.zeros((N, H), dtype=torch.bfloat16))
        run(vit_pre, f"pre{N}", p, PRE_ORDER)
        t = dict(qs=p["q_out"], kc=p["k_out"], vc=p["v_out"], nblk=nblk, kmask=kmask, out=torch.zeros((N, H), dtype=torch.bfloat16))
        run(vit_attn, f"attn{N}", t, ATTN_ORDER)
        if os.environ.get("ONLY") == "attn":
            qf, kf, vf = p["q_out"][:n].float(), p["k_out"][:n].float().view(n, NH, HD), p["v_out"][:n].float().view(n, NH, HD)
            att = torch.einsum("hqk,khd->qhd", torch.softmax(torch.einsum("qhd,khd->hqk", qf, kf), -1), vf).reshape(n, H)
            logger.info(f"vit attn N={N} QB={QB} BC={BC} (n={n}): cos vs torch {cos(t['out'][:n], att):.6f}")
            return
        o = dict(x=x_in, attn=t["out"], w_o=g(blk.attn.proj.weight), b_o=g(blk.attn.proj.bias), ln2_w=g(blk.norm2.weight),
                 ln2_b=g(blk.norm2.bias), w_up=g(blk.mlp.linear_fc1.weight), b_up=g(blk.mlp.linear_fc1.bias),
                 w_down=g(blk.mlp.linear_fc2.weight), b_down=g(blk.mlp.linear_fc2.bias), out=torch.zeros((N, H), dtype=torch.bfloat16))
        run(vit_post, f"post{N}", o, POST_ORDER)
        # torch reference of the attention alone from the kernel's own q / k / v
        qf, kf, vf = p["q_out"][:n].float(), p["k_out"][:n].float().view(n, NH, HD), p["v_out"][:n].float().view(n, NH, HD)
        att = torch.softmax(torch.einsum("qhd,khd->hqk", qf, kf), -1)
        att = torch.einsum("hqk,khd->qhd", att, vf).reshape(n, H)
        logger.info(f"vit layer {i} N={N} (n={n}): attn vs torch on its q/k/v cos {cos(t['out'][:n], att):.6f}, "
                    f"layer out vs HF cos {cos(o['out'][:n], rec[i]):.6f}")
        x_prev = o["out"]

    last = len(vm.blocks) - 1
    mg = vm.merger
    m = dict(x=x_prev if chain else bf(pad(rec[last])), ln_w=bf(mg.norm.weight.detach()), ln_b=bf(mg.norm.bias.detach()),
             w0=bf(mg.linear_fc1.weight.detach()), b0=bf(mg.linear_fc1.bias.detach()), w2=bf(mg.linear_fc2.weight.detach()),
             b2=bf(mg.linear_fc2.bias.detach()), out=torch.zeros((N // (MERGE * MERGE), OUT), dtype=torch.float32))
    run(vit_merge, f"merge{N}", m, MERGE_ORDER)
    logger.info(f"vit_merge N={N}: out vs HF cos {cos(m['out'][: n // (MERGE * MERGE)], ref):.6f}")

    if out_dir:
        tag = f"h{H}"
        for sub, fn in built:
            (path,) = glob.glob(f"{BUILD}/{sub}/cache/{fn}.*.compiled.axmodel")
            dst = board_io.kernel_file(out_dir, f"qwen35_{fn}{N}_{tag}", CORES)
            shutil.copy(path, dst)
            logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
