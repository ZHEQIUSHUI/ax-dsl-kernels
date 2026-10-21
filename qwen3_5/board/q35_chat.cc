// On-board Qwen3.5-0.8B greedy generation, every layer and the lm_head on the NPU.
// Per token: embedding lookup (CPU, from the CMM copy of the tied embedding) -> 24 layer calls on two shared
// axmodels (GDN layer / gated attention layer, weights as runtime inputs) -> V/VC lm_head calls whose weights are
// slices of the same embedding buffer -> argmax (CPU) -> decode bytes from vocab.bin.
// The prompt arrives as token ids (prompt.ids, int32; tokenizer encoding stays on the host for now).
// The prompt is fed token by token through the decode kernels; prefill and decode times are reported apart.
// Writes <out_ids> = prompt + generated ids (int32) for checking against transformers.
// The lm_head may be bf16 (weights = the embedding buffer) or MX (lm_w.bin / lm_s.bin, when the model has an "s" input).
// max_new_tokens = 0 scores the ids instead: per-position NLL of the next token -> <out_ids> as float32, plus perplexity.
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct Buf {
    AX_U64 phy = 0;
    void* vir = nullptr;
    uint32_t size = 0;
};

static Buf alloc(uint64_t size, bool cached = false) {
    Buf b;
    b.size = size;
    auto fn = cached ? AX_SYS_MemAllocCached : AX_SYS_MemAlloc;
    if (fn(&b.phy, &b.vir, size, 128, (const AX_S8*)"q35") != 0) {
        fprintf(stderr, "CMM alloc %lu failed\n", (unsigned long)size);
        exit(6);
    }
    memset(b.vir, 0, size);
    return b;
}

static Buf slice(const Buf& b, uint64_t off, uint32_t size) {
    Buf s;
    s.phy = b.phy + off;
    s.vir = (char*)b.vir + off;
    s.size = size;
    return s;
}

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing %s\n", path.c_str());
        exit(7);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void load_into(const std::string& path, Buf& b) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "missing %s\n", path.c_str());
        exit(7);
    }
    if ((uint64_t)f.tellg() != b.size) {
        fprintf(stderr, "%s: %ld bytes, tensor wants %u\n", path.c_str(), (long)f.tellg(), b.size);
        exit(7);
    }
    f.seekg(0);
    f.read((char*)b.vir, b.size);
}

struct Model {
    AX_ENGINE_HANDLE handle;
    AX_ENGINE_IO_INFO_T* info;
};

static Model load_model(const std::string& path) {
    Model m;
    auto data = read_file(path);
    if (AX_ENGINE_CreateHandle(&m.handle, data.data(), data.size()) != 0) exit(3);
    if (AX_ENGINE_CreateContext(m.handle) != 0) exit(4);
    if (AX_ENGINE_GetIOInfo(m.handle, &m.info) != 0) exit(5);
    return m;
}

static uint32_t size_of(const Model& m, const char* name, bool input) {
    uint32_t n = input ? m.info->nInputSize : m.info->nOutputSize;
    auto* t = input ? m.info->pInputs : m.info->pOutputs;
    for (uint32_t i = 0; i < n; ++i)
        if (strcmp(t[i].pName, name) == 0) return t[i].nSize;
    fprintf(stderr, "no tensor %s\n", name);
    exit(8);
}

static AX_ENGINE_IO_BUFFER_T io_buf(const Buf& b) {
    AX_ENGINE_IO_BUFFER_T x;
    memset(&x, 0, sizeof(x));
    x.phyAddr = b.phy;
    x.pVirAddr = b.vir;
    x.nSize = b.size;
    return x;
}

static AX_ENGINE_IO_T make_io(const Model& m) {
    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.nInputSize = m.info->nInputSize;
    io.nOutputSize = m.info->nOutputSize;
    io.pInputs = new AX_ENGINE_IO_BUFFER_T[io.nInputSize]();
    io.pOutputs = new AX_ENGINE_IO_BUFFER_T[io.nOutputSize]();
    return io;
}

static uint16_t bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);  // round to nearest even，和 torch 一致
}

static float from_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s data_dir prompt.ids max_new_tokens out.ids timing.txt\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const int max_new = atoi(argv[3]);
    int LMAX, BC, H, V, VC, ROT;
    double THETA;
    {
        std::ifstream f(dir + "/meta.txt");
        f >> LMAX >> BC >> H >> V >> VC >> ROT >> THETA;
    }
    std::set<int> eos;
    {
        std::ifstream f(dir + "/eos.txt");
        for (int e; f >> e;) eos.insert(e);
    }
    std::vector<std::string> types;
    {
        std::ifstream f(dir + "/layers.txt");
        for (std::string s; f >> s;) types.push_back(s);
    }
    std::vector<std::string> vocab(V);
    {
        auto d = read_file(dir + "/vocab.bin");
        size_t p = 0;
        for (int i = 0; i < V; ++i) {
            uint32_t n;
            memcpy(&n, d.data() + p, 4);
            vocab[i].assign(d.data() + p + 4, n);
            p += 4 + n;
        }
    }
    std::vector<int32_t> tokens;
    {
        auto d = read_file(argv[2]);
        tokens.resize(d.size() / 4);
        memcpy(tokens.data(), d.data(), d.size());
    }
    const int n_prompt = tokens.size();
    const bool score = max_new == 0;
    if (n_prompt + max_new > LMAX) {
        fprintf(stderr, "prompt %d + new %d exceeds LMAX %d\n", n_prompt, max_new, LMAX);
        return 1;
    }
    const int L = types.size();

    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;
    Model gdn = load_model(dir + "/gdn_npu1.axmodel"), attn = load_model(dir + "/attn_npu1.axmodel");
    Model head = load_model(dir + "/lm_head_npu1.axmodel");

    auto t_load = std::chrono::steady_clock::now();
    Buf X = alloc(size_of(gdn, "x", true)), OUT = alloc(size_of(gdn, "out", false));
    Buf POS = alloc(size_of(attn, "pos", true)), NBLK = alloc(size_of(attn, "nblk", true));
    Buf MASK = alloc(size_of(attn, "mask", true));
    Buf COS = alloc(size_of(attn, "cos", true)), SIN = alloc(size_of(attn, "sin", true));
    std::map<std::string, Buf> shared = {{"x", X}, {"out", OUT}, {"pos", POS}, {"nblk", NBLK},
                                         {"mask", MASK}, {"cos", COS}, {"sin", SIN}};

    struct Layer {
        Model* m;
        AX_ENGINE_IO_T io;
        int si = -1, so = -1, ci = -1, co = -1;  // GDN 状态的输入/输出下标，每步交换
    };
    std::vector<Layer> layers(L);
    uint64_t weight_bytes = 0;
    for (int l = 0; l < L; ++l) {
        Layer& ly = layers[l];
        ly.m = types[l] == "gdn" ? &gdn : &attn;
        auto* info = ly.m->info;
        ly.io = make_io(*ly.m);
        char ldir[16];
        snprintf(ldir, sizeof(ldir), "/L%02d/", l);
        for (uint32_t i = 0; i < info->nInputSize; ++i) {
            std::string n = info->pInputs[i].pName;
            if (shared.count(n)) {
                ly.io.pInputs[i] = io_buf(shared[n]);
                continue;
            }
            Buf b = alloc(info->pInputs[i].nSize);
            if (n == "state_in") {
                ly.si = i;
            } else if (n == "conv_in") {
                ly.ci = i;
            } else {
                load_into(dir + ldir + n + ".bin", b);
                weight_bytes += b.size;
            }
            ly.io.pInputs[i] = io_buf(b);
        }
        for (uint32_t i = 0; i < info->nOutputSize; ++i) {
            std::string n = info->pOutputs[i].pName;
            if (shared.count(n)) {
                ly.io.pOutputs[i] = io_buf(shared[n]);
                continue;
            }
            if (n == "state_out") ly.so = i;
            if (n == "conv_out") ly.co = i;
            ly.io.pOutputs[i] = io_buf(alloc(info->pOutputs[i].nSize));  // kc/vc：常驻 cache，初值 0
        }
    }

    // lm_head：bf16 时每块的权重是 embedding 的一段；MX 时权重和 scale 单独加载。输出写进整块 logits 的对应位置
    Buf EMB = alloc((uint64_t)V * H * 2), NORM = alloc(size_of(head, "norm", true));
    bool head_mx = false;
    for (uint32_t i = 0; i < head.info->nInputSize; ++i) head_mx |= strcmp(head.info->pInputs[i].pName, "s") == 0;
    const uint32_t w_chunk = size_of(head, "w", true), s_chunk = head_mx ? size_of(head, "s", true) : 0;
    Buf LMW = EMB, LMS;
    if (head_mx) {
        LMW = alloc((uint64_t)w_chunk * (V / VC));
        LMS = alloc((uint64_t)s_chunk * (V / VC));
        load_into(dir + "/lm_w.bin", LMW);
        load_into(dir + "/lm_s.bin", LMS);
    }
    // logits 用 cached 内存：CPU 逐个读 non-cacheable 的 CMM 很慢，NPU 写完后先 invalidate 再读
    Buf LOGITS = alloc((uint64_t)V * 2, true);
    AX_SYS_MflushCache(LOGITS.phy, LOGITS.vir, LOGITS.size);
    load_into(dir + "/embed.bin", EMB);
    load_into(dir + "/final_norm.bin", NORM);
    const int n_chunk = V / VC;
    std::vector<AX_ENGINE_IO_T> head_io(n_chunk);
    for (int c = 0; c < n_chunk; ++c) {
        head_io[c] = make_io(head);
        for (uint32_t i = 0; i < head.info->nInputSize; ++i) {
            std::string n = head.info->pInputs[i].pName;
            Buf b = n == "x"      ? OUT
                    : n == "norm" ? NORM
                    : n == "s"    ? slice(LMS, (uint64_t)c * s_chunk, s_chunk)
                                  : slice(LMW, (uint64_t)c * w_chunk, w_chunk);
            head_io[c].pInputs[i] = io_buf(b);
        }
        head_io[c].pOutputs[0] = io_buf(slice(LOGITS, (uint64_t)c * VC * 2, VC * 2));
    }
    printf("loaded: layers=%d weights=%.1f MB + embedding %.1f MB, LMAX=%d BC=%d, lm_head %s %d x %d rows (%.1f MB), %.1f s\n",
           L, weight_bytes / 1e6, (double)V * H * 2 / 1e6, LMAX, BC, head_mx ? "MX" : "bf16", n_chunk, VC,
           (double)(w_chunk + s_chunk) * n_chunk / 1e6, ms_since(t_load) / 1e3);

    std::vector<float> inv(ROT / 2);
    for (int i = 0; i < ROT / 2; ++i) inv[i] = (float)(1.0 / std::pow(THETA, (double)(2 * i) / ROT));
    {
        auto* m = (uint16_t*)MASK.vir;
        for (int j = 0; j < LMAX; ++j) m[j] = bf16(-3e38f);
    }

    double prefill_ms = 0, decode_layers_ms = 0, decode_head_ms = 0, decode_argmax_ms = 0, decode_ms = 0;
    int n_decode = 0;
    std::vector<float> nll;
    auto run = [](Model* m, AX_ENGINE_IO_T* io, const char* what, int pos) {
        int ret = AX_ENGINE_RunSync(m->handle, io);
        if (ret != 0) {
            fprintf(stderr, "RunSync failed 0x%x (%s, pos %d)\n", ret, what, pos);
            exit(9);
        }
    };
    printf("---\n");
    fflush(stdout);
    for (int t = 0; t + 1 < n_prompt + max_new; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        memcpy(X.vir, (char*)EMB.vir + (size_t)tokens[t] * H * 2, H * 2);
        auto* c = (uint16_t*)COS.vir;
        auto* s = (uint16_t*)SIN.vir;
        for (int i = 0; i < ROT / 2; ++i) {
            float f = (float)t * inv[i];
            c[i] = bf16(std::cos(f));
            s[i] = bf16(std::sin(f));
        }
        *(int32_t*)POS.vir = t;
        *(int32_t*)NBLK.vir = t / BC + 1;
        ((uint16_t*)MASK.vir)[t] = 0;

        for (int l = 0; l < L; ++l) {
            Layer& ly = layers[l];
            run(ly.m, &ly.io, "layer", t);
            if (l + 1 < L) memcpy(X.vir, OUT.vir, H * 2);
            if (ly.si >= 0) {
                std::swap(ly.io.pInputs[ly.si], ly.io.pOutputs[ly.so]);
                std::swap(ly.io.pInputs[ly.ci], ly.io.pOutputs[ly.co]);
            }
        }
        if (t + 1 < n_prompt && !score) {  // prompt 中间的位置不需要 logits
            prefill_ms += ms_since(t0);
            continue;
        }
        double layers_ms = ms_since(t0);
        auto th = std::chrono::steady_clock::now();
        for (int k = 0; k < n_chunk; ++k) run(&head, &head_io[k], "lm_head", t);
        double head_ms = ms_since(th);
        auto ta = std::chrono::steady_clock::now();
        AX_SYS_MinvalidateCache(LOGITS.phy, LOGITS.vir, LOGITS.size);
        const auto* lg = (const uint16_t*)LOGITS.vir;
        int best = 0;
        float best_v = -INFINITY;
        for (int i = 0; i < V; ++i) {
            float v = from_bf16(lg[i]);
            if (v > best_v) best_v = v, best = i;
        }
        double argmax_ms = ms_since(ta);
        if (score) {  // 打分：下一个 token 的负对数似然
            if (t + 1 >= n_prompt) break;
            double sum = 0;
            for (int i = 0; i < V; ++i) sum += std::exp((double)from_bf16(lg[i]) - best_v);
            nll.push_back((float)(std::log(sum) + best_v - from_bf16(lg[tokens[t + 1]])));
            if (t % 100 == 0) fprintf(stderr, "scored %d/%d\n", t + 1, n_prompt - 1);
            continue;
        }
        if (t + 1 == n_prompt) {
            prefill_ms += ms_since(t0);  // 第一个生成 token 算在首 token 延迟里
        } else {
            decode_layers_ms += layers_ms;
            decode_head_ms += head_ms;
            decode_argmax_ms += argmax_ms;
            decode_ms += ms_since(t0);
            ++n_decode;
        }
        tokens.push_back(best);
        fwrite(vocab[best].data(), 1, vocab[best].size(), stdout);
        fflush(stdout);
        if (eos.count(best)) break;
    }
    if (score) {
        double s = 0;
        for (float v : nll) s += v;
        std::ofstream(argv[4], std::ios::binary).write((const char*)nll.data(), nll.size() * 4);
        char line[256];
        snprintf(line, sizeof(line), "scored=%zu mean_nll=%.5f ppl=%.4f\n", nll.size(), s / nll.size(),
                 std::exp(s / nll.size()));
        printf("RESULT %s", line);
        std::ofstream(argv[5]) << line;
        return 0;
    }
    const int n_gen = tokens.size() - n_prompt;
    printf("\n---\n");
    std::ofstream(argv[4], std::ios::binary).write((const char*)tokens.data(), tokens.size() * 4);
    char line[512];
    snprintf(line, sizeof(line),
             "prompt_tokens=%d gen_tokens=%d first_token_ms=%.1f (prompt fed through decode kernels, %.2f ms/token) "
             "decode_ms_per_token=%.2f (layers %.2f + lm_head %.2f + argmax %.2f) decode_tok_per_s=%.1f\n",
             n_prompt, n_gen, prefill_ms, prefill_ms / n_prompt, decode_ms / n_decode, decode_layers_ms / n_decode,
             decode_head_ms / n_decode, decode_argmax_ms / n_decode, 1e3 * n_decode / decode_ms);
    printf("RESULT %s", line);
    std::ofstream(argv[5]) << line;
    AX_ENGINE_DestroyHandle(gdn.handle);
    AX_ENGINE_DestroyHandle(attn.handle);
    AX_ENGINE_DestroyHandle(head.handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
