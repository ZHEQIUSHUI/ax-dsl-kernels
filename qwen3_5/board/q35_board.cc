// On-board Qwen3.5-0.8B decode: 24 layer calls per token on two shared axmodels (GDN layer, gated attention
// layer), weights as runtime inputs loaded once per layer. GDN state/conv state ping-pong between input and
// output buffers; the KV cache of each attention layer is a resident buffer bound to the kc/vc outputs.
// Tokens are teacher-forced from x.bin; every layer output is written to layer_out.bin [T, L, H] (bf16) and
// the per-step latency is printed.
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct Buf {
    AX_U64 phy = 0;
    void* vir = nullptr;
    uint32_t size = 0;
};

static Buf alloc(uint32_t size) {
    Buf b;
    b.size = size;
    if (AX_SYS_MemAlloc(&b.phy, &b.vir, size, 128, (const AX_S8*)"q35") != 0) {
        fprintf(stderr, "CMM alloc %u failed\n", size);
        exit(6);
    }
    memset(b.vir, 0, size);
    return b;
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
    auto d = read_file(path);
    if (d.size() != b.size) {
        fprintf(stderr, "%s: %zu bytes, tensor wants %u\n", path.c_str(), d.size(), b.size);
        exit(7);
    }
    memcpy(b.vir, d.data(), d.size());
}

struct Model {
    AX_ENGINE_HANDLE handle;
    AX_ENGINE_IO_INFO_T* info;
};

static Model load_model(const char* path) {
    Model m;
    auto data = read_file(path);
    if (AX_ENGINE_CreateHandle(&m.handle, data.data(), data.size()) != 0) exit(3);
    if (AX_ENGINE_CreateContext(m.handle) != 0) exit(4);
    if (AX_ENGINE_GetIOInfo(m.handle, &m.info) != 0) exit(5);
    return m;
}

static AX_ENGINE_IO_BUFFER_T io_buf(const Buf& b) {
    AX_ENGINE_IO_BUFFER_T x;
    memset(&x, 0, sizeof(x));
    x.phyAddr = b.phy;
    x.pVirAddr = b.vir;
    x.nSize = b.size;
    return x;
}

static uint16_t bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s data_dir gdn.axmodel attn.axmodel\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    int T, LMAX, BC, H;
    {
        std::ifstream f(dir + "/meta.txt");
        f >> T >> LMAX >> BC >> H;
    }
    std::vector<std::string> types;
    {
        std::ifstream f(dir + "/layers.txt");
        for (std::string s; f >> s;) types.push_back(s);
    }
    const int L = types.size();

    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;
    Model gdn = load_model(argv[2]), attn = load_model(argv[3]);
    for (Model* m : {&gdn, &attn}) {
        for (uint32_t i = 0; i < m->info->nInputSize; ++i)
            printf("  in  %-12s %u\n", m->info->pInputs[i].pName, m->info->pInputs[i].nSize);
        for (uint32_t i = 0; i < m->info->nOutputSize; ++i)
            printf("  out %-12s %u\n", m->info->pOutputs[i].pName, m->info->pOutputs[i].nSize);
    }
    fflush(stdout);

    // 所有层共用的逐步输入：x、out、pos/nblk/mask/cos/sin
    auto size_of = [](AX_ENGINE_IO_INFO_T* info, const char* name, bool input) -> uint32_t {
        uint32_t n = input ? info->nInputSize : info->nOutputSize;
        auto* m = input ? info->pInputs : info->pOutputs;
        for (uint32_t i = 0; i < n; ++i)
            if (strcmp(m[i].pName, name) == 0) return m[i].nSize;
        fprintf(stderr, "no tensor %s\n", name);
        exit(8);
    };
    Buf X = alloc(size_of(gdn.info, "x", true)), OUT = alloc(size_of(gdn.info, "out", false));
    Buf POS = alloc(size_of(attn.info, "pos", true)), NBLK = alloc(size_of(attn.info, "nblk", true)), MASK = alloc(size_of(attn.info, "mask", true));
    Buf COS = alloc(size_of(attn.info, "cos", true)), SIN = alloc(size_of(attn.info, "sin", true));
    std::map<std::string, Buf> shared = {{"x", X}, {"out", OUT}, {"pos", POS}, {"nblk", NBLK},
                                         {"mask", MASK}, {"cos", COS}, {"sin", SIN}};

    struct Layer {
        Model* m;
        AX_ENGINE_IO_T io;
        int si = -1, so = -1, ci = -1, co = -1;  // GDN 状态的输入/输出下标，每步交换
    };
    std::vector<Layer> layers(L);
    size_t weight_bytes = 0;
    for (int l = 0; l < L; ++l) {
        Layer& ly = layers[l];
        ly.m = types[l] == "gdn" ? &gdn : &attn;
        auto* info = ly.m->info;
        memset(&ly.io, 0, sizeof(ly.io));
        ly.io.nInputSize = info->nInputSize;
        ly.io.nOutputSize = info->nOutputSize;
        ly.io.pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize]();
        ly.io.pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize]();
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
    printf("layers=%d weights=%.1f MB T=%d LMAX=%d BC=%d\n", L, weight_bytes / 1e6, T, LMAX, BC);

    auto xs = read_file(dir + "/x.bin");
    auto cs = read_file(dir + "/cos.bin");
    auto ss = read_file(dir + "/sin.bin");
    std::vector<uint16_t> layer_out((size_t)T * L * H);
    std::vector<double> step_ms;
    for (int t = 0; t < T; ++t) {
        memcpy(X.vir, xs.data() + (size_t)t * X.size, X.size);
        memcpy(COS.vir, cs.data() + (size_t)t * COS.size, COS.size);
        memcpy(SIN.vir, ss.data() + (size_t)t * SIN.size, SIN.size);
        *(int32_t*)POS.vir = t;
        *(int32_t*)NBLK.vir = t / BC + 1;
        auto* m = (uint16_t*)MASK.vir;
        for (int j = 0; j < LMAX; ++j) m[j] = j <= t ? 0 : bf16(-3e38f);

        auto t0 = std::chrono::steady_clock::now();
        for (int l = 0; l < L; ++l) {
            Layer& ly = layers[l];
            int ret = AX_ENGINE_RunSync(ly.m->handle, &ly.io);
            if (ret != 0) {
                fprintf(stderr, "RunSync failed 0x%x step %d layer %d\n", ret, t, l);
                return 9;
            }
            memcpy(&layer_out[((size_t)t * L + l) * H], OUT.vir, H * 2);
            memcpy(X.vir, OUT.vir, H * 2);
            if (ly.si >= 0) {
                std::swap(ly.io.pInputs[ly.si], ly.io.pOutputs[ly.so]);
                std::swap(ly.io.pInputs[ly.ci], ly.io.pOutputs[ly.co]);
            }
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        step_ms.push_back(ms);
        printf("step %2d: %.3f ms (24 layers)\n", t, ms);
    }
    std::ofstream(dir + "/layer_out.bin", std::ios::binary)
        .write((const char*)layer_out.data(), layer_out.size() * 2);
    double sum = 0;
    for (size_t i = 1; i < step_ms.size(); ++i) sum += step_ms[i];
    printf("RESULT steps=%d avg_step_ms(excl. first)=%.3f\n", T, sum / (T - 1));
    AX_ENGINE_DestroyHandle(gdn.handle);
    AX_ENGINE_DestroyHandle(attn.handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
