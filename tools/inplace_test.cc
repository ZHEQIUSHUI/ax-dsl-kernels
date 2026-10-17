// Can a recurrent-state kernel update its state in place? Runs the same kernel S steps twice from the same
// pseudo-random inputs: (A) state_in/state_out and conv_in/conv_out in separate buffers swapped every step, (B) each
// pair bound to one buffer. Reports whether the per-step outputs and the final states are bit-identical.
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Buf {
    AX_U64 phy = 0;
    void * vir = nullptr;
    uint32_t size = 0;
};

static Buf alloc(uint32_t size) {
    Buf b;
    b.size = size;
    if (AX_SYS_MemAlloc(&b.phy, &b.vir, size, 128, (const AX_S8 *) "inplace") != 0) exit(6);
    memset(b.vir, 0, size);
    return b;
}

static void fill(Buf & b, uint32_t seed) {  // small bf16 values (|x| < 2^-4)
    auto * h = (uint16_t *) b.vir;
    for (uint32_t i = 0; i < b.size / 2; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h[i] = (uint16_t) (0x3d00 + ((seed >> 16) & 0x00ff)) | (uint16_t) ((seed >> 31) << 15);
    }
}

static AX_ENGINE_IO_BUFFER_T io(const Buf & b) {
    AX_ENGINE_IO_BUFFER_T x;
    memset(&x, 0, sizeof(x));
    x.phyAddr = b.phy;
    x.pVirAddr = b.vir;
    x.nSize = b.size;
    return x;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s gdn.axmodel steps\n", argv[0]);
        return 1;
    }
    const int S = atoi(argv[2]);
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;
    AX_ENGINE_HANDLE h;
    AX_ENGINE_IO_INFO_T * info;
    if (AX_ENGINE_CreateHandle(&h, data.data(), data.size()) || AX_ENGINE_CreateContext(h) || AX_ENGINE_GetIOInfo(h, &info))
        return 3;

    std::vector<Buf> in(info->nInputSize);
    int si = -1, ci = -1, so = -1, co = -1, oi = -1, xi = -1;
    for (uint32_t i = 0; i < info->nInputSize; ++i) {
        std::string n = info->pInputs[i].pName;
        in[i] = alloc(info->pInputs[i].nSize);
        if (n == "state_in") si = i; else if (n == "conv_in") ci = i; else fill(in[i], 1000 + i);
        if (n == "x") xi = i;
    }
    for (uint32_t i = 0; i < info->nOutputSize; ++i) {
        std::string n = info->pOutputs[i].pName;
        if (n == "state_out") so = i; else if (n == "conv_out") co = i; else if (n == "out") oi = i;
    }
    if (si < 0 || ci < 0 || so < 0 || co < 0 || oi < 0 || xi < 0) return 4;
    const uint32_t ssz = info->pInputs[si].nSize, csz = info->pInputs[ci].nSize, osz = info->pOutputs[oi].nSize;

    std::vector<std::vector<uint8_t>> outs[2];
    std::vector<uint8_t> state[2], conv[2];
    for (int mode = 0; mode < 2; ++mode) {
        Buf s0 = alloc(ssz), s1 = alloc(ssz), c0 = alloc(csz), c1 = alloc(csz), out = alloc(osz);
        Buf x = alloc(info->pInputs[xi].nSize);
        std::vector<AX_ENGINE_IO_BUFFER_T> ib(info->nInputSize), ob(info->nOutputSize);
        for (uint32_t i = 0; i < info->nInputSize; ++i) ib[i] = io(in[i]);
        ib[xi] = io(x);
        ob[oi] = io(out);
        ib[si] = io(s0);
        ib[ci] = io(c0);
        ob[so] = io(mode == 0 ? s1 : s0);  // mode 1: in place
        ob[co] = io(mode == 0 ? c1 : c0);
        AX_ENGINE_IO_T set;
        memset(&set, 0, sizeof(set));
        set.pInputs = ib.data();
        set.nInputSize = ib.size();
        set.pOutputs = ob.data();
        set.nOutputSize = ob.size();
        for (int t = 0; t < S; ++t) {
            fill(x, 77 + t);  // a different token every step
            if (AX_ENGINE_RunSync(h, &set) != 0) {
                fprintf(stderr, "RunSync failed mode %d step %d\n", mode, t);
                return 9;
            }
            outs[mode].emplace_back((uint8_t *) out.vir, (uint8_t *) out.vir + osz);
            if (mode == 0) {
                std::swap(ib[si], ob[so]);
                std::swap(ib[ci], ob[co]);
            }
        }
        auto * sf = (uint8_t *) ib[si].pVirAddr;  // latest state (mode 0: swapped into the input slot)
        auto * cf = (uint8_t *) ib[ci].pVirAddr;
        state[mode].assign(sf, sf + ssz);
        conv[mode].assign(cf, cf + csz);
    }
    int diff_steps = 0;
    for (int t = 0; t < S; ++t) diff_steps += outs[0][t] != outs[1][t];
    printf("RESULT steps=%d out_differs_in=%d state_identical=%d conv_identical=%d\n", S, diff_steps,
           state[0] == state[1], conv[0] == conv[1]);
    AX_ENGINE_DestroyHandle(h);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
