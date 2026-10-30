// Stress one multi-core axmodel: R rounds of RunSync on fixed inputs; count failures and check every round's outputs
// match round 0 bit for bit (a race between cores shows as a mismatch). ALIAS=1: bind state_in/state_out and
// conv_in/conv_out to one buffer (in place, as the llama.cpp backend does); then outputs are not compared.
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<char> read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing %s\n", path);
        exit(7);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void * alloc(uint32_t size, AX_U64 * phy) {
    void * vir;
    if (AX_SYS_MemAllocCached(phy, &vir, size, 128, (const AX_S8 *) "mcs") != 0) {
        fprintf(stderr, "CMM alloc %u failed\n", size);
        exit(6);
    }
    return vir;
}

static void fill(void * p, uint32_t size, uint32_t seed) {
    auto * h = (uint16_t *) p;
    for (uint32_t i = 0; i < size / 2; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h[i] = (uint16_t) (0x3d00 + ((seed >> 16) & 0x00ff)) | (uint16_t) ((seed >> 31) << 15);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.axmodel rounds\n", argv[0]);
        return 1;
    }
    const int    R     = atoi(argv[2]);
    const bool   alias = getenv("ALIAS") && atoi(getenv("ALIAS"));
    auto         data  = read_file(argv[1]);
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;
    AX_ENGINE_HANDLE h;
    if (AX_ENGINE_CreateHandle(&h, data.data(), data.size()) != 0) return 3;
    if (AX_ENGINE_CreateContext(h) != 0) return 4;
    AX_ENGINE_IO_INFO_T * info;
    if (AX_ENGINE_GetIOInfo(h, &info) != 0) return 5;
    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.nInputSize  = info->nInputSize;
    io.nOutputSize = info->nOutputSize;
    io.pInputs     = new AX_ENGINE_IO_BUFFER_T[io.nInputSize]();
    io.pOutputs    = new AX_ENGINE_IO_BUFFER_T[io.nOutputSize]();
    for (uint32_t i = 0; i < io.nInputSize; ++i) {
        auto & b = io.pInputs[i];
        b.nSize  = info->pInputs[i].nSize;
        b.pVirAddr = alloc(b.nSize, &b.phyAddr);
        fill(b.pVirAddr, b.nSize, 12345u + i);
        AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
    }
    for (uint32_t i = 0; i < io.nOutputSize; ++i) {
        auto &            b = io.pOutputs[i];
        const std::string n = info->pOutputs[i].pName;
        b.nSize             = info->pOutputs[i].nSize;
        int src             = -1;
        if (alias && (n == "state_out" || n == "conv_out")) {
            const std::string in = n == "state_out" ? "state_in" : "conv_in";
            for (uint32_t j = 0; j < io.nInputSize; ++j) {
                src = in == info->pInputs[j].pName ? (int) j : src;
            }
        }
        if (src >= 0) {
            b.pVirAddr = io.pInputs[src].pVirAddr;
            b.phyAddr  = io.pInputs[src].phyAddr;
            printf("alias %s -> %s\n", n.c_str(), info->pInputs[src].pName);
        } else {
            b.pVirAddr = alloc(b.nSize, &b.phyAddr);
            memset(b.pVirAddr, 0, b.nSize);
            AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
        }
    }
    std::vector<std::vector<char>> ref(io.nOutputSize);
    int fails = 0, last = 0, mism = 0, first_mism = -1;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) {
        const int e = AX_ENGINE_RunSync(h, &io);
        if (e != 0) {
            ++fails, last = e;
            printf("round %d: RunSync 0x%x\n", r, e);
            fflush(stdout);
            continue;
        }
        if (alias) {
            continue;
        }
        for (uint32_t i = 0; i < io.nOutputSize; ++i) {
            auto & b = io.pOutputs[i];
            AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
            if (ref[i].empty()) {
                ref[i].assign((char *) b.pVirAddr, (char *) b.pVirAddr + b.nSize);
            } else if (memcmp(ref[i].data(), b.pVirAddr, b.nSize)) {
                ++mism;
                if (first_mism < 0) {
                    first_mism = r;
                    printf("round %d: output %s differs from round 0\n", r, info->pOutputs[i].pName);
                }
            }
        }
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
    printf("RESULT %s rounds=%d ms=%.3f fails=%d last=0x%x mismatches=%d\n", argv[1], R, ms, fails, last, mism);
    AX_ENGINE_DestroyHandle(h);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
