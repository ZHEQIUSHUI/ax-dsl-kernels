// Can single-core axmodels run on the four NPU cores at once and add up their bandwidth?
// N handles of the same single-core model, handle i bound to core i with AX_ENGINE_SetAffinity, each with its own
// copy of every input (same pseudo-random content). N threads call RunSync together, with a barrier after every
// call (the per-layer sync point a tensor-parallel split would need). Reports per-round time and the aggregate
// bandwidth of the inputs read, and checks that every core's outputs match core 0 bit for bit.
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

struct Inst {
    AX_ENGINE_HANDLE handle;
    AX_ENGINE_IO_INFO_T* info;
    AX_ENGINE_IO_T io;
};

static std::vector<char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing %s\n", path);
        exit(7);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void* alloc(uint32_t size, AX_U64* phy) {
    void* vir;
    if (AX_SYS_MemAlloc(phy, &vir, size, 128, (const AX_S8*)"par") != 0) {
        fprintf(stderr, "CMM alloc %u failed\n", size);
        exit(6);
    }
    return vir;
}

// 小幅度的 bf16 伪随机数（|x| < 2^-4），避免 GDN 递推和 MLP 溢出
static void fill(void* p, uint32_t size, uint32_t seed) {
    auto* h = (uint16_t*)p;
    for (uint32_t i = 0; i < size / 2; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h[i] = (uint16_t)(0x3d00 + ((seed >> 16) & 0x00ff)) | (uint16_t)((seed >> 31) << 15);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.axmodel n_cores rounds [first_core]\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[2]), R = atoi(argv[3]), C0 = argc > 4 ? atoi(argv[4]) : 0;
    auto data = read_file(argv[1]);
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;

    std::vector<Inst> inst(N);
    uint64_t in_bytes = 0;
    for (int c = 0; c < N; ++c) {
        Inst& k = inst[c];
        // 创建时指定核（CreateHandleV2 的 nNpuSet），创建 context 后再 SetAffinity 一次；把实际生效的值打印出来
        AX_ENGINE_HANDLE_EXTRA_T extra;
        memset(&extra, 0, sizeof(extra));
        extra.nNpuSet = 0x3u << (2 * (C0 + c));  // 掩码每个核占 2 位：0x3 0xc 0x30 0xc0
        extra.pName = (AX_S8*)"par";
        if (AX_ENGINE_CreateHandleV2(&k.handle, data.data(), data.size(), &extra) != 0) return 3;
        AX_ENGINE_NPU_SET_T got = 0;
        AX_ENGINE_GetAffinity(k.handle, &got);
        if (AX_ENGINE_CreateContext(k.handle) != 0) return 4;
        int ret = AX_ENGINE_SetAffinity(k.handle, 0x3u << (2 * (C0 + c)));
        AX_ENGINE_NPU_SET_T got2 = 0;
        AX_ENGINE_GetAffinity(k.handle, &got2);
        printf("handle %d -> core %d: affinity after CreateHandleV2 0x%x, after SetAffinity(ret 0x%x) 0x%x\n", c, C0 + c, got, ret, got2);
        if (AX_ENGINE_GetIOInfo(k.handle, &k.info) != 0) return 5;
        memset(&k.io, 0, sizeof(k.io));
        k.io.nInputSize = k.info->nInputSize;
        k.io.nOutputSize = k.info->nOutputSize;
        k.io.pInputs = new AX_ENGINE_IO_BUFFER_T[k.io.nInputSize]();
        k.io.pOutputs = new AX_ENGINE_IO_BUFFER_T[k.io.nOutputSize]();
        for (uint32_t i = 0; i < k.io.nInputSize; ++i) {
            auto& b = k.io.pInputs[i];
            b.nSize = k.info->pInputs[i].nSize;
            b.pVirAddr = alloc(b.nSize, &b.phyAddr);
            fill(b.pVirAddr, b.nSize, 12345u + i);
            if (c == 0) in_bytes += b.nSize;
        }
        for (uint32_t i = 0; i < k.io.nOutputSize; ++i) {
            auto& b = k.io.pOutputs[i];
            b.nSize = k.info->pOutputs[i].nSize;
            b.pVirAddr = alloc(b.nSize, &b.phyAddr);
            memset(b.pVirAddr, 0, b.nSize);
        }
    }

    std::barrier sync(N);
    std::vector<int> fails(N, 0), last_err(N, 0);
    auto worker = [&](int c) {
        for (int r = 0; r < R; ++r) {
            int e = AX_ENGINE_RunSync(inst[c].handle, &inst[c].io);
            if (e != 0) ++fails[c], last_err[c] = e;
            sync.arrive_and_wait();
        }
    };
    // 预热一轮
    {
        std::vector<std::thread> th;
        for (int c = 0; c < N; ++c) th.emplace_back([&, c] { AX_ENGINE_RunSync(inst[c].handle, &inst[c].io); });
        for (auto& t : th) t.join();
    }
    auto t0 = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> th;
        for (int c = 0; c < N; ++c) th.emplace_back(worker, c);
        for (auto& t : th) t.join();
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;

    int mismatch = 0;
    for (int c = 1; c < N; ++c)
        for (uint32_t i = 0; i < inst[0].io.nOutputSize; ++i)
            if (memcmp(inst[0].io.pOutputs[i].pVirAddr, inst[c].io.pOutputs[i].pVirAddr, inst[0].io.pOutputs[i].nSize))
                ++mismatch;
    for (int c = 0; c < N; ++c)
        if (fails[c]) printf("handle %d (core %d): %d failures, last error 0x%x\n", c, C0 + c, fails[c], last_err[c]);
    int nfail = 0;
    for (int f : fails) nfail += f;
    printf("RESULT cores=%d rounds=%d per_round_ms=%.3f input_MB_per_core=%.1f aggregate_GBps=%.1f run_failures=%d "
           "output_mismatch_vs_core0=%d\n",
           N, R, ms, in_bytes / 1e6, N * in_bytes / ms / 1e6, nfail, mismatch);
    for (auto& k : inst) AX_ENGINE_DestroyHandle(k.handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
