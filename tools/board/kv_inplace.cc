// Phase-0 check: in-place KV cache update on AX8860.
// The kv_write axmodel has inputs (kn, vn, pos) and outputs (kc, vc): the compiler drops kc/vc from the
// inputs because the kernel only writes them. The host binds the outputs to persistent, pre-filled cache
// memory; after writing several positions we check that written rows hold the new K/V and every other
// row is untouched (the runtime must not clear output buffers).
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.axmodel HKV LMAX D\n", argv[0]);
        return 1;
    }
    const int HKV = atoi(argv[2]), LMAX = atoi(argv[3]), D = atoi(argv[4]);
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;

    auto model = read_file(argv[1]);
    AX_ENGINE_HANDLE handle;
    if (AX_ENGINE_CreateHandle(&handle, model.data(), model.size()) != 0) return 3;
    if (AX_ENGINE_CreateContext(handle) != 0) return 4;
    AX_ENGINE_IO_INFO_T* info;
    if (AX_ENGINE_GetIOInfo(handle, &info) != 0) return 5;

    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.nInputSize = info->nInputSize;
    io.nOutputSize = info->nOutputSize;
    io.pInputs = new AX_ENGINE_IO_BUFFER_T[io.nInputSize]();
    io.pOutputs = new AX_ENGINE_IO_BUFFER_T[io.nOutputSize]();
    int idx_kc = -1, idx_vc = -1, idx_kn = -1, idx_vn = -1, idx_pos = -1;
    for (uint32_t i = 0; i < info->nInputSize; ++i) {
        auto& m = info->pInputs[i];
        auto& b = io.pInputs[i];
        b.nSize = m.nSize;
        if (AX_SYS_MemAlloc((AX_U64*)&b.phyAddr, &b.pVirAddr, m.nSize, 128, (const AX_S8*)"kv-inplace") != 0) return 6;
        std::string n = m.pName;
        printf("input  %u %-6s %u bytes\n", i, m.pName, m.nSize);
        if (n == "kn") idx_kn = i;
        if (n == "vn") idx_vn = i;
        if (n == "pos") idx_pos = i;
    }
    if (idx_kn < 0 || idx_vn < 0 || idx_pos < 0) {
        fprintf(stderr, "unexpected input names\n");
        return 7;
    }
    // 输出 kc、vc 就是常驻的 KV cache：自己分配并预先填好内容
    for (uint32_t i = 0; i < info->nOutputSize; ++i) {
        auto& m = info->pOutputs[i];
        auto& b = io.pOutputs[i];
        b.nSize = m.nSize;
        if (AX_SYS_MemAlloc((AX_U64*)&b.phyAddr, &b.pVirAddr, m.nSize, 128, (const AX_S8*)"kv-inplace") != 0) return 6;
        std::string n = m.pName;
        printf("output %u %-6s %u bytes\n", i, m.pName, m.nSize);
        if (n == "kc") idx_kc = i;
        if (n == "vc") idx_vc = i;
    }
    if (idx_kc < 0 || idx_vc < 0) {
        fprintf(stderr, "unexpected output names\n");
        return 8;
    }

    const size_t row = (size_t)D;  // bf16 elements per row
    auto* kc = (uint16_t*)io.pOutputs[idx_kc].pVirAddr;
    auto* vc = (uint16_t*)io.pOutputs[idx_vc].pVirAddr;
    auto* kn = (uint16_t*)io.pInputs[idx_kn].pVirAddr;
    auto* vn = (uint16_t*)io.pInputs[idx_vn].pVirAddr;
    auto* pos = (int32_t*)io.pInputs[idx_pos].pVirAddr;
    auto cell = [&](int h, int t, int d, int which) -> uint16_t {
        return (uint16_t)((h * 7919 + t * 31 + d * 3 + which * 101) & 0x3fff);  // 有限的 bf16 位型
    };
    for (int h = 0; h < HKV; ++h)
        for (int t = 0; t < LMAX; ++t)
            for (int d = 0; d < D; ++d) {
                kc[((size_t)h * LMAX + t) * row + d] = cell(h, t, d, 0);
                vc[((size_t)h * LMAX + t) * row + d] = cell(h, t, d, 1);
            }

    const int positions[] = {0, 1, 777, 4096, LMAX - 1};
    double total_ms = 0;
    for (int p : positions) {
        for (int h = 0; h < HKV; ++h)
            for (int d = 0; d < D; ++d) {
                kn[(size_t)h * row + d] = (uint16_t)(0x4000 | ((p + h + d) & 0xff));
                vn[(size_t)h * row + d] = (uint16_t)(0x4100 | ((p * 3 + h + d) & 0xff));
            }
        *pos = p;
        auto t0 = std::chrono::steady_clock::now();
        int ret = AX_ENGINE_RunSync(handle, &io);
        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ret != 0) {
            fprintf(stderr, "RunSync failed 0x%x at pos %d\n", ret, p);
            return 9;
        }
    }

    long bad_written = 0, bad_other = 0;
    for (int h = 0; h < HKV; ++h)
        for (int t = 0; t < LMAX; ++t) {
            bool written = false;
            for (int p : positions) written |= (p == t);
            for (int d = 0; d < D; ++d) {
                uint16_t k = kc[((size_t)h * LMAX + t) * row + d], v = vc[((size_t)h * LMAX + t) * row + d];
                if (written) {
                    bad_written += k != (uint16_t)(0x4000 | ((t + h + d) & 0xff));
                    bad_written += v != (uint16_t)(0x4100 | ((t * 3 + h + d) & 0xff));
                } else {
                    bad_other += k != cell(h, t, d, 0);
                    bad_other += v != cell(h, t, d, 1);
                }
            }
        }
    printf("RESULT runs=%zu avg_ms=%.3f bad_written=%ld bad_other=%ld -> %s\n", sizeof(positions) / sizeof(int),
           total_ms / (sizeof(positions) / sizeof(int)), bad_written, bad_other,
           (bad_written == 0 && bad_other == 0) ? "IN-PLACE OK" : "MISMATCH");

    for (uint32_t i = 0; i < io.nInputSize; ++i) AX_SYS_MemFree(io.pInputs[i].phyAddr, io.pInputs[i].pVirAddr);
    for (uint32_t i = 0; i < io.nOutputSize; ++i) AX_SYS_MemFree(io.pOutputs[i].phyAddr, io.pOutputs[i].pVirAddr);
    AX_ENGINE_DestroyHandle(handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return (bad_written == 0 && bad_other == 0) ? 0 : 10;
}
