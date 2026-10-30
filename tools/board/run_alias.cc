// Generic AX8860 runner for phase-0 kernels.
// Inputs are read from <in_dir>/<name>.bin. An output whose name matches an input name is bound to that
// input's buffer (in-place). An output that is not an input but has <in_dir>/<name>.bin is a resident
// buffer the kernel reads and writes (e.g. a KV cache the compiler kept as output only): it is pre-filled
// from that file. Other outputs get fresh buffers. After
// `repeat` runs the outputs are written to <out_dir>/<name>.bin and the average time is printed.
// Note: with in-place outputs, every run mutates the input, so only run 1 sees the pristine data; the
// outputs written are those of the first run (a copy is taken right after it).
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_into(const std::string& path, void* dst, size_t size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    size_t n = f.tellg();
    if (n != size) {
        fprintf(stderr, "%s: %zu bytes, tensor wants %zu\n", path.c_str(), n, size);
        return false;
    }
    f.seekg(0);
    f.read((char*)dst, size);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.axmodel in_dir out_dir [repeat]\n", argv[0]);
        return 1;
    }
    const std::string in_dir = argv[2], out_dir = argv[3];
    const int repeat = argc > 4 ? atoi(argv[4]) : 10;
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;
    std::ifstream mf(argv[1], std::ios::binary);
    std::vector<char> model((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
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
    for (uint32_t i = 0; i < info->nInputSize; ++i) {
        auto& m = info->pInputs[i];
        printf("input  %-6s %10u bytes\n", m.pName, m.nSize);
        auto& b = io.pInputs[i];
        b.nSize = m.nSize;
        if (AX_SYS_MemAlloc((AX_U64*)&b.phyAddr, &b.pVirAddr, m.nSize, 128, (const AX_S8*)"phase0") != 0) return 6;
        if (!read_into(in_dir + "/" + m.pName + ".bin", b.pVirAddr, m.nSize)) {
            fprintf(stderr, "missing/bad input %s\n", m.pName);
            return 7;
        }
    }
    std::vector<int> alias(info->nOutputSize, -1);
    for (uint32_t i = 0; i < info->nOutputSize; ++i) {
        auto& m = info->pOutputs[i];
        for (uint32_t j = 0; j < info->nInputSize; ++j)
            if (strcmp(m.pName, info->pInputs[j].pName) == 0 && m.nSize == info->pInputs[j].nSize) alias[i] = j;
        if (alias[i] >= 0) {
            io.pOutputs[i] = io.pInputs[alias[i]];
        } else {
            auto& b = io.pOutputs[i];
            b.nSize = m.nSize;
            if (AX_SYS_MemAlloc((AX_U64*)&b.phyAddr, &b.pVirAddr, m.nSize, 128, (const AX_S8*)"phase0") != 0) return 8;
            std::ifstream probe(in_dir + "/" + m.pName + ".bin");
            if (probe.good()) {
                probe.close();
                if (!read_into(in_dir + "/" + m.pName + ".bin", b.pVirAddr, m.nSize)) return 8;
                alias[i] = -2;  // 预填的常驻 buffer
            }
        }
        printf("output %-6s %10u bytes %s\n", m.pName, m.nSize,
               alias[i] >= 0 ? "(in-place, bound to input)" : alias[i] == -2 ? "(resident, pre-filled)" : "");
    }

    std::vector<std::vector<char>> first(info->nOutputSize);
    double total = 0;
    for (int r = 0; r < repeat; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        int ret = AX_ENGINE_RunSync(handle, &io);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ret != 0) {
            fprintf(stderr, "RunSync failed 0x%x\n", ret);
            return 9;
        }
        if (r == 0) {
            for (uint32_t i = 0; i < info->nOutputSize; ++i) {
                auto* p = (char*)io.pOutputs[i].pVirAddr;
                first[i].assign(p, p + info->pOutputs[i].nSize);
            }
        } else {
            total += ms;
        }
    }
    for (uint32_t i = 0; i < info->nOutputSize; ++i) {
        std::ofstream f(out_dir + "/" + info->pOutputs[i].pName + ".bin", std::ios::binary);
        f.write(first[i].data(), first[i].size());
    }
    printf("RESULT runs=%d avg_ms(excl. first)=%.3f\n", repeat, repeat > 1 ? total / (repeat - 1) : 0.0);
    AX_ENGINE_DestroyHandle(handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
