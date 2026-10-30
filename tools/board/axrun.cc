// axrun <model.axmodel> <dir> [reps]: run one kernel on the board with inputs <dir>/<input name>.bin (raw bytes, the
// IO's size; missing inputs zero), write <dir>/out_<output name>.bin, then time reps more runs (RunSync, us per run).
// AXRUN_ALIAS="state_out:state_in,..." runs those outputs in place of the inputs (written after the first run only);
// AXRUN_NC=1 allocates the IO buffers non-cached
#include <ax_engine_api.h>
#include <ax_sys_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static AX_ENGINE_IO_BUFFER_T alloc(AX_U32 size) {
    AX_ENGINE_IO_BUFFER_T b;
    memset(&b, 0, sizeof(b));
    static const bool nc = getenv("AXRUN_NC") != nullptr;  // non-cached IO buffers (runtime cache maintenance cost)
    if ((nc ? AX_SYS_MemAlloc((AX_U64 *) &b.phyAddr, &b.pVirAddr, size, 4096, (const AX_S8 *) "axrun")
            : AX_SYS_MemAllocCached((AX_U64 *) &b.phyAddr, &b.pVirAddr, size, 4096, (const AX_S8 *) "axrun")) != 0) {
        fprintf(stderr, "axrun: CMM alloc of %u bytes failed\n", size);
        exit(1);
    }
    b.nSize = size;
    memset(b.pVirAddr, 0, size);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: axrun <model.axmodel> <dir> [reps]\n");
        return 2;
    }
    const std::string dir  = argv[2];
    const int         reps = argc > 3 ? atoi(argv[3]) : 20;
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_ENGINE_Init(&attr);
    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "axrun: cannot open %s\n", argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> model(n);
    if (fread(model.data(), 1, n, f) != (size_t) n) {
        return 1;
    }
    fclose(f);
    AX_ENGINE_HANDLE h;
    AX_ENGINE_IO_INFO_T * info;
    if (AX_ENGINE_CreateHandle(&h, model.data(), n) != 0 || AX_ENGINE_CreateContext(h) != 0 ||
        AX_ENGINE_GetIOInfo(h, &info) != 0) {
        fprintf(stderr, "axrun: cannot load %s\n", argv[1]);
        return 1;
    }
    std::vector<AX_ENGINE_IO_BUFFER_T> in(info->nInputSize), out(info->nOutputSize);
    for (AX_U32 i = 0; i < info->nInputSize; ++i) {
        in[i]           = alloc(info->pInputs[i].nSize);
        std::string p   = dir + "/" + info->pInputs[i].pName + ".bin";
        FILE *      fi  = fopen(p.c_str(), "rb");
        size_t      got = 0;
        if (fi) {
            got = fread(in[i].pVirAddr, 1, in[i].nSize, fi);
            fclose(fi);
        }
        printf("in  %-12s %10u bytes %s\n", info->pInputs[i].pName, in[i].nSize,
               !fi ? "(zero)" : got == in[i].nSize ? "" : "(SHORT FILE)");
        AX_SYS_MflushCache(in[i].phyAddr, in[i].pVirAddr, in[i].nSize);
    }
    // AXRUN_ALIAS="out_name:in_name,...": that output in the input's buffer (in place, as the backend's recurrent state)
    const char * alias = getenv("AXRUN_ALIAS");
    for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
        const std::string on = info->pOutputs[i].pName;
        int               ai = -1;
        if (alias) {
            const std::string a   = std::string(",") + alias + ",";
            const size_t      pos = a.find("," + on + ":");
            if (pos != std::string::npos) {
                const size_t s0 = pos + on.size() + 2;
                const std::string in_name = a.substr(s0, a.find(',', s0) - s0);
                for (AX_U32 j = 0; j < info->nInputSize; ++j) {
                    if (in_name == info->pInputs[j].pName && info->pInputs[j].nSize == info->pOutputs[i].nSize) {
                        ai = (int) j;
                    }
                }
            }
        }
        if (ai >= 0) {
            out[i] = in[ai];
            printf("out %-12s in place of input %s\n", on.c_str(), info->pInputs[ai].pName);
            continue;
        }
        out[i] = alloc(info->pOutputs[i].nSize);
        AX_SYS_MflushCache(out[i].phyAddr, out[i].pVirAddr, out[i].nSize);
    }
    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.pInputs     = in.data();
    io.nInputSize  = info->nInputSize;
    io.pOutputs    = out.data();
    io.nOutputSize = info->nOutputSize;
    int ret = AX_ENGINE_RunSync(h, &io);
    if (ret != 0) {
        fprintf(stderr, "axrun: RunSync failed 0x%x\n", ret);
        return 1;
    }
    for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
        AX_SYS_MinvalidateCache(out[i].phyAddr, out[i].pVirAddr, out[i].nSize);
        std::string p  = dir + "/out_" + info->pOutputs[i].pName + ".bin";
        FILE *      fo = fopen(p.c_str(), "wb");
        fwrite(out[i].pVirAddr, 1, out[i].nSize, fo);
        fclose(fo);
        printf("out %-12s %10u bytes -> %s\n", info->pOutputs[i].pName, out[i].nSize, p.c_str());
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps && ret == 0; ++r) {
        ret = AX_ENGINE_RunSync(h, &io);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    printf("time %.1f us per run (%d runs)%s\n", reps ? us / reps : 0.0, reps, ret ? " FAILED" : "");
    AX_ENGINE_DestroyHandle(h);
    return ret != 0;
}
