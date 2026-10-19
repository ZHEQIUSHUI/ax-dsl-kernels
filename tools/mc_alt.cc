// Alternate several axmodels like the llama.cpp layer chain: each round calls every model once, in order, each with
// its own IO. Counts RunSync failures per model and checks outputs against round 0 (inputs fixed: no aliasing).
// Inputs named nblk are left as a valid single run: [c, c + 1] blocks per core for [C, 2], 1 for a scalar.
// SENTINEL=1: before each call fill the outputs with 0xff bytes; right after RunSync compare them with round 0 and,
// if they differ, poll (invalidate + compare) until they match: shows a RunSync that returns before the outputs are
// complete, and for how long (first differing byte offset / output size, the wait in us).
// LOADALL=<dir>: first create a handle + context for every .axmodel in dir (not run), as a llama.cpp process does.
// CHAIN=n: the models as a layer chain: inputs base / part of model i are outputs base_out / part_out of model i - 1
// (cyclic), state_out / conv_out alias state_in / conv_in; trials of n calls from the same start, the first with a 2
// ms wait after every call (reference), the others back to back; reports trials whose final outputs differ.
#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <dirent.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

struct M {
    const char *                   path;
    AX_ENGINE_HANDLE               h;
    AX_ENGINE_IO_INFO_T *          info;
    AX_ENGINE_IO_T                 io;
    std::vector<std::vector<char>> ref;
    int                            fails = 0, mism = 0;
};

static void * alloc(uint32_t size, AX_U64 * phy) {
    void * vir;
    if (AX_SYS_MemAllocCached(phy, &vir, size, 128, (const AX_S8 *) "mca") != 0) {
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
        fprintf(stderr, "usage: %s rounds a.axmodel [b.axmodel ...]\n", argv[0]);
        return 1;
    }
    const int R = atoi(argv[1]);
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) return 2;
    const bool     sentinel = getenv("SENTINEL") != nullptr;
    int            late = 0;
    double         late_max = 0;
    std::vector<AX_ENGINE_HANDLE> extra;
    if (const char * dir = getenv("LOADALL")) {
        if (DIR * d = opendir(dir)) {
            while (dirent * f = readdir(d)) {
                std::string n = f->d_name;
                if (n.size() < 8 || n.substr(n.size() - 8) != ".axmodel") continue;
                std::ifstream     fs(std::string(dir) + "/" + n, std::ios::binary);
                std::vector<char> data((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
                AX_ENGINE_HANDLE  h;
                if (AX_ENGINE_CreateHandle(&h, data.data(), data.size()) == 0 && AX_ENGINE_CreateContext(h) == 0) {
                    extra.push_back(h);
                }
            }
            closedir(d);
        }
        printf("loaded %zu extra handles\n", extra.size());
    }
    std::vector<M> ms(argc - 2);
    for (size_t k = 0; k < ms.size(); ++k) {
        M & m = ms[k];
        m.path = argv[k + 2];
        std::ifstream f(m.path, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (data.empty() || AX_ENGINE_CreateHandle(&m.h, data.data(), data.size()) != 0 || AX_ENGINE_CreateContext(m.h) != 0 ||
            AX_ENGINE_GetIOInfo(m.h, &m.info) != 0) {
            fprintf(stderr, "load %s failed\n", m.path);
            return 3;
        }
        memset(&m.io, 0, sizeof(m.io));
        m.io.nInputSize  = m.info->nInputSize;
        m.io.nOutputSize = m.info->nOutputSize;
        m.io.pInputs     = new AX_ENGINE_IO_BUFFER_T[m.io.nInputSize]();
        m.io.pOutputs    = new AX_ENGINE_IO_BUFFER_T[m.io.nOutputSize]();
        for (uint32_t i = 0; i < m.io.nInputSize; ++i) {
            auto & b = m.io.pInputs[i];
            b.nSize  = m.info->pInputs[i].nSize;
            b.pVirAddr = alloc(b.nSize, &b.phyAddr);
            if (std::string(m.info->pInputs[i].pName) == "nblk") {
                auto * v = (int32_t *) b.pVirAddr;
                const uint32_t n = b.nSize / 4;
                for (uint32_t j = 0; j < n; ++j) {
                    v[j] = n == 1 ? 1 : (int32_t) (j / 2 + j % 2);  // core c: blocks [c, c + 1)
                }
            } else {
                fill(b.pVirAddr, b.nSize, 12345u + i);
            }
            AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
        }
        for (uint32_t i = 0; i < m.io.nOutputSize; ++i) {
            auto & b = m.io.pOutputs[i];
            b.nSize  = m.info->pOutputs[i].nSize;
            b.pVirAddr = alloc(b.nSize, &b.phyAddr);
            memset(b.pVirAddr, 0, b.nSize);
            AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
        }
        m.ref.resize(m.io.nOutputSize);
    }
    const int chain = getenv("CHAIN") ? atoi(getenv("CHAIN")) : 0;
    if (chain > 0) {
        auto find = [](M & m, bool in, const char * n) {
            const uint32_t cnt = in ? m.io.nInputSize : m.io.nOutputSize;
            for (uint32_t i = 0; i < cnt; ++i) {
                if (std::string(in ? m.info->pInputs[i].pName : m.info->pOutputs[i].pName) == n) return (int) i;
            }
            return -1;
        };
        const size_t n = ms.size();
        for (size_t k = 0; k < n; ++k) {  // rebind
            M & m = ms[k], &prev = ms[(k + n - 1) % n];
            for (const char * pr : { "base", "part" }) {
                const int i = find(m, true, pr), o = find(prev, false, (std::string(pr) + "_out").c_str());
                if (i >= 0 && o >= 0 && m.io.pInputs[i].nSize == prev.io.pOutputs[o].nSize) {
                    m.io.pInputs[i] = prev.io.pOutputs[o];
                }
            }
            for (const char * st : { "state", "conv" }) {
                const int i = find(m, true, (std::string(st) + "_in").c_str()), o = find(m, false, (std::string(st) + "_out").c_str());
                if (i >= 0 && o >= 0) m.io.pOutputs[o] = m.io.pInputs[i];
            }
        }
        // start: every model's outputs (incl. the aliased state) small pseudo-random
        std::vector<std::vector<char>> start;
        for (auto & m : ms) {
            for (uint32_t i = 0; i < m.io.nOutputSize; ++i) {
                auto & b = m.io.pOutputs[i];
                if (m.info->pOutputs[i].eDataType == AX_ENGINE_DT_FLOAT32) {
                    memset(b.pVirAddr, 0, b.nSize);
                } else {
                    fill(b.pVirAddr, b.nSize, 777u + i);
                }
                start.emplace_back((char *) b.pVirAddr, (char *) b.pVirAddr + b.nSize);
            }
        }
        auto reset = [&] {
            size_t j = 0;
            for (auto & m : ms) {
                for (uint32_t i = 0; i < m.io.nOutputSize; ++i, ++j) {
                    auto & b = m.io.pOutputs[i];
                    memcpy(b.pVirAddr, start[j].data(), b.nSize);
                    AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
                }
            }
        };
        auto snapshot = [&] {
            std::vector<char> v;
            for (auto & m : ms) {
                for (uint32_t i = 0; i < m.io.nOutputSize; ++i) {
                    auto & b = m.io.pOutputs[i];
                    AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
                    v.insert(v.end(), (char *) b.pVirAddr, (char *) b.pVirAddr + b.nSize);
                }
            }
            return v;
        };
        std::vector<char> ref;
        int               bad = 0, fails = 0;
        for (int r = 0; r < R; ++r) {
            reset();
            for (int c = 0; c < chain; ++c) {
                M & m = ms[c % n];
                if (AX_ENGINE_RunSync(m.h, &m.io) != 0 && fails++ < 3) {
                    printf("trial %d call %d %s: RunSync failed\n", r, c, m.path);
                }
                if (r == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            auto v = snapshot();
            if (r == 0) {
                ref = v;
            } else if (v != ref && bad++ < 5) {
                size_t off = 0;
                while (off < v.size() && v[off] == ref[off]) ++off;
                printf("trial %d: final outputs differ from the reference (first byte %zu of %zu)\n", r, off, v.size());
            }
        }
        printf("RESULT chain=%d trials=%d differing=%d run_failures=%d\n", chain, R, bad, fails);
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) {
        for (auto & m : ms) {
            if (sentinel && r > 0) {
                for (uint32_t i = 0; i < m.io.nOutputSize; ++i) {
                    auto & b = m.io.pOutputs[i];
                    memset(b.pVirAddr, 0xff, b.nSize);
                    AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
                }
            }
            const int e = AX_ENGINE_RunSync(m.h, &m.io);
            if (sentinel && r > 0 && e == 0) {
                auto t1 = std::chrono::steady_clock::now();
                for (uint32_t i = 0; i < m.io.nOutputSize; ++i) {
                    auto & b = m.io.pOutputs[i];
                    AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
                    if (memcmp(m.ref[i].data(), b.pVirAddr, b.nSize) == 0) {
                        continue;
                    }
                    size_t off = 0;
                    while (off < b.nSize && ((char *) b.pVirAddr)[off] == m.ref[i][off]) ++off;
                    int polls = 0;
                    while (memcmp(m.ref[i].data(), b.pVirAddr, b.nSize) && polls < 1000000) {
                        AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
                        ++polls;
                    }
                    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
                    late_max = std::max(late_max, us);
                    if (late++ < 10) {
                        printf("round %d %s: output %s incomplete at return (first diff at %zu of %u), complete after %.1f us%s\n",
                               r, m.path, m.info->pOutputs[i].pName, off, b.nSize, us, polls >= 1000000 ? " (never)" : "");
                        fflush(stdout);
                    }
                }
            }
            if (e != 0) {
                if (m.fails++ < 3) {
                    printf("round %d %s: RunSync 0x%x\n", r, m.path, e);
                    fflush(stdout);
                }
                continue;
            }
            if (getenv("NOCHECK") || (sentinel && r > 0)) {
                continue;
            }
            for (uint32_t i = 0; i < m.io.nOutputSize; ++i) {
                auto & b = m.io.pOutputs[i];
                AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
                if (m.ref[i].empty()) {
                    m.ref[i].assign((char *) b.pVirAddr, (char *) b.pVirAddr + b.nSize);
                } else if (memcmp(m.ref[i].data(), b.pVirAddr, b.nSize) && m.mism++ < 3) {
                    printf("round %d %s: output %s differs from round 0\n", r, m.path, m.info->pOutputs[i].pName);
                }
            }
        }
    }
    double ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
    for (auto & m : ms) {
        printf("RESULT %s fails=%d mismatches=%d\n", m.path, m.fails, m.mism);
    }
    printf("RESULT rounds=%d ms_per_round=%.3f late_outputs=%d late_max_us=%.1f\n", R, ms_, late, late_max);
    for (auto & m : ms) AX_ENGINE_DestroyHandle(m.h);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
