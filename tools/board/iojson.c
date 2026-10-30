// the IO tables of kernels as JSON (the model is not run): iojson <x.axmodel>... > io.json
// {"<file>": {"in": [[name, bytes, dtype, [d0, ...]], ...], "out": [...]}, ...}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ax_sys_api.h"
#include "ax_engine_api.h"
int main(int argc, char **argv) {
    if (AX_SYS_Init()) return 1;
    AX_ENGINE_NPU_ATTR_T a; memset(&a, 0, sizeof a); a.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&a)) return 1;
    printf("{");
    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[i]); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *d = malloc(n); if (fread(d, 1, n, f) != (size_t) n) return 1; fclose(f);
        AX_ENGINE_HANDLE h; AX_ENGINE_IO_INFO_T *io;
        int r = AX_ENGINE_CreateHandle(&h, d, n);
        if (!r) r = AX_ENGINE_GetIOInfo(h, &io);
        if (r) { fprintf(stderr, "%s: rc 0x%x\n", argv[i], r); return 1; }
        const char *base = strrchr(argv[i], '/'); base = base ? base + 1 : argv[i];
        printf("%s\n\"%s\": {", i > 1 ? "," : "", base);
        for (int k = 0; k < 2; k++) {
            unsigned cnt = k ? io->nOutputSize : io->nInputSize;
            printf("%s\"%s\": [", k ? ", " : "", k ? "out" : "in");
            for (unsigned j = 0; j < cnt; j++) {
                AX_ENGINE_IOMETA_T *m = k ? &io->pOutputs[j] : &io->pInputs[j];
                printf("%s[\"%s\", %u, %d, [", j ? ", " : "", m->pName, m->nSize, (int) m->eDataType);
                for (unsigned s = 0; s < m->nShapeSize; s++) printf("%s%d", s ? ", " : "", m->pShape[s]);
                printf("]]");
            }
            printf("]");
        }
        printf("}");
        AX_ENGINE_DestroyHandle(h);
        free(d);
        fprintf(stderr, "%s\n", base);
    }
    printf("\n}\n");
    return 0;
}
