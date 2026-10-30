#!/bin/bash
# Qwen3.8-27B kernels: build.sh <out dir>
#   NPU_DECODE   npu-codebase checkout at 7913a0599 (the axnn of the decode group), its .venv set up
#   NPU_PREFILL  npu-codebase checkout at cad0db3d6 (fix/sglang_axera_opt: axnn + sgl-kernel-axera of the prefill kernel)
# Writes the decode group (+ .layout) and the whole-model prefill kernel, both as the board runtime loads them
# (board_io: FP4 / E8M0 IO declared u8), and checks them against kernels.md5. The kernel directory adds the output head
# and the 4-core ViT kernels of the delivered directory (unchanged; see README.md) and manifest.json.
# A build runs the simulator only up to the compiled axmodel (minutes for the prefill, ~1.5 h for the decode group:
# random weights of the config's shapes are generated and packed first).
set -e
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
D=$(cd "$(dirname "$0")" && pwd)
: "${NPU_DECODE:?}" "${NPU_PREFILL:?}"
DEC=qwen38_27b_group16_gggagggagggaggga_mx4_kvc_c16384_npu4
PRE=qwen38_27b_prefill256_mx4_kvc_c16384_npu4
# run "$@" until the compiled axmodel matching $1 appears, then stop it
until_compiled() {
    local pat=$1; shift
    "$@" &
    local p=$!
    while kill -0 $p 2>/dev/null; do ls $pat > /dev/null 2>&1 && { sleep 60; kill -TERM $p; break; }; sleep 30; done
    wait $p || true
}
(   # decode: 16 layers per call, MXFP4 layer weights in packs, the prefill kernel's state layouts
    source "$NPU_DECODE/.venv/bin/activate"; source "$NPU_DECODE/script/npu_dev" > /dev/null 2>&1
    export PYTHONPATH=$D/decode QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 MODEL=$D/cfg WFMT4=1
    rm -rf "$OUT/o_dec"
    KVX=1 DEC_SPLIT=1 LMAX=16384 STEPS=1 G=16 FIRST=0 DUMP="$OUT/o_dec" \
        until_compiled "$OUT/o_dec/grp/cache/group_kernel.*.compiled.axmodel" python "$D/decode/q35_group.py" "$OUT/dec" > "$OUT/dec.log" 2>&1
    cp "$(ls -t "$OUT"/o_dec/grp/cache/group_kernel.*.compiled.axmodel | head -1)" "$OUT/$DEC.axmodel"
    cp "$OUT/dec/qwen35_group16_gggagggagggaggga_mx4_kvc_c16384_npu4.layout" "$OUT/$DEC.layout"
)
(   # prefill: the whole model in one kernel, a runtime loop over the 16 blocks (sgl forward_qwen38_blocks), MXFP4
    source "$NPU_PREFILL/.venv/bin/activate"; source "$NPU_PREFILL/script/npu_dev" > /dev/null 2>&1
    export PYTHONPATH=$NPU_PREFILL/sgl-kernel-axera/tests:$NPU_PREFILL/sgl-kernel-axera/python:$D/prefill:$PYTHONPATH
    rm -rf "$OUT/o_pre"
    BLOCKS=16 ROWS=256 until_compiled "$OUT/o_pre/forward_qwen38_blocks.*.compiled.axmodel" \
        python "$D/prefill/build27.py" "$OUT/o_pre" > "$OUT/pre.log" 2>&1
    cp "$(ls -t "$OUT"/o_pre/forward_qwen38_blocks.*.compiled.axmodel | head -1)" "$OUT/$PRE.axmodel"
    python "$D/board_io.py" "$OUT/$DEC.axmodel" "$OUT/$PRE.axmodel" > /dev/null
)
cd "$OUT" && grep -v mul_mat "$D/kernels.md5" | md5sum -c - || echo "differs from kernels.md5 (see README.md: reproducibility)"
