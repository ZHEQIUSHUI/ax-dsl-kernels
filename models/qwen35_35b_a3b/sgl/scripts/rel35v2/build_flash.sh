#!/bin/bash
set -e
source /data/junda/work/q35lc/env.sh
S=/data/junda/work/qsmall/pf35/ax-dsl-kernels/models/qwen35_35b_a3b/sgl
export PYTHONPATH=$S/python:$S/tests:/data/junda/work/npu-sgl36/axnn
cd /data/junda/work/qsmall/pf35
OUT=${1:?out dir}
UNROLL=${2:-1}
ROWS=${ROWS:-256}
CTX=${CTX:-262144}
echo "build flash out=$OUT unroll=$UNROLL rows=$ROWS ctx=$CTX"
export PAGED_ATTN_PAGE_UNROLL=$UNROLL ROWS CTX RUN_MODE=CRUN
nice -n 10 python "$S/scripts/build_prefill.py" flash "$OUT"
