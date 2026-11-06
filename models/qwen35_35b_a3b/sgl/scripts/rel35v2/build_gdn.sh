#!/bin/bash
set -e
source /data/junda/work/q35lc/env.sh
S=/data/junda/work/qsmall/pf35/ax-dsl-kernels/models/qwen35_35b_a3b/sgl
export PYTHONPATH=$S/python:$S/tests:/data/junda/work/npu-sgl36/axnn
cd /data/junda/work/qsmall/pf35
OUT=${1:?out dir}
ROWS=${GDN_ROWS:-256}
VALID=${GDN_VALID:-$ROWS}
CHUNK=${GDN_CHUNK_SIZE:-32}
echo "build gdn out=$OUT rows=$ROWS valid=$VALID chunk=$CHUNK"
export GDN_ROWS=$ROWS GDN_VALID=$VALID GDN_CHUNK_SIZE=$CHUNK RUN_MODE=CRUN
nice -n 10 python "$S/scripts/build_prefill.py" gdn "$OUT"
