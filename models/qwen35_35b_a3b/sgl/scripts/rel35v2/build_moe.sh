#!/bin/bash
set -e
source /data/junda/work/q35lc/env.sh
S=/data/junda/work/qsmall/pf35/ax-dsl-kernels/models/qwen35_35b_a3b/sgl
export PYTHONPATH=$S/python:$S/tests:/data/junda/work/npu-sgl36/axnn
cd /data/junda/work/qsmall/pf35
OUT=${1:?out dir}
EXP=${2:-0}
ROWS=${ROWS:-4096}
CTX=${CTX:-8192}
echo "build moe out=$OUT expert_tokens=${EXP:-default} rows=$ROWS ctx=$CTX"
if [ "$EXP" = 0 ]; then unset MOE_EXPERT_TOKENS; else export MOE_EXPERT_TOKENS=$EXP; fi
export ROWS CTX RUN_MODE=CRUN
nice -n 10 python "$S/scripts/build_prefill.py" moe "$OUT"
