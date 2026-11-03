#!/bin/bash
set -e
source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
cd /data/junda/work/qsmall/lmh27
export PYTHONPATH=/data/junda/work/llama.cpp-axera/gguf-py:$PYTHONPATH
nice python build_lm4.py "$@"
