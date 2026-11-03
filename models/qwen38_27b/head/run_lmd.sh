#!/bin/bash
source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
cd /data/junda/work/qs_mtp/lmd
for c in "$@"; do LM_C=$c nice python build_lmd.py > build_c$c.log 2>&1; tail -4 build_c$c.log; done
