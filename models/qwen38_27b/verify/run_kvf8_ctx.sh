#!/bin/bash
# the snapshot verify kernel at a larger context (CRUN build + the harness check): run_ctx.sh <LMAX>
source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
cd /data/junda/work/qs_mtp/verbks_m4attn
export BANKS=1 KSTRIDE=1 KV8=2 LMAX=$1
O=out_kvf8_c$1
RUN_MODE=CRUN nice python build_verify_m4.py $O > build_kvf8_c$1.log 2>&1
echo BUILD_RC=$? >> build_kvf8_c$1.log
BUILD=$O/t4 OUT=$O nice python check_verify_m4_crun.py > check_kvf8_c$1.log 2>&1
echo CHECK_RC=$? >> check_kvf8_c$1.log
rm -f $O/t4/run/simulator_cycle_trace.json
