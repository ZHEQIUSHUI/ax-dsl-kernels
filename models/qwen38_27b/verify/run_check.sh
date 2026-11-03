#!/bin/bash
source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
cd /data/junda/work/qs_mtp/verbks_lastguard
export BANKS=1 KSTRIDE=1
RUN_MODE=CRUN nice python build_verify_m4.py out_crun > build_crun.log 2>&1
echo BUILD_RC=$? >> build_crun.log
BUILD=out_crun/t4 OUT=out_crun nice python check_verify_m4_crun.py > check_crun.log 2>&1
echo CHECK_RC=$? >> check_crun.log
rm -f out_crun/t4/run/simulator_cycle_trace.json
