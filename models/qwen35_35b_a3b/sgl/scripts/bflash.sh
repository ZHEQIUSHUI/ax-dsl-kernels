#!/bin/bash
cd /data/junda/work/q35lc; source env.sh
CTX=262144 RUN_MODE=CRUN nice -n 10 python $S/scripts/build_prefill.py flash out_c256k
