#!/bin/bash
# 256k (1024 pages) decode, KV per layer; NLAYERS=4 first for a quick structure check
cd /data/junda/work/q35lc; source env.sh
N=${NLAYERS:-40}; O=out_dec_${N}_${PAGES:-1024}
NLAYERS=$N DUMP=0 PROCS=16 RUN_MODE=CRUN nice -n 10 python $S/scripts/build_decode_split.py /data/junda/hf/Qwen3.5-35B-A3B $O ${PAGES:-1024}
