#!/bin/bash
# 需先激活 axnn 环境；用法：echo "rt 1 4096 12288 256" | bash run_gemv.sh out_dir
HERE=$(cd $(dirname $0) && pwd)
OUT=$1; shift; mkdir -p $OUT; cd $OUT
one() { set -- $1; tag="$1_m$2_k$3_n$4_t$5"; t=$(date +%s)
  WMODE=$1 M=$2 K=$3 N=$4 TN=$5 timeout 5400 python $HERE/gemv_bench.py > $tag.log 2>&1; rc=$?
  cyc=$(grep -o "simulator cycles: [0-9,]*" $tag.log | tail -1 | tr -d ","); cs=$(grep -o "cosine=[0-9.]*" $tag.log)
  echo "$tag rc=$rc $(( $(date +%s)-t ))s ${cyc#simulator cycles: } $cs" >> summary.txt; }
export -f one; export HERE
cat | xargs -P 16 -I{} bash -c "one \"{}\""
echo ALLDONE >> summary.txt
