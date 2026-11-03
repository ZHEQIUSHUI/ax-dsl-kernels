#!/bin/bash
# compile_one.sh mtp_t1|mtp_prompt|verify4
set +u
cd /data/junda/work/qsmall/mtp27
source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
export MODEL=/data/junda/hf/Qwen3.8-27B PYTHONPATH=/data/junda/work/q27m/src:/data/junda/work/qsmall/mtp27
export KVL=cell BC=256 CORES=4 LMAX=16384 WFMT=mx8 WFMT4=1 BLOCK=1 KVX=1 DEC_SPLIT=1 QW_RANDOM=1 RUN_MODE=CRUN
mode=$1
case $mode in
 mtp_t1)
   OUT=out_mtp_crun; SUB=t1; PAT='mtp_t1.*.compiled.axmodel'; NAME=qwen35_mtp_t1_mx4_kvc_c16384_npu4
   CMD="env ONLY=t1 python build_mtp.py $OUT" ;;
 mtp_prompt)
   OUT=out_mtp_crun; SUB=prompt; PAT='mtp_prompt.*.compiled.axmodel'; NAME=qwen35_mtp_prefill256_mx4_kvc_c16384_npu4
   CMD="env ONLY=prompt python build_mtp.py $OUT" ;;
 verify4)
   OUT=out_verify_crun; SUB=t4; PAT='verify_kernel.*.compiled.axmodel'; NAME=qwen35_group16_verify4_gggagggagggaggga_mx4_kvc_c16384_npu4
   CMD="env ONLY=t4 python build_verify.py $OUT" ;;
 *) echo bad mode; exit 2;;
esac
mkdir -p $OUT
rm -rf $OUT/$SUB/cache
nice -n 10 $CMD > ${mode}.compile.log 2>&1 &
P=$!
f=""
while kill -0 $P 2>/dev/null; do
  f=$(ls $OUT/$SUB/cache/$PAT 2>/dev/null | head -1)
  if [ -n "$f" ]; then sleep 5; kill -TERM $P 2>/dev/null; break; fi
  sleep 10
done
wait $P 2>/dev/null
if [ -z "$f" ]; then f=$(ls $OUT/$SUB/cache/$PAT 2>/dev/null | head -1); fi
if [ -n "$f" ]; then
  cp "$f" "$OUT/$NAME.axmodel"
  echo "COMPILED $mode $f -> $OUT/$NAME.axmodel $(stat -c%s $OUT/$NAME.axmodel)" | tee -a ${mode}.compile.log
else
  echo "FAILED $mode" | tee -a ${mode}.compile.log
fi
