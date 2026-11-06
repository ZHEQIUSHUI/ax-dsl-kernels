#!/bin/bash
# step 3: the same whole-model prefill kernel for a $R-row tail page (same banks / IO names, KV one IO per block)
cd /data/junda/work/q27f && source /data/junda/work/sgl_env.sh >/dev/null 2>&1
export PYTHONPATH=/data/junda/work/npu-sgl/sgl-kernel-axera/tests:$PYTHONPATH
O=/data/junda/work/q27f/out_pf27/pf80; mkdir -p $O; s=$(date +%s)
KV8=2 CTX=262144 BLOCKS=16 ROWS=80 SPLIT_KV=1 nice -n 10 python -u build27.py $O > b27_16_c256k_kvf8_split_t80.log 2>&1 &
P=$!
echo $P > b27_16_c256k_kvf8_split_t80.pid
while kill -0 $P 2>/dev/null; do
  f=$(ls $O/forward_qwen38_blocks_split.*.compiled.axmodel 2>/dev/null | head -1)
  [ -n "$f" ] && { sleep 30; kill -TERM $P 2>/dev/null; break; }
  sleep 20
done
wait $P 2>/dev/null
echo "DONE $(( $(date +%s) - s ))s $(ls -la $O/*.compiled.axmodel 2>&1)" >> b27_16_c256k_kvf8_split_t80.log
