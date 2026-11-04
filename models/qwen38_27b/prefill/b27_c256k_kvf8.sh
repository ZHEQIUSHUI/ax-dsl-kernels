# the full 27B prefill kernel (16 blocks, T=256): stop once the compiled axmodel exists (no full-model simulation)
cd /data/junda/work/npu-sgl && source /data/junda/work/sgl_env.sh >/dev/null 2>&1
cd /data/junda/work/q27f && export PYTHONPATH=/data/junda/work/npu-sgl/sgl-kernel-axera/tests:$PYTHONPATH
O=/data/junda/work/q27f/out_b1_c256k_kvf8; rm -rf $O; s=$(date +%s)
KV8=2 CTX=262144 BLOCKS=1 ROWS=256 nice python build27.py $O > b27_1_c256k_kvf8.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do ls $O/*.compiled.axmodel > /dev/null 2>&1 && { sleep 30; kill -TERM $P; break; }; sleep 20; done
wait $P; echo "DONE $(( $(date +%s) - s ))s $(ls -la $O/*.compiled.axmodel 2>&1)" >> b27_1_c256k_kvf8.log
