#!/bin/bash
source /data/junda/work/sgl_env.sh >/dev/null 2>&1
set -u
cd "$(dirname "$0")"

MODEL=${MODEL:?set MODEL}
LAYERS=${LAYERS:-28}
OUT=${OUT:-out/build_${MODEL}}
LOG=${LOG:-build_${MODEL}.log}
export RUN_MODE=CRUN
mkdir -p "$OUT"
marker="$OUT/.compile_start"
touch "$marker"
start=$(date +%s)
nice -n 10 python build_qwen3.py "$OUT" --model "$MODEL" --layers "$LAYERS" > "$LOG" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  count=$(find "$OUT" -name '*.compiled.axmodel' -newer "$marker" | wc -l)
  if [ "$count" -ge 2 ]; then
    sleep 10
    kill -TERM "$pid" 2>/dev/null || true
    break
  fi
  sleep 10
done
wait "$pid" 2>/dev/null || true
echo "DONE $(( $(date +%s) - start ))s" >> "$LOG"
find "$OUT" -name '*.compiled.axmodel' -printf '%p %s\n' >> "$LOG"
