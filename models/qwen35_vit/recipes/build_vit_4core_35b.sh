cd /data/junda/k4090/lineA; source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
export PYTHONPATH=/data/junda/k4090/lineA/dslnmc/qwen3_5 MODEL=/data/junda/hf/Qwen3.5-35B-A3B CORES=4
mkdir -p kvit4_35; : > vit35.done
for n in 256 1024 2048 3072; do
  ( s=$(date +%s); N=$n DUMP=/data/junda/k4090/lineA/ovit35_$n nice python dslnmc/qwen3_5/q35_vit.py /data/junda/k4090/lineA/kvit4_35 > bvit35_$n.log 2>&1
    echo "$n rc=$? $(( $(date +%s) - s ))s $(grep -oE 'cos [0-9.]+' bvit35_$n.log | tr '\n' ' ') $(grep -oE '(Assertion|Value|Runtime|Type|Index)Error.*|AX8860CModelError.*' bvit35_$n.log | head -1 | cut -c1-200)" >> vit35.done ) &
done
wait; echo ALL >> vit35.done
