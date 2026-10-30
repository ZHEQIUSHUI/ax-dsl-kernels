cd /data/junda/k4090/lineA; source /data/junda/work/axdsl_env.sh >/dev/null 2>&1
# 27B GDN prefill pv kernels without MatMul NMC input-channel accumulation (not reproducible on the board):
# MX projections in 256-row tiles over the whole K, K slices of 4096 summed by explicit fp32 adds (as the attention)
export PYTHONPATH=/data/junda/k4090/lineA/dslnmc/qwen3_5 QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 DEC_SPLIT=1 STEPS=2 MODEL=/data/junda/k4090/cfg/Qwen3.8-27B WFMT4=1 REF_HALF=128
job() { t=$1; shift; ( s=$(date +%s); env "$@" DUMP=/data/junda/k4090/lineA/of_$t nice python dslnmc/qwen3_5/q35_layer_pk.py /data/junda/k4090/lineA/knmc_f > bf_$t.log 2>&1
  echo "$t rc=$? $(( $(date +%s) - s ))s $(grep -oE 'step [0-9]: out cos [0-9.]+, (state|K) cos [0-9.]+' bf_$t.log | tr '\n' ' ') $(grep -oE '(Assertion|Value|Runtime)Error.*|AX8860CModelError.*' bf_$t.log | head -1 | cut -c1-150)" >> kf.done ) & }
# the 27B GDN prefill kernels: PRJ_IC=1280 (input tiles summed in fp32: reproducible; 1.702 vs 1.694 ms per 128-row
# call with the NMC accumulation that is not)
mkdir -p knmc_f; : > kf.done
for P in 128 64 16 4; do job pv$P PRJ_ROWS=256 PRJ_IC=1280 PRJ_ACC=part LMAX=4096 PCHUNK=$P ONLY=gdn PAD=1 NVROWS=$((P * 3 / 4)); done
wait; echo ALL >> kf.done
