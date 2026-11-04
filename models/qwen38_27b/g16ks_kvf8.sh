# the 27B decode group reading the row-split weights from the prefill banks (BANKS=1 KSTRIDE=1): compile + the harness check
source /data/junda/work/axdsl_env.sh > /dev/null 2>&1
cd /data/junda/work/q27m
export PYTHONPATH=/data/junda/work/q27m/src_ks QW_RANDOM=1 KVL=cell BC=256 CORES=4 WFMT=mx8 BLOCK=1 BANKS=1 KSTRIDE=1 MODEL=/data/junda/k4090/cfg/Qwen3.8-27B WFMT4=1
O=/data/junda/work/q27m/o_g16ks_kvf8; rm -rf $O; s=$(date +%s)
KV8=2 NAMESFX=_kvf8 KVX=1 DEC_SPLIT=1 LMAX=16384 STEPS=1 G=16 FIRST=0 DUMP=$O nice python src_ks/q35_group.py /data/junda/work/q27m/k_g16ks_kvf8 > g16ks_kvf8.log 2>&1
echo "DONE rc=$? $(( $(date +%s) - s ))s" >> g16ks_kvf8.log
