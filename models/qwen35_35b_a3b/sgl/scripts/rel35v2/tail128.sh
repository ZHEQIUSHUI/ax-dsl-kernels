#!/bin/sh
# 128-row tail attention vs the release 256-row kernel, same cache state.
set -u
export LD_LIBRARY_PATH=/opt/lib
R=/data/qiushui/pf35
T=$R/test
AX=/data/qiushui/gap35/bin/axrun
B=$R/big256
D=$B/in128
mkdir -p "$B/logs"
run() {
    name=$1 model=$2 dir=$3 reps=${4:-3}
    echo "=== $name ==="
    rm -f "$dir"/out_*.bin
    "$AX" "$model" "$dir" "$reps" >"$B/logs/$name.log" 2>&1
    rc=$?
    grep -E '^time ' "$B/logs/$name.log" || true
    if [ $rc -ne 0 ]; then echo "FAILED $name rc=$rc"; tail -8 "$B/logs/$name.log"; exit $rc; fi
    ls -l "$dir"/out_output.bin
}
# current state: x.bin = 128 rows, position_ids.bin = [3,128]
run attn128 "$T/kern256/sgl35_flash128.axmodel" "$D" 3 || exit 1
mv "$D"/out_output.bin "$B"/out_128.bin
# 256-row page input and positions, same cache file
mv "$D"/x.bin "$D"/x128.bin
mv "$D"/x256.bin "$D"/x.bin
mv "$D"/position_ids.bin "$D"/position_ids128.bin
mv "$D"/position_ids_orig.bin "$D"/position_ids.bin
run attn256ref /data/qiushui/gap35/rel/kernels/sgl35_flash.axmodel "$D" 3 || exit 1
mv "$D"/out_output.bin "$B"/out_256.bin
echo TAIL128_DONE
