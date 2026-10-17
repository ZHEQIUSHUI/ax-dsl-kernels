#!/bin/sh
# 板端：bf16 与 MXFP8 两套数据各跑中英文生成 + 中英文 900 token 打分，结果在各自目录
export LD_LIBRARY_PATH=/opt/lib
cd ${1:-/tmp/qiushui}
for d in board_chat board_mx; do
  cd $d
  for l in zh en; do
    ./q35_chat . p_$l.ids 200 out_$l.ids t_$l.txt > run_$l.log 2>&1
    ./q35_chat . s_$l.ids 0 nll_$l.f32 score_$l.txt > score_$l.log 2>&1
  done
  cd ..
done
echo ALL_DONE > done.flag
