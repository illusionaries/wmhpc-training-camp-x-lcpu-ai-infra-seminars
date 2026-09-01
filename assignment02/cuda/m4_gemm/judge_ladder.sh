#!/usr/bin/env bash
# M4 梯子表:依次构建并运行 4.1/4.2/4.3,打印三行。
# 用法:./judge_ladder.sh [M N K](默认 4096^3;在 m4_gemm/ 下执行)
# 每个进程套 timeout -k:流水写错的典型故障是挂死,SIGTERM 杀不死
# 卡在 CUDA 同步里的进程,必须 -k 强杀,否则僵尸进程占着 GPU。
set -u
cd "$(dirname "$0")"
SHAPE=${*:-""}
fail=0
for t in 01_tiled 02_tma 03_pipeline 04; do
    (cd .. && make -s bin/m4_gemm/$t) || { echo "$t: 编译失败"; fail=1; continue; }
    timeout -k 5 180 ../bin/m4_gemm/$t $SHAPE || fail=1
done
[[ $fail == 0 ]] && echo "JUDGE: PASS" || { echo "JUDGE: FAIL"; exit 1; }
