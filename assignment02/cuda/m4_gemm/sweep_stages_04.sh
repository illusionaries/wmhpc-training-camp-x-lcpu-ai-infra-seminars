#!/usr/bin/env bash
# 问题 4.3 的 stages 扫描:S ∈ {2,3,4,6},两个形状(4096^3 与
# M=256 N=4096 K=16384)。用法:./sweep_stages.sh(在 m4_gemm/ 下执行)
# 注意:改 STAGES 必须 make -B(-D 变了文件没变,make 认为无需重编)。
# 测性能前后建议 nvidia-smi --query-compute-apps=pid,name --format=csv
# 查一下有没有别的进程占卡,有残留数字整体失真。
set -u
cd "$(dirname "$0")"
for shape in "4096 4096 4096" "256 4096 16384"; do
    echo "== 形状 $shape =="
    for s in 2 3 4 6; do
        (cd .. && STAGES=$s make -sB bin/m4_gemm/04) || exit 1
        timeout -k 5 180 ../bin/m4_gemm/04 $shape || echo "S=$s: 失败或挂死(超时被杀)"
    done
done
