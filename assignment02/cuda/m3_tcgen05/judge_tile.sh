#!/usr/bin/env bash
# 问题 3.2 的多 seed 判测。用法:./judge_tile.sh(在 cuda/ 下先 make)
set -e
BIN=./bin/m3_tcgen05/02_single_tile
[[ -x $BIN ]] || { echo "先 make bin/m3_tcgen05/02_single_tile"; exit 1; }
ok=1
for s in 1 7 42 1234 99999; do
    out=$($BIN $s) || ok=0
    echo "$out"
    [[ "$out" == PASS* ]] || ok=0
done
[[ $ok == 1 ]] && echo "JUDGE: PASS" || { echo "JUDGE: FAIL"; exit 1; }
