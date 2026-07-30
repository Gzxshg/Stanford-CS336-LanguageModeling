#!/usr/bin/env bash
# batch size 对比实验: 固定 token 预算(327.68M)与全部超参, 只改 batch size
# bs=128 即基线(已跑, best valid 1.3163), 这里补 64 / 256 两档
# 用法: setsid nohup ./run_batch_size.sh > logs/batch_size.log 2>&1 &
set -uo pipefail   # 不用 -e: 单组失败不中断另一组
cd "$(dirname "$0")"

run_bs() {
  local bs="$1"
  echo "=== $(date '+%H:%M:%S') batch_size: $bs ==="
  BATCH_SIZE="$bs" CKPT_DIR="checkpoints_bs${bs}" \
    ./run_train.sh --save-every 100000 > "logs/bs_${bs}.log" 2>&1
  echo "=== $(date '+%H:%M:%S') bs=$bs exit=$? ==="
}

run_bs 64
run_bs 256
echo "batch size experiments done"
