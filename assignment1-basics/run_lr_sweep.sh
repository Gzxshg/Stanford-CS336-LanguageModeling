#!/usr/bin/env bash
# LR sweep on TinyStories: 串行跑全量预算(327.68M tokens / 10000 iters)。
# 基线 lr=3e-3 已完成(best valid 1.3163), 这里补 1e-3 / 1e-2 / 3e-2 三档,
# 3e-2 用于制造发散曲线。min_lr 一律取 lr 的 1/10。
#
# 用法: setsid nohup ./run_lr_sweep.sh > logs/lr_sweep.log 2>&1 &
set -uo pipefail   # 不用 -e: 某一档跑挂不能中断整个 sweep
cd "$(dirname "$0")"

for LR in 1e-3 1e-2 3e-2; do
  case "$LR" in
    1e-3) MIN_LR=1e-4 ;;
    1e-2) MIN_LR=1e-3 ;;
    3e-2) MIN_LR=3e-3 ;;
  esac
  CKPT_DIR="checkpoints_lr_${LR}"
  LOG="logs/lr_sweep_${LR}.log"
  echo "=== $(date '+%H:%M:%S') LR=$LR min_lr=$MIN_LR ckpt=$CKPT_DIR log=$LOG ==="
  LR="$LR" MIN_LR="$MIN_LR" CKPT_DIR="$CKPT_DIR" \
    ./run_train.sh --save-every 10001 > "$LOG" 2>&1
  echo "=== $(date '+%H:%M:%S') LR=$LR exit=$? ==="
done
echo "sweep done"
