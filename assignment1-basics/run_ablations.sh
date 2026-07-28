#!/usr/bin/env bash
# 四个消融实验: 串行跑全量预算(TinyStories, lr=3e-3 基线配置, 与基线仅差一个开关)
# 用法: setsid nohup ./run_ablations.sh > logs/ablations.log 2>&1 &
set -uo pipefail   # 不用 -e: 某个消融跑挂(如 post-norm 发散)不能中断其余
cd "$(dirname "$0")"

run_abl() {
  local name="$1"; shift
  echo "=== $(date '+%H:%M:%S') ablation: $name (flags: $*) ==="
  CKPT_DIR="checkpoints_abl_${name}" \
    ./run_train.sh --save-every 10001 "$@" > "logs/abl_${name}.log" 2>&1
  echo "=== $(date '+%H:%M:%S') $name exit=$? ==="
}

run_abl no_layer_norm --no-layer-norm
run_abl no_pre_norm   --no-pre-norm
run_abl no_pos_emb    --no-pos-emb
run_abl no_swiglu     --no-swiglu
echo "ablations done"
