#!/usr/bin/env bash
# Driver for train.py with the assignment's TinyStories baseline config.
#
# Usage:
#   ./run_train.sh                          # baseline run (327.68M token budget)
#   BATCH_SIZE=64 LR=1e-3 ./run_train.sh    # override common knobs via env vars
#   ./run_train.sh --total-iters 200 --log-every 5   # extra args are passed to train.py
#   DATA_PREFIX=owt ./run_train.sh          # train on OpenWebText instead
set -euo pipefail
cd "$(dirname "$0")"

# ---- run knobs (env-var overridable) ----
DATA_PREFIX=${DATA_PREFIX:-tinystories}
VOCAB_SIZE=${VOCAB_SIZE:-10000}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-3e-3}
MIN_LR=${MIN_LR:-3e-4}
WARMUP_ITERS=${WARMUP_ITERS:-300}
TOTAL_TOKENS=${TOTAL_TOKENS:-327680000}
CKPT_DIR=${CKPT_DIR:-checkpoints}
BF16=${BF16:-1}
WANDB=${WANDB:-0}

CONTEXT_LENGTH=256
TOTAL_ITERS=$(( TOTAL_TOKENS / (BATCH_SIZE * CONTEXT_LENGTH) ))

EXTRA_FLAGS=()
[ "$BF16" = "1" ] && EXTRA_FLAGS+=(--bf16)
[ "$WANDB" = "1" ] && EXTRA_FLAGS+=(--wandb)

echo "data=${DATA_PREFIX} batch=${BATCH_SIZE} lr=${LR} iters=${TOTAL_ITERS} ckpt=${CKPT_DIR}"

exec uv run train.py \
  --train-data "data/${DATA_PREFIX}_train_tokens.npy" \
  --valid-data "data/${DATA_PREFIX}_valid_tokens.npy" \
  --vocab-size "$VOCAB_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size "$BATCH_SIZE" \
  --total-iters "$TOTAL_ITERS" \
  --lr "$LR" --min-lr "$MIN_LR" \
  --warmup-iters "$WARMUP_ITERS" \
  --ckpt-dir "$CKPT_DIR" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
