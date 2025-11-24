#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# You can override these via environment variables before running the script.
PRETRAIN_SAVE="${PRETRAIN_SAVE:-$SCRIPT_DIR/models/pretrained_ppo.pth}"
FINAL_SAVE="${FINAL_SAVE:-$SCRIPT_DIR/models/policy_ppo.pth}"

PRETRAIN_ARGS=()
TRAIN_ARGS=()
PHASE="pretrain"

for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    PHASE="train"
    continue
  fi

  if [[ "$PHASE" == "pretrain" ]]; then
    PRETRAIN_ARGS+=("$arg")
  else
    TRAIN_ARGS+=("$arg")
  fi
done

echo "=== Behavior Cloning (PPO) ==="
python "$SCRIPT_DIR/pretrain_ppo.py" --save-path "$PRETRAIN_SAVE" "${PRETRAIN_ARGS[@]+"${PRETRAIN_ARGS[@]}"}"

echo "=== Online PPO Training ==="
python "$SCRIPT_DIR/train_ppo.py" --load-path "$PRETRAIN_SAVE" --save-path "$FINAL_SAVE" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"

echo "✅ Pipeline finished."

