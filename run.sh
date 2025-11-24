#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# You can override these via environment variables before running the script.
PRETRAIN_SAVE="${PRETRAIN_SAVE:-$SCRIPT_DIR/models/pretrained_bc_policy.zip}"
FINAL_SAVE="${FINAL_SAVE:-$SCRIPT_DIR/models/maskable_ppo.zip}"

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

echo "=== Behavior Cloning (Maskable Policy) ==="
python "$SCRIPT_DIR/pretrain_ppo.py" --save-path "$PRETRAIN_SAVE" "${PRETRAIN_ARGS[@]+"${PRETRAIN_ARGS[@]}"}"

echo "=== Testing Pretrained BC Policy ==="
python "$SCRIPT_DIR/policy_test.py" --model-path "$PRETRAIN_SAVE" --wrap

echo "=== Online Maskable PPO Training ==="
python "$SCRIPT_DIR/train_ppo.py" --bc-path "$PRETRAIN_SAVE" --save-path "$FINAL_SAVE" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"

echo "=== Testing Final Trained PPO Policy ==="
python "$SCRIPT_DIR/policy_test.py" --model-path "$FINAL_SAVE"

echo "✅ Pipeline finished."

