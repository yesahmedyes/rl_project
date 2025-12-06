#!/usr/bin/env bash
set -e  # Exit on error

echo "=============================================="
echo "Ludo BC Data + Training + Eval"
echo "=============================================="

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ENCODING_TYPE="handcrafted"
NUM_EPISODES=50000
MAX_STEPS=500
NUM_ITERS=10000
CHECKPOINT_FREQ=1000
NUM_GAMES=1000      # eval games per opponent
DEVICE="cuda:1"
REWARD_TYPE="sparse"

DATA_ROOT="$PROJECT_ROOT/offline_data"
COMBINED_DIR="$DATA_ROOT/combined_${ENCODING_TYPE}"
RESULTS_DIR="$PROJECT_ROOT/bc_results"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/$RUN_TAG"
MAIN_LOG="$LOG_DIR/run.log"

mkdir -p "$COMBINED_DIR" "$RESULTS_DIR" "$LOG_DIR"
find "$COMBINED_DIR" -name '*.json' -delete

exec >"$MAIN_LOG" 2>&1

echo "Encoding: $ENCODING_TYPE"
echo "Episodes per opponent: $NUM_EPISODES"
echo "Max steps per episode: $MAX_STEPS"
echo "Logs: $LOG_DIR"
echo ""

OPPONENTS=(random heuristic milestone2)

for OPP in "${OPPONENTS[@]}"; do
  OUT_DIR="$DATA_ROOT/heuristic_vs_${OPP}"
  LOG_FILE="$LOG_DIR/collect_${OPP}.log"

  echo "=== Collecting heuristic vs $OPP -> $OUT_DIR ==="
  python3 "$PROJECT_ROOT/collect_offline_data.py" \
    --num_episodes "$NUM_EPISODES" \
    --encoding_type "$ENCODING_TYPE" \
    --opponent_type "$OPP" \
    --expert_type "heuristic" \
    --reward_type "$REWARD_TYPE" \
    --output_dir "$OUT_DIR" \
    --max_steps "$MAX_STEPS" >"$LOG_FILE" 2>&1

  find "$OUT_DIR" -name '*.json' ! -name 'dataset_info.json' -exec cp {} "$COMBINED_DIR" \;
done

echo "=== Training BC on combined dataset ($COMBINED_DIR) ==="
TRAIN_LOG="$LOG_DIR/train_bc.log"
python3 "$PROJECT_ROOT/train_bc.py" \
  --data_dir "$COMBINED_DIR" \
  --encoding_type "$ENCODING_TYPE" \
  --num_iterations "$NUM_ITERS" \
  --checkpoint_freq "$CHECKPOINT_FREQ" \
  --output_dir "$RESULTS_DIR" >"$TRAIN_LOG" 2>&1

CKPT_PATH="$(grep -oE 'Final checkpoint saved to:.*' "$TRAIN_LOG" | awk -F': ' '{print $2}' | tail -1)"
if [[ -z "$CKPT_PATH" ]]; then
  echo "ERROR: Could not find checkpoint path in $TRAIN_LOG"
  exit 1
fi

echo "=== Evaluating checkpoint ($CKPT_PATH) ==="
EVAL_LOG="$LOG_DIR/eval.log"
python3 "$PROJECT_ROOT/policy_test.py" \
  --model_path "$CKPT_PATH" \
  --device "$DEVICE" \
  --encoding_type "$ENCODING_TYPE" \
  --num_games "$NUM_GAMES" >"$EVAL_LOG" 2>&1

echo "=== Done ==="
echo "Checkpoint: $CKPT_PATH"
echo "Logs: $LOG_DIR"

