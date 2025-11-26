set -e  # Exit on error

echo "=============================================="
echo "Ludo PPO Training"
echo "=============================================="

# Configuration
N_GPUS=6

# Create log directory for training outputs
LAUNCH_LOG_DIR="./training_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LAUNCH_LOG_DIR"

# Launch training for handcrafted encoding on GPU 0
echo "=============================================="
echo "Launching Handcrafted encoding training on GPU 0"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python -u train_ppo.py \
    --encoding handcrafted \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

# Launch training for handcrafted encoding with BC loss on GPU 1
echo "=============================================="
echo "Launching Handcrafted+BC encoding training on GPU 1"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 python -u train_ppo.py \
    --encoding handcrafted \
    --gpu 0 \
    --use-bc-loss \
    > "$LAUNCH_LOG_DIR/handcrafted_bc.log" 2>&1 &
HANDCRAFTED_BC_PID=$!
echo "Started handcrafted+bc training (PID: $HANDCRAFTED_BC_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_bc.log"
echo ""

# Launch training for onehot encoding on GPU 1
echo "=============================================="
echo "Launching One-hot encoding training on GPU 1"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2 python -u train_ppo.py \
    --encoding onehot \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/onehot.log" 2>&1 &
ONEHOT_PID=$!
echo "Started one-hot training (PID: $ONEHOT_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot.log"
echo ""

# Launch training for onehot encoding with BC loss on GPU 1
# echo "=============================================="  
# echo "Launching One-hot+BC encoding training on GPU 1"
# echo "=============================================="
# CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
#     --encoding onehot \
#     --gpu 0 \
#     --use-bc-loss \
#     > "$LAUNCH_LOG_DIR/onehot_bc.log" 2>&1 &
# ONEHOT_BC_PID=$!
# echo "Started one-hot+bc training (PID: $ONEHOT_BC_PID)"
# echo "Log file: $LAUNCH_LOG_DIR/onehot_bc.log"
# echo ""