set -e  # Exit on error

echo "=============================================="
echo "Ludo PPO Training"
echo "=============================================="

# Configuration
N_GPUS=6

# Create log directory for training outputs
LAUNCH_LOG_DIR="./training_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LAUNCH_LOG_DIR"

echo "=============================================="
echo "Launching Handcrafted encoding training"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/latest_handcrafted_lr5e-04_ent0.15_arch512.zip \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

echo "=============================================="
echo "Launching One-hot encoding training"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_onehot_lr5e-04_ent0.15_arch512.zip \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/onehot.log" 2>&1 &
ONEHOT_PID=$!
echo "Started one-hot training (PID: $ONEHOT_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot.log"
echo ""

echo "=============================================="
echo "Launching One-hot encoding training with 1024"
echo "=============================================="
CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_onehot_lr5e-04_ent0.15_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/onehot_1024.log" 2>&1 &
ONEHOT_PID=$!
echo "Started one-hot training (PID: $ONEHOT_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot_1024.log"
echo ""
