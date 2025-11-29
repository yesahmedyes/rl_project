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
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --ent-coef 0.05 \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted_ent0.05.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_ent0.05.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training"
echo "=============================================="
CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --batch-size 1024 \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted_batch1024.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_batch1024.log"
echo ""