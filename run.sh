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
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_handcrafted_lr3e-04_ent0.10_arch512.zip \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training with 5e-4 learning rate"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_handcrafted_lr3e-04_ent0.10_arch512.zip \
    --learning-rate 5e-4 \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted_lr5e-04.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_lr5e-04.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training with 0.15 entropy coefficient"
echo "=============================================="
CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_handcrafted_lr3e-04_ent0.10_arch512.zip \
    --ent-coef 0.15 \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted_ent0.15.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_ent0.15.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training with 0.15 entropy coefficient and 5e-4 learning rate"
echo "=============================================="
CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_handcrafted_lr5e-04_ent0.15_arch512.zip \
    --learning-rate 5e-4 \
    --ent-coef 0.15 \
    --reload-configs \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted_lr5e-04_ent0.15.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_lr5e-04_ent0.15.log"
echo ""

# echo "=============================================="
# echo "Launching One-hot encoding training"
# echo "=============================================="
# CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
#     --encoding onehot \
#     --resume /home/ubuntu/ahmed-etri/rl_project/models/v2/best_onehot_lr5e-04_ent0.15_arch1024.zip \
#     --reload-configs \
#     --gpu 0 \
#     > "$LAUNCH_LOG_DIR/onehot.log" 2>&1 &
# ONEHOT_PID=$!
# echo "Started one-hot training (PID: $ONEHOT_PID)"
# echo "Log file: $LAUNCH_LOG_DIR/onehot.log"
# echo ""
