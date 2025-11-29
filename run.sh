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
CUDA_VISIBLE_DEVICES=0,1 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --reload-configs \
    --gpu 0 \
    --self-play-opponent-device cuda:1 \
    --self-play-opponent /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training with batch size 1024"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2,3 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --reload-configs \
    --gpu 0 \
    --self-play-opponent-device cuda:1 \
    --self-play-opponent /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --batch-size 1024 \
    > "$LAUNCH_LOG_DIR/handcrafted_batch1024.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_batch1024.log"
echo ""

echo "=============================================="
echo "Launching Handcrafted encoding training with entropy coefficient 0.05"
echo "=============================================="
CUDA_VISIBLE_DEVICES=4,5 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --reload-configs \
    --gpu 0 \
    --self-play-opponent-device cuda:1 \
    --self-play-opponent /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --ent-coef 0.05 \
    > "$LAUNCH_LOG_DIR/handcrafted_ent0.05.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted_ent0.05.log"
echo ""
