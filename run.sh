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
echo "Launching Handcrafted encoding training with batch size 1024"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2,3 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/best_stage1_handcrafted_lr3e-04_ent0.1_arch512.zip \
    --reload-configs \
    --ent-coef 0.01 \
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
echo "Launching Onehot encoding training with batch size 1024"
echo "=============================================="
CUDA_VISIBLE_DEVICES=4,5 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v4/latest_checkpoint_stage1_onehot_lr3e-04_ent0.1_batch1024_arch1024.zip \
    --reload-configs \
    --ent-coef 0.01 \
    --gpu 0 \
    --self-play-opponent-device cuda:1 \
    --self-play-opponent /home/ubuntu/ahmed-etri/rl_project/models/v4/latest_checkpoint_stage1_onehot_lr3e-04_ent0.1_batch1024_arch1024.zip \
    --net-arch 1024 1024 512 \
    --batch-size 1024 \
    > "$LAUNCH_LOG_DIR/onehot_batch1024.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot_batch1024.log"
echo ""