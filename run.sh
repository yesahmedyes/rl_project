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
CUDA_VISIBLE_DEVICES=2,3 python -u train_ppo.py \
    --encoding handcrafted \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v5/best_stage3_handcrafted_lr3e-04_ent0.01_batch1024_arch512.zip \
    --reload-configs \
    --ent-coef 0.01 \
    --gpu 0 \
    --self-play-opponent-device cuda:1 \
    --self-play-opponent /home/ubuntu/ahmed-etri/rl_project/models/v5/best_stage3_handcrafted_lr3e-04_ent0.01_batch1024_arch512.zip \
    --batch-size 4096 \
    --learning-rate 1e-4 \
    --n-epochs 10 \
    --total-timesteps-stage1 0 \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

echo "=============================================="
echo "Launching Onehot encoding training"
echo "=============================================="
CUDA_VISIBLE_DEVICES=4,5 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v5/best_stage3_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip \
    --reload-configs \
    --ent-coef 0.01 \
    --gpu 0 \
    --self-play-opponent-device cuda:1 \
    --self-play-opponent /home/ubuntu/ahmed-etri/rl_project/models/v5/best_stage3_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip \
    --net-arch 1024 1024 512 \
    --batch-size 4096 \
    --learning-rate 1e-4 \
    --n-epochs 10 \
    --total-timesteps-stage1 0 \
    > "$LAUNCH_LOG_DIR/onehot.log" 2>&1 &
ONEHOT_PID=$!
echo "Started onehot training (PID: $ONEHOT_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot.log"
echo ""

echo "=============================================="
echo "Launching Onehot encoding training"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python -u train_ppo.py \
    --encoding onehot \
    --gpu 0 \
    --net-arch 2048 2048 1024 \
    > "$LAUNCH_LOG_DIR/onehot_2048.log" 2>&1 &
ONEHOT_PID=$!
echo "Started onehot 2048 training (PID: $ONEHOT_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot_2048.log"
echo ""
