set -e  # Exit on error

echo "=============================================="
echo "Ludo PPO Training - 6 Configurations"
echo "=============================================="

# Create log directory for training outputs
LAUNCH_LOG_DIR="./training_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LAUNCH_LOG_DIR"

echo "=============================================="
echo "Config 6.0: Default"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v6/latest_checkpoint_stage0_onehot_dense_lr2e-04_ent0.005_batch2048_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    --dense-reward \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 2e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config6.0_onehot_dense_lr2e-04.log" 2>&1 &
PID6=$!
echo "Started Config 6.0 (PID: $PID6)"
echo "Log: $LAUNCH_LOG_DIR/config6.0_onehot_dense_lr2e-04.log"
echo ""

echo "=============================================="
echo "Config 6.1: Higher Learning Rate"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v6/latest_checkpoint_stage0_onehot_dense_lr2e-04_ent0.005_batch2048_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    --dense-reward \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 3e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config6.1_onehot_dense_lr3e-04.log" 2>&1 &
PID61=$!
echo "Started Config 6.1 (PID: $PID61)"
echo "Log: $LAUNCH_LOG_DIR/config6.1_onehot_dense_lr3e-04.log"
echo ""

echo "=============================================="
echo "Config 6.2: Lower Learning Rate"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v6/latest_checkpoint_stage0_onehot_dense_lr2e-04_ent0.005_batch2048_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    --dense-reward \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 1e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config6.2_onehot_dense_lr1e-04.log" 2>&1 &
PID62=$!
echo "Started Config 6.2 (PID: $PID62)"
echo "Log: $LAUNCH_LOG_DIR/config6.2_onehot_dense_lr1e-04.log"
echo ""

echo "=============================================="
echo "Config 6.3: Higher Entropy Coefficient"
echo "=============================================="
CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v6/latest_checkpoint_stage0_onehot_dense_lr2e-04_ent0.005_batch2048_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    --dense-reward \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 2e-4 \
    --ent-coef 0.008 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config6.3_onehot_dense_ent0.008.log" 2>&1 &
PID63=$!
echo "Started Config 6.3 (PID: $PID63)"
echo "Log: $LAUNCH_LOG_DIR/config6.3_onehot_dense_ent0.008.log"
echo ""

echo "=============================================="
echo "Config 6.4: Lower Entropy Coefficient"
echo "=============================================="
CUDA_VISIBLE_DEVICES=4 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v6/latest_checkpoint_stage0_onehot_dense_lr2e-04_ent0.005_batch2048_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    --dense-reward \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 2e-4 \
    --ent-coef 0.002 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config6.4_onehot_dense_ent0.002.log" 2>&1 &
PID64=$!
echo "Started Config 6.4 (PID: $PID64)"
echo "Log: $LAUNCH_LOG_DIR/config6.4_onehot_dense_ent0.002.log"
echo ""

echo "=============================================="
echo "Config 6.5: Sparse Reward"
echo "=============================================="
CUDA_VISIBLE_DEVICES=5 python -u train_ppo.py \
    --encoding onehot \
    --resume /home/ubuntu/ahmed-etri/rl_project/models/v6/latest_checkpoint_stage0_onehot_dense_lr2e-04_ent0.005_batch2048_arch1024.zip \
    --reload-configs \
    --gpu 0 \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 2e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config6.5_onehot_sparse.log" 2>&1 &
PID65=$!
echo "Started Config 6.5 (PID: $PID65)"
echo "Log: $LAUNCH_LOG_DIR/config6.5_onehot_sparse.log"
echo ""