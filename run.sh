set -e  # Exit on error

echo "=============================================="
echo "Ludo PPO Training - 6 Configurations"
echo "=============================================="

# Create log directory for training outputs
LAUNCH_LOG_DIR="./training_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LAUNCH_LOG_DIR"

echo "=============================================="
echo "Config 1: Handcrafted + Sparse + [256,256,128]"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python -u train_ppo.py \
    --encoding handcrafted \
    --gpu 0 \
    --net-arch 256 256 128 \
    --n-envs 16 \
    --n-steps 512 \
    --learning-rate 3e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.995 \
    > "$LAUNCH_LOG_DIR/config1_handcrafted_sparse_256.log" 2>&1 &
PID1=$!
echo "Started Config 1 (PID: $PID1)"
echo "Log: $LAUNCH_LOG_DIR/config1_handcrafted_sparse_256.log"
echo ""

echo "=============================================="
echo "Config 2: Handcrafted + Dense + [256,256,128]"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 python -u train_ppo.py \
    --encoding handcrafted \
    --gpu 0 \
    --dense-reward \
    --net-arch 256 256 128 \
    --n-envs 16 \
    --n-steps 512 \
    --learning-rate 3e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config2_handcrafted_dense_256.log" 2>&1 &
PID2=$!
echo "Started Config 2 (PID: $PID2)"
echo "Log: $LAUNCH_LOG_DIR/config2_handcrafted_dense_256.log"
echo ""

echo "=============================================="
echo "Config 3: Handcrafted + Sparse + [512,512,256]"
echo "=============================================="
CUDA_VISIBLE_DEVICES=2 python -u train_ppo.py \
    --encoding handcrafted \
    --gpu 0 \
    --net-arch 512 512 256 \
    --n-envs 16 \
    --n-steps 512 \
    --learning-rate 3e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.995 \
    > "$LAUNCH_LOG_DIR/config3_handcrafted_sparse_512.log" 2>&1 &
PID3=$!
echo "Started Config 3 (PID: $PID3)"
echo "Log: $LAUNCH_LOG_DIR/config3_handcrafted_sparse_512.log"
echo ""

echo "=============================================="
echo "Config 4: Onehot + Sparse + [512,512,256]"
echo "=============================================="
CUDA_VISIBLE_DEVICES=3 python -u train_ppo.py \
    --encoding onehot \
    --gpu 0 \
    --net-arch 512 512 256 \
    --n-envs 16 \
    --n-steps 768 \
    --learning-rate 2e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.995 \
    > "$LAUNCH_LOG_DIR/config4_onehot_sparse_512.log" 2>&1 &
PID4=$!
echo "Started Config 4 (PID: $PID4)"
echo "Log: $LAUNCH_LOG_DIR/config4_onehot_sparse_512.log"
echo ""

echo "=============================================="
echo "Config 5: Onehot + Dense + [512,512,256]"
echo "=============================================="
CUDA_VISIBLE_DEVICES=4 python -u train_ppo.py \
    --encoding onehot \
    --gpu 0 \
    --dense-reward \
    --net-arch 512 512 256 \
    --n-envs 16 \
    --n-steps 768 \
    --learning-rate 2e-4 \
    --batch-size 2048 \
    --n-epochs 3 \
    --gamma 0.99 \
    > "$LAUNCH_LOG_DIR/config5_onehot_dense_512.log" 2>&1 &
PID5=$!
echo "Started Config 5 (PID: $PID5)"
echo "Log: $LAUNCH_LOG_DIR/config5_onehot_dense_512.log"
echo ""

echo "=============================================="
echo "Config 6: Onehot + Sparse + [1024,1024,512]"
echo "=============================================="
CUDA_VISIBLE_DEVICES=5 python -u train_ppo.py \
    --encoding onehot \
    --gpu 0 \
    --net-arch 1024 1024 512 \
    --n-envs 16 \
    --n-steps 1024 \
    --learning-rate 1.5e-4 \
    --batch-size 4096 \
    --n-epochs 2 \
    --gamma 0.995 \
    > "$LAUNCH_LOG_DIR/config6_onehot_sparse_1024.log" 2>&1 &
PID6=$!
echo "Started Config 6 (PID: $PID6)"
echo "Log: $LAUNCH_LOG_DIR/config6_onehot_sparse_1024.log"
echo ""