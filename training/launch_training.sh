#!/bin/bash

# Multi-GPU training launch script for Ludo PPO
# This script launches training for both encoding types on separate GPUs

set -e  # Exit on error

echo "=============================================="
echo "Ludo PPO Multi-GPU Training"
echo "=============================================="

# Configuration
N_GPUS=6
PROJECT_DIR="/Users/ahmedharoon/Desktop/Fall 2025/Reinforcement Learning/Project"

# Check if we're in the right directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Create log directory for training outputs
LAUNCH_LOG_DIR="./training_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LAUNCH_LOG_DIR"

echo ""
echo "Launch configuration:"
echo "  - Number of GPUs: $N_GPUS"
echo "  - Project directory: $PROJECT_DIR"
echo "  - Log directory: $LAUNCH_LOG_DIR"
echo ""

# Function to check if GPU is available
check_gpu() {
    local gpu_id=$1
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi -i $gpu_id &> /dev/null
        return $?
    else
        echo "Warning: nvidia-smi not found, skipping GPU check"
        return 0
    fi
}

# Check GPU availability
echo "Checking GPU availability..."
for ((i=0; i<N_GPUS; i++)); do
    if check_gpu $i; then
        echo "  GPU $i: Available"
    else
        echo "  GPU $i: Not available"
    fi
done
echo ""

# Launch training for handcrafted encoding on GPU 0
echo "=============================================="
echo "Launching Handcrafted encoding training on GPU 0"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python -u training/train_ppo.py \
    --encoding handcrafted \
    --gpu 0 \
    > "$LAUNCH_LOG_DIR/handcrafted.log" 2>&1 &
HANDCRAFTED_PID=$!
echo "Started handcrafted training (PID: $HANDCRAFTED_PID)"
echo "Log file: $LAUNCH_LOG_DIR/handcrafted.log"
echo ""

# Launch training for onehot encoding on GPU 1
echo "=============================================="
echo "Launching One-hot encoding training on GPU 1"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 python -u training/train_ppo.py \
    --encoding onehot \
    --gpu 1 \
    > "$LAUNCH_LOG_DIR/onehot.log" 2>&1 &
ONEHOT_PID=$!
echo "Started one-hot training (PID: $ONEHOT_PID)"
echo "Log file: $LAUNCH_LOG_DIR/onehot.log"
echo ""

# Store PIDs for monitoring
echo "$HANDCRAFTED_PID" > "$LAUNCH_LOG_DIR/handcrafted.pid"
echo "$ONEHOT_PID" > "$LAUNCH_LOG_DIR/onehot.pid"

echo "=============================================="
echo "Both training processes launched!"
echo "=============================================="
echo ""
echo "To monitor training:"
echo "  - Handcrafted: tail -f $LAUNCH_LOG_DIR/handcrafted.log"
echo "  - One-hot:     tail -f $LAUNCH_LOG_DIR/onehot.log"
echo ""
echo "To monitor with TensorBoard:"
echo "  tensorboard --logdir=./tensorboard_logs"
echo ""
echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To stop training:"
echo "  kill $HANDCRAFTED_PID  # Stop handcrafted"
echo "  kill $ONEHOT_PID       # Stop one-hot"
echo "  OR"
echo "  bash training/stop_training.sh $LAUNCH_LOG_DIR"
echo ""

# Wait for both processes (optional - comment out to return to shell)
# echo "Waiting for training processes to complete..."
# wait $HANDCRAFTED_PID
# echo "Handcrafted training completed!"
# wait $ONEHOT_PID
# echo "One-hot training completed!"
# echo "All training complete!"

