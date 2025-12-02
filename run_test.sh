#!/bin/bash

set -e  # Exit on error

echo "=============================================="
echo "Running Policy Tests in Parallel"
echo "=============================================="

# Check for available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
else
    echo "nvidia-smi not found, assuming 1 GPU"
    GPU_COUNT=1
fi

# Create log directory for test outputs
TEST_LOG_DIR="./test_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_LOG_DIR"

echo "Running comprehensive policy tests across $GPU_COUNT GPU(s)..."
echo "Log directory: $TEST_LOG_DIR"
echo ""

# Define all model paths
MODEL_PATHS=(
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage1_handcrafted_dense_lr3e-04_ent0.005_batch2048_arch256.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage1_handcrafted_sparse_lr3e-04_ent0.005_batch2048_arch256.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage1_handcrafted_sparse_lr3e-04_ent0.005_batch2048_arch512.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage2_handcrafted_dense_lr3e-04_ent0.005_batch2048_arch256.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage2_handcrafted_sparse_lr3e-04_ent0.005_batch2048_arch256.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage2_handcrafted_sparse_lr3e-04_ent0.005_batch2048_arch512.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/latest_checkpoint_stage1_handcrafted_dense_lr3e-04_ent0.005_batch2048_arch256.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/latest_checkpoint_stage1_handcrafted_sparse_lr3e-04_ent0.005_batch2048_arch256.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/latest_checkpoint_stage1_handcrafted_sparse_lr3e-04_ent0.005_batch2048_arch512.zip"
)

# Shared log file for all tests
SHARED_LOG="$TEST_LOG_DIR/all_tests.log"
echo "All tests will be logged to: $SHARED_LOG"
echo ""

# Start tests for each model, distributing across GPUs
pids=()
gpu_counter=0

for model_path in "${MODEL_PATHS[@]}"; do
    model_name=$(basename "$model_path" .zip)
    
    # Extract encoding type from model name (handcrafted or onehot)
    if [[ "$model_name" == *"handcrafted"* ]]; then
        encoding_type="handcrafted"
    else
        encoding_type="onehot"
    fi
    
    echo "Starting tests for $model_name on GPU $gpu_counter (encoding: $encoding_type)"
    CUDA_VISIBLE_DEVICES=$gpu_counter python -u policy_test.py \
        --model_path "$model_path" \
        --device "cuda:0" \
        --encoding_type "$encoding_type" \
        >> "$SHARED_LOG" 2>&1 &
    
    pids+=($!)
    
    # Cycle through GPUs
    gpu_counter=$(( (gpu_counter + 1) % GPU_COUNT ))
done

echo "All test processes started. Waiting for completion..."
echo "PIDs: ${pids[@]}"

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo "=============================================="
echo "Policy tests completed!"
echo "Results saved to: $SHARED_LOG"
echo "=============================================="