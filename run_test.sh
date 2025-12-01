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
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage1_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage2_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/best_stage3_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/latest_checkpoint_stage1_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/latest_checkpoint_stage2_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip"
    "/home/ubuntu/ahmed-etri/rl_project/models/latest_checkpoint_stage3_onehot_lr3e-04_ent0.01_batch1024_arch1024.zip"
)

# Function to run tests for a specific model on a specific GPU
run_model_tests() {
    local model_path=$1
    local gpu_id=$2
    local model_name=$(basename "$model_path" .zip)
    local log_file="$TEST_LOG_DIR/${model_name}_gpu${gpu_id}.log"
    
    echo "Starting tests for $model_name on GPU $gpu_id (log: $log_file)"
    CUDA_VISIBLE_DEVICES=$gpu_id python -u policy_test.py \
        --model_path "$model_path" \
        --device "cuda:0" \
        > "$log_file" 2>&1 &
    echo $!  # Return PID
}

# Start tests for each model, distributing across GPUs
pids=()
gpu_counter=0

for model_path in "${MODEL_PATHS[@]}"; do
    pid=$(run_model_tests "$model_path" $gpu_counter)
    pids+=($pid)
    
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
echo "Results saved to: $TEST_LOG_DIR/"
echo "=============================================="