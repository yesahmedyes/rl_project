#!/bin/bash

set -e  # Exit on error

echo "=============================================="
echo "Running Policy Tests in Parallel"
echo "=============================================="

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <models_directory>"
    echo "Example: $0 /path/to/models"
    exit 1
fi

MODELS_DIR="$1"

# Check if directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Directory '$MODELS_DIR' does not exist"
    exit 1
fi

# Find all .zip model files in the directory
MODEL_PATHS=($(find "$MODELS_DIR" -maxdepth 1 -name "*.zip" -type f | sort))

# Check if any models were found
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Error: No .zip model files found in '$MODELS_DIR'"
    exit 1
fi

echo "Found ${#MODEL_PATHS[@]} model(s) in $MODELS_DIR"
echo ""

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