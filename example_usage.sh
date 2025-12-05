#!/bin/bash

# Example usage script for offline RL training
# This demonstrates the complete workflow

echo "=================================================="
echo "Offline RL Training Pipeline for Ludo"
echo "=================================================="

# Install dependencies (run once)
echo "Step 0: Installing dependencies..."
# pip install -r requirements_offline_rl.txt

# Option 1: Run everything with one command (RECOMMENDED)
echo ""
echo "Option 1: Running complete pipeline..."
echo ""
python run_all_experiments.py \
    --num_data_episodes 1000 \
    --encoding_type handcrafted \
    --expert_type heuristic \
    --opponent_type heuristic \
    --num_train_iterations 100 \
    --num_eval_episodes 100

# Option 2: Run step by step
echo ""
echo "Option 2: Step-by-step execution..."
echo ""

# Step 1: Collect offline data
echo "Step 1: Collecting offline data..."
python collect_offline_data.py \
    --num_episodes 1000 \
    --encoding_type handcrafted \
    --expert_type heuristic \
    --opponent_type heuristic \
    --output_dir offline_data

# Step 2: Train each algorithm
echo ""
echo "Step 2a: Training Behavior Cloning..."
python train_bc.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 1e-4 \
    --train_batch_size 512

echo ""
echo "Step 2b: Training Conservative Q-Learning..."
python train_cql.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 3e-4 \
    --train_batch_size 256 \
    --bc_iters 20000

echo ""
echo "Step 2c: Training Implicit Q-Learning..."
python train_iql.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 3e-4 \
    --train_batch_size 256 \
    --expectile 0.7

echo ""
echo "Step 2d: Training MARWIL..."
python train_marwil.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 1e-4 \
    --train_batch_size 512 \
    --beta 1.0

# Step 3: Evaluate all policies
echo ""
echo "Step 3: Evaluating all trained policies..."
python evaluate_policies.py \
    --bc_checkpoint checkpoints/bc/BC_heuristic_handcrafted/final \
    --cql_checkpoint checkpoints/cql/CQL_heuristic_handcrafted/final \
    --iql_checkpoint checkpoints/iql/IQL_heuristic_handcrafted/final \
    --marwil_checkpoint checkpoints/marwil/MARWIL_heuristic_handcrafted_beta1.0/final \
    --encoding_type handcrafted \
    --num_episodes 100

echo ""
echo "=================================================="
echo "Training complete! Check evaluation_results.json"
echo "=================================================="

