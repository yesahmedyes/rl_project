#!/bin/bash

# Script to stop running training processes

if [ -z "$1" ]; then
    echo "Usage: bash training/stop_training.sh <launch_log_dir>"
    echo "Example: bash training/stop_training.sh ./training_logs/20231125_120000"
    exit 1
fi

LAUNCH_LOG_DIR=$1

if [ ! -d "$LAUNCH_LOG_DIR" ]; then
    echo "Error: Directory not found: $LAUNCH_LOG_DIR"
    exit 1
fi

echo "Stopping training processes..."

# Stop handcrafted training
if [ -f "$LAUNCH_LOG_DIR/handcrafted.pid" ]; then
    HANDCRAFTED_PID=$(cat "$LAUNCH_LOG_DIR/handcrafted.pid")
    if ps -p $HANDCRAFTED_PID > /dev/null; then
        echo "Stopping handcrafted training (PID: $HANDCRAFTED_PID)..."
        kill $HANDCRAFTED_PID
        echo "Stopped handcrafted training"
    else
        echo "Handcrafted training process not running"
    fi
else
    echo "No handcrafted PID file found"
fi

# Stop one-hot training
if [ -f "$LAUNCH_LOG_DIR/onehot.pid" ]; then
    ONEHOT_PID=$(cat "$LAUNCH_LOG_DIR/onehot.pid")
    if ps -p $ONEHOT_PID > /dev/null; then
        echo "Stopping one-hot training (PID: $ONEHOT_PID)..."
        kill $ONEHOT_PID
        echo "Stopped one-hot training"
    else
        echo "One-hot training process not running"
    fi
else
    echo "No one-hot PID file found"
fi

echo "Done!"

