#!/bin/bash

# Script to monitor training progress

if [ -z "$1" ]; then
    echo "Usage: bash training/monitor_training.sh <launch_log_dir>"
    echo "Example: bash training/monitor_training.sh ./training_logs/20231125_120000"
    echo ""
    echo "Available options:"
    echo "  handcrafted - Monitor handcrafted encoding training"
    echo "  onehot      - Monitor one-hot encoding training"
    echo "  both        - Monitor both (split screen)"
    exit 1
fi

LAUNCH_LOG_DIR=$1
MODE=${2:-both}

if [ ! -d "$LAUNCH_LOG_DIR" ]; then
    echo "Error: Directory not found: $LAUNCH_LOG_DIR"
    exit 1
fi

case $MODE in
    handcrafted)
        echo "Monitoring handcrafted training..."
        tail -f "$LAUNCH_LOG_DIR/handcrafted.log"
        ;;
    onehot)
        echo "Monitoring one-hot training..."
        tail -f "$LAUNCH_LOG_DIR/onehot.log"
        ;;
    both)
        echo "Monitoring both trainings..."
        # Check if tmux is available for split screen
        if command -v tmux &> /dev/null; then
            tmux new-session \; \
                split-window -v \; \
                send-keys "tail -f $LAUNCH_LOG_DIR/handcrafted.log" C-m \; \
                select-pane -t 0 \; \
                send-keys "tail -f $LAUNCH_LOG_DIR/onehot.log" C-m \;
        else
            # Fallback: use tail with both files
            tail -f "$LAUNCH_LOG_DIR/handcrafted.log" "$LAUNCH_LOG_DIR/onehot.log"
        fi
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: handcrafted, onehot, both"
        exit 1
        ;;
esac

