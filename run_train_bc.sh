TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_bc_${TIMESTAMP}.txt"

echo "Starting training at $(date)"
echo "Logging to: $LOG_FILE"

python train_bc.py --data_dir "offline_data/heuristic_handcrafted" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "Training failed with exit code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE

