import os
from pathlib import Path
import glob
import torch
import numpy as np
import csv
from datetime import datetime

# Global flag for graceful shutdown
interrupted = False


def signal_handler(sig, frame):
    global interrupted

    print("\n\n⚠️  Keyboard interrupt received!")
    print("Finishing current batch and saving...")

    interrupted = True


def ensure_directories():
    directories = ["models", "logs", "plots", "checkpoints"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    return directories


def cleanup_old_checkpoints(save_path, keep_last=3):
    base_name = save_path.replace(".pth", "")

    checkpoint_pattern = f"checkpoints/{base_name}_ep*.pth"

    # Get all checkpoint files sorted by modification time
    checkpoints = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)

    # Remove old checkpoints, keeping only the last N
    if len(checkpoints) > keep_last:
        for old_checkpoint in checkpoints[:-keep_last]:
            try:
                os.remove(old_checkpoint)
                print(f"  Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                print(f"  Warning: Could not remove {old_checkpoint}: {e}")
