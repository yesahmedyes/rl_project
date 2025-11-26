# PPO Training for Ludo Environment

This directory contains a complete implementation of PPO (Proximal Policy Optimization) training for the Ludo game environment with curriculum learning.

## Overview

The training pipeline implements a three-stage curriculum:

1. **Stage 1**: Train against a random policy until achieving >75% win rate
2. **Stage 2**: Train against a heuristic policy until achieving >75% win rate
3. **Stage 3**: Self-play training to further refine the policy

Both **handcrafted features** (70-dim) and **one-hot encoding** (946-dim) state representations are supported.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For CUDA support (recommended for GPU training):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Installation

```python
python -c "import gymnasium; import stable_baselines3; import sb3_contrib; print('All dependencies installed!')"
```

## Quick Start

### Single GPU Training

Train with handcrafted encoding on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_ppo.py --encoding handcrafted --gpu 0
```

Train with one-hot encoding on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python training/train_ppo.py --encoding onehot --gpu 1
```

### Multi-GPU Training (Recommended)

Launch training for both encodings simultaneously:

```bash
bash training/launch_training.sh
```

This will:

- Train handcrafted encoding on GPU 0
- Train one-hot encoding on GPU 1
- Save logs to `./training_logs/<timestamp>/`

## Monitoring Training

### Monitor Training Logs

```bash
# Monitor both trainings
bash training/monitor_training.sh ./training_logs/<timestamp>/ both

# Monitor only handcrafted
bash training/monitor_training.sh ./training_logs/<timestamp>/ handcrafted

# Monitor only one-hot
bash training/monitor_training.sh ./training_logs/<timestamp>/ onehot
```

### TensorBoard

```bash
tensorboard --logdir=./tensorboard_logs
```

Then open http://localhost:6006 in your browser.

### GPU Monitoring

```bash
watch -n 1 nvidia-smi
```

## Stopping Training

```bash
bash training/stop_training.sh ./training_logs/<timestamp>/
```

Or manually:

```bash
kill <PID>  # Use PID from launch output
```

## Directory Structure

```
training/
├── __init__.py              # Module initialization
├── config.py                # Training configurations
├── vec_env_factory.py       # Vectorized environment creation
├── evaluate.py              # Evaluation utilities
├── train_ppo.py            # Main training script
├── launch_training.sh       # Multi-GPU launch script
├── stop_training.sh         # Stop training script
└── monitor_training.sh      # Monitor training script

models/                      # Saved model checkpoints
logs/                        # Environment logs
tensorboard_logs/            # TensorBoard logs
training_logs/               # Training output logs
```

## Configuration

Training hyperparameters can be modified in `training/config.py`:

### Handcrafted Encoding Config

- State dimension: 70
- Network: [128, 128] for both policy and value
- Learning rate: 3e-4
- Batch size: 64
- 16 parallel environments

### One-Hot Encoding Config

- State dimension: 946
- Network: [256, 256, 128] for both policy and value
- Learning rate: 2e-4
- Batch size: 128
- 16 parallel environments

### Curriculum Settings

- Stage 1 threshold: 75% win rate vs random
- Stage 2 threshold: 75% win rate vs heuristic
- Evaluation frequency: Every 50,000 timesteps
- Evaluation episodes: 100

### Training Duration

- Stage 1: 5M timesteps (vs random)
- Stage 2: 10M timesteps (vs heuristic)
- Stage 3: 20M timesteps (self-play)

## Advanced Usage

### Resume Training

```bash
python training/train_ppo.py --encoding handcrafted --resume models/stage1_handcrafted_1000000_steps.zip
```

### Custom Configuration

Modify `training/config.py` or create your own config:

```python
from training.config import TrainingConfig

custom_config = TrainingConfig(
    encoding_type="handcrafted",
    n_envs=32,  # More parallel environments
    learning_rate=1e-4,  # Lower learning rate
    total_timesteps_stage1=10_000_000,  # More training
)
```

### Evaluation Only

```python
from sb3_contrib import MaskablePPO
from training.evaluate import evaluate_alternating_players

# Load trained model
model = MaskablePPO.load("models/final_stage3_handcrafted.zip")

# Evaluate vs random
results = evaluate_alternating_players(
    model=model,
    opponent_type="random",
    n_eval_episodes=1000,
    encoding_type="handcrafted",
    verbose=True,
)

# Evaluate vs heuristic
results = evaluate_alternating_players(
    model=model,
    opponent_type="heuristic",
    n_eval_episodes=1000,
    encoding_type="handcrafted",
    verbose=True,
)
```

## Hardware Recommendations

### Minimum

- 1 GPU (6GB VRAM)
- 16 CPU cores
- 32GB RAM

### Recommended

- 2-6 GPUs (8GB+ VRAM each)
- 48+ CPU cores
- 64GB+ RAM

### Your Setup (as specified)

- 6 GPUs
- 96 CPU cores
- Perfect for training both encodings simultaneously with 16 envs per GPU

## Troubleshooting

### CUDA Out of Memory

- Reduce `n_envs` in config
- Reduce `batch_size` in config
- Use handcrafted encoding (smaller network)

### Training is Slow

- Increase `n_envs` for more parallelism
- Use `SubprocVecEnv` (already default)
- Ensure using GPU (`device="cuda"`)

### Low Win Rate

- Train longer (increase `total_timesteps`)
- Adjust reward function in `misc/dense_reward.py`
- Try different network architectures in config
- Ensure curriculum thresholds are being met

### Import Errors

- Ensure all dependencies installed: `pip install -r requirements.txt`
- Ensure you're in the project root directory
- Check Python path includes project directory

## Expected Results

After full curriculum training, you should see:

- **vs Random**: >95% win rate
- **vs Heuristic**: >75% win rate (target), potentially higher with more training
- **Self-play**: Continuously improving policy

## Model Checkpoints

Models are saved at:

- `models/best_model_stage<N>_<encoding>.zip` - Best model for each stage
- `models/final_stage<N>_<encoding>.zip` - Final model after each stage
- `models/stage<N>_<encoding>_<steps>_steps.zip` - Periodic checkpoints

## Next Steps

After training completes:

1. Evaluate final models against all opponents
2. Visualize games with trained agent
3. Compare handcrafted vs one-hot encoding performance
4. Fine-tune hyperparameters based on results
5. Consider extending to 4-player Ludo

## Citation

If you use this code, please cite the following libraries:

- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium

## Support

For issues or questions:

1. Check TensorBoard for training progress
2. Review training logs for errors
3. Verify GPU availability with `nvidia-smi`
4. Ensure all dependencies are installed
