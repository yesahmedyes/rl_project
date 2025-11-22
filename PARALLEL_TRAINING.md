# Parallel Training Implementation

This document describes the parallel training architecture implemented for the DQN agent.

## Overview

The training has been parallelized to use multiple CPU cores for episode collection while keeping learning centralized on a single GPU. This follows a similar architecture to A3C/PPO but adapted for off-policy DQN.

## Architecture

```
┌─────────────────────────────────────────┐
│  Main Process (GPU)                     │
│  - Policy/Target Networks               │
│  - Replay Buffer                        │
│  - Learning (_learn())                  │
│  - Optimizer                            │
└─────────────────────────────────────────┘
            ↑                    ↓
    (trajectories)        (weights)
            │                    │
┌───────────┴────────────────────┴─────────┐
│  N Worker Processes (CPU)                │
│  - Each: Environment + Opponent Manager  │
│  - Collect episodes in parallel          │
│  - No learning, just rollouts            │
└──────────────────────────────────────────┘
```

## Key Changes

### 1. `policy_dqn.py`

Added methods for distributed training:

- `get_weights()`: Returns policy network weights for workers
- `set_weights()`: Updates policy from main process
- `add_trajectory_to_buffer()`: Batch add episode trajectories

### 2. `parallel_worker.py` (NEW)

Contains:

- `InferencePolicy`: Lightweight policy for workers (no optimizer/buffer)
- `rollout_worker()`: Worker function that collects episodes

### 3. `train_dqn.py`

Major refactor:

- Uses multiprocessing pool for parallel episode collection
- Collects batches of episodes from all workers
- Performs multiple learning updates per batch
- Synchronized weight distribution to workers

## Usage

### Basic Usage

```python
from train_dqn import train_dqn

agent = train_dqn(
    n_episodes=100000,
    num_workers=None,  # Auto-detect (uses CPU count - 2)
    episodes_per_batch=4,  # Each worker collects 4 episodes
    learning_steps_per_batch=None,  # Auto: num_workers * episodes_per_batch
)
```

### Parameters

- **num_workers**: Number of parallel workers. Default: `cpu_count() - 2`
- **episodes_per_batch**: Episodes each worker collects per iteration. Default: `4`
- **learning_steps_per_batch**: Gradient updates per batch. Default: `num_workers * episodes_per_batch`

### Running the Test

```bash
python test_parallel.py
```

### Full Training

```bash
python train_dqn.py
```

## Performance

### Before (Sequential)

- 1 CPU core
- ~50-100 episodes/minute

### After (Parallel with 48 cores)

- 48 CPU cores
- Expected: ~2000-4000 episodes/minute (40-50x speedup)

### GPU Usage

- Training uses **1 GPU** by default
- Workers use CPU for rollouts
- If you have multiple GPUs, you can:
  - Increase batch size and use DataParallel
  - Train multiple agents simultaneously
  - Use GPU for worker inference (not recommended)

## Configuration for Your System

With **48 cores and 6 GPUs**, recommended configuration:

### Option 1: Single Agent (Recommended)

```python
train_dqn(
    num_workers=46,  # Use 46 cores (leave 2 for system)
    episodes_per_batch=4,
    learning_steps_per_batch=184,  # 46 * 4
    batch_size=256,  # Larger batch for better GPU utilization
)
```

### Option 2: Multiple Agents (Advanced)

Train 6 agents simultaneously (one per GPU):

```python
# Run 6 separate processes, each with:
train_dqn(
    num_workers=7,  # 48 / 6 ≈ 8 cores per agent
    episodes_per_batch=4,
    device='cuda:0'  # Change for each agent: cuda:0, cuda:1, ..., cuda:5
)
```

## Technical Details

### Epsilon Decay

- Epsilon decays after each batch (not per episode)
- All workers use the same epsilon value per batch
- Synced from main process

### Replay Buffer

- Centralized in main process
- Thread-safe by design (only main process writes during learning)
- Workers don't maintain local buffers

### Opponent Management

- Snapshots taken periodically by main process
- Distributed to workers as weight dictionaries
- Workers create local InferencePolicy instances

### Multiprocessing

- Uses `spawn` method for compatibility
- Each worker is independent
- No shared memory (weights copied per batch)

## Troubleshooting

### "Too many open files" error

Increase system limits:

```bash
ulimit -n 4096
```

### Slow performance

- Check `num_workers` isn't too high
- Monitor CPU usage
- Reduce `episodes_per_batch` if memory constrained

### GPU out of memory

- Reduce `batch_size`
- Reduce `buffer_size`
- Reduce `learning_steps_per_batch`

## Future Improvements

1. **Shared replay buffer**: Use multiprocessing shared memory
2. **Asynchronous updates**: Workers write to buffer continuously
3. **GPU worker inference**: Distribute workers across GPUs
4. **Adaptive learning frequency**: Learn based on buffer size, not fixed steps
5. **Per-worker epsilon**: Different exploration rates per worker
