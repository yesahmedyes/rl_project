"""
Train a policy using Behavior Cloning (BC) on offline data.
BC is pure imitation learning - learns to directly mimic the expert policy.
"""

import ray
import gymnasium as gym
import numpy as np
import argparse
import json
from pathlib import Path
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.algorithms.bc.bc import BC
from ray import tune, air
from gymnasium import spaces

from env.ludo_gym_env import LudoGymEnv


def load_episodes_to_rllib_format(data_dir):
    """
    Load episodes from JSON files and convert to RLlib format.
    """
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episodes_*.json"))

    all_episodes = []
    for file in episode_files:
        with open(file, "r") as f:
            episodes = json.load(f)
            all_episodes.extend(episodes)

    print(f"Loaded {len(all_episodes)} episodes from {data_dir}")

    # Convert to RLlib batch format
    batch_data = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "terminateds": [],
        "truncateds": [],
        "infos": [],
    }

    for episode in all_episodes:
        # All observations except the last one
        batch_data["obs"].extend(episode["observations"][:-1])
        batch_data["actions"].extend(episode["actions"])
        batch_data["rewards"].extend(episode["rewards"])
        batch_data["terminateds"].extend(episode["terminateds"])
        batch_data["truncateds"].extend(episode["truncateds"])
        batch_data["infos"].extend(episode["infos"])

    return batch_data


def train_bc(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    lr=1e-4,
    train_batch_size=512,
    output_dir="checkpoints/bc",
    experiment_name=None,
):
    """
    Train a BC policy on offline data.

    Args:
        data_dir: Directory containing offline data
        encoding_type: State encoding type
        num_iterations: Number of training iterations
        lr: Learning rate
        train_batch_size: Training batch size
        output_dir: Directory to save checkpoints
        experiment_name: Name for the experiment
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Load dataset info
    info_file = Path(data_dir) / "dataset_info.json"
    with open(info_file, "r") as f:
        dataset_info = json.load(f)

    print("=" * 50)
    print("Training Behavior Cloning (BC)")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Encoding type: {encoding_type}")
    print(f"Dataset info: {dataset_info}")
    print(f"Learning rate: {lr}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Number of iterations: {num_iterations}")
    print("=" * 50)

    # Define observation and action spaces
    if encoding_type == "handcrafted":
        state_dim = 70
    elif encoding_type == "onehot":
        state_dim = 946
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}")

    observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
    )
    action_space = spaces.Discrete(12)

    # Create BC config
    config = BCConfig()

    # Enable new API stack
    config = config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    # Set environment
    config = config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Set training parameters
    config = config.training(
        lr=lr,
        train_batch_size_per_learner=train_batch_size,
    )

    # Set offline data source
    config = config.offline_data(
        input_=[Path(data_dir).as_posix()],
        input_read_method="read_json",
        dataset_num_iters_per_learner=1,
    )

    # Set resources
    config = config.resources(
        num_learners=1,
    )

    # Set evaluation
    config = config.evaluation(
        evaluation_interval=10,
        evaluation_duration=10,
        evaluation_config={
            "explore": False,
        },
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set experiment name
    if experiment_name is None:
        expert_type = dataset_info.get("expert_type", "unknown")
        experiment_name = f"BC_{expert_type}_{encoding_type}"

    # Create trainer
    print("\nBuilding BC trainer...")
    trainer = config.build()

    # Training loop
    print("\nStarting training...")
    best_reward = -float("inf")

    for iteration in range(num_iterations):
        result = trainer.train()

        # Print progress
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        print(
            f"  Mean loss: {result.get('info', {}).get('learner', {}).get('default_policy', {}).get('total_loss', 'N/A')}"
        )

        # Save checkpoint periodically
        if (iteration + 1) % 10 == 0:
            checkpoint_dir = output_path / experiment_name / f"iter_{iteration + 1:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.save(checkpoint_dir.as_posix())
            print(f"  Checkpoint saved to {checkpoint_dir}")

    # Save final checkpoint
    final_checkpoint = output_path / experiment_name / "final"
    final_checkpoint.mkdir(parents=True, exist_ok=True)
    trainer.save(final_checkpoint.as_posix())
    print(f"\nFinal checkpoint saved to {final_checkpoint}")

    # Cleanup
    trainer.stop()
    ray.shutdown()

    print("\nTraining complete!")
    return final_checkpoint.as_posix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC on offline Ludo data")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing offline data"
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "onehot"],
        help="State encoding type",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=512,
        help="Training batch size per learner",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/bc",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )

    args = parser.parse_args()

    train_bc(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
