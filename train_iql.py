"""
Train a policy using Implicit Q-Learning (IQL) on offline data.
IQL learns separate value and Q-functions using expectile regression to avoid bootstrapping errors.
"""

import ray
import gymnasium as gym
import numpy as np
import argparse
import json
from pathlib import Path
from ray.rllib.algorithms.iql import IQLConfig
from ray import tune, air
from gymnasium import spaces

from env.ludo_gym_env import LudoGymEnv


def train_iql(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    lr=3e-4,
    train_batch_size=256,
    expectile=0.7,
    output_dir="checkpoints/iql",
    experiment_name=None,
):
    """
    Train an IQL policy on offline data.

    Args:
        data_dir: Directory containing offline data
        encoding_type: State encoding type
        num_iterations: Number of training iterations
        lr: Learning rate
        train_batch_size: Training batch size
        expectile: Expectile for value function learning (0.7 is typical)
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
    print("Training Implicit Q-Learning (IQL)")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Encoding type: {encoding_type}")
    print(f"Dataset info: {dataset_info}")
    print(f"Learning rate: {lr}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Expectile: {expectile}")
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

    # Create IQL config
    config = IQLConfig()

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
        gamma=0.99,
        expectile=expectile,  # Expectile for value function
        tau=0.005,  # Target network update rate
    )

    # Set offline data source
    config = config.offline_data(
        input_=[Path(data_dir).as_posix()],
        input_read_method="read_json",
        dataset_num_iters_per_learner=1,
    )

    # Set evaluation (disabled for offline training)
    config = config.evaluation(
        evaluation_interval=None,  # Disable evaluation during training
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set experiment name
    if experiment_name is None:
        expert_type = dataset_info.get("expert_type", "unknown")
        experiment_name = f"IQL_{expert_type}_{encoding_type}"

    # Create trainer
    print("\nBuilding IQL trainer...")
    trainer = config.build()

    # Training loop
    print("\nStarting training...")
    best_reward = -float("inf")

    for iteration in range(num_iterations):
        result = trainer.train()

        # Print progress
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        learner_info = (
            result.get("info", {}).get("learner", {}).get("default_policy", {})
        )
        print(f"  Value Loss: {learner_info.get('value_loss', 'N/A')}")
        print(f"  Q Loss: {learner_info.get('q_loss', 'N/A')}")
        print(f"  Policy Loss: {learner_info.get('policy_loss', 'N/A')}")
        print(f"  Mean Q: {learner_info.get('mean_q', 'N/A')}")

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
    parser = argparse.ArgumentParser(description="Train IQL on offline Ludo data")
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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Training batch size per learner",
    )
    parser.add_argument(
        "--expectile",
        type=float,
        default=0.7,
        help="Expectile for value function learning",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/iql",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )

    args = parser.parse_args()

    train_iql(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        expectile=args.expectile,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
