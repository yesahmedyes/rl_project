"""
Train a policy using Monotonic Advantage Re-Weighted Imitation Learning (MARWIL) on offline data.
MARWIL is a hybrid between imitation learning and policy gradient methods using advantage weighting.
When beta=0, it reduces to pure BC. Higher beta values incorporate policy gradient learning.
"""

import ray
import gymnasium as gym
import numpy as np
import argparse
import json
from pathlib import Path
from ray.rllib.algorithms.marwil import MARWILConfig
from ray import tune, air
from gymnasium import spaces

from env.ludo_gym_env import LudoGymEnv


def train_marwil(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    lr=1e-4,
    train_batch_size=512,
    beta=1.0,
    vf_coeff=1.0,
    output_dir="checkpoints/marwil",
    experiment_name=None,
):
    """
    Train a MARWIL policy on offline data.

    Args:
        data_dir: Directory containing offline data
        encoding_type: State encoding type
        num_iterations: Number of training iterations
        lr: Learning rate
        train_batch_size: Training batch size
        beta: Advantage weighting parameter (0=pure BC, higher=more RL)
        vf_coeff: Value function coefficient
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
    print("Training MARWIL (Monotonic Advantage Re-Weighted Imitation Learning)")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Encoding type: {encoding_type}")
    print(f"Dataset info: {dataset_info}")
    print(f"Learning rate: {lr}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Beta (advantage weighting): {beta}")
    print(f"Value function coefficient: {vf_coeff}")
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

    # Create MARWIL config
    config = MARWILConfig()

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
        beta=beta,  # Advantage weighting (0=BC, higher=more RL)
        vf_coeff=vf_coeff,  # Value function coefficient
        bc_logstd_coeff=0.0,  # Coefficient for BC entropy regularization
        moving_average_sqd_adv_norm_start=100.0,  # Starting value for advantage norm
        moving_average_sqd_adv_norm_update_rate=1e-8,  # Update rate for advantage norm
    )

    # Set offline data source
    config = config.offline_data(
        input_=[Path(data_dir).as_posix()],
        input_read_method="read_json",  # Read JSONL files
        input_read_method_kwargs={"lines": True},  # Line-delimited JSON
        dataset_num_iters_per_learner=1,
    )

    # Set evaluation (disabled for offline training)
    config = config.evaluation(
        evaluation_interval=None,  # Disable evaluation during training
        off_policy_estimation_methods={},  # Explicitly set to empty to avoid deprecation warning
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set experiment name
    if experiment_name is None:
        expert_type = dataset_info.get("expert_type", "unknown")
        experiment_name = f"MARWIL_{expert_type}_{encoding_type}_beta{beta}"

    # Create trainer
    print("\nBuilding MARWIL trainer...")
    trainer = config.build_algo()

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
        print(f"  Total Loss: {learner_info.get('total_loss', 'N/A')}")
        print(f"  Policy Loss: {learner_info.get('policy_loss', 'N/A')}")
        print(f"  VF Loss: {learner_info.get('vf_loss', 'N/A')}")
        print(f"  Mean Advantages: {learner_info.get('mean_advantages', 'N/A')}")

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
    parser = argparse.ArgumentParser(description="Train MARWIL on offline Ludo data")
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
        "--beta",
        type=float,
        default=1.0,
        help="Advantage weighting parameter (0=pure BC, higher=more RL)",
    )
    parser.add_argument(
        "--vf_coeff", type=float, default=1.0, help="Value function coefficient"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/marwil",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )

    args = parser.parse_args()

    train_marwil(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        beta=args.beta,
        vf_coeff=args.vf_coeff,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
