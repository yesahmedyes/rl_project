import numpy as np
import argparse
from pathlib import Path
import gymnasium as gym
from ray.rllib.algorithms.bc import BCConfig
import ray
import json


def count_dataset_samples(data_dir):
    data_path = Path(data_dir)
    total_samples = 0
    total_episodes = 0

    for json_file in data_path.glob("*.json"):
        if json_file.name == "dataset_info.json":
            continue

        try:
            with open(json_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    num_transitions = len(data.get("actions", []))
                    total_samples += num_transitions
                    total_episodes += 1
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")

    return total_samples, total_episodes


def train_bc(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    checkpoint_freq=10,
    eval_env_type="heuristic",
    output_dir="bc_results",
):  # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Get data path
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    # Collect individual JSON episode files (RLlib needs file paths, not just a directory)
    json_files = [
        str(p.resolve())
        for p in data_path.glob("*.json")
        if p.name != "dataset_info.json"
    ]

    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_path}")

    print(f"Training BC on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")

    # Define observation and action spaces based on encoding type
    if encoding_type == "handcrafted":
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(70,),
            dtype=np.float32,
        )
    else:  # onehot
        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(946,),
            dtype=np.float32,
        )

    # Action space: 12 actions (3 dice rolls × 4 pieces)
    action_space = gym.spaces.Discrete(12)

    config = BCConfig()

    # Use the old API stack (pre-RLModule/new env runner) for compatibility with the
    # JSON offline dataset format produced by JsonWriter.
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    # Define the environment spaces
    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Set training parameters
    config.training(
        lr=1e-4,
        train_batch_size_per_learner=64,  # Reduced from 512 to work with smaller datasets
        # BC uses standard supervised learning
    )

    # Configure offline data
    config.offline_data(
        input_="json_reader",
        input_config={"paths": json_files},
        dataset_num_iters_per_learner=1,
    )

    config.evaluation(
        evaluation_interval=None,
    )

    # Count and print dataset size
    num_samples, num_episodes = count_dataset_samples(data_path)

    print("\nDataset Statistics:")
    print(f"  Total samples: {num_samples}")
    print(f"  Total episodes: {num_episodes}")

    if num_episodes > 0:
        print(f"  Average episode length: {num_samples / num_episodes:.2f}")

    print("\nBC Configuration:")
    print(f"  Learning rate: {config.lr}")
    print(f"  Train batch size per learner: {config.train_batch_size_per_learner}")
    print(f"  Dataset iterations per learner: {config.dataset_num_iters_per_learner}")
    print(f"  Number of training iterations: {num_iterations}")

    # Build the algorithm
    algo = config.build()

    print("\nStarting training...")

    # Training loop
    best_loss = float("inf")

    for iteration in range(num_iterations):
        result = algo.train()

        # Extract relevant metrics
        info = result.get("info", {})
        learner_info = info.get("learner", {})

        # Get loss from learner info
        loss = learner_info.get("total_loss", learner_info.get("loss", None))

        print(f"\nIteration {iteration + 1}/{num_iterations}")

        if loss is not None:
            print(f"  Loss: {loss:.6f}")
            if loss < best_loss:
                best_loss = loss
                print("  New best loss!")

        # Print other relevant metrics
        if "learner" in info:
            for key, value in learner_info.items():
                if key != "total_loss" and isinstance(value, (int, float)):
                    print(f"  {key}: {value}")

        # Save checkpoint
        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_dir = algo.save(checkpoint_dir=output_dir)
            print(f"  Checkpoint saved to: {checkpoint_dir}")

    # Save final checkpoint
    final_checkpoint = algo.save(checkpoint_dir=output_dir)
    print("\nTraining complete!")
    print(f"Final checkpoint saved to: {final_checkpoint}")
    print(f"Best loss achieved: {best_loss:.6f}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    return final_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Behavior Cloning model using Ray RLlib"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing offline data",
    )

    parser.add_argument(
        "--encoding_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "onehot"],
        help="State encoding type",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Checkpoint frequency",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="bc_results",
        help="Output directory for checkpoints and results",
    )

    args = parser.parse_args()

    train_bc(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        output_dir=args.output_dir,
    )
