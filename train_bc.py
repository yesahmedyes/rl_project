import numpy as np
import argparse
from pathlib import Path
import gymnasium as gym
from ray.rllib.algorithms.bc import BCConfig
from ray import tune, train
import ray


def train_bc(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    checkpoint_freq=10,
    eval_env_type="heuristic",
    output_dir="bc_results",
):
    """
    Train a Behavior Cloning model using Ray RLlib's new API stack.

    Args:
        data_dir: Directory containing offline data collected by collect_offline_data.py
        encoding_type: State encoding type ('handcrafted' or 'onehot')
        num_iterations: Number of training iterations
        checkpoint_freq: Checkpoint frequency
        eval_env_type: Type of opponent for evaluation environment
        output_dir: Directory to save training results
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Get data path
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    print(f"Training BC on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")

    # Define observation and action spaces based on encoding type
    if encoding_type == "handcrafted":
        # Handcrafted encoding: 36 features
        # Per piece (4 pieces): position (1), in_home (1), in_goal (1), vulnerable (1),
        # safe (1), can_capture (1), blocks_opponent (1), progress (1), escaped_home (1)
        # = 9 features per piece * 4 pieces = 36 features
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32,
        )
    else:  # onehot
        # One-hot encoding: more features depending on implementation
        # This needs to match your actual one-hot encoding size
        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(200,),  # Adjust based on your actual one-hot size
            dtype=np.float32,
        )

    # Action space: 4 pieces to choose from
    action_space = gym.spaces.Discrete(4)

    # Create BC configuration with new API stack
    config = BCConfig()

    # Enable the new API stack
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    # Define the environment spaces
    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Set training parameters
    config.training(
        lr=1e-4,
        train_batch_size_per_learner=512,
        # BC uses standard supervised learning
    )

    # Configure offline data
    config.offline_data(
        input_=[data_path.as_posix()],
        # Number of passes through the dataset per training iteration
        dataset_num_iters_per_learner=1,
    )

    # Set resources
    config.resources(
        num_learners=1,
        num_gpus_per_learner=0,  # Set to 1 if you have GPU
    )

    # Configure evaluation (optional - only if you want to evaluate during training)
    # Note: For offline RL, evaluation requires setting up an actual environment
    # which we skip here for simplicity
    config.evaluation(
        evaluation_interval=None,  # Disable online evaluation
        evaluation_duration=0,
    )

    # Set reporting
    config.reporting(
        min_sample_timesteps_per_iteration=1000,
    )

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
                print(f"  New best loss!")

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
    print(f"\nTraining complete!")
    print(f"Final checkpoint saved to: {final_checkpoint}")
    print(f"Best loss achieved: {best_loss:.6f}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    return final_checkpoint


def train_bc_with_tune(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    output_dir="bc_tune_results",
):
    """
    Train BC using Ray Tune for hyperparameter tuning.

    Args:
        data_dir: Directory containing offline data
        encoding_type: State encoding type
        num_iterations: Number of training iterations
        output_dir: Directory to save tuning results
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    # Define observation and action spaces
    if encoding_type == "handcrafted":
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32,
        )
    else:
        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(200,),
            dtype=np.float32,
        )

    action_space = gym.spaces.Discrete(4)

    # Create BC configuration
    config = BCConfig()

    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Use Ray Tune for hyperparameter search
    config.training(
        lr=tune.grid_search([1e-3, 1e-4, 1e-5]),
        train_batch_size_per_learner=tune.choice([256, 512, 1024]),
    )

    config.offline_data(
        input_=[data_path.as_posix()],
        dataset_num_iters_per_learner=1,
    )

    config.resources(
        num_learners=1,
        num_gpus_per_learner=0,
    )

    config.evaluation(
        evaluation_interval=None,
        evaluation_duration=0,
    )

    print("Starting hyperparameter tuning with Ray Tune...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # Set up the tuner
    tuner = tune.Tuner(
        "BC",
        param_space=config.to_dict(),
        run_config=train.RunConfig(
            stop={"training_iteration": num_iterations},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            storage_path=output_dir,
            name="bc_tuning",
        ),
    )

    # Run the tuning
    results = tuner.fit()

    print("\nTuning complete!")
    print(f"Results saved to: {output_dir}")

    # Get best result
    best_result = results.get_best_result(metric="info/learner/total_loss", mode="min")
    print(f"\nBest configuration:")
    print(f"  Learning rate: {best_result.config['lr']}")
    print(f"  Batch size: {best_result.config['train_batch_size_per_learner']}")
    print(f"  Best loss: {best_result.metrics.get('info/learner/total_loss', 'N/A')}")

    ray.shutdown()

    return best_result


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

    parser.add_argument(
        "--use_tune",
        action="store_true",
        help="Use Ray Tune for hyperparameter tuning",
    )

    args = parser.parse_args()

    if args.use_tune:
        train_bc_with_tune(
            data_dir=args.data_dir,
            encoding_type=args.encoding_type,
            num_iterations=args.num_iterations,
            output_dir=args.output_dir,
        )
    else:
        train_bc(
            data_dir=args.data_dir,
            encoding_type=args.encoding_type,
            num_iterations=args.num_iterations,
            checkpoint_freq=args.checkpoint_freq,
            output_dir=args.output_dir,
        )
