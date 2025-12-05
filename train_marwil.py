import numpy as np
import argparse
from pathlib import Path
import gymnasium as gym
from ray.rllib.algorithms.marwil import MARWILConfig
from ray import tune, train
import ray


def train_marwil(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    checkpoint_freq=10,
    beta=1.0,
    lr=1e-4,
    output_dir="marwil_results",
):
    """
    Train a MARWIL model using Ray RLlib's new API stack.

    MARWIL is a hybrid imitation learning and policy gradient algorithm.
    When beta=0, it reduces to BC. Higher beta values use advantage re-weighting.

    Args:
        data_dir: Directory containing offline data
        encoding_type: State encoding type ('handcrafted' or 'onehot')
        num_iterations: Number of training iterations
        checkpoint_freq: Checkpoint frequency
        beta: Scaling of advantages (0=BC, higher=more RL-like)
        lr: Learning rate
        output_dir: Directory to save training results
    """
    if not ray.is_initialized():
        ray.init()

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    print(f"Training MARWIL on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")
    print(f"Beta (advantage scaling): {beta}")

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

    # Create MARWIL configuration
    config = MARWILConfig()

    # Enable new API stack
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Set training parameters
    config.training(
        beta=beta,  # Advantage scaling (0=BC, higher=more advantage-weighted)
        lr=lr,
        gamma=0.99,  # Discount factor for advantage computation
        train_batch_size_per_learner=512,
        vf_coeff=1.0,  # Value function loss coefficient
        grad_clip=40.0,  # Gradient clipping
    )

    # Configure offline data
    config.offline_data(
        input_=[data_path.as_posix()],
        dataset_num_iters_per_learner=1,
    )

    # Set resources
    config.resources(
        num_learners=1,
        num_gpus_per_learner=0,
    )

    config.evaluation(
        evaluation_interval=None,
        evaluation_duration=0,
    )

    config.reporting(
        min_sample_timesteps_per_iteration=1000,
    )

    print("\nMARWIL Configuration:")
    print(f"  Beta: {config.beta}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Train batch size: {config.train_batch_size_per_learner}")
    print(f"  Value function coeff: {config.vf_coeff}")

    # Build and train
    algo = config.build()

    print("\nStarting training...")

    best_loss = float("inf")
    for iteration in range(num_iterations):
        result = algo.train()

        info = result.get("info", {})
        learner_info = info.get("learner", {})
        loss = learner_info.get("total_loss", None)

        print(f"\nIteration {iteration + 1}/{num_iterations}")
        if loss is not None:
            print(f"  Total Loss: {loss:.6f}")
            if loss < best_loss:
                best_loss = loss

        # Print MARWIL-specific metrics
        for key in ["policy_loss", "vf_loss", "entropy"]:
            if key in learner_info:
                print(f"  {key}: {learner_info[key]:.6f}")

        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_dir = algo.save(checkpoint_dir=output_dir)
            print(f"  Checkpoint saved to: {checkpoint_dir}")

    final_checkpoint = algo.save(checkpoint_dir=output_dir)
    print(f"\nTraining complete!")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Best loss: {best_loss:.6f}")

    algo.stop()
    ray.shutdown()

    return final_checkpoint


def train_marwil_with_tune(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    output_dir="marwil_tune_results",
):
    """Train MARWIL with hyperparameter tuning."""
    if not ray.is_initialized():
        ray.init()

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    if encoding_type == "handcrafted":
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )
    else:
        observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(200,), dtype=np.float32
        )

    action_space = gym.spaces.Discrete(4)

    config = MARWILConfig()

    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Hyperparameter search space
    config.training(
        beta=tune.grid_search(
            [0.0, 0.5, 1.0, 2.0]
        ),  # Test different advantage scalings
        lr=tune.grid_search([1e-3, 1e-4, 1e-5]),
        gamma=0.99,
        train_batch_size_per_learner=tune.choice([256, 512, 1024]),
        vf_coeff=1.0,
        grad_clip=40.0,
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

    print("Starting MARWIL hyperparameter tuning...")

    tuner = tune.Tuner(
        "MARWIL",
        param_space=config.to_dict(),
        run_config=train.RunConfig(
            stop={"training_iteration": num_iterations},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            storage_path=output_dir,
            name="marwil_tuning",
        ),
    )

    results = tuner.fit()

    print("\nTuning complete!")

    best_result = results.get_best_result(metric="info/learner/total_loss", mode="min")
    print(f"\nBest configuration:")
    print(f"  Beta: {best_result.config['beta']}")
    print(f"  Learning rate: {best_result.config['lr']}")
    print(f"  Batch size: {best_result.config['train_batch_size_per_learner']}")

    ray.shutdown()

    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MARWIL model using Ray RLlib")

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
        "--beta",
        type=float,
        default=1.0,
        help="Advantage scaling (0=BC, higher=more advantage-weighted)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="marwil_results",
        help="Output directory for checkpoints and results",
    )

    parser.add_argument(
        "--use_tune",
        action="store_true",
        help="Use Ray Tune for hyperparameter tuning",
    )

    args = parser.parse_args()

    if args.use_tune:
        train_marwil_with_tune(
            data_dir=args.data_dir,
            encoding_type=args.encoding_type,
            num_iterations=args.num_iterations,
            output_dir=args.output_dir,
        )
    else:
        train_marwil(
            data_dir=args.data_dir,
            encoding_type=args.encoding_type,
            num_iterations=args.num_iterations,
            checkpoint_freq=args.checkpoint_freq,
            beta=args.beta,
            lr=args.lr,
            output_dir=args.output_dir,
        )
