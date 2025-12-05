import numpy as np
import argparse
from pathlib import Path
import gymnasium as gym
from ray.rllib.algorithms.cql import CQLConfig
from ray import tune, train
import ray


def train_cql(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    checkpoint_freq=10,
    bc_iters=0,
    lr=3e-4,
    output_dir="cql_results",
):
    """
    Train a CQL (Conservative Q-Learning) model using Ray RLlib's new API stack.

    CQL is an offline RL algorithm that learns conservative Q-functions to avoid
    overestimation bias on out-of-distribution actions.

    Args:
        data_dir: Directory containing offline data
        encoding_type: State encoding type ('handcrafted' or 'onehot')
        num_iterations: Number of training iterations
        checkpoint_freq: Checkpoint frequency
        bc_iters: Number of BC iterations before CQL training
        lr: Learning rate
        output_dir: Directory to save training results
    """
    if not ray.is_initialized():
        ray.init()

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    print(f"Training CQL on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")
    print(f"BC iterations: {bc_iters}")

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

    # Create CQL configuration
    config = CQLConfig()

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
        bc_iters=bc_iters,  # Behavior cloning warmup iterations
        temperature=1.0,  # CQL temperature for logsumexp
        lr=lr,
        train_batch_size_per_learner=256,
        gamma=0.99,
        # CQL-specific parameters
        min_q_weight=5.0,  # Weight for CQL conservative penalty
        # Twin Q-networks
        twin_q=True,
        # Target network update
        tau=0.005,
        target_network_update_freq=1,
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

    print("\nCQL Configuration:")
    print(f"  BC iterations: {config.bc_iters}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Min Q weight: {config.min_q_weight}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Twin Q: {config.twin_q}")
    print(f"  Tau: {config.tau}")

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

        # Print CQL-specific metrics
        for key in ["critic_loss", "actor_loss", "alpha_loss", "cql_loss"]:
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


def train_cql_with_tune(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    output_dir="cql_tune_results",
):
    """Train CQL with hyperparameter tuning."""
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

    config = CQLConfig()

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
        bc_iters=tune.choice([0, 20, 50]),
        temperature=tune.grid_search([0.5, 1.0, 2.0]),
        lr=tune.grid_search([1e-3, 3e-4, 1e-4]),
        train_batch_size_per_learner=256,
        gamma=0.99,
        min_q_weight=tune.grid_search([1.0, 5.0, 10.0]),
        twin_q=True,
        tau=0.005,
        target_network_update_freq=1,
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

    print("Starting CQL hyperparameter tuning...")

    tuner = tune.Tuner(
        "CQL",
        param_space=config.to_dict(),
        run_config=train.RunConfig(
            stop={"training_iteration": num_iterations},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            storage_path=output_dir,
            name="cql_tuning",
        ),
    )

    results = tuner.fit()

    print("\nTuning complete!")

    best_result = results.get_best_result(metric="info/learner/total_loss", mode="min")
    print(f"\nBest configuration:")
    print(f"  BC iters: {best_result.config['bc_iters']}")
    print(f"  Temperature: {best_result.config['temperature']}")
    print(f"  Learning rate: {best_result.config['lr']}")
    print(f"  Min Q weight: {best_result.config['min_q_weight']}")

    ray.shutdown()

    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CQL model using Ray RLlib")

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
        "--bc_iters",
        type=int,
        default=0,
        help="Number of BC warmup iterations",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cql_results",
        help="Output directory for checkpoints and results",
    )

    parser.add_argument(
        "--use_tune",
        action="store_true",
        help="Use Ray Tune for hyperparameter tuning",
    )

    args = parser.parse_args()

    if args.use_tune:
        train_cql_with_tune(
            data_dir=args.data_dir,
            encoding_type=args.encoding_type,
            num_iterations=args.num_iterations,
            output_dir=args.output_dir,
        )
    else:
        train_cql(
            data_dir=args.data_dir,
            encoding_type=args.encoding_type,
            num_iterations=args.num_iterations,
            checkpoint_freq=args.checkpoint_freq,
            bc_iters=args.bc_iters,
            lr=args.lr,
            output_dir=args.output_dir,
        )
