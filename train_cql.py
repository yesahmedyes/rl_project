import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.cql import CQLConfig


def checkpoint_to_path(ckpt_obj):
    if isinstance(ckpt_obj, str):
        return ckpt_obj

    inner_ckpt = getattr(ckpt_obj, "checkpoint", None)
    if inner_ckpt is not None:
        ckpt_obj = inner_ckpt

    path_attr = getattr(ckpt_obj, "path", None)
    if path_attr:
        return str(path_attr)

    if hasattr(ckpt_obj, "to_directory"):
        try:
            return ckpt_obj.to_directory()
        except Exception:
            pass

    return str(ckpt_obj)


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


def train_cql(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=200,
    checkpoint_freq=20,
    output_dir="cql_results",
    learning_rate=3e-4,
    train_batch_size=256,
    min_q_weight=5.0,
    bc_iters=20000,
    temperature=1.0,
    gamma=0.99,
    tau=0.005,
    model_layers=None,
):
    if not ray.is_initialized():
        ray.init()

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    json_files = [
        str(p.resolve())
        for p in data_path.glob("*.json")
        if p.name != "dataset_info.json"
    ]

    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_path}")

    print(f"Training CQL on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")

    if encoding_type == "handcrafted":
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(70,),
            dtype=np.float32,
        )
    else:
        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(946,),
            dtype=np.float32,
        )

    action_space = gym.spaces.Discrete(12)

    config = CQLConfig()
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    # Configure model architecture for both actor and critics
    hidden_layers = model_layers or [256, 256]
    config.model.update({"fcnet_hiddens": hidden_layers})

    if hasattr(config, "q_model_config") and isinstance(config.q_model_config, dict):
        config.q_model_config.update({"fcnet_hiddens": hidden_layers})

    if hasattr(config, "policy_model_config") and isinstance(
        config.policy_model_config, dict
    ):
        config.policy_model_config.update({"fcnet_hiddens": hidden_layers})

    config.rollouts(num_rollout_workers=0)

    config.training(
        lr=learning_rate,
        train_batch_size=train_batch_size,
        gamma=gamma,
        tau=tau,
        min_q_weight=min_q_weight,
        bc_iters=bc_iters,
        temperature=temperature,
    )

    config.offline_data(
        input_=json_files,
        dataset_num_iters_per_learner=1,
    )

    config.evaluation(evaluation_interval=None)

    num_samples, num_episodes = count_dataset_samples(data_path)

    print("\nDataset Statistics:")
    print(f"  Total samples: {num_samples}")
    print(f"  Total episodes: {num_episodes}")

    if num_episodes > 0:
        print(f"  Average episode length: {num_samples / num_episodes:.2f}")

    print("\nCQL Configuration:")
    print(f"  Learning rate: {config.lr}")
    print(f"  Train batch size: {config.train_batch_size}")
    print(f"  Min Q weight: {config.min_q_weight}")
    print(f"  BC iters: {config.bc_iters}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Tau: {config.tau}")
    print(f"  Number of training iterations: {num_iterations}")
    print(f"  Model hidden sizes: {hidden_layers}")

    algo = config.build()

    print("\nStarting training...")

    best_loss = float("inf")
    loss_history = []

    for iteration in range(num_iterations):
        result = algo.train()

        info = result.get("info", {})
        learner_info = info.get("learner", {})

        metrics = {}
        if isinstance(learner_info, dict) and learner_info:
            first_value = next(iter(learner_info.values()))
            metrics = first_value if isinstance(first_value, dict) else learner_info

        loss = metrics.get("total_loss", metrics.get("loss", None))
        cql_loss = metrics.get("cql_loss", metrics.get("min_q_loss", None))
        actor_loss = metrics.get("actor_loss", None)
        critic_loss = metrics.get("critic_loss", None)

        print(f"\nIteration {iteration + 1}/{num_iterations}")

        if loss is not None:
            print(f"  Total loss: {float(loss):.6f}")
            best_loss = min(best_loss, float(loss))
            loss_history.append(
                {
                    "iteration": iteration + 1,
                    "loss": float(loss),
                    "best_loss": float(best_loss),
                    "cql_loss": float(cql_loss) if cql_loss is not None else None,
                    "actor_loss": float(actor_loss) if actor_loss is not None else None,
                    "critic_loss": float(critic_loss)
                    if critic_loss is not None
                    else None,
                }
            )

        if cql_loss is not None:
            print(f"  Conservative loss: {float(cql_loss):.6f}")
        if actor_loss is not None:
            print(f"  Actor loss: {float(actor_loss):.6f}")
        if critic_loss is not None:
            print(f"  Critic loss: {float(critic_loss):.6f}")

        extra_keys = ("alpha_loss", "min_q_weight")
        for key in extra_keys:
            if key in metrics and isinstance(metrics[key], (int, float)):
                print(f"  {key}: {metrics[key]}")

        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_obj = algo.save(checkpoint_dir=str(output_path))
            checkpoint_path = checkpoint_to_path(checkpoint_obj)
            print(f"  Checkpoint saved to: {checkpoint_path}")

    final_checkpoint_obj = algo.save(checkpoint_dir=str(output_path))
    final_checkpoint_path = checkpoint_to_path(final_checkpoint_obj)
    history_path = output_path / "loss_history.jsonl"

    with open(history_path, "w") as f:
        for entry in loss_history:
            f.write(json.dumps(entry) + "\n")

    best_loss_value = best_loss if best_loss < float("inf") else None

    print("\nTraining complete!")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")

    if best_loss_value is not None:
        print(f"Best loss achieved: {best_loss_value:.6f}")
    else:
        print("Best loss achieved: N/A (loss not reported)")

    print(f"Loss history logged to: {history_path}")

    algo.stop()
    ray.shutdown()

    return final_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Conservative Q-Learning model using Ray RLlib"
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
        default=200,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=20,
        help="Checkpoint frequency",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cql_results",
        help="Output directory for checkpoints and results",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for CQL",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Training batch size",
    )

    parser.add_argument(
        "--min_q_weight",
        type=float,
        default=5.0,
        help="Weight for conservative Q loss",
    )

    parser.add_argument(
        "--bc_iters",
        type=int,
        default=20000,
        help="Number of BC pretraining iterations",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for advantage-weighted actor updates",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Target network smoothing coefficient",
    )
    parser.add_argument(
        "--model_layers",
        type=json.loads,
        default="[256, 256]",
        help="JSON list of hidden sizes",
    )

    args = parser.parse_args()

    train_cql(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        min_q_weight=args.min_q_weight,
        bc_iters=args.bc_iters,
        temperature=args.temperature,
        gamma=args.gamma,
        tau=args.tau,
        model_layers=args.model_layers,
    )
