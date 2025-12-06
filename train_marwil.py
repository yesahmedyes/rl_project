import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.marwil import MARWILConfig


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


def train_marwil(
    data_dir,
    encoding_type="handcrafted",
    num_iterations=100,
    checkpoint_freq=10,
    output_dir="marwil_results",
    learning_rate=1e-5,
    train_batch_size=2000,
    beta=1.0,
    gamma=0.99,
    model_layers=None,
    num_gpus=1,
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

    print(f"Training MARWIL on data from: {data_path}")
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

    config = MARWILConfig()
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    config.environment(
        observation_space=observation_space,
        action_space=action_space,
    )

    hidden_layers = model_layers or [256, 256]
    config.model.update({"fcnet_hiddens": hidden_layers})

    # Request GPU resources if available
    config.resources(num_gpus=num_gpus)

    config.training(
        lr=learning_rate,
        train_batch_size_per_learner=train_batch_size,
        beta=beta,
        gamma=gamma,
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

    print("\nMARWIL Configuration:")
    print(f"  Learning rate: {config.lr}")
    print(f"  Train batch size per learner: {config.train_batch_size_per_learner}")
    print(f"  Beta (advantage scale): {config.beta}")
    print(f"  Gamma: {config.gamma}")
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
        policy_loss = metrics.get("policy_loss", None)
        vf_loss = metrics.get("vf_loss", None)

        print(f"\nIteration {iteration + 1}/{num_iterations}")

        if loss is not None:
            print(f"  Total loss: {float(loss):.6f}")
            best_loss = min(best_loss, float(loss))
            loss_history.append(
                {
                    "iteration": iteration + 1,
                    "loss": float(loss),
                    "best_loss": float(best_loss),
                    "policy_loss": float(policy_loss)
                    if policy_loss is not None
                    else None,
                    "vf_loss": float(vf_loss) if vf_loss is not None else None,
                }
            )

        if policy_loss is not None:
            print(f"  Policy loss: {float(policy_loss):.6f}")
        if vf_loss is not None:
            print(f"  Value loss: {float(vf_loss):.6f}")

        for key, value in metrics.items():
            if key in ("total_loss", "loss", "policy_loss", "vf_loss"):
                continue
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")

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
        description="Train MARWIL offline RL model using Ray RLlib (old API stack)"
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
        default="marwil_results",
        help="Output directory for checkpoints and results",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for MARWIL",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2000,
        help="Train batch size per learner",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Advantage scaling term; 0 reduces to behavior cloning",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    parser.add_argument(
        "--model_layers",
        type=json.loads,
        default="[256, 256]",
        help="JSON list of hidden sizes",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to allocate to the trainer",
    )

    args = parser.parse_args()

    train_marwil(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        beta=args.beta,
        gamma=args.gamma,
        model_layers=args.model_layers,
        num_gpus=args.num_gpus,
    )
