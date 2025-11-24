import argparse
import glob
import os
import pickle
from typing import Tuple

import numpy as np
from gymnasium import spaces
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from features import MAX_ACTIONS, STATE_DIM


def load_demonstrations(demonstrations_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    pattern = os.path.join(demonstrations_dir, "*.pkl")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No demonstration files found in {demonstrations_dir}. "
            "Run collect_demonstrations.py first."
        )

    all_states, all_actions = [], []
    for filepath in files:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        all_states.append(data["states"])
        all_actions.append(data["actions"])

    states = np.concatenate(all_states, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.int64)
    return states, actions


def build_transitions(states: np.ndarray, actions: np.ndarray) -> Transitions:
    n = len(states)
    return Transitions(
        obs=states,
        acts=actions,
        next_obs=states,
        dones=np.zeros(n, dtype=bool),
        infos=[{} for _ in range(n)],
    )


def make_policy(observation_space, action_space, learning_rate, net_arch):
    def lr_schedule(_):
        return learning_rate

    return MaskableActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        net_arch=net_arch,
    )


def pretrain_with_bc(
    demonstrations_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_path: str,
    net_arch,
):
    states, actions = load_demonstrations(demonstrations_dir)
    transitions = build_transitions(states, actions)

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
    action_space = spaces.Discrete(MAX_ACTIONS)

    policy = make_policy(obs_space, action_space, learning_rate, net_arch)
    bc_trainer = BC(
        observation_space=obs_space,
        action_space=action_space,
        policy=policy,
        demonstrations=transitions,
        batch_size=batch_size,
    )

    bc_trainer.train(n_epochs=epochs)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    policy.save(save_path)
    print(f"✅ Saved BC policy to {save_path}")


def parse_arch(text: str):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "net-arch must be comma separated integers, e.g. 256,256"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "net-arch must be comma separated integers"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral cloning using imitation + MaskableActorCritic policy"
    )
    parser.add_argument("--demonstrations-dir", type=str, default="demonstrations")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--save-path", type=str, default="models/pretrained_bc_policy.zip"
    )
    parser.add_argument("--net-arch", type=parse_arch, default=parse_arch("256,256"))

    args = parser.parse_args()

    pretrain_with_bc(
        demonstrations_dir=args.demonstrations_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        net_arch=args.net_arch,
    )


if __name__ == "__main__":
    main()
