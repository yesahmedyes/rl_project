import argparse
import os
from typing import Callable, Sequence, Tuple

import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ludo_maskable_env import LudoMaskableEnv
from win_rate_callback import WinRateCallback


def mask_fn(env: LudoMaskableEnv) -> np.ndarray:
    return env.get_action_mask()


def make_env(
    opponents: Sequence[str],
    dense_rewards: bool,
    alternate_start: bool,
    base_seed: int,
    rank: int,
) -> Callable[[], LudoMaskableEnv]:
    def _init():
        env = LudoMaskableEnv(
            opponents=opponents,
            dense_rewards=dense_rewards,
            alternate_start=alternate_start,
            seed=base_seed + rank,
        )

        env = ActionMasker(env, mask_fn)

        return env

    return _init


def parse_arch(text: str) -> Tuple[int, ...]:
    layers = [p.strip() for p in text.split(",") if p.strip()]

    if not layers:
        raise argparse.ArgumentTypeError("net-arch must list hidden sizes")

    try:
        return tuple(int(p) for p in layers)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "net-arch must be comma separated integers"
        ) from exc


def train_maskable_ppo(args):
    set_random_seed(args.seed)

    opponents = tuple(op.strip() for op in args.opponents.split(",") if op.strip())
    dense_rewards = not args.sparse

    env_fns = [
        make_env(opponents, dense_rewards, not args.disable_alt_start, args.seed, rank)
        for rank in range(args.n_envs)
    ]

    if args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    policy_kwargs = dict(net_arch=dict(pi=list(args.net_arch), vf=list(args.net_arch)))

    # Setup TensorBoard logging
    if args.tensorboard_log:
        os.makedirs(args.tensorboard_log, exist_ok=True)
        print(f"📊 TensorBoard logs will be saved to {args.tensorboard_log}")

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
    )

    if args.bc_path and os.path.exists(args.bc_path):
        print(f"🔁 Loading BC weights from {args.bc_path}")
        bc_policy = MaskableActorCriticPolicy.load(args.bc_path, device="auto")
        model.policy.load_state_dict(bc_policy.state_dict())

    callbacks = []
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // args.n_envs),
                save_path=args.checkpoint_dir,
                name_prefix="maskable_ppo",
            )
        )

    # Add win rate evaluation callback
    if args.eval_freq > 0:
        callbacks.append(
            WinRateCallback(
                eval_freq=args.eval_freq,
                n_eval_games=args.n_eval_games,
                dense_rewards=dense_rewards,
                log_path=args.win_rate_log,
                verbose=1,
            )
        )

    callback_list = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=args.total_timesteps, callback=callback_list)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.save(args.save_path)
    print(f"✅ Saved trained MaskablePPO to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO with SB3-Contrib")
    parser.add_argument("--opponents", type=str, default="heuristic,random,milestone2")
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of parallel environments",
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", type=parse_arch, default=parse_arch("256,256"))
    parser.add_argument("--save-path", type=str, default="models/maskable_ppo.zip")
    parser.add_argument("--bc-path", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Frequency to evaluate win rates",
    )
    parser.add_argument(
        "--n-eval-games",
        type=int,
        default=100,
        help="Number of games to play per opponent during evaluation",
    )
    parser.add_argument(
        "--win-rate-log",
        type=str,
        default="logs/win_rates.csv",
        help="Path to save win rate CSV file",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/tensorboard",
        help="Directory for TensorBoard logs. View with: tensorboard --logdir=logs/tensorboard",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'auto'). Default: 'auto'",
    )
    parser.add_argument(
        "--sparse", action="store_true", help="use sparse win/loss rewards"
    )
    parser.add_argument(
        "--disable-alt-start",
        action="store_true",
        help="disable alternating which player the agent controls",
    )

    args = parser.parse_args()
    train_maskable_ppo(args)


if __name__ == "__main__":
    main()
