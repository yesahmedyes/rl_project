import argparse
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
from tqdm import trange
from typing import Optional

from ludo import Ludo
from milestone2 import Policy_Milestone2
from policy_heuristic import Policy_Heuristic
from policy_ppo import PolicyPPO, calculate_dense_reward
from policy_random import Policy_Random
from utils import ensure_directories


def select_opponent(opponents):
    return random.choice(opponents)


def train_ppo(
    n_episodes: int = 20000,
    update_timesteps: int = 2048,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    mini_batch_size: int = 256,
    ppo_epochs: int = 4,
    save_path: str = "models/policy_ppo.pth",
    load_path: Optional[str] = None,
    dense_rewards: bool = True,
    eval_window: int = 200,
    seed: int = 42,
):
    ensure_directories()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = PolicyPPO(
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        update_timesteps=update_timesteps,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        policy_path=save_path,
        training_mode=True,
    )

    if load_path:
        agent.load(load_path)
        print(f"✅ Loaded PPO weights from {load_path}")

    opponents = [
        ("Random", Policy_Random()),
        ("Heuristic", Policy_Heuristic()),
        ("Milestone2", Policy_Milestone2()),
    ]

    moving_rewards = deque(maxlen=eval_window)
    moving_wins = deque(maxlen=eval_window)
    opponent_histories = {name: deque(maxlen=eval_window) for name, _ in opponents}
    best_win_rate = 0.0
    if save_path.endswith(".pth"):
        best_model_path = save_path.replace(".pth", "_best.pth")
    else:
        best_model_path = f"{save_path}_best"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"ppo_training_{timestamp}.csv")

    with open(log_path, "w") as log_file:
        log_file.write(
            "episode,opponent,episode_reward,agent_won,win_rate_overall,win_rate_random,win_rate_heuristic,win_rate_milestone2,avg_external_win,reward_avg,buffer_size\n"
        )

    progress = trange(1, n_episodes + 1, desc="Training PPO")
    for episode in progress:
        env = Ludo()
        state = env.reset()
        agent_player = episode % 2
        player_turn = state[4]

        opponent_name, opponent_policy = select_opponent(opponents)
        if agent_player == 0:
            policies = [agent, opponent_policy]
        else:
            policies = [opponent_policy, agent]

        episode_reward = 0.0
        agent_won = False

        while not env.terminated:
            action_space = env.get_action_space()
            if not action_space:
                state = env.step(None)
                player_turn = state[4]
                continue

            if player_turn == agent_player:
                action, info = agent.act(state, action_space, deterministic=False)
                if action is None:
                    state = env.step(None)
                    player_turn = state[4]
                    continue

                next_state = env.step(action)
                terminated = next_state[3]
                winner = next_state[4] if terminated else None
                agent_won_step = terminated and winner == agent_player

                if dense_rewards:
                    reward = calculate_dense_reward(
                        state, next_state, terminated, agent_won_step, agent_player
                    )
                else:
                    reward = 1.0 if agent_won_step else 0.0

                agent.store_transition(
                    state_tensor=info["state_tensor"],
                    action_idx=info["action_idx"],
                    log_prob=info["log_prob"],
                    value=info["value"],
                    reward=reward,
                    done=terminated,
                    action_mask=info["action_mask"],
                )

                episode_reward += reward

                if agent.ready_to_update():
                    bootstrap_value = (
                        0.0 if terminated else agent.evaluate_state_value(next_state)
                    )
                    update_stats = agent.update(last_value=bootstrap_value)
                    progress.set_postfix(
                        {
                            "pol_loss": f"{update_stats['policy_loss']:.3f}",
                            "val_loss": f"{update_stats['value_loss']:.3f}",
                        }
                    )

                state = next_state
                player_turn = next_state[4]
                if terminated:
                    agent_won = agent_won_step
            else:
                opponent_action = policies[player_turn].get_action(state, action_space)
                state = env.step(opponent_action)
                player_turn = state[4]

        moving_rewards.append(episode_reward)
        moving_wins.append(1 if agent_won else 0)

        opponent_histories[opponent_name].append(1 if agent_won else 0)

        avg_reward = np.mean(moving_rewards)
        win_rate = np.mean(moving_wins) if moving_wins else 0.0

        def get_history_mean(name):
            history = opponent_histories.get(name, [])
            return float(np.mean(history)) if len(history) > 0 else 0.0

        win_rate_random = get_history_mean("Random")
        win_rate_heuristic = get_history_mean("Heuristic")
        win_rate_milestone2 = get_history_mean("Milestone2")
        external_average = np.mean(
            [win_rate_random, win_rate_heuristic, win_rate_milestone2]
        )

        with open(log_path, "a") as log_file:
            log_file.write(
                f"{episode},{opponent_name},{episode_reward:.2f},{int(agent_won)},{win_rate:.3f},{win_rate_random:.3f},{win_rate_heuristic:.3f},{win_rate_milestone2:.3f},{external_average:.3f},{avg_reward:.3f},{len(agent.buffer)}\n"
            )

        progress.set_postfix(
            {
                "win_rate": f"{win_rate:.2%}",
                "avg_reward": f"{avg_reward:.2f}",
                "ext_win": f"{external_average:.2%}",
            }
        )

        if external_average > best_win_rate and episode >= eval_window:
            best_win_rate = external_average
            agent.save(best_model_path)
            print(
                f"\n💾 New best model saved (external win {best_win_rate:.2%}) at episode {episode}"
            )

    if len(agent.buffer) > 0:
        agent.update(last_value=0.0)

    # Final save
    agent.save(save_path)
    print(f"\n✅ Training complete. Final weights saved to {save_path}")
    print(
        f"⭐ Best external model stored at {best_model_path} with win rate {best_win_rate:.2%}"
    )
    print(f"📄 Training log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Ludo")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--update-steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--mini-batch", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="models/policy_ppo.pth")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--sparse", action="store_true", help="use sparse rewards")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_ppo(
        n_episodes=args.episodes,
        update_timesteps=args.update_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        mini_batch_size=args.mini_batch,
        ppo_epochs=args.ppo_epochs,
        save_path=args.save_path,
        load_path=args.load_path,
        dense_rewards=not args.sparse,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
