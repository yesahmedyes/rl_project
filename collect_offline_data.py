import numpy as np
import json
import argparse
import multiprocessing as mp
import atexit
from pathlib import Path
from tqdm import tqdm
from env.ludo_gym_env import LudoGymEnv
from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic
from policies.milestone2 import Policy_Milestone2

# Worker-scoped globals for multiprocessing
_worker_env = None
_worker_policy = None
_worker_max_steps = None


class JsonEpisodeWriter:
    def __init__(self, output_dir: Path, episodes_per_file: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file_counter = 0
        self._episodes_per_file = episodes_per_file
        self._buffer = []

    def _flush(self):
        if not self._buffer:
            return

        path = self.output_dir / f"episodes_{self._file_counter:06d}.json"
        self._file_counter += 1

        with path.open("w") as f:
            f.write(json.dumps(self._buffer))

        self._buffer = []

    def write(self, episode_data: dict):
        self._buffer.append(episode_data)

        if len(self._buffer) >= self._episodes_per_file:
            self._flush()

    def close(self):
        self._flush()


def _init_worker(encoding_type, opponent_type, expert_type, reward_type, max_steps):
    global _worker_env, _worker_policy, _worker_max_steps

    _worker_env = LudoGymEnv(
        encoding_type=encoding_type,
        opponent_type=opponent_type,
        agent_player=None,
        use_dense_reward=reward_type == "dense",
    )

    if expert_type == "random":
        _worker_policy = Policy_Random()
    elif expert_type == "heuristic":
        _worker_policy = Policy_Heuristic()
    elif expert_type == "milestone2":
        _worker_policy = Policy_Milestone2()
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")

    _worker_max_steps = max_steps

    atexit.register(_worker_env.close)


def _collect_episode_worker(episode_id):
    return collect_episode(
        _worker_env, _worker_policy, episode_id, max_steps=_worker_max_steps
    )


def collect_episode(env, policy, episode_id, max_steps=1000):
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []

    obs, info = env.reset()
    observations.append(obs.tolist())
    infos.append(
        {"action_mask": info["action_mask"].tolist() if "action_mask" in info else None}
    )

    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated and step_count < max_steps:
        # Get action mask
        action_mask = info.get("action_mask", np.ones(env.action_space.n))

        # Get valid actions
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            action = 0  # Dummy action
        else:
            # Get state from environment for policy
            state = env.current_state

            # Convert valid discrete actions to tuple format for policy
            action_space = env.env.get_action_space()

            # Get action from policy
            policy_action = policy.get_action(state, action_space)

            # Convert policy action (tuple) to discrete action
            if policy_action is None:
                action = valid_actions[0] if len(valid_actions) > 0 else 0
            else:
                action = env._tuple_to_action(policy_action)

                # Ensure action is valid
                if action not in valid_actions:
                    action = valid_actions[0] if len(valid_actions) > 0 else 0

        # Take action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store transition
        actions.append(int(action))
        rewards.append(float(reward))
        observations.append(next_obs.tolist())
        dones.append(bool(terminated or truncated))
        infos.append(
            {
                "action_mask": info["action_mask"].tolist()
                if "action_mask" in info
                else None,
                "agent_won": info.get("agent_won", False),
            }
        )

        obs = next_obs
        step_count += 1

    timesteps = list(range(len(actions)))

    episode_data = {
        "obs": observations[:-1],
        "new_obs": observations[1:],
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "infos": infos[1:],
        "eps_id": [episode_id] * len(actions),
        "agent_index": [0] * len(actions),
        "unroll_id": [episode_id] * len(actions),
        "t": timesteps,
    }

    episode_stats = {
        "episode_length": step_count,
        "episode_return": sum(rewards),
        "agent_won": infos[-1].get("agent_won", False) if infos else False,
    }

    return episode_data, episode_stats


def collect_dataset(
    num_episodes,
    encoding_type="handcrafted",
    opponent_types=("random", "heuristic", "milestone2"),
    expert_type="heuristic",
    reward_type="dense",
    output_dir="offline_data",
    max_steps_per_episode=1000,
    num_workers=1,
):
    if isinstance(opponent_types, str):
        opponent_types = [opponent_types]

    opponent_types = list(opponent_types)

    # Create output directory
    output_path = Path(output_dir) / f"{expert_type}_{encoding_type}_{reward_type}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Collecting {num_episodes} episodes per opponent using {expert_type} policy..."
    )
    print(f"Encoding: {encoding_type}, Opponents: {opponent_types}")
    print(f"Output directory: {output_path}")
    print(f"Workers: {num_workers}")
    print("Episodes per file: 1000")

    writer = JsonEpisodeWriter(output_path, episodes_per_file=1000)

    total_return = 0
    total_steps = 0
    win_count = 0
    total_episodes = 0

    for opponent_type in opponent_types:
        print(f"\n==> Collecting vs opponent: {opponent_type}")

        if num_workers <= 1:
            env = LudoGymEnv(
                encoding_type=encoding_type,
                opponent_type=opponent_type,
                agent_player=None,
                use_dense_reward=reward_type == "dense",
            )

            if expert_type == "random":
                expert_policy = Policy_Random()
            elif expert_type == "heuristic":
                expert_policy = Policy_Heuristic()
            elif expert_type == "milestone2":
                expert_policy = Policy_Milestone2()
            else:
                raise ValueError(f"Unknown expert_type: {expert_type}")

            for local_idx in tqdm(
                range(num_episodes), desc=f"Collecting vs {opponent_type}"
            ):
                episode_id = total_episodes + local_idx
                episode_data, episode_stats = collect_episode(
                    env,
                    expert_policy,
                    episode_id=episode_id,
                    max_steps=max_steps_per_episode,
                )

                episode_data["opponent_type"] = opponent_type
                episode_data["expert_type"] = expert_type

                writer.write(episode_data)
                total_return += episode_stats["episode_return"]
                total_steps += episode_stats["episode_length"]
                if episode_stats["agent_won"]:
                    win_count += 1

            env.close()
        else:
            ctx = mp.get_context("spawn")

            with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(
                    encoding_type,
                    opponent_type,
                    expert_type,
                    reward_type,
                    max_steps_per_episode,
                ),
            ) as pool:
                start_id = total_episodes
                for episode_data, episode_stats in tqdm(
                    pool.imap_unordered(
                        _collect_episode_worker,
                        range(start_id, start_id + num_episodes),
                    ),
                    total=num_episodes,
                    desc=f"Collecting vs {opponent_type}",
                ):
                    episode_data["opponent_type"] = opponent_type
                    episode_data["expert_type"] = expert_type

                    writer.write(episode_data)
                    total_return += episode_stats["episode_return"]
                    total_steps += episode_stats["episode_length"]
                    if episode_stats["agent_won"]:
                        win_count += 1

        total_episodes += num_episodes

    writer.close()

    avg_return = total_return / total_episodes if total_episodes else 0.0
    avg_steps = total_steps / total_episodes if total_episodes else 0.0
    win_rate = (win_count / total_episodes) * 100 if total_episodes else 0.0

    print("\nDataset Collection Complete!")
    print(f"Total episodes: {total_episodes}")
    print(f"Average return: {avg_return:.2f}")
    print(f"Average episode length: {avg_steps:.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Data saved to: {output_path}")

    info_file = output_path / "dataset_info.json"

    with open(info_file, "w") as f:
        json.dump(
            {
                "num_episodes_per_opponent": num_episodes,
                "total_episodes": total_episodes,
                "encoding_type": encoding_type,
                "opponent_types": opponent_types,
                "expert_type": expert_type,
                "reward_type": reward_type,
                "avg_return": avg_return,
                "avg_episode_length": avg_steps,
                "win_rate": win_rate,
                "max_steps_per_episode": max_steps_per_episode,
                "episodes_per_file": 1000,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect offline dataset for Ludo")

    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of episodes to collect"
    )

    parser.add_argument(
        "--encoding_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "onehot"],
        help="State encoding type",
    )

    parser.add_argument(
        "--opponent_types",
        type=str,
        nargs="+",
        default=["heuristic"],
        choices=["random", "heuristic", "milestone2"],
        help="Opponent policy types (space separated)",
    )

    parser.add_argument(
        "--expert_type",
        type=str,
        default="heuristic",
        choices=["random", "heuristic", "milestone2"],
        help="Expert policy to collect from",
    )

    parser.add_argument(
        "--reward_type",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Reward type used during data collection",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="offline_data",
        help="Output directory for collected data",
    )

    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for data collection",
    )

    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.num_episodes,
        encoding_type=args.encoding_type,
        opponent_types=args.opponent_types,
        expert_type=args.expert_type,
        reward_type=args.reward_type,
        output_dir=args.output_dir,
        max_steps_per_episode=args.max_steps,
        num_workers=args.num_workers,
    )
