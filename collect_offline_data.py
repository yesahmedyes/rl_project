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
    """Minimal JSON episode writer to replace Ray's JsonWriter."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def write(self, episode_data: dict):
        path = self.output_dir / f"episode_{self._counter:06d}.json"
        self._counter += 1
        with path.open("w") as f:
            f.write(json.dumps(episode_data))


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

    # Ensure worker env is cleaned up when the process exits
    atexit.register(_worker_env.close)


def _collect_episode_worker(episode_id):
    """Collect a single episode using worker-local env/policy."""
    return collect_episode(
        _worker_env, _worker_policy, episode_id, max_steps=_worker_max_steps
    )


def collect_episode(env, policy, episode_id, max_steps=1000):
    observations = []
    actions = []
    rewards = []
    dones = []  # RLlib uses "dones" instead of "terminateds"
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
            # No valid actions, environment should auto-pass
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
        dones.append(bool(terminated or truncated))  # RLlib combines both
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

    # Create episode data in RLlib JsonWriter format
    timesteps = list(range(len(actions)))

    episode_data = {
        "obs": observations[:-1],  # All observations except the last
        "new_obs": observations[1:],  # All observations except the first
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "infos": infos[1:],  # Match with actions (exclude first info)
        "eps_id": [episode_id] * len(actions),
        "agent_index": [0] * len(actions),
        "unroll_id": [episode_id] * len(actions),
        "t": timesteps,  # Timestep indices
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
    opponent_type="heuristic",
    expert_type="heuristic",
    reward_type="dense",
    output_dir="offline_data",
    max_steps_per_episode=1000,
    num_workers=1,
):
    # Create output directory
    output_path = (
        Path(output_dir)
        / f"{expert_type}_{encoding_type}_{reward_type}"
        / f"{expert_type}_vs_{opponent_type}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {num_episodes} episodes using {expert_type} policy...")
    print(f"Encoding: {encoding_type}, Opponent: {opponent_type}")
    print(f"Output directory: {output_path}")
    print(f"Workers: {num_workers}")

    # Minimal JSON writer (one file per episode)
    writer = JsonEpisodeWriter(output_path)

    total_return = 0
    total_steps = 0
    win_count = 0

    if num_workers <= 1:
        # Create environment and policy in the main process
        env = LudoGymEnv(
            encoding_type=encoding_type,
            opponent_type=opponent_type,
            agent_player=None,  # Randomize which player the agent controls
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

        for episode_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
            episode_data, episode_stats = collect_episode(
                env,
                expert_policy,
                episode_id=episode_idx,
                max_steps=max_steps_per_episode,
            )

            writer.write(episode_data)
            total_return += episode_stats["episode_return"]
            total_steps += episode_stats["episode_length"]
            if episode_stats["agent_won"]:
                win_count += 1

        env.close()
    else:
        # Use multiple worker processes; keep writing in the main process
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
            for episode_data, episode_stats in tqdm(
                pool.imap_unordered(_collect_episode_worker, range(num_episodes)),
                total=num_episodes,
                desc="Collecting episodes",
            ):
                writer.write(episode_data)
                total_return += episode_stats["episode_return"]
                total_steps += episode_stats["episode_length"]
                if episode_stats["agent_won"]:
                    win_count += 1

    # Print statistics
    avg_return = total_return / num_episodes
    avg_steps = total_steps / num_episodes
    win_rate = (win_count / num_episodes) * 100

    print("\nDataset Collection Complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Average return: {avg_return:.2f}")
    print(f"Average episode length: {avg_steps:.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Data saved to: {output_path}")

    # Save dataset info
    info_file = output_path / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(
            {
                "num_episodes": num_episodes,
                "encoding_type": encoding_type,
                "opponent_type": opponent_type,
                "expert_type": expert_type,
                "reward_type": reward_type,
                "avg_return": avg_return,
                "avg_episode_length": avg_steps,
                "win_rate": win_rate,
                "max_steps_per_episode": max_steps_per_episode,
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
        "--opponent_type",
        type=str,
        default="heuristic",
        choices=["random", "heuristic", "milestone2"],
        help="Opponent policy type",
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
        opponent_type=args.opponent_type,
        expert_type=args.expert_type,
        reward_type=args.reward_type,
        output_dir=args.output_dir,
        max_steps_per_episode=args.max_steps,
        num_workers=args.num_workers,
    )
