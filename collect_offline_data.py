import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from ray.rllib.offline.json_writer import JsonWriter
from env.ludo_gym_env import LudoGymEnv
from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic
from policies.milestone2 import Policy_Milestone2


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
    output_dir="offline_data",
    max_steps_per_episode=1000,
):
    # Create output directory
    output_path = Path(output_dir) / f"{expert_type}_{encoding_type}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = LudoGymEnv(
        encoding_type=encoding_type,
        opponent_type=opponent_type,
        agent_player=None,  # Randomize which player the agent controls
        use_dense_reward=True,
    )

    # Create expert policy
    if expert_type == "random":
        expert_policy = Policy_Random()
    elif expert_type == "heuristic":
        expert_policy = Policy_Heuristic()
    elif expert_type == "milestone2":
        expert_policy = Policy_Milestone2()
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")

    print(f"Collecting {num_episodes} episodes using {expert_type} policy...")
    print(f"Encoding: {encoding_type}, Opponent: {opponent_type}")
    print(f"Output directory: {output_path}")

    # Initialize RLlib JsonWriter
    writer = JsonWriter(str(output_path))

    total_return = 0
    total_steps = 0
    win_count = 0

    for episode_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Collect episode
        episode_data, episode_stats = collect_episode(
            env, expert_policy, episode_id=episode_idx, max_steps=max_steps_per_episode
        )

        # Write episode using JsonWriter
        writer.write(episode_data)

        # Update statistics
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
                "avg_return": avg_return,
                "avg_episode_length": avg_steps,
                "win_rate": win_rate,
                "max_steps_per_episode": max_steps_per_episode,
            },
            f,
            indent=2,
        )

    env.close()


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
        "--output_dir",
        type=str,
        default="offline_data",
        help="Output directory for collected data",
    )

    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )

    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.num_episodes,
        encoding_type=args.encoding_type,
        opponent_type=args.opponent_type,
        expert_type=args.expert_type,
        output_dir=args.output_dir,
        max_steps_per_episode=args.max_steps,
    )
