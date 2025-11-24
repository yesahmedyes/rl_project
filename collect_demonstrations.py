import numpy as np
import torch
from tqdm import tqdm
import os
import pickle
from datetime import datetime

from features import encode_ludo_state, MAX_ACTIONS
from policy_heuristic import Policy_Heuristic
from policy_random import Policy_Random
from milestone2 import Policy_Milestone2
from ludo import Ludo


def collect_expert_demonstrations(num_episodes=5000, opponent_policy=None, device=None):
    heuristic_policy = Policy_Heuristic()

    expert_data = []

    iterator = tqdm(range(num_episodes), desc="Collecting expert data")

    for episode in iterator:
        env = Ludo()
        state = env.reset()
        player_turn = state[4]

        # Randomly assign which player uses heuristic (for diversity)
        heuristic_player = episode % 2

        while not env.terminated:
            action_space = env.get_action_space()

            if not action_space:
                state = env.step(None)
                player_turn = state[4]
                continue

            # Get action from heuristic if it's heuristic player's turn
            if player_turn == heuristic_player:
                action = heuristic_policy.get_action(state, action_space)

                if action is not None:
                    # Encode state using PPO encoder
                    state_encoded = encode_ludo_state(state)

                    # Convert action to action index
                    dice_idx, goti_idx = action
                    action_idx = dice_idx * 4 + goti_idx
                    action_idx = max(0, min(action_idx, MAX_ACTIONS - 1))

                    expert_data.append((state_encoded, action_idx))
            else:
                # Opponent's turn
                action = opponent_policy.get_action(state, action_space)

            state = env.step(action)
            player_turn = state[4]

    return expert_data


def save_demonstrations(expert_data, filepath):
    """Save demonstrations to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Extract states and actions
    states = np.array([data[0] for data in expert_data], dtype=np.float32)
    actions = np.array([data[1] for data in expert_data], dtype=np.int64)

    # Save as a dictionary
    data_dict = {"states": states, "actions": actions, "num_samples": len(expert_data)}

    with open(filepath, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"✅ Saved {len(expert_data)} demonstrations to {filepath}")


def collect_and_save_demonstrations(
    num_episodes=5000, demonstrations_dir="demonstrations", device=None
):
    # Set device
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    os.makedirs(demonstrations_dir, exist_ok=True)

    # Generate timestamp for this collection run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    opponents = [
        ("random", Policy_Random()),
        ("heuristic", Policy_Heuristic()),
        ("milestone2", Policy_Milestone2()),
    ]

    all_data = []

    for opponent_name, opponent_policy in opponents:
        print(f"Collecting demonstrations against {opponent_name} opponent...")

        expert_data = collect_expert_demonstrations(
            num_episodes=num_episodes,
            opponent_policy=opponent_policy,
            device=device,
        )

        # Save to separate file with timestamp
        filepath = os.path.join(
            demonstrations_dir, f"{opponent_name}_demonstrations_{timestamp}.pkl"
        )

        save_demonstrations(expert_data, filepath)

        all_data.extend(expert_data)
        print()

    # Also save combined file with timestamp
    combined_filepath = os.path.join(
        demonstrations_dir, f"all_demonstrations_{timestamp}.pkl"
    )
    save_demonstrations(all_data, combined_filepath)

    print(f"\n✅ Total demonstrations collected: {len(all_data)}")

    return all_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations and save to files"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of episodes to collect per opponent (default: 5000)",
    )
    parser.add_argument(
        "--demonstrations-dir",
        type=str,
        default="demonstrations",
        help="Directory to save demonstrations (default: demonstrations)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to use (default: cuda:1)",
    )

    args = parser.parse_args()

    collect_and_save_demonstrations(
        num_episodes=args.episodes,
        demonstrations_dir=args.demonstrations_dir,
        device=args.device,
    )
