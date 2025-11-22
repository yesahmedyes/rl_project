"""
Parallel worker functions for distributed episode collection.
Each worker runs episodes and collects trajectories without doing any learning.
"""

import random
import numpy as np
import torch
from ludo import Ludo
from policy_random import Policy_Random
from policy_heuristic import Policy_Heuristic
from models import DuelingDQNNetwork


class InferencePolicy:
    """
    Lightweight inference-only policy for workers.
    No replay buffer, no optimizer, just forward pass.
    """

    def __init__(self, weights, state_dim=12, max_actions=12, device="cpu"):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.max_actions = max_actions
        self.epsilon = weights["epsilon"]

        # Create network and load weights
        self.policy_net = DuelingDQNNetwork(
            state_dim, hidden_dim=256, max_actions=max_actions
        ).to(self.device)
        self.policy_net.load_state_dict(weights["policy_net"])
        self.policy_net.eval()

    def encode_state(self, state):
        """Same encoding as Policy_DQN"""
        gotis_red, gotis_yellow, dice_roll, _, player_turn = state

        if player_turn == 0:
            my_gotis = [g.position for g in gotis_red.gotis]
            opp_gotis = [g.position for g in gotis_yellow.gotis]
        else:
            my_gotis = [g.position for g in gotis_yellow.gotis]
            opp_gotis = [g.position for g in gotis_red.gotis]

        def normalize_position(pos):
            return (pos + 1) / 57.0

        my_positions = [normalize_position(pos) for pos in my_gotis]
        opp_positions = [normalize_position(pos) for pos in opp_gotis]

        dice_normalized = [0.0, 0.0, 0.0]
        if dice_roll:
            for i, d in enumerate(dice_roll[:3]):
                dice_normalized[i] = d / 6.0

        state_vector = (
            my_positions + opp_positions + dice_normalized + [float(player_turn)]
        )

        return np.array(state_vector, dtype=np.float32)

    def get_action(self, state, action_space):
        """Get action using current policy (epsilon-greedy for exploration)"""
        if not action_space:
            return None

        # Single action case
        if len(action_space) == 1:
            return action_space[0]

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(action_space)

        # Greedy action
        state_encoded = self.encode_state(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)

            # Create action mask
            action_mask = self._create_action_mask(action_space)
            action_mask_tensor = (
                torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            )

            # Get Q-values with masking
            q_values = (
                self.policy_net(state_tensor, action_mask_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        # Select best action
        valid_q_values = []
        for action in action_space:
            dice_idx, goti_idx = action
            action_net_idx = dice_idx * 4 + goti_idx
            action_net_idx = max(0, min(action_net_idx, self.max_actions - 1))
            valid_q_values.append((q_values[action_net_idx], action))

        if valid_q_values:
            return max(valid_q_values, key=lambda x: x[0])[1]
        else:
            return action_space[0]

    def _create_action_mask(self, action_space):
        """Create action mask for valid actions"""
        mask = [False] * self.max_actions
        for dice_idx, goti_idx in action_space:
            action_idx = dice_idx * 4 + goti_idx
            action_idx = max(0, min(action_idx, self.max_actions - 1))
            mask[action_idx] = True
        return mask


def rollout_worker(
    worker_id, num_episodes, weights, agent_player_positions, opponent_snapshots
):
    """
    Worker function that collects episodes in parallel.

    Args:
        worker_id: Unique worker identifier
        num_episodes: Number of episodes to collect
        weights: Dictionary with policy network weights and epsilon
        agent_player_positions: List of player positions (0 or 1) for each episode
        opponent_snapshots: List of opponent snapshots for self-play

    Returns:
        List of trajectories, where each trajectory is a list of
        (state_encoded, action_idx, reward, next_state_encoded, done) tuples
    """
    # Set random seed for reproducibility (each worker gets different seed)
    np.random.seed(worker_id * 1000 + np.random.randint(1000))
    random.seed(worker_id * 1000 + np.random.randint(1000))

    # Create environment
    env = Ludo()

    # Create inference policy (CPU only for workers)
    agent_policy = InferencePolicy(weights, device="cpu")

    # Create opponent policies
    random_policy = Policy_Random()
    heuristic_policy = Policy_Heuristic()

    # Create self-play policy if snapshots available
    if opponent_snapshots:
        # Use a random snapshot for self-play
        snapshot_weights = random.choice(opponent_snapshots)
        self_play_policy = InferencePolicy(snapshot_weights, device="cpu")
    else:
        self_play_policy = None

    all_trajectories = []

    for ep_idx in range(num_episodes):
        # Select opponent for this episode
        opponent_type = np.random.choice(
            ["random", "heuristic", "self_play"], p=[0.25, 0.25, 0.5]
        )

        if opponent_type == "random":
            opponent_policy = random_policy
        elif opponent_type == "heuristic":
            opponent_policy = heuristic_policy
        else:
            if self_play_policy is not None:
                opponent_policy = self_play_policy
            else:
                opponent_policy = random_policy

        # Determine agent player position
        agent_player = (
            agent_player_positions[ep_idx]
            if ep_idx < len(agent_player_positions)
            else ep_idx % 2
        )

        # Set up policies
        if agent_player == 0:
            policies = [agent_policy, opponent_policy]
        else:
            policies = [opponent_policy, agent_policy]

        # Run episode and collect trajectory
        trajectory = []
        state = env.reset()
        terminated = False
        player_turn = 0
        episode_length = 0

        while not terminated:
            action_space = env.get_action_space()
            action = policies[player_turn].get_action(state, action_space)

            # Encode state before step
            if player_turn == agent_player:
                state_encoded = agent_policy.encode_state(state)
                dice_idx, goti_idx = action
                action_idx = dice_idx * 4 + goti_idx
                action_idx = max(0, min(action_idx, agent_policy.max_actions - 1))

            next_state = env.step(action)
            terminated = next_state[3]
            next_player_turn = next_state[4]
            episode_length += 1

            # Store transition if agent's turn
            if player_turn == agent_player and not terminated:
                next_state_encoded = agent_policy.encode_state(next_state)
                reward = -0.0001  # Small step penalty
                trajectory.append(
                    (state_encoded, action_idx, reward, next_state_encoded, False)
                )

            state = next_state
            player_turn = next_player_turn

        # Add final transition with terminal reward
        if trajectory:
            winner = state[4]
            agent_won = winner == agent_player
            final_reward = 1.0 if agent_won else -1.0

            # Replace last transition with terminal state
            if len(trajectory) > 0:
                last_state, last_action, _, _, _ = trajectory[-1]
                next_state_encoded = np.zeros(agent_policy.state_dim, dtype=np.float32)
                trajectory[-1] = (
                    last_state,
                    last_action,
                    final_reward,
                    next_state_encoded,
                    True,
                )
            else:
                # Edge case: episode ended immediately
                state_encoded = agent_policy.encode_state(state)
                next_state_encoded = np.zeros(agent_policy.state_dim, dtype=np.float32)
                trajectory.append(
                    (state_encoded, 0, final_reward, next_state_encoded, True)
                )

        all_trajectories.append(trajectory)

    return all_trajectories
