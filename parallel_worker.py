import random
import numpy as np

import torch
from ludo import Ludo
from policy_random import Policy_Random
from policy_heuristic import Policy_Heuristic
from milestone2 import Policy_Milestone2
from models import DuelingDQNNetwork


class InferencePolicy:
    def __init__(self, weights, state_dim=12, max_actions=12, device="cpu"):
        self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.max_actions = max_actions
        self.epsilon = weights["epsilon"]

        # Create network on CPU only
        self.policy_net = DuelingDQNNetwork(
            state_dim, hidden_dim=256, max_actions=max_actions
        )

        state_dict = {}
        for key, value in weights["policy_net"].items():
            state_dict[key] = value.cpu() if value.is_cuda else value

        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()

        for param in self.policy_net.parameters():
            param.requires_grad = False

    def encode_state(self, state):
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
        mask = [False] * self.max_actions

        for dice_idx, goti_idx in action_space:
            action_idx = dice_idx * 4 + goti_idx
            action_idx = max(0, min(action_idx, self.max_actions - 1))
            mask[action_idx] = True

        return mask


def rollout_worker(
    worker_id, num_episodes, weights, agent_player_positions, opponent_snapshots
):
    # Each worker gets different seed
    np.random.seed(worker_id * 1000 + np.random.randint(1000))
    random.seed(worker_id * 1000 + np.random.randint(1000))

    # Create environment
    env = Ludo()

    agent_policy = InferencePolicy(weights, device=torch.device("cpu"))

    # Create opponent policies
    random_policy = Policy_Random()
    heuristic_policy = Policy_Heuristic()
    milestone2_policy = Policy_Milestone2()

    # Create self-play policy if snapshots available (CPU only)
    if opponent_snapshots:
        snapshot_weights = random.choice(opponent_snapshots)
        self_play_policy = InferencePolicy(snapshot_weights, device=torch.device("cpu"))
    else:
        self_play_policy = None

    all_trajectories = []

    for ep_idx in range(num_episodes):
        opponent_type = np.random.choice(
            ["random", "heuristic", "milestone2", "self_play"],
            p=[0.25, 0.25, 0.25, 0.25],
        )

        if opponent_type == "random":
            opponent_policy = random_policy
        elif opponent_type == "heuristic":
            opponent_policy = heuristic_policy
        elif opponent_type == "milestone2":
            opponent_policy = milestone2_policy
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

            # Encode state before step (only if agent's turn and action exists)
            if player_turn == agent_player and action is not None:
                state_encoded = agent_policy.encode_state(state)
                dice_idx, goti_idx = action
                action_idx = dice_idx * 4 + goti_idx
                action_idx = max(0, min(action_idx, agent_policy.max_actions - 1))
            else:
                action_idx = None
                state_encoded = None

            next_state = env.step(action)
            terminated = next_state[3]
            next_player_turn = next_state[4]
            episode_length += 1

            # Store transition if agent's turn and action was valid
            if (
                player_turn == agent_player
                and not terminated
                and action is not None
                and state_encoded is not None
            ):
                next_state_encoded = agent_policy.encode_state(next_state)
                reward = -0.0001  # Small step penalty
                trajectory.append(
                    (state_encoded, action_idx, reward, next_state_encoded, False)
                )

            state = next_state
            player_turn = next_player_turn

        # Add final transition with terminal reward
        if len(trajectory) > 0:
            winner = state[4]
            agent_won = winner == agent_player
            final_reward = 1.0 if agent_won else -1.0

            last_state, last_action, _, _, _ = trajectory[-1]
            next_state_encoded = np.zeros(agent_policy.state_dim, dtype=np.float32)
            trajectory[-1] = (
                last_state,
                last_action,
                final_reward,
                next_state_encoded,
                True,
            )

        all_trajectories.append(
            {
                "trajectory": trajectory,
                "opponent_type": opponent_type,
                "agent_won": agent_won if len(trajectory) > 0 else False,
                "episode_length": episode_length,
                "final_reward": final_reward if len(trajectory) > 0 else 0,
            }
        )

    return all_trajectories
