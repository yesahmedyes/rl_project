import random
import numpy as np

import torch
from ludo import Ludo, STARTING, DESTINATION, SAFE_SQUARES
from policy_random import Policy_Random
from policy_heuristic import Policy_Heuristic
from milestone2 import Policy_Milestone2
from models import DuelingDQNNetwork


def calculate_dense_reward(
    state_before, state_after, terminated, agent_won, agent_player
):
    """
    Calculate shaped reward based on game progress and key events.

    Dense reward shaping to provide richer learning signals:
    - Terminal: ±10.0 for win/loss
    - Reaching destination: +0.5 per piece
    - Capturing opponent: +0.3 per capture
    - Getting captured: -0.3 per capture
    - Exiting home: +0.1 per piece
    - Forward progress: +0.01 per position
    - Step penalty: -0.001 (efficiency incentive)
    """

    # Terminal rewards (strong signal)
    if terminated:
        return 10.0 if agent_won else -10.0

    # Initialize reward with small step penalty
    reward = -0.001

    # Extract agent's and opponent's pieces before and after
    gotis_red_before, gotis_yellow_before, _, _, _ = state_before
    gotis_red_after, gotis_yellow_after, _, _, _ = state_after

    if agent_player == 0:
        my_gotis_before = gotis_red_before.gotis
        my_gotis_after = gotis_red_after.gotis
        opp_gotis_before = gotis_yellow_before.gotis
        opp_gotis_after = gotis_yellow_after.gotis
    else:
        my_gotis_before = gotis_yellow_before.gotis
        my_gotis_after = gotis_yellow_after.gotis
        opp_gotis_before = gotis_red_before.gotis
        opp_gotis_after = gotis_red_after.gotis

    # 1. FORWARD PROGRESS REWARD
    # Small reward for moving pieces toward destination
    total_progress_before = sum(max(0, g.position) for g in my_gotis_before)
    total_progress_after = sum(max(0, g.position) for g in my_gotis_after)
    progress_delta = total_progress_after - total_progress_before

    if progress_delta > 0:
        reward += progress_delta * 0.01  # 0.01 per position advanced

    # 2. REACHING DESTINATION REWARD
    # Significant reward for completing a piece
    pieces_home_before = sum(1 for g in my_gotis_before if g.position == DESTINATION)
    pieces_home_after = sum(1 for g in my_gotis_after if g.position == DESTINATION)

    if pieces_home_after > pieces_home_before:
        reward += 0.5  # Major milestone reward

    # 3. CAPTURING OPPONENT REWARD
    # Reward for sending opponent back to start
    opp_at_start_before = sum(1 for g in opp_gotis_before if g.position == STARTING)
    opp_at_start_after = sum(1 for g in opp_gotis_after if g.position == STARTING)

    if opp_at_start_after > opp_at_start_before:
        reward += 0.3 * (opp_at_start_after - opp_at_start_before)

    # 4. GETTING CAPTURED PENALTY
    # Penalty for being sent back to start
    my_at_start_before = sum(1 for g in my_gotis_before if g.position == STARTING)
    my_at_start_after = sum(1 for g in my_gotis_after if g.position == STARTING)

    if my_at_start_after > my_at_start_before:
        reward -= 0.3 * (my_at_start_after - my_at_start_before)

    # 5. EXITING HOME REWARD
    # Encourage getting pieces into play
    my_in_play_before = sum(
        1 for g in my_gotis_before if g.position >= 0 and g.position < DESTINATION
    )
    my_in_play_after = sum(
        1 for g in my_gotis_after if g.position >= 0 and g.position < DESTINATION
    )

    if my_in_play_after > my_in_play_before:
        reward += 0.1 * (my_in_play_after - my_in_play_before)

    return reward


class InferencePolicy:
    def __init__(self, weights, state_dim=28, max_actions=12, device="cpu"):
        self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.max_actions = max_actions
        self.epsilon = weights["epsilon"]

        # Create network on CPU only
        self.policy_net = DuelingDQNNetwork(
            state_dim, hidden_dim=128, max_actions=max_actions
        )

        state_dict = {}
        for key, value in weights["policy_net"].items():
            state_dict[key] = value.cpu() if value.is_cuda else value

        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()

        for param in self.policy_net.parameters():
            param.requires_grad = False

        # Reset noise for deterministic inference (uses mean weights in eval mode)
        self.policy_net.reset_noise()

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

        # 1. Distance to goal for each of my pieces (4 features)
        my_distances = [
            (DESTINATION - pos) / 57.0 if pos >= 0 else 1.0 for pos in my_gotis
        ]

        # 2. Count pieces at home for both players (2 features)
        my_at_home = sum(1 for pos in my_gotis if pos == STARTING) / 4.0
        opp_at_home = sum(1 for pos in opp_gotis if pos == STARTING) / 4.0

        # 3. Count pieces at destination for both players (2 features)
        my_at_dest = sum(1 for pos in my_gotis if pos == DESTINATION) / 4.0
        opp_at_dest = sum(1 for pos in opp_gotis if pos == DESTINATION) / 4.0

        # 4. Count pieces on safe squares for both players (2 features)
        my_on_safe = sum(1 for pos in my_gotis if pos in SAFE_SQUARES) / 4.0
        opp_on_safe = sum(1 for pos in opp_gotis if pos in SAFE_SQUARES) / 4.0

        # 5. Average progress for both players (2 features)
        def calc_progress(positions):
            active = [p for p in positions if p >= 0]
            if active:
                return sum(p for p in active) / (len(active) * DESTINATION)
            return 0.0

        my_avg_progress = calc_progress(my_gotis)
        opp_avg_progress = calc_progress(opp_gotis)

        # 6. Pieces in danger - my pieces on unsafe squares with opponent nearby (1 feature)
        pieces_in_danger = 0

        for my_pos in my_gotis:
            if my_pos >= 0 and my_pos not in SAFE_SQUARES:
                # Check if any opponent within striking distance (1-6 squares behind)
                for opp_pos in opp_gotis:
                    if opp_pos >= 0 and 1 <= (my_pos - opp_pos) <= 6:
                        pieces_in_danger += 1
                        break

        pieces_in_danger_norm = pieces_in_danger / 4.0

        # 7. Capture opportunities - my pieces that could capture opponent (1 feature)
        capture_opportunities = 0

        for my_pos in my_gotis:
            if my_pos >= 0:
                # Check if any opponent within striking distance ahead
                for opp_pos in opp_gotis:
                    if opp_pos >= 0 and opp_pos not in SAFE_SQUARES:
                        if 1 <= (opp_pos - my_pos) <= 6:
                            capture_opportunities += 1
                            break

        capture_opportunities_norm = capture_opportunities / 4.0

        # 8. Can move out of home - have a 6 in dice (1 feature)
        has_six = 1.0 if (dice_roll and 6 in dice_roll) else 0.0

        # 9. Number of dice available (1 feature)
        num_dice = len(dice_roll) / 3.0 if dice_roll else 0.0

        # Combine all features (total: 28 features)
        state_vector = (
            my_positions  # 4 features
            + opp_positions  # 4 features
            + dice_normalized  # 3 features
            + [float(player_turn)]  # 1 feature
            + my_distances  # 4 features
            + [my_at_home, opp_at_home]  # 2 features
            + [my_at_dest, opp_at_dest]  # 2 features
            + [my_on_safe, opp_on_safe]  # 2 features
            + [my_avg_progress, opp_avg_progress]  # 2 features
            + [pieces_in_danger_norm]  # 1 feature
            + [capture_opportunities_norm]  # 1 feature
            + [has_six]  # 1 feature
            + [num_dice]  # 1 feature
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
                and action is not None
                and state_encoded is not None
            ):
                next_state_encoded = agent_policy.encode_state(next_state)

                # Calculate dense shaped reward
                winner = next_state[4] if terminated else None
                agent_won_step = (winner == agent_player) if terminated else False
                reward = calculate_dense_reward(
                    state, next_state, terminated, agent_won_step, agent_player
                )

                trajectory.append(
                    (state_encoded, action_idx, reward, next_state_encoded, terminated)
                )

            state = next_state
            player_turn = next_player_turn

        # Determine episode outcome
        if len(trajectory) > 0:
            winner = state[4]
            agent_won = winner == agent_player

            # Get final reward from last transition (already calculated with dense rewards)
            final_reward = trajectory[-1][2]
        else:
            agent_won = False
            final_reward = 0.0

        all_trajectories.append(
            {
                "trajectory": trajectory,
                "opponent_type": opponent_type,
                "agent_won": agent_won,
                "episode_length": episode_length,
                "final_reward": final_reward,
            }
        )

    return all_trajectories
