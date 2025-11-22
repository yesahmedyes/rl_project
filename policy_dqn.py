import random
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from models import DuelingDQNNetwork, PrioritizedReplayBuffer, ReplayBuffer


class Policy_DQN:
    def __init__(
        self,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        learning_rate=0.001,
        gamma=0.99,
        batch_size=2048,
        buffer_size=100000,
        target_update_freq=1000,
        training_mode=False,
        device=None,
        policy_path="models/policy_dqn.pth",
        use_prioritized_replay=True,
        per_alpha=0.6,
        per_beta=0.4,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_mode = training_mode
        self.policy_path = policy_path
        self.use_prioritized_replay = use_prioritized_replay

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.state_dim = 12  # 4 my gotis + 4 opp gotis + 3 dice + 1 player turn
        self.max_actions = 12  # Max possible actions (3 dice × 4 gotis)

        self.policy_net = DuelingDQNNetwork(
            self.state_dim, hidden_dim=256, max_actions=self.max_actions
        ).to(self.device)
        self.target_net = DuelingDQNNetwork(
            self.state_dim, hidden_dim=256, max_actions=self.max_actions
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size, alpha=per_alpha, beta=per_beta, device=self.device
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

        self.last_state_encoded = None
        self.last_action_idx = None
        self.last_action_space = None
        self.steps_done = 0
        self.episode_count = 0

        if not self.training_mode:
            self.load(policy_path)

    def encode_state(self, state):
        gotis_red, gotis_yellow, dice_roll, _, player_turn = state

        # Get current player and opponent gotis
        if player_turn == 0:
            my_gotis = [g.position for g in gotis_red.gotis]
            opp_gotis = [g.position for g in gotis_yellow.gotis]
        else:
            my_gotis = [g.position for g in gotis_yellow.gotis]
            opp_gotis = [g.position for g in gotis_red.gotis]

        # Normalize positions: STARTING (-1) → 0, positions 0-56 → normalized to [0,1]
        def normalize_position(pos):
            return (pos + 1) / 57.0  # Maps -1 to 0, 56 to 1

        my_positions = [normalize_position(pos) for pos in my_gotis]
        opp_positions = [normalize_position(pos) for pos in opp_gotis]

        # Normalize dice roll (pad to 3 values)
        dice_normalized = [0.0, 0.0, 0.0]
        if dice_roll:
            for i, d in enumerate(dice_roll[:3]):
                dice_normalized[i] = d / 6.0

        # Combine all features
        state_vector = (
            my_positions + opp_positions + dice_normalized + [float(player_turn)]
        )

        return np.array(state_vector, dtype=np.float32)

    def get_action(self, state, action_space):
        if not action_space:
            return None

        # Encode state
        state_encoded = self.encode_state(state)

        # Single action case
        if len(action_space) == 1:
            action = action_space[0]

            if self.training_mode:
                self._store_transition(state_encoded, action, action_space)

            return action

        # Epsilon-greedy action selection
        if self.training_mode and random.random() < self.epsilon:
            # Random action
            action = random.choice(action_space)

            if self.training_mode:
                self._store_transition(state_encoded, action, action_space)

            return action

        # Greedy action (use Q-network with action masking)
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

        # Select action with highest Q-value among valid actions
        valid_q_values = []

        for action in action_space:
            dice_idx, goti_idx = action
            # Map action to network output index (0-11)
            action_net_idx = dice_idx * 4 + goti_idx
            # Clamp to valid range as safety check
            action_net_idx = max(0, min(action_net_idx, self.max_actions - 1))
            valid_q_values.append((q_values[action_net_idx], action))

        if valid_q_values:
            best_action = max(valid_q_values, key=lambda x: x[0])[1]
        else:
            best_action = action_space[0]

        if self.training_mode:
            self._store_transition(state_encoded, best_action, action_space)

        return best_action

    def _create_action_mask(self, action_space):
        mask = [False] * self.max_actions

        for dice_idx, goti_idx in action_space:
            action_idx = dice_idx * 4 + goti_idx
            action_idx = max(0, min(action_idx, self.max_actions - 1))
            mask[action_idx] = True

        return mask

    def _store_transition(self, state_encoded, action, action_space):
        self.last_state_encoded = state_encoded
        self.last_action_space = action_space

        dice_idx, goti_idx = action
        action_idx = dice_idx * 4 + goti_idx

        action_idx = max(0, min(action_idx, self.max_actions - 1))

        self.last_action_idx = action_idx

    def update(self, reward, next_state, next_action_space):
        if not self.training_mode or self.last_state_encoded is None:
            return

        next_state_encoded = self.encode_state(next_state)
        done = False

        self.replay_buffer.push(
            self.last_state_encoded,
            self.last_action_idx,
            reward,
            next_state_encoded,
            done,
        )

        if len(self.replay_buffer) >= self.batch_size:
            self._learn()

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def episode_end(self, reward):
        if not self.training_mode or self.last_state_encoded is None:
            return

        next_state_encoded = np.zeros(self.state_dim, dtype=np.float32)

        done = True

        self.replay_buffer.push(
            self.last_state_encoded,
            self.last_action_idx,
            reward,
            next_state_encoded,
            done,
        )

        if len(self.replay_buffer) >= self.batch_size:
            self._learn()

        self.last_state_encoded = None
        self.last_action_idx = None
        self.last_action_space = None
        self.episode_count += 1

    def _learn(self):
        if self.use_prioritized_replay:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ) = self.replay_buffer.sample(self.batch_size)

            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )

            weights = torch.ones(self.batch_size).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Clamp actions to valid range (safety check)
        actions = torch.clamp(actions, 0, self.max_actions - 1)

        # Compute current Q-values
        current_q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Double DQN: Use policy network to select best action, target network to evaluate
        with torch.no_grad():
            # Use policy network to select best actions for next states
            next_actions = self.policy_net(next_states).max(1)[1]

            # Use target network to evaluate the selected actions
            next_q_values = (
                self.target_net(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )

            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        # Compute weighted loss (for PER importance sampling)
        elementwise_loss = F.mse_loss(
            current_q_values, target_q_values, reduction="none"
        )
        loss = (weights * elementwise_loss).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in replay buffer
        if self.use_prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_errors)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_epsilon(self):
        return self.epsilon

    def set_training_mode(self, mode):
        self.training_mode = mode

        if mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def save(self, filename=None):
        if filename is None:
            filename = self.policy_path

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "episode_count": self.episode_count,
            },
            filename,
        )

        print(f"DQN model saved to {filename}")

    def load(self, filename):
        if not os.path.exists(filename):
            print(f"File {filename} not found. Starting with fresh DQN model.")
            return

        checkpoint = torch.load(filename, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_start)
        self.steps_done = checkpoint.get("steps_done", 0)
        self.episode_count = checkpoint.get("episode_count", 0)

        print(f"DQN model loaded from {filename}")
        print(f"Episodes trained: {self.episode_count}, Steps: {self.steps_done}")

    def reset_traces(self):
        self.last_state_encoded = None
        self.last_action_idx = None
        self.last_action_space = None

    def get_weights(self):
        cpu_state_dict = {
            key: value.cpu().detach().clone()
            for key, value in self.policy_net.state_dict().items()
        }
        return {
            "policy_net": cpu_state_dict,
            "epsilon": self.epsilon,
        }

    def set_weights(self, weights):
        self.policy_net.load_state_dict(weights["policy_net"])
        self.epsilon = weights["epsilon"]

    def add_trajectory_to_buffer(self, trajectory):
        for transition in trajectory:
            state, action_idx, reward, next_state, done = transition
            self.replay_buffer.push(state, action_idx, reward, next_state, done)
