import random
import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention mechanism over gotis to focus on relevant pieces"""

    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim**0.5

    def forward(self, x, num_items):
        """
        Args:
            x: tensor of shape (batch_size, total_features)
            num_items: number of items to attend over (e.g., 8 for my+opp gotis)
        Returns:
            attended features of shape (batch_size, hidden_dim)
        """
        batch_size = x.shape[0]
        feature_dim = x.shape[1] // num_items

        # Reshape to (batch_size, num_items, feature_dim)
        items = x[:, : num_items * feature_dim].reshape(
            batch_size, num_items, feature_dim
        )

        # Compute attention scores
        q = self.query(items)  # (batch_size, num_items, hidden_dim)
        k = self.key(items)  # (batch_size, num_items, hidden_dim)
        v = self.value(items)  # (batch_size, num_items, hidden_dim)

        # Attention weights: (batch_size, num_items, num_items)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)

        # Apply attention: (batch_size, num_items, hidden_dim)
        attended = torch.matmul(weights, v)

        # Global pooling: (batch_size, hidden_dim)
        output = attended.mean(dim=1)

        return output


class DuelingDQNNetwork(nn.Module):
    """
    Dueling Deep Q-Network with Attention for Ludo game

    Separates state value V(s) from action advantages A(s,a):
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, state_dim=12, hidden_dim=256, max_actions=12):
        super(DuelingDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.max_actions = max_actions

        # Attention over gotis (8 gotis: 4 mine + 4 opponent)
        # Each goti has 1 feature (normalized position)
        self.goti_attention = AttentionLayer(input_dim=1, hidden_dim=hidden_dim // 2)

        # Feature extraction layers
        # Input: 12 features (8 goti positions + 3 dice + 1 turn)
        # After attention: 64 (goti attention) + 3 (dice) + 1 (turn) = 68
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Combine attention with FC features
        self.feature_combine = nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim)

        # Dueling streams
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, action_mask=None):
        """
        Forward pass through the network with optional action masking

        Args:
            x: state tensor of shape (batch_size, state_dim)
            action_mask: optional boolean tensor of shape (batch_size, max_actions)
                         True for valid actions, False for invalid

        Returns:
            Q-values tensor of shape (batch_size, max_actions)
        """
        batch_size = x.shape[0]

        # Extract goti positions (first 8 features)
        goti_features = x[:, :8]

        # Apply attention over gotis
        attended_gotis = self.goti_attention(goti_features, num_items=8)

        # Standard feature extraction
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))

        # Combine attention with features
        combined = torch.cat([features, attended_gotis], dim=1)
        combined = F.relu(self.feature_combine(combined))

        # Dueling architecture
        value = self.value_stream(combined)  # (batch_size, 1)
        advantages = self.advantage_stream(combined)  # (batch_size, max_actions)

        # Combine value and advantages
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        # Apply action masking if provided
        if action_mask is not None:
            # Set invalid actions to very negative value
            q_values = torch.where(
                action_mask, q_values, torch.tensor(-1e9, device=x.device)
            )

        return q_values


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer

    Samples transitions based on their TD error magnitude, allowing the agent
    to learn more from surprising/important transitions.
    """

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta: Importance sampling weight (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Add a transition with maximum priority"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Calculate sampling probabilities
        priorities = self.priorities[: self.size]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(
            self.size, batch_size, p=probabilities, replace=False
        )

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        # Increment beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        # Extract samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to ensure non-zero

    def __len__(self):
        return self.size


class Policy_DQN:
    """Deep Q-Network policy for Ludo game with Dueling, Double DQN, Attention, and PER"""

    def __init__(
        self,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        learning_rate=0.001,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000,
        target_update_freq=1000,
        training_mode=False,
        device=None,
        policy_path="models/policy_dqn.pth",
        use_prioritized_replay=True,
        per_alpha=0.6,
        per_beta=0.4,
    ):
        # Hyperparameters
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

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # State and action space
        self.state_dim = 12  # 4 my gotis + 4 opp gotis + 3 dice + 1 player turn
        self.max_actions = 12  # Max possible actions (3 dice × 4 gotis)

        # Networks - Using Dueling DQN with Attention
        self.policy_net = DuelingDQNNetwork(
            self.state_dim, hidden_dim=256, max_actions=self.max_actions
        ).to(self.device)
        self.target_net = DuelingDQNNetwork(
            self.state_dim, hidden_dim=256, max_actions=self.max_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer - Prioritized or standard
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size, alpha=per_alpha, beta=per_beta
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

        # Training state
        self.last_state_encoded = None
        self.last_action_idx = None
        self.last_action_space = None
        self.steps_done = 0
        self.episode_count = 0

        # Load pretrained model if not in training mode
        if not self.training_mode:
            self.load(policy_path)

    def encode_state(self, state):
        """
        Encode the game state into a fixed-size vector for the neural network

        State components:
        - My 4 gotis positions (normalized to [0,1])
        - Opponent 4 gotis positions (normalized to [0,1])
        - Dice roll (padded to 3 values, normalized to [0,1])
        - Player turn (0 or 1)

        Returns: numpy array of shape (12,)
        """
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
        """
        Select an action using epsilon-greedy strategy with action masking

        Args:
            state: Game state tuple
            action_space: List of valid actions

        Returns:
            Selected action tuple (dice_index, goti_index)
        """
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
        """
        Create a boolean mask for valid actions

        Args:
            action_space: List of valid (dice_idx, goti_idx) tuples

        Returns:
            Boolean array of shape (max_actions,) with True for valid actions
        """
        mask = [False] * self.max_actions
        for dice_idx, goti_idx in action_space:
            action_idx = dice_idx * 4 + goti_idx
            action_idx = max(0, min(action_idx, self.max_actions - 1))
            mask[action_idx] = True
        return mask

    def _store_transition(self, state_encoded, action, action_space):
        """Store the current state and action for later update"""
        self.last_state_encoded = state_encoded
        self.last_action_space = action_space

        # Map action to network output index (0-11)
        dice_idx, goti_idx = action
        action_idx = dice_idx * 4 + goti_idx

        # Clamp to valid range as safety check
        action_idx = max(0, min(action_idx, self.max_actions - 1))

        self.last_action_idx = action_idx

    def update(self, reward, next_state, next_action_space):
        """
        Store transition in replay buffer and perform learning step

        Args:
            reward: Immediate reward
            next_state: Next game state
            next_action_space: Valid actions in next state
        """
        if not self.training_mode or self.last_state_encoded is None:
            return

        # Encode next state
        next_state_encoded = self.encode_state(next_state)
        done = False

        # Add transition to replay buffer
        self.replay_buffer.push(
            self.last_state_encoded,
            self.last_action_idx,
            reward,
            next_state_encoded,
            done,
        )

        # Perform learning step if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._learn()

        self.steps_done += 1

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def episode_end(self, reward):
        """
        Handle episode termination

        Args:
            reward: Final reward (typically large for win/loss)
        """
        if not self.training_mode or self.last_state_encoded is None:
            return

        # Create a zero next state (terminal state)
        next_state_encoded = np.zeros(self.state_dim, dtype=np.float32)
        done = True

        # Add final transition to replay buffer
        self.replay_buffer.push(
            self.last_state_encoded,
            self.last_action_idx,
            reward,
            next_state_encoded,
            done,
        )

        # Perform learning step
        if len(self.replay_buffer) >= self.batch_size:
            self._learn()

        # Reset episode state
        self.last_state_encoded = None
        self.last_action_idx = None
        self.last_action_space = None
        self.episode_count += 1

    def _learn(self):
        """
        Perform one step of Double DQN learning with Prioritized Experience Replay

        Double DQN: Use policy network to select action, target network to evaluate it
        PER: Weight updates by importance sampling weights and update priorities
        """
        # Sample batch from replay buffer
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

        # Convert to tensors
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
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_epsilon(self):
        """Get current epsilon value"""
        return self.epsilon

    def set_training_mode(self, mode):
        """Set training mode"""
        self.training_mode = mode
        if mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def save(self, filename=None):
        """Save the DQN model"""
        if filename is None:
            filename = self.policy_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save model state
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
        """Load the DQN model"""
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
        """Reset episode traces (for compatibility with training script)"""
        self.last_state_encoded = None
        self.last_action_idx = None
        self.last_action_space = None
