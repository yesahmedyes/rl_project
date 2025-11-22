import random
import torch.nn as nn
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim**0.5

    def forward(self, x, num_items):
        batch_size = x.shape[0]
        feature_dim = x.shape[1] // num_items

        items = x[:, : num_items * feature_dim].reshape(
            batch_size, num_items, feature_dim
        )

        # Compute attention scores
        q = self.query(items)
        k = self.key(items)
        v = self.value(items)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(weights, v)

        output = attended.mean(dim=1)

        return output


class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim=12, hidden_dim=256, max_actions=12):
        super(DuelingDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.max_actions = max_actions

        self.goti_attention = AttentionLayer(input_dim=1, hidden_dim=hidden_dim // 2)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.feature_combine = nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim)

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, action_mask=None):
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
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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
    def __init__(
        self, capacity=500000, alpha=0.6, beta=0.4, beta_increment=0.001, device=None
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0

        # Use CUDA if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Store experiences on CPU (standard practice for large buffers)
        self.buffer = []
        # Store priorities on GPU for fast sampling
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = (
            self.priorities[: self.size].max().item() if self.size > 0 else 1.0
        )

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Use GPU for priority calculations
        priorities = self.priorities[: self.size]
        probabilities = priorities**self.alpha
        probabilities = probabilities / probabilities.sum()

        # Move to CPU for numpy random choice (faster for small arrays)
        probabilities_cpu = probabilities.cpu().numpy()
        indices = np.random.choice(
            self.size, batch_size, p=probabilities_cpu, replace=False
        )

        # Calculate importance sampling weights on GPU
        indices_tensor = torch.from_numpy(indices).to(self.device)
        selected_probs = probabilities[indices_tensor]
        weights = (self.size * selected_probs) ** (-self.beta)
        weights = weights / weights.max()
        weights_cpu = weights.cpu().numpy()

        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        # Sample experiences from buffer
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights_cpu,
        )

    def update_priorities(self, indices, priorities):
        # Convert to tensors if needed and update on GPU
        if isinstance(priorities, np.ndarray):
            priorities = torch.from_numpy(priorities).to(self.device)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).to(self.device)

        self.priorities[indices] = priorities + 1e-5

    def __len__(self):
        return self.size
