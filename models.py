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
    def __init__(self, capacity=500000, alpha=0.6, beta=0.4, beta_increment=0.001):
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
        max_priority = self.priorities.max() if self.size > 0 else 1.0

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

        priorities = self.priorities[: self.size]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            self.size, batch_size, p=probabilities, replace=False
        )

        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(self.max_beta, self.beta + self.beta_increment)

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
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5

    def __len__(self):
        return self.size
