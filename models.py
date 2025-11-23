import random
import torch.nn as nn
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters for mean
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # Noise buffers (not trained, reset each forward pass)
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)

        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            # Use noisy weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim=28, hidden_dim=128, max_actions=12, use_noisy=True):
        super(DuelingDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.max_actions = max_actions
        self.use_noisy = use_noisy

        # Feature extraction with 3-layer MLP + LayerNorm
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Dueling streams with NoisyLinear (where exploration matters most)
        if use_noisy:
            self.value_fc1 = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.value_fc2 = NoisyLinear(hidden_dim // 2, 1)

            self.advantage_fc1 = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.advantage_fc2 = NoisyLinear(hidden_dim // 2, max_actions)
        else:
            self.value_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.value_fc2 = nn.Linear(hidden_dim // 2, 1)

            self.advantage_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.advantage_fc2 = nn.Linear(hidden_dim // 2, max_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def reset_noise(self):
        """Reset noise in all noisy layers"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def forward(self, x, action_mask=None):
        # Feature extraction with LayerNorm (improves stability)
        features = self.fc1(x)
        features = self.ln1(features)
        features = F.relu(features)

        features = self.fc2(features)
        features = self.ln2(features)
        features = F.relu(features)

        features = self.fc3(features)
        features = self.ln3(features)
        features = F.relu(features)

        # Dueling architecture with noisy layers
        value = F.relu(self.value_fc1(features))
        value = self.value_fc2(value)

        advantages = F.relu(self.advantage_fc1(features))
        advantages = self.advantage_fc2(advantages)

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
