import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ludo import DESTINATION, SAFE_SQUARES, STARTING


STATE_DIM = 28
MAX_ACTIONS = 12


def encode_ludo_state(state) -> np.ndarray:
    """
    Convert an environment state tuple to a flat feature vector.

    Mirrors the previous DQN feature engineering so that demonstrations and
    downstream tooling remain compatible.
    """
    gotis_red, gotis_yellow, dice_roll, _, player_turn = state

    if player_turn == 0:
        my_gotis = [g.position for g in gotis_red.gotis]
        opp_gotis = [g.position for g in gotis_yellow.gotis]
    else:
        my_gotis = [g.position for g in gotis_yellow.gotis]
        opp_gotis = [g.position for g in gotis_red.gotis]

    def normalize_position(pos: int) -> float:
        return (pos + 1) / 57.0  # Maps STARTING (-1) -> 0, DESTINATION (56) -> 1

    my_positions = [normalize_position(pos) for pos in my_gotis]
    opp_positions = [normalize_position(pos) for pos in opp_gotis]

    dice_normalized = [0.0, 0.0, 0.0]
    if dice_roll:
        for i, d in enumerate(dice_roll[:3]):
            dice_normalized[i] = d / 6.0

    my_distances = [(DESTINATION - pos) / 57.0 if pos >= 0 else 1.0 for pos in my_gotis]

    my_at_home = sum(1 for pos in my_gotis if pos == STARTING) / 4.0
    opp_at_home = sum(1 for pos in opp_gotis if pos == STARTING) / 4.0

    my_at_dest = sum(1 for pos in my_gotis if pos == DESTINATION) / 4.0
    opp_at_dest = sum(1 for pos in opp_gotis if pos == DESTINATION) / 4.0

    my_on_safe = sum(1 for pos in my_gotis if pos in SAFE_SQUARES) / 4.0
    opp_on_safe = sum(1 for pos in opp_gotis if pos in SAFE_SQUARES) / 4.0

    def calc_progress(positions: Sequence[int]) -> float:
        active = [p for p in positions if p >= 0]
        if not active:
            return 0.0
        return sum(active) / (len(active) * DESTINATION)

    my_avg_progress = calc_progress(my_gotis)
    opp_avg_progress = calc_progress(opp_gotis)

    pieces_in_danger = 0
    for my_pos in my_gotis:
        if my_pos >= 0 and my_pos not in SAFE_SQUARES:
            for opp_pos in opp_gotis:
                if opp_pos >= 0 and 1 <= (my_pos - opp_pos) <= 6:
                    pieces_in_danger += 1
                    break
    pieces_in_danger_norm = pieces_in_danger / 4.0

    capture_opportunities = 0
    for my_pos in my_gotis:
        if my_pos >= 0:
            for opp_pos in opp_gotis:
                if opp_pos >= 0 and opp_pos not in SAFE_SQUARES:
                    if 1 <= (opp_pos - my_pos) <= 6:
                        capture_opportunities += 1
                        break
    capture_opportunities_norm = capture_opportunities / 4.0

    has_six = 1.0 if (dice_roll and 6 in dice_roll) else 0.0
    num_dice = len(dice_roll) / 3.0 if dice_roll else 0.0

    state_vector = (
        my_positions
        + opp_positions
        + dice_normalized
        + [float(player_turn)]
        + my_distances
        + [my_at_home, opp_at_home]
        + [my_at_dest, opp_at_dest]
        + [my_on_safe, opp_on_safe]
        + [my_avg_progress, opp_avg_progress]
        + [pieces_in_danger_norm]
        + [capture_opportunities_norm]
        + [has_six]
        + [num_dice]
    )

    return np.array(state_vector, dtype=np.float32)


def calculate_dense_reward(
    state_before,
    state_after,
    terminated: bool,
    agent_won: bool,
    agent_player: int,
) -> float:
    """Dense shaping signal borrowed from the previous DQN pipeline."""
    if terminated:
        return 10.0 if agent_won else -10.0

    reward = -0.001  # small step penalty

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

    total_progress_before = sum(max(0, g.position) for g in my_gotis_before)
    total_progress_after = sum(max(0, g.position) for g in my_gotis_after)
    progress_delta = total_progress_after - total_progress_before
    if progress_delta > 0:
        reward += progress_delta * 0.01

    pieces_home_before = sum(1 for g in my_gotis_before if g.position == DESTINATION)
    pieces_home_after = sum(1 for g in my_gotis_after if g.position == DESTINATION)
    if pieces_home_after > pieces_home_before:
        reward += 0.5

    opp_at_start_before = sum(1 for g in opp_gotis_before if g.position == STARTING)
    opp_at_start_after = sum(1 for g in opp_gotis_after if g.position == STARTING)
    if opp_at_start_after > opp_at_start_before:
        reward += 0.3 * (opp_at_start_after - opp_at_start_before)

    my_at_start_before = sum(1 for g in my_gotis_before if g.position == STARTING)
    my_at_start_after = sum(1 for g in my_gotis_after if g.position == STARTING)
    if my_at_start_after > my_at_start_before:
        reward -= 0.3 * (my_at_start_after - my_at_start_before)

    my_in_play_before = sum(1 for g in my_gotis_before if 0 <= g.position < DESTINATION)
    my_in_play_after = sum(1 for g in my_gotis_after if 0 <= g.position < DESTINATION)
    if my_in_play_after > my_in_play_before:
        reward += 0.1 * (my_in_play_after - my_in_play_before)

    return reward


class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=256, action_dim=MAX_ACTIONS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    action_mask: torch.Tensor


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
    ):
        self.storage.append(
            Transition(
                state=state,
                action=torch.tensor(action, dtype=torch.long, device=state.device),
                log_prob=log_prob.detach(),
                value=value.detach(),
                reward=torch.tensor(reward, dtype=torch.float32, device=state.device),
                done=torch.tensor(done, dtype=torch.float32, device=state.device),
                action_mask=action_mask.detach(),
            )
        )

    def clear(self):
        self.storage: List[Transition] = []

    def as_tensors(self) -> Transition:
        states = torch.stack([t.state for t in self.storage])
        actions = torch.stack([t.action for t in self.storage])
        log_probs = torch.stack([t.log_prob for t in self.storage])
        values = torch.stack([t.value for t in self.storage])
        rewards = torch.stack([t.reward for t in self.storage])
        dones = torch.stack([t.done for t in self.storage])
        masks = torch.stack([t.action_mask for t in self.storage])
        return Transition(states, actions, log_probs, values, rewards, dones, masks)


class PolicyPPO:
    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_timesteps: int = 2048,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        device: Optional[str] = None,
        policy_path: str = "models/policy_ppo.pth",
        create_optimizer: bool = True,
        training_mode: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_timesteps = update_timesteps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.policy_path = policy_path

        self.model = ActorCritic().to(self.device)
        self.buffer = RolloutBuffer()
        self.training_mode = training_mode
        self.total_steps = 0

        if create_optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = None

        if policy_path and os.path.exists(policy_path):
            self.load(policy_path)

        if training_mode:
            self.model.train()
        else:
            self.model.eval()

    def set_training(self, mode: bool):
        self.training_mode = mode
        self.model.train(mode)

    @property
    def max_actions(self) -> int:
        return MAX_ACTIONS

    def encode_state(self, state) -> np.ndarray:
        return encode_ludo_state(state)

    def _action_mask_tensor(
        self, action_space: Sequence[Tuple[int, int]]
    ) -> torch.Tensor:
        mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=self.device)
        for dice_idx, goti_idx in action_space:
            action_idx = dice_idx * 4 + goti_idx
            action_idx = max(0, min(action_idx, MAX_ACTIONS - 1))
            mask[action_idx] = True
        return mask

    def _apply_mask_to_logits(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        large_neg = torch.full_like(logits, -1e9)
        return torch.where(mask, logits, large_neg)

    def act(
        self,
        state,
        action_space: Sequence[Tuple[int, int]],
        deterministic: bool = False,
    ):
        if not action_space:
            return None, None

        state_vec = torch.from_numpy(self.encode_state(state)).to(self.device)
        action_mask = self._action_mask_tensor(action_space)

        with torch.set_grad_enabled(self.training_mode):
            logits, value = self.model(state_vec.unsqueeze(0))
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            masked_logits = self._apply_mask_to_logits(logits, action_mask)

            dist = Categorical(logits=masked_logits)
            if deterministic:
                action_idx = torch.argmax(dist.probs)
            else:
                action_idx = dist.sample()

            log_prob = dist.log_prob(action_idx)

        action_idx_int = int(action_idx.item())
        dice_idx = action_idx_int // 4
        goti_idx = action_idx_int % 4

        if action_idx_int >= len(action_mask) or not action_mask[action_idx_int]:
            valid_indices = torch.nonzero(action_mask, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                return None, None
            action_idx_int = int(valid_indices[0].item())
            dice_idx = action_idx_int // 4
            goti_idx = action_idx_int % 4
            action_idx_tensor = torch.tensor(action_idx_int, device=self.device)
            log_prob = dist.log_prob(action_idx_tensor)

        action = (dice_idx, goti_idx)
        info = {
            "state_tensor": state_vec,
            "action_idx": action_idx_int,
            "log_prob": log_prob,
            "value": value,
            "action_mask": action_mask,
        }
        return action, info

    def get_action(self, state, action_space, deterministic: bool = False):
        action, _ = self.act(state, action_space, deterministic=deterministic)
        return action

    def evaluate_state_value(self, state) -> float:
        state_vec = torch.from_numpy(self.encode_state(state)).to(self.device)

        with torch.no_grad():
            _, value = self.model(state_vec.unsqueeze(0))

        return float(value.squeeze(0).item())

    def store_transition(
        self,
        state_tensor: torch.Tensor,
        action_idx: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
    ):
        self.buffer.add(
            state=state_tensor.detach(),
            action=action_idx,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            action_mask=action_mask.detach(),
        )
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        return len(self.buffer) >= self.update_timesteps

    def _compute_advantages(self, rewards, values, dones, last_value=0.0):
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=rewards.device)
        next_value = torch.tensor(last_value, device=rewards.device)
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * next_value * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            next_value = values[step]
        returns = advantages + values
        return advantages, returns

    def update(self, last_value: Optional[float] = None):
        if len(self.buffer) == 0:
            return {}

        data = self.buffer.as_tensors()
        bootstrap_value = 0.0 if last_value is None else float(last_value)
        advantages, returns = self._compute_advantages(
            rewards=data.reward,
            values=data.value,
            dones=data.done,
            last_value=bootstrap_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(self.buffer)
        indices = np.arange(batch_size)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                minibatch_idx = indices[start:end]

                mb_states = data.state[minibatch_idx]
                mb_actions = data.action[minibatch_idx]
                mb_old_log_probs = data.log_prob[minibatch_idx]
                mb_returns = returns[minibatch_idx]
                mb_advantages = advantages[minibatch_idx]
                mb_masks = data.action_mask[minibatch_idx]

                logits, values = self.model(mb_states)
                masked_logits = torch.where(
                    mb_masks, logits, torch.full_like(logits, -1e9)
                )
                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self.buffer.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, path: Optional[str] = None):
        path = path or self.policy_path
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict()
                if self.optimizer is not None
                else None,
            },
            path,
        )

    def load(self, path: Optional[str] = None):
        path = path or self.policy_path
        if not path or not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if self.optimizer is not None and checkpoint.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
