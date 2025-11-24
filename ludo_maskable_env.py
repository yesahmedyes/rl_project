from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from features import MAX_ACTIONS, STATE_DIM, calculate_dense_reward, encode_ludo_state
from ludo import Ludo
from milestone2 import Policy_Milestone2
from policy_heuristic import Policy_Heuristic
from policy_random import Policy_Random

OpponentName = str


class LudoMaskableEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        opponents: Sequence[OpponentName] = ("heuristic", "random", "milestone2"),
        dense_rewards: bool = True,
        alternate_start: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(MAX_ACTIONS)

        if not opponents:
            raise ValueError("At least one opponent must be specified")

        self._opponent_names = tuple(opponents)
        self._dense_rewards = dense_rewards
        self._alternate_start = alternate_start

        self._rng = np.random.default_rng(seed)

        self._ludo: Optional[Ludo] = None
        self._state = None
        self._agent_player = 0
        self._opponent_policy = None
        self._last_mask: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._ludo = Ludo()
        self._state = self._ludo.reset()
        self._agent_player = self._choose_agent_player()
        self._opponent_policy = self._make_opponent()

        self._state, _ = self._auto_to_agent(self._state)
        observation = encode_ludo_state(self._state)
        self._last_mask = self._compute_action_mask()
        info = {"action_mask": self._last_mask}

        return observation, info

    def step(self, action: int):
        if self._ludo is None or self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self._state[3]:
            return self._terminal_observation()

        action_tuple = self._index_to_action(action)
        valid_actions = self._ludo.get_action_space()

        if action_tuple not in valid_actions:
            observation = encode_ludo_state(self._state)
            info = {"action_mask": self._last_mask}

            return observation, -5.0, True, False, info

        next_state = self._ludo.step(action_tuple)
        reward = self._transition_reward(self._state, next_state)
        self._state = next_state

        self._state, auto_reward = self._auto_to_agent(self._state)
        reward += auto_reward

        observation = encode_ludo_state(self._state)
        self._last_mask = self._compute_action_mask()
        done = self._state[3]
        info = {"action_mask": self._last_mask}

        return observation, reward, done, False, info

    def get_action_mask(self) -> np.ndarray:
        if self._last_mask is None:
            self._last_mask = self._compute_action_mask()

        return self._last_mask

    def _auto_to_agent(self, state) -> Tuple:
        total_reward = 0.0

        while not state[3]:
            if state[4] == self._agent_player:
                action_space = self._ludo.get_action_space()

                if action_space:
                    break

                next_state = self._ludo.step(None)
                total_reward += self._transition_reward(state, next_state)
                state = next_state

                continue

            action_space = self._ludo.get_action_space()

            if action_space:
                action = self._opponent_policy.get_action(state, action_space)
            else:
                action = None

            next_state = self._ludo.step(action)
            total_reward += self._transition_reward(state, next_state)

            state = next_state

        return state, total_reward

    def _compute_action_mask(self) -> np.ndarray:
        if self._state is None or self._state[3]:
            return np.zeros(MAX_ACTIONS, dtype=bool)

        mask = np.zeros(MAX_ACTIONS, dtype=bool)

        for dice_idx, goti_idx in self._ludo.get_action_space():
            action_idx = dice_idx * 4 + goti_idx
            action_idx = max(0, min(action_idx, MAX_ACTIONS - 1))
            mask[action_idx] = True

        if not mask.any():
            # Should not happen because _auto_to_agent ensures at least one action.
            mask[0] = True

        return mask

    def _transition_reward(self, state_before, state_after):
        if not self._dense_rewards:
            if state_after[3]:
                return 1.0 if state_after[4] == self._agent_player else 0.0
            return 0.0

        terminated = state_after[3]
        agent_won = terminated and state_after[4] == self._agent_player

        return calculate_dense_reward(
            state_before, state_after, terminated, agent_won, self._agent_player
        )

    def _index_to_action(self, action_index: int) -> Tuple[int, int]:
        dice_idx = max(0, action_index // 4)
        goti_idx = max(0, action_index % 4)

        return dice_idx, goti_idx

    def _choose_agent_player(self) -> int:
        if not self._alternate_start:
            return 0

        return int(self._rng.integers(0, 2))

    def _make_opponent(self):
        choice = self._rng.choice(self._opponent_names)

        if choice == "random":
            return Policy_Random()
        if choice == "heuristic":
            return Policy_Heuristic()
        if choice == "milestone2":
            return Policy_Milestone2()

        raise ValueError(f"Unsupported opponent '{choice}'")

    def _terminal_observation(self):
        observation = encode_ludo_state(self._state)

        info = {"action_mask": np.zeros(MAX_ACTIONS, dtype=bool)}

        return observation, 0.0, True, False, info

    def agent_won(self) -> bool:
        if self._state is None or not self._state[3]:
            return False

        return self._state[4] == self._agent_player
