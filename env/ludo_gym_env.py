import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from env.ludo import Ludo
from misc.state_encoding import encode_handcrafted_state, encode_onehot_state
from misc.dense_reward import calculate_dense_reward
from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic


class LudoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        encoding_type: str = "handcrafted",
        opponent_policy: Optional[Any] = None,
        opponent_type: str = "random",
        agent_player: int = 0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.encoding_type = encoding_type
        self.agent_player = agent_player
        self.render_mode = render_mode
        self.opponent_type = opponent_type

        # Initialize the Ludo environment
        self.env = Ludo(render_mode="" if render_mode is None else render_mode)

        # Set up opponent policy
        if opponent_policy is not None:
            self.opponent_policy = opponent_policy
        elif opponent_type == "random":
            self.opponent_policy = Policy_Random()
        elif opponent_type == "heuristic":
            self.opponent_policy = Policy_Heuristic()
        elif opponent_type == "self":
            # Will be set externally for self-play
            self.opponent_policy = None
        else:
            raise ValueError(f"Unknown opponent_type: {opponent_type}")

        # Determine state dimensions based on encoding type
        if encoding_type == "handcrafted":
            self.state_dim = 70
        elif encoding_type == "onehot":
            self.state_dim = 946
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

        # Maximum possible actions: 3 dice rolls × 4 gotis = 12
        self.max_actions = 12

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.max_actions)

        # State tracking
        self.current_state = None
        self.prev_state = None
        self.steps_taken = 0

    def _encode_state(self, state) -> np.ndarray:
        """Encode the game state based on the encoding type."""
        if self.encoding_type == "handcrafted":
            _, _, encoded = encode_handcrafted_state(state)
        elif self.encoding_type == "onehot":
            _, _, encoded = encode_onehot_state(state)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        return encoded

    def _action_to_tuple(self, action: int) -> Tuple[int, int]:
        """
        Convert discrete action (0-11) to (dice_index, goti_index).

        Mapping:
        - action 0-3: dice_index=0, goti_index=0-3
        - action 4-7: dice_index=1, goti_index=0-3
        - action 8-11: dice_index=2, goti_index=0-3
        """
        dice_index = action // 4
        goti_index = action % 4
        return (dice_index, goti_index)

    def _tuple_to_action(self, action_tuple: Tuple[int, int]) -> int:
        """Convert (dice_index, goti_index) to discrete action."""
        dice_index, goti_index = action_tuple
        return dice_index * 4 + goti_index

    def _get_action_mask(self) -> np.ndarray:
        """
        Get a binary mask indicating which actions are valid.

        Returns:
            np.ndarray: Binary mask of shape (max_actions,)
        """
        mask = np.zeros(self.max_actions, dtype=np.int8)

        # Get valid actions from the environment
        action_space = self.env.get_action_space()

        # Convert each valid action tuple to discrete action and mark as valid
        for action_tuple in action_space:
            action = self._tuple_to_action(action_tuple)
            if action < self.max_actions:
                mask[action] = 1

        return mask

    def _play_opponent_turn(self) -> Tuple[Any, bool]:
        """
        Let the opponent play until it's the agent's turn again.

        Returns:
            Tuple of (state, terminated)
        """
        state = self.current_state
        terminated = state[3]
        player_turn = state[4]

        while not terminated and player_turn != self.agent_player:
            action_space = self.env.get_action_space()

            if self.opponent_policy is None:
                # For self-play when opponent policy not yet set
                action = None if not action_space else action_space[0]
            else:
                action = self.opponent_policy.get_action(state, action_space)

            state = self.env.step(action)
            terminated = state[3]
            player_turn = state[4]

        return state, terminated

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset the Ludo environment
        self.current_state = self.env.reset()
        self.prev_state = None
        self.steps_taken = 0

        # If agent is player 1, let opponent (player 0) play first
        if self.agent_player == 1:
            self.current_state, terminated = self._play_opponent_turn()
            if terminated:
                # Game ended before agent could play
                obs = self._encode_state(self.current_state)
                info = {
                    "action_mask": self._get_action_mask(),
                    "terminated": True,
                    "agent_won": False,
                }
                return obs, info

        # Encode observation
        obs = self._encode_state(self.current_state)

        # Get action mask
        info = {
            "action_mask": self._get_action_mask(),
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Discrete action (0-11)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.steps_taken += 1

        # Store previous state for reward calculation
        self.prev_state = self.current_state

        # Get action mask BEFORE converting action
        action_mask = self._get_action_mask()

        if np.sum(action_mask) == 0:
            # No valid moves - automatically pass turn
            new_state = self.env.step(None)  # Pass None to trigger auto-pass
            self.current_state = new_state
            terminated = new_state[3]

            # If not terminated, let opponent play
            if not terminated:
                player_turn = new_state[4]

                if player_turn != self.agent_player:
                    self.current_state, terminated = self._play_opponent_turn()

            reward = -0.05 / 100
            obs = self._encode_state(self.current_state)

            info = {
                "action_mask": self._get_action_mask(),
                "agent_won": False,
                "steps": self.steps_taken,
                "no_valid_actions": True,
            }

            return obs, reward, terminated, False, info

        # Convert discrete action to tuple
        action_tuple = self._action_to_tuple(action)

        # Validate action
        if action >= self.max_actions or action_mask[action] == 0:
            # Invalid action - penalize and terminate
            obs = self._encode_state(self.current_state)
            reward = -10.0
            terminated = True
            truncated = False
            info = {
                "action_mask": action_mask,
                "invalid_action": True,
                "agent_won": False,
            }
            return obs, reward, terminated, truncated, info

        # Execute action
        new_state = self.env.step(action_tuple)
        terminated = new_state[3]
        player_turn = new_state[4]

        # Check if agent won
        agent_won = False
        if terminated and player_turn != self.agent_player:
            # If terminated and it's not agent's turn, agent won
            agent_won = True

        # Update current state
        self.current_state = new_state

        # If game not terminated and it's opponent's turn, let opponent play
        if not terminated and player_turn != self.agent_player:
            self.current_state, terminated = self._play_opponent_turn()

            # Check if opponent won
            if terminated:
                player_turn = self.current_state[4]
                if player_turn == self.agent_player:
                    agent_won = False  # Opponent won

        # Calculate reward
        reward = calculate_dense_reward(
            self.prev_state,
            self.current_state,
            terminated,
            agent_won,
            self.agent_player,
        )

        # Encode observation
        obs = self._encode_state(self.current_state)

        # Get new action mask
        action_mask = self._get_action_mask()

        truncated = False  # We don't use truncation in this environment

        info = {
            "action_mask": action_mask,
            "agent_won": agent_won,
            "steps": self.steps_taken,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        """Close the environment."""
        pass

    def set_opponent_policy(self, policy):
        """Set the opponent policy (useful for self-play)."""
        self.opponent_policy = policy
