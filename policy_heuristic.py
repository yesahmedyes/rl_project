import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import DQN

from ludo import DESTINATION, SAFE_SQUARES, STARTING

ActionType = Tuple[int, int]
StateType = Tuple
StateEncoder = Callable[[StateType, Optional[ActionType]], np.ndarray]
ActionEncoder = Callable[[ActionType], int]
TransitionFn = Callable[[StateType, ActionType], Tuple[StateType, float, bool]]
ActionSpaceFn = Callable[[StateType], Sequence[ActionType]]
TerminalFn = Callable[[StateType, bool], bool]
RewardFn = Callable[[StateType, float, bool], float]


class _MCTSNode:
    def __init__(
        self,
        state: StateType,
        parent: Optional["_MCTSNode"],
        action: Optional[ActionType],
        terminal: bool,
        prior_reward: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.terminal = terminal
        self.prior_reward = prior_reward
        self.children: Dict[ActionType, "_MCTSNode"] = {}
        self.untried_actions: List[ActionType] = []
        self.visits = 0
        self.total_value = 0.0

    def average_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class SimpleMCTS:
    def __init__(
        self,
        transition_fn: TransitionFn,
        action_space_fn: ActionSpaceFn,
        terminal_fn: TerminalFn,
        reward_fn: RewardFn,
        rollouts: int = 64,
        max_depth: int = 4,
        exploration_c: float = 1.4,
        discount: float = 0.95,
    ) -> None:
        self.transition_fn = transition_fn
        self.action_space_fn = action_space_fn
        self.terminal_fn = terminal_fn
        self.reward_fn = reward_fn
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.exploration_c = exploration_c
        self.discount = discount

    def evaluate(
        self, state: StateType, root_actions: Sequence[ActionType]
    ) -> Dict[ActionType, float]:
        root = _MCTSNode(
            state=state,
            parent=None,
            action=None,
            terminal=self.terminal_fn(state, False),
        )
        root.untried_actions = list(root_actions)

        if not root.untried_actions:
            return {}

        for _ in range(self.rollouts):
            node = self._tree_policy(root)
            simulation_value = self._default_policy(node)
            self._backup(node, simulation_value)

        scores: Dict[ActionType, float] = {}
        for action in root_actions:
            child = root.children.get(action)
            if child is None:
                scores[action] = 0.0
                continue
            scores[action] = child.average_value()
        return scores

    def _tree_policy(self, node: _MCTSNode) -> _MCTSNode:
        current = node
        while not current.terminal:
            if current.untried_actions:
                return self._expand(current)
            if not current.children:
                break
            current = self._best_child(current)
        return current

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        next_state, env_reward, done = self.transition_fn(node.state, action)
        terminal = self.terminal_fn(next_state, done)
        reward = self.reward_fn(next_state, env_reward, terminal)
        child = _MCTSNode(
            state=next_state,
            parent=node,
            action=action,
            terminal=terminal,
            prior_reward=reward,
        )
        if not terminal:
            child.untried_actions = list(self.action_space_fn(next_state))
        node.children[action] = child
        return child

    def _best_child(self, node: _MCTSNode) -> _MCTSNode:
        best_score = float("-inf")
        best_child: Optional[_MCTSNode] = None
        for child in node.children.values():
            exploitation = child.average_value()
            exploration = (
                self.exploration_c
                * (np.log(node.visits + 1) / (child.visits + 1e-9)) ** 0.5
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child or random.choice(list(node.children.values()))

    def _default_policy(self, node: _MCTSNode) -> float:
        cumulative_reward = node.prior_reward
        depth = 1
        current_state = node.state
        terminal = node.terminal

        while not terminal and depth < self.max_depth:
            actions = list(self.action_space_fn(current_state))
            if not actions:
                break
            action = random.choice(actions)
            current_state, env_reward, done = self.transition_fn(current_state, action)
            terminal = self.terminal_fn(current_state, done)
            reward = self.reward_fn(current_state, env_reward, terminal)
            cumulative_reward += (self.discount**depth) * reward
            depth += 1
        return cumulative_reward

    def _backup(self, node: _MCTSNode, value: float) -> None:
        current_value = value
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += current_value
            current_value *= self.discount
            current = current.parent


class Policy_Heuristic:
    def __init__(
        self,
        heuristic_weight: float = 0.4,
        dqn_weight: float = 0.4,
        mcts_weight: float = 0.2,
        dqn_model_path: Optional[str] = None,
        dqn_model: Optional[DQN] = None,
        state_encoder: Optional[StateEncoder] = None,
        action_encoder: Optional[ActionEncoder] = None,
        mcts_transition_fn: Optional[TransitionFn] = None,
        mcts_action_space_fn: Optional[ActionSpaceFn] = None,
        mcts_terminal_fn: Optional[TerminalFn] = None,
        mcts_reward_fn: Optional[RewardFn] = None,
        mcts_rollouts: int = 64,
        mcts_max_depth: int = 4,
        mcts_exploration_c: float = 1.4,
        mcts_discount: float = 0.95,
    ) -> None:
        self.heuristic_weight = heuristic_weight
        self.dqn_weight = dqn_weight
        self.mcts_weight = mcts_weight
        self._normalize_weights()

        if dqn_model is not None:
            self.dqn_model = dqn_model
        elif dqn_model_path is not None:
            self.dqn_model = DQN.load(dqn_model_path)
        else:
            self.dqn_model = None

        self.dqn_device = self.dqn_model.device if self.dqn_model is not None else "cpu"
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        if (
            mcts_transition_fn is not None
            and mcts_action_space_fn is not None
            and mcts_terminal_fn is not None
            and mcts_reward_fn is not None
        ):
            self.mcts = SimpleMCTS(
                transition_fn=mcts_transition_fn,
                action_space_fn=mcts_action_space_fn,
                terminal_fn=mcts_terminal_fn,
                reward_fn=mcts_reward_fn,
                rollouts=mcts_rollouts,
                max_depth=mcts_max_depth,
                exploration_c=mcts_exploration_c,
                discount=mcts_discount,
            )
        else:
            self.mcts = None

    def _normalize_weights(self) -> None:
        total = self.heuristic_weight + self.dqn_weight + self.mcts_weight
        if total <= 0:
            # Default back to heuristic-only if weights are invalid
            self.heuristic_weight, self.dqn_weight, self.mcts_weight = 1.0, 0.0, 0.0
            return
        self.heuristic_weight /= total
        self.dqn_weight /= total
        self.mcts_weight /= total

    def get_action(
        self, state: StateType, action_space: Sequence[ActionType]
    ) -> Optional[ActionType]:
        if not action_space:
            return None

        if len(action_space) == 1:
            return action_space[0]

        mcts_scores = self._evaluate_mcts(state, action_space)

        best_action: Optional[ActionType] = None
        best_score = float("-inf")
        for action in action_space:
            heuristic_score = self.evaluate_heuristic(state, action)
            dqn_score = self._evaluate_dqn(state, action)
            mcts_score = mcts_scores.get(action, 0.0)
            blended_score = (
                self.heuristic_weight * heuristic_score
                + self.dqn_weight * dqn_score
                + self.mcts_weight * mcts_score
            )
            if blended_score > best_score:
                best_score = blended_score
                best_action = action

        return best_action

    def _evaluate_dqn(self, state: StateType, action: ActionType) -> float:
        if (
            self.dqn_model is None
            or self.state_encoder is None
            or self.action_encoder is None
        ):
            return 0.0

        obs = self.state_encoder(state, action)
        action_idx = self.action_encoder(action)
        if action_idx is None:
            return 0.0

        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]

        obs_tensor = torch.as_tensor(
            obs_arr, dtype=torch.float32, device=self.dqn_device
        )

        with torch.no_grad():
            q_values = self.dqn_model.policy.q_net(obs_tensor)

        q_values_np = q_values.cpu().numpy()[0]
        if action_idx >= len(q_values_np):
            return 0.0
        return float(q_values_np[action_idx])

    def _evaluate_mcts(
        self, state: StateType, action_space: Sequence[ActionType]
    ) -> Dict[ActionType, float]:
        if self.mcts is None:
            return {}
        return self.mcts.evaluate(state, action_space)

    def evaluate_heuristic(self, state, action):
        def get_detailed_state_info(state):
            gotis_red, gotis_yellow, dice_roll, _, player_turn = state

            if player_turn == 0:
                my_gotis = [g.position for g in gotis_red.gotis]
                opp_gotis = [g.position for g in gotis_yellow.gotis]
            else:
                my_gotis = [g.position for g in gotis_yellow.gotis]
                opp_gotis = [g.position for g in gotis_red.gotis]

            return my_gotis, opp_gotis, dice_roll, player_turn

        def convert_to_opponent_position(position):
            """
            Same logic as Goti.convert_into_opponent_position in the env.
            Convert *our* board position into how the opponent sees it.
            """
            if position <= STARTING or position > 50 or position == 25:
                return -2  # cannot be converted

            if position <= 24:
                return position + 26

            return position - 26

        def calculate_danger(position, opp_gotis):
            """
            Estimate how dangerous a given position is based on opponent pieces.
            Higher value = more likely to be killed within 1–6 steps.
            """
            # Off-board or home or safe squares are not "dangerous"
            if position in SAFE_SQUARES:
                return 0.0

            if position < 0 or position >= DESTINATION:
                return 0.0

            opp_view_pos = convert_to_opponent_position(position)
            if opp_view_pos == -2:
                return 0.0

            danger = 0.0

            for opp_pos in opp_gotis:
                # Ignore opponents that are in base or already in home stretch
                if opp_pos < 0 or opp_pos >= 51:
                    continue

                # Distance along opponent’s loop (0..50)
                if opp_view_pos >= opp_pos:
                    distance = opp_view_pos - opp_pos
                else:
                    distance = opp_view_pos + 51 - opp_pos

                if 0 < distance <= 6:
                    danger += 0.8 / distance  # closer opponent = more danger

            return danger

        my_gotis, opp_gotis, dice_roll, player_turn = get_detailed_state_info(state)

        dice_index, goti_index = action
        dice_value = dice_roll[dice_index]
        current_pos = my_gotis[goti_index]

        # Compute new position according to game rules approximation
        if current_pos == STARTING:
            # Action space guarantees dice_value == 6 if this action is valid
            new_pos = 0
        else:
            new_pos = current_pos + dice_value

        score = 0.0

        # --- 0. DANGER BEFORE / AFTER (used later for escape bonus) ---
        danger_before = calculate_danger(current_pos, opp_gotis)
        danger_after = calculate_danger(new_pos, opp_gotis)

        # --- 1. BRINGING GOTI OUT OF BASE ---
        if current_pos == STARTING and dice_value == 6:
            # Bringing a new piece into play is valuable
            if 0 not in opp_gotis:
                score += 1.0  # safe(ish) to bring out
            else:
                score += 0.5  # still valuable but opponent is sitting on your spawn

        # --- 2. CAPTURING OPPONENT ---
        if new_pos != DESTINATION and new_pos >= 0:
            opp_view_pos = convert_to_opponent_position(new_pos)
            # In the env, killing happens when opponent.position == opp_view_pos
            if (
                opp_view_pos != -2
                and opp_view_pos in opp_gotis
                and opp_view_pos not in SAFE_SQUARES
            ):
                score += 2.0  # capturing is very valuable

        # --- 3. REACHING HOME ---
        if new_pos == DESTINATION:
            score += 3.0  # finishing a piece is highest value

        # --- 4. PROGRESS TOWARDS HOME ---
        if current_pos >= 0 and new_pos <= DESTINATION:
            # Reward actual distance progressed, scaled by how advanced the piece already is.
            progress = new_pos - current_pos
            # Piece further along gets slightly more value for progress
            progress_factor = 0.5 + (current_pos / DESTINATION) * 0.5  # 0.5..1.0
            score += progress * progress_factor * 0.05

        # --- 5. MOVING TO SAFETY / LEAVING SAFETY ---
        # Entering a safe square
        if new_pos in SAFE_SQUARES and current_pos not in SAFE_SQUARES:
            score += 0.4

        # Leaving a safe square into a non-safe one (unless we are starting)
        if (
            current_pos in SAFE_SQUARES
            and new_pos not in SAFE_SQUARES
            and current_pos != STARTING
        ):
            score -= 0.35

        # --- 6. DANGER ADJUSTMENT (AVOID / ESCAPE DANGER) ---
        # Penalize final danger at new position
        score -= danger_after

        # Bonus for escaping danger (danger_after < danger_before)
        if danger_after < danger_before:
            score += (danger_before - danger_after) * 0.5

        # --- 7. "WASTE OF SIX" LOGIC ---
        # If there are gotis in base, we prefer not to spend a 6 on a very advanced piece
        any_in_base = any(pos == STARTING for pos in my_gotis)
        if (
            dice_value == 6
            and current_pos > 40
            and new_pos < DESTINATION
            and any_in_base
        ):
            score -= 0.3

        # --- 8. SPREAD STRATEGY (DISCOURAGE CLUSTERING) ---
        future_positions = my_gotis.copy()
        future_positions[goti_index] = new_pos

        active_positions = [p for p in future_positions if 0 <= p < DESTINATION]
        duplicates = len(active_positions) - len(set(active_positions))
        if duplicates > 0:
            score -= duplicates * 0.1

        # --- 9. ENDGAME STRATEGY ---
        gotis_at_home = sum(1 for pos in my_gotis if pos == DESTINATION)
        if gotis_at_home >= 2 and 0 <= current_pos < DESTINATION:
            # In endgame, prioritize pieces closer to home more strongly
            score += (current_pos / DESTINATION) * 0.5

        # --- 10. MULTI-DICE MICRO-STRATEGY ---
        # Slightly prefer using smaller dice for low-impact moves when all else is equal.
        # This is a small adjustment so it doesn't override big strategic decisions.
        score -= dice_value * 0.02

        return score
