import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm
from env.ludo_gym_env import LudoGymEnv


class PPOPolicyWrapper:
    """
    Wrapper to make PPO model compatible with the policy interface used by opponents.
    """

    def __init__(self, model, use_deterministic: bool = True):
        """
        Initialize the PPO policy wrapper.

        Args:
            model: Trained PPO model
            use_deterministic: If True, use deterministic actions (no sampling)
        """
        self.model = model
        self.use_deterministic = use_deterministic

    def get_action(self, state, action_space):
        """
        Get action from the PPO model.

        Args:
            state: Game state tuple
            action_space: List of valid action tuples

        Returns:
            Action tuple or None
        """
        if not action_space:
            return None

        # Create a temporary environment to encode the state
        temp_env = LudoGymEnv(
            encoding_type=self.model.observation_space.shape[0] == 70
            and "handcrafted"
            or "onehot",
            opponent_type="random",  # Doesn't matter for encoding
        )
        temp_env.current_state = state

        # Encode state
        obs = temp_env._encode_state(state)

        # Get action mask
        action_mask = temp_env._get_action_mask()

        # Predict action
        action, _ = self.model.predict(
            obs, deterministic=self.use_deterministic, action_masks=action_mask
        )

        # Convert discrete action to tuple
        action_tuple = temp_env._action_to_tuple(int(action))

        # Verify action is valid
        if action_tuple in action_space:
            return action_tuple

        # Fallback: return first valid action if prediction is invalid
        return action_space[0] if action_space else None


def evaluate_agent(
    model,
    opponent_type: str = "random",
    opponent_policy: Optional[Any] = None,
    n_eval_episodes: int = 100,
    encoding_type: str = "handcrafted",
    agent_player: int = 0,
    use_deterministic: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained PPO agent against a specified opponent.

    Args:
        model: Trained PPO model
        opponent_type: Type of opponent ("random", "heuristic", or "self")
        opponent_policy: Custom opponent policy object
        n_eval_episodes: Number of episodes to evaluate
        encoding_type: State encoding type
        agent_player: Which player is the agent (0 or 1)
        use_deterministic: If True, use deterministic actions
        seed: Random seed
        verbose: If True, show progress bar

    Returns:
        Dictionary with evaluation statistics
    """
    # Create evaluation environment
    env = LudoGymEnv(
        encoding_type=encoding_type,
        opponent_type=opponent_type,
        opponent_policy=opponent_policy,
        agent_player=agent_player,
    )

    wins = 0
    losses = 0
    invalid_actions = 0
    total_steps = []
    total_rewards = []

    iterator = range(n_eval_episodes)
    if verbose:
        iterator = tqdm(iterator, desc=f"Evaluating vs {opponent_type}")

    for episode in iterator:
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        episode_reward = 0
        steps = 0

        while not terminated:
            action_mask = info.get("action_mask")

            # Predict action
            action, _ = model.predict(
                obs, deterministic=use_deterministic, action_masks=action_mask
            )

            # Step environment
            obs, reward, terminated, truncated, info = env.step(int(action))

            episode_reward += reward
            steps += 1

            if info.get("invalid_action", False):
                invalid_actions += 1
                break

        # Record results
        if info.get("agent_won", False):
            wins += 1
        else:
            losses += 1

        total_steps.append(steps)
        total_rewards.append(episode_reward)

    env.close()

    # Calculate statistics
    win_rate = wins / n_eval_episodes
    avg_steps = np.mean(total_steps)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    results = {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "invalid_actions": invalid_actions,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "n_episodes": n_eval_episodes,
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Evaluation Results (vs {opponent_type}):")
        print(f"{'=' * 50}")
        print(f"Win Rate: {win_rate * 100:.2f}%")
        print(f"Wins: {wins}/{n_eval_episodes}")
        print(f"Invalid Actions: {invalid_actions}")
        print(f"Avg Steps per Episode: {avg_steps:.2f}")
        print(f"Avg Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"{'=' * 50}\n")

    return results


def evaluate_alternating_players(
    model,
    opponent_type: str = "random",
    opponent_policy: Optional[Any] = None,
    n_eval_episodes: int = 100,
    encoding_type: str = "handcrafted",
    use_deterministic: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate agent playing as both player 0 and player 1.

    Args:
        model: Trained PPO model
        opponent_type: Type of opponent
        opponent_policy: Custom opponent policy object
        n_eval_episodes: Number of episodes (split evenly between both positions)
        encoding_type: State encoding type
        use_deterministic: If True, use deterministic actions
        seed: Random seed
        verbose: If True, show progress bar

    Returns:
        Dictionary with combined evaluation statistics
    """
    n_per_position = n_eval_episodes // 2

    # Evaluate as player 0
    results_p0 = evaluate_agent(
        model=model,
        opponent_type=opponent_type,
        opponent_policy=opponent_policy,
        n_eval_episodes=n_per_position,
        encoding_type=encoding_type,
        agent_player=0,
        use_deterministic=use_deterministic,
        seed=seed,
        verbose=False,
    )

    # Evaluate as player 1
    results_p1 = evaluate_agent(
        model=model,
        opponent_type=opponent_type,
        opponent_policy=opponent_policy,
        n_eval_episodes=n_per_position,
        encoding_type=encoding_type,
        agent_player=1,
        use_deterministic=use_deterministic,
        seed=seed + 10000,
        verbose=False,
    )

    # Combine results
    combined = {
        "win_rate": (results_p0["win_rate"] + results_p1["win_rate"]) / 2,
        "win_rate_p0": results_p0["win_rate"],
        "win_rate_p1": results_p1["win_rate"],
        "wins": results_p0["wins"] + results_p1["wins"],
        "losses": results_p0["losses"] + results_p1["losses"],
        "invalid_actions": results_p0["invalid_actions"]
        + results_p1["invalid_actions"],
        "avg_steps": (results_p0["avg_steps"] + results_p1["avg_steps"]) / 2,
        "avg_reward": (results_p0["avg_reward"] + results_p1["avg_reward"]) / 2,
        "n_episodes": n_eval_episodes,
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Alternating Evaluation Results (vs {opponent_type}):")
        print(f"{'=' * 50}")
        print(f"Overall Win Rate: {combined['win_rate'] * 100:.2f}%")
        print(f"  - As Player 0: {combined['win_rate_p0'] * 100:.2f}%")
        print(f"  - As Player 1: {combined['win_rate_p1'] * 100:.2f}%")
        print(f"Wins: {combined['wins']}/{n_eval_episodes}")
        print(f"Invalid Actions: {combined['invalid_actions']}")
        print(f"Avg Steps per Episode: {combined['avg_steps']:.2f}")
        print(f"Avg Reward: {combined['avg_reward']:.4f}")
        print(f"{'=' * 50}\n")

    return combined


def quick_eval(
    model,
    opponent_type: str = "random",
    n_episodes: int = 100,
    encoding_type: str = "handcrafted",
) -> float:
    """
    Quick evaluation returning just the win rate.

    Args:
        model: Trained PPO model
        opponent_type: Type of opponent
        n_episodes: Number of episodes
        encoding_type: State encoding type

    Returns:
        Win rate as a float
    """
    results = evaluate_alternating_players(
        model=model,
        opponent_type=opponent_type,
        n_eval_episodes=n_episodes,
        encoding_type=encoding_type,
        use_deterministic=True,
        verbose=False,
    )

    return results["win_rate"]
