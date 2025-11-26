import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm
from env.vec_env_factory import make_eval_env


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
    env = make_eval_env(
        encoding_type=encoding_type,
        opponent_type=opponent_type,
        opponent_policy=opponent_policy,
        agent_player=agent_player,
        seed=seed,
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
            action_masks = info.get("action_mask")

            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=use_deterministic
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
):
    results = evaluate_alternating_players(
        model=model,
        opponent_type=opponent_type,
        n_eval_episodes=n_episodes,
        encoding_type=encoding_type,
        use_deterministic=True,
        verbose=False,
    )

    return results["win_rate"]
