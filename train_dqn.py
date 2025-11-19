from ludo import Ludo
from policy_dqn import Policy_DQN
from policy_random import Policy_Random
from policy_heuristic import Policy_Heuristic
from tqdm import tqdm
import numpy as np
import copy
from misc import plot_training_curves


class OpponentManager:
    """
    Manages opponent selection and self-play snapshots for diverse training
    """

    def __init__(self, agent, max_snapshots=10):
        """
        Args:
            agent: The DQN agent being trained
            max_snapshots: Maximum number of snapshots to keep
        """
        self.agent = agent
        self.max_snapshots = max_snapshots
        self.snapshots = []  # List of (episode, state_dict) tuples

        # Initialize opponent policies
        self.random_policy = Policy_Random()
        self.heuristic_policy = Policy_Heuristic()

        # Create self-play agent (shares device with main agent)
        # Don't load a model - we'll load snapshots dynamically
        self.self_play_agent = Policy_DQN(
            training_mode=False,
            device=agent.device,
            use_prioritized_replay=agent.use_prioritized_replay,
            policy_path="dummy_path_no_load.pth",  # Dummy path that doesn't exist
        )

    def save_snapshot(self, episode):
        """Save current agent state as a snapshot for self-play"""
        snapshot = copy.deepcopy(self.agent.policy_net.state_dict())
        self.snapshots.append((episode, snapshot))

        # Keep only recent snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def select_opponent(self, episode, n_episodes):
        progress = episode / n_episodes

        if progress < 0.2:  # First 20%: Learn basics
            probs = [0.7, 0.2, 0.1]  # Random, Heuristic, Self-play
        elif progress < 0.6:  # 20-60%: Transition to advanced
            probs = [0.1, 0.7, 0.2]
        else:  # 60%+: Master advanced play
            probs = [0.1, 0.2, 0.7]

        opponent_type = np.random.choice(["random", "heuristic", "self_play"], p=probs)

        if opponent_type == "random":
            return self.random_policy, "Random"

        elif opponent_type == "heuristic":
            return self.heuristic_policy, "Heuristic"

        else:  # self_play
            # Load a snapshot for diverse self-play
            if self.snapshots:
                # Sample from recent snapshots (last 5 or all if less)
                recent_snapshots = self.snapshots[-5:]
                snapshot_episode, snapshot_state = recent_snapshots[
                    np.random.randint(len(recent_snapshots))
                ]

                # Load snapshot into self-play agent
                self.self_play_agent.policy_net.load_state_dict(snapshot_state)
                self.self_play_agent.target_net.load_state_dict(snapshot_state)

                return self.self_play_agent, f"Self-Play (ep {snapshot_episode})"
            else:
                # No snapshots yet, use current agent
                return self.agent, "Self-Play (Current)"


def train_dqn(
    n_episodes=50000,
    opponent="mixed",  # "random", "heuristic", "self", or "mixed"
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.9995,
    learning_rate=0.001,
    gamma=0.99,
    batch_size=64,
    buffer_size=100000,
    target_update_freq=1000,
    save_path="policy_dqn.pth",
    eval_interval=1000,
    save_interval=5000,
    snapshot_interval=2000,  # How often to save snapshots for self-play
    load_checkpoint=False,
    use_prioritized_replay=True,
    per_alpha=0.6,
    per_beta=0.4,
):
    env = Ludo()

    # Initialize DQN agent with improved architecture
    agent = Policy_DQN(
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        training_mode=True,
        policy_path=f"models/{save_path}",
        use_prioritized_replay=use_prioritized_replay,
        per_alpha=per_alpha,
        per_beta=per_beta,
    )

    # Load checkpoint if requested
    if load_checkpoint:
        agent.load(f"models/{save_path}")

    # Initialize opponent system
    use_mixed_opponents = opponent == "mixed"

    if use_mixed_opponents:
        opponent_manager = OpponentManager(agent, max_snapshots=10)
    else:
        # Single opponent mode for backward compatibility
        if opponent == "random":
            opponent_policy = Policy_Random()
        elif opponent == "heuristic":
            opponent_policy = Policy_Heuristic()
        elif opponent == "self":
            opponent_policy = agent
        else:
            raise ValueError(f"Unknown opponent type: {opponent}")

    # Training metrics
    wins = []
    avg_rewards = []
    episode_lengths = []
    win_rate_history = []

    # Track wins by opponent type (for mixed training)
    opponent_stats = {
        "Random": {"wins": 0, "games": 0},
        "Heuristic": {"wins": 0, "games": 0},
        "Self-Play": {"wins": 0, "games": 0},
    }

    print("=" * 60)
    print("STARTING IMPROVED DQN TRAINING")
    print("=" * 60)
    print("Architecture: Dueling DQN + Attention + Double DQN")
    print(f"Replay Buffer: {'Prioritized' if use_prioritized_replay else 'Standard'}")
    print(f"Episodes: {n_episodes}")
    print(f"Device: {agent.device}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size}")
    print(f"Learning rate: {learning_rate}")
    if use_prioritized_replay:
        print(f"PER Alpha: {per_alpha}, Beta: {per_beta}")
    print("=" * 60 + "\n")

    for episode in tqdm(range(n_episodes), desc="Training DQN"):
        # Select opponent for this episode
        if use_mixed_opponents:
            opponent_policy, opponent_name = opponent_manager.select_opponent(
                episode, n_episodes
            )
            # Categorize opponent for stats
            if "Random" in opponent_name:
                opponent_category = "Random"
            elif "Heuristic" in opponent_name:
                opponent_category = "Heuristic"
            else:
                opponent_category = "Self-Play"
        else:
            opponent_name = opponent
            opponent_category = opponent.capitalize()

        state = env.reset()
        terminated = False
        player_turn = 0
        episode_length = 0

        # Determine if agent plays first or second
        agent_player = episode % 2

        policies = (
            [agent, opponent_policy] if agent_player == 0 else [opponent_policy, agent]
        )

        while not terminated:
            action_space = env.get_action_space()
            action = policies[player_turn].get_action(state, action_space)

            next_state = env.step(action)
            terminated = next_state[3]
            next_player_turn = next_state[4]
            episode_length += 1

            # Update Q-network for agent
            if player_turn == agent_player and not terminated:
                # Small step penalty to encourage faster wins
                reward = -0.0001
                next_action_space = (
                    env.get_action_space() if next_player_turn == agent_player else []
                )
                agent.update(reward, next_state, next_action_space)

            state = next_state
            player_turn = next_player_turn

        # Episode ended - give final reward
        winner = state[4]
        agent_won = winner == agent_player
        final_reward = 1.0 if agent_won else -1.0
        agent.episode_end(final_reward)

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Track metrics
        wins.append(1 if agent_won else 0)
        avg_rewards.append(final_reward)
        episode_lengths.append(episode_length)

        # Track opponent-specific stats
        if use_mixed_opponents:
            opponent_stats[opponent_category]["games"] += 1
            if agent_won:
                opponent_stats[opponent_category]["wins"] += 1

        # Save snapshots for self-play
        if use_mixed_opponents and (episode + 1) % snapshot_interval == 0:
            opponent_manager.save_snapshot(episode + 1)

        # Evaluation and logging
        if (episode + 1) % eval_interval == 0:
            win_rate = sum(wins[-eval_interval:]) / min(eval_interval, len(wins))
            win_rate_history.append((episode + 1, win_rate))
            avg_length = np.mean(episode_lengths[-eval_interval:])

            print(f"\n\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Win Rate (last {eval_interval}): {win_rate:.2%}")
            print(f"  Avg Episode Length: {avg_length:.1f}")
            print(f"  Replay Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Total Steps: {agent.steps_done}")

            # Show opponent-specific stats
            if use_mixed_opponents:
                print("\n  Opponent Win Rates (all-time):")
                for opp_name, stats in opponent_stats.items():
                    if stats["games"] > 0:
                        opp_wr = stats["wins"] / stats["games"]
                        print(
                            f"    vs {opp_name}: {opp_wr:.2%} ({stats['wins']}/{stats['games']})"
                        )
                print(f"  Snapshots saved: {len(opponent_manager.snapshots)}")
            print()

        # Save checkpoint periodically
        if (episode + 1) % save_interval == 0:
            agent.save(f"models/{save_path}")
            print(f"Checkpoint saved at episode {episode + 1}")

    # Save final model
    agent.save(f"models/{save_path}")

    # Plot training curves
    plot_training_curves(win_rate_history, episode_lengths, avg_rewards, save_path)

    # Final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(
        f"Final win rate: {sum(wins[-eval_interval:]) / min(eval_interval, len(wins)):.2%}"
    )

    if use_mixed_opponents:
        print("\nFinal Win Rates by Opponent Type:")
        for opp_name, stats in opponent_stats.items():
            if stats["games"] > 0:
                opp_wr = stats["wins"] / stats["games"]
                print(
                    f"  vs {opp_name}: {opp_wr:.2%} ({stats['wins']}/{stats['games']})"
                )

    print(f"\nReplay buffer size: {len(agent.replay_buffer)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total episodes: {agent.episode_count}")
    print(f"Total steps: {agent.steps_done}")
    print(f"Model saved to: models/{save_path}")
    print("=" * 60 + "\n")

    return agent


if __name__ == "__main__":
    # Train DQN agent with improved architecture and mixed opponents
    agent = train_dqn(
        n_episodes=50000,
        opponent="mixed",  # Use mixed opponents (random + heuristic + self-play)
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        learning_rate=0.001,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000,
        target_update_freq=1000,
        eval_interval=1000,
        save_interval=5000,
        snapshot_interval=2000,  # Save agent snapshot every 2000 episodes
        load_checkpoint=False,
        use_prioritized_replay=True,
        per_alpha=0.6,
        per_beta=0.4,
    )
