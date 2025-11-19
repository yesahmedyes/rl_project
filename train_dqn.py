from ludo import Ludo
from policy_dqn import Policy_DQN
from policy_random import Policy_Random
from policy_heuristic import Policy_Heuristic
from tqdm import tqdm
import numpy as np
import copy
import os
import csv
import pickle
import signal
import torch
from datetime import datetime
from pathlib import Path
from misc import plot_training_curves


# Global flag for graceful shutdown
interrupted = False


def signal_handler(sig, frame):
    """Handle keyboard interrupt gracefully"""
    global interrupted
    print("\n\n⚠️  Keyboard interrupt received!")
    print("Finishing current episode and saving...")
    interrupted = True


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ["models", "logs", "plots", "checkpoints"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return directories


def cleanup_old_checkpoints(save_path, keep_last=3):
    """Keep only the most recent checkpoints to save disk space"""
    import glob

    # Get base name without extension
    base_name = save_path.replace(".pth", "")
    checkpoint_pattern = f"checkpoints/{base_name}_ep*.pth"

    # Get all checkpoint files sorted by modification time
    checkpoints = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)

    # Remove old checkpoints, keeping only the last N
    if len(checkpoints) > keep_last:
        for old_checkpoint in checkpoints[:-keep_last]:
            try:
                os.remove(old_checkpoint)
                print(f"  Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                print(f"  Warning: Could not remove {old_checkpoint}: {e}")


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
        if episode < 10000:
            probs = [0.8, 0.1, 0.1]  # Random, Heuristic, Self-play
        elif episode < 50000:
            probs = [0.2, 0.6, 0.2]
        elif episode < 200000:
            probs = [0.2, 0.2, 0.6]
        else:
            probs = [0.1, 0.1, 0.8]

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
    eval_interval=5000,  # Quick stats logging
    save_interval=50000,  # Full checkpoint + plot every 50k
    snapshot_interval=10000,  # Self-play snapshots
    csv_log_interval=1000,  # CSV logging frequency
    load_checkpoint=False,
    checkpoint_path=None,  # Specific checkpoint to load
    use_prioritized_replay=True,
    per_alpha=0.6,
    per_beta=0.4,
):
    global interrupted

    # Ensure all directories exist
    ensure_directories()

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
    start_episode = 0
    if load_checkpoint:
        checkpoint_file = checkpoint_path if checkpoint_path else f"models/{save_path}"
        if os.path.exists(checkpoint_file):
            agent.load(checkpoint_file)
            print(f"✅ Loaded checkpoint from {checkpoint_file}")
        else:
            print(f"⚠️  Checkpoint {checkpoint_file} not found, starting fresh")

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

    # Training metrics with memory management
    MAX_HISTORY = 100000  # Prevent memory overflow on long runs
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

    # Best model tracking
    best_win_rate = 0.0
    best_episode = 0

    # Initialize CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "win_rate",
                "avg_length",
                "epsilon",
                "buffer_size",
                "steps_done",
                "timestamp",
                "gpu_memory_allocated_gb",
                "gpu_memory_max_gb",
            ]
        )

    print("=" * 60)
    print("STARTING IMPROVED DQN TRAINING")
    print("=" * 60)
    print("Architecture: Dueling DQN + Attention + Double DQN")
    print(f"Replay Buffer: {'Prioritized' if use_prioritized_replay else 'Standard'}")
    print(
        f"Max Episodes: {n_episodes if n_episodes != float('inf') else 'Infinite (Ctrl+C to stop)'}"
    )
    print(f"Device: {agent.device}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size}")
    print(f"Learning rate: {learning_rate}")
    if use_prioritized_replay:
        print(f"PER Alpha: {per_alpha}, Beta: {per_beta}")
    print("\nOutput directories:")
    print("  Models: models/")
    print("  Checkpoints: checkpoints/")
    print("  Logs: logs/")
    print("  Plots: plots/")
    print(f"\nLog file: {log_file}")
    print("=" * 60 + "\n")

    # Convert infinite episodes to very large number for range
    n_episodes_actual = int(n_episodes) if n_episodes != float("inf") else 10**9

    for episode in tqdm(
        range(start_episode, n_episodes_actual),
        desc="Training DQN",
        initial=start_episode,
    ):
        # Check for interrupt
        if interrupted:
            print("\n🛑 Stopping training due to keyboard interrupt...")
            break

        # Select opponent for this episode
        if use_mixed_opponents:
            opponent_policy, opponent_name = opponent_manager.select_opponent(
                episode, n_episodes_actual
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

        # Memory management - trim lists if too large
        if len(wins) > MAX_HISTORY:
            wins = wins[-MAX_HISTORY:]
            avg_rewards = avg_rewards[-MAX_HISTORY:]
            episode_lengths = episode_lengths[-MAX_HISTORY:]

        # Track opponent-specific stats
        if use_mixed_opponents:
            opponent_stats[opponent_category]["games"] += 1

            if agent_won:
                opponent_stats[opponent_category]["wins"] += 1

        # Save snapshots for self-play
        if use_mixed_opponents and (episode + 1) % snapshot_interval == 0:
            opponent_manager.save_snapshot(episode + 1)

        # CSV logging (frequent)
        if (episode + 1) % csv_log_interval == 0:
            recent_win_rate = sum(wins[-csv_log_interval:]) / min(
                csv_log_interval, len(wins)
            )
            recent_avg_length = np.mean(episode_lengths[-csv_log_interval:])

            # GPU memory stats
            gpu_mem_alloc = (
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            )
            gpu_mem_max = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0
            )

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode + 1,
                        recent_win_rate,
                        recent_avg_length,
                        agent.epsilon,
                        len(agent.replay_buffer),
                        agent.steps_done,
                        datetime.now().isoformat(),
                        gpu_mem_alloc,
                        gpu_mem_max,
                    ]
                )

        # Evaluation and logging (less frequent)
        if (episode + 1) % eval_interval == 0:
            win_rate = sum(wins[-eval_interval:]) / min(eval_interval, len(wins))
            win_rate_history.append((episode + 1, win_rate))
            avg_length = np.mean(episode_lengths[-eval_interval:])

            print(f"\n\nEpisode {episode + 1}/{n_episodes_actual}")
            print(f"  Win Rate (last {eval_interval}): {win_rate:.2%}")
            print(f"  Avg Episode Length: {avg_length:.1f}")
            print(f"  Replay Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Total Steps: {agent.steps_done}")

            # GPU memory monitoring
            if torch.cuda.is_available():
                gpu_mem_alloc = torch.cuda.memory_allocated() / 1e9
                gpu_mem_max = torch.cuda.max_memory_allocated() / 1e9
                print(f"  GPU Memory: {gpu_mem_alloc:.2f}GB / {gpu_mem_max:.2f}GB max")
                torch.cuda.reset_peak_memory_stats()

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

            # Track best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_episode = episode + 1
                agent.save(f"models/best_{save_path}")
                print(f"\n  🏆 New best win rate: {win_rate:.2%}")

            print()

        # Save checkpoint periodically (every 50k episodes)
        if (episode + 1) % save_interval == 0:
            # Save versioned checkpoint
            checkpoint_name = f"{save_path.replace('.pth', '')}_ep{episode + 1}.pth"
            agent.save(f"checkpoints/{checkpoint_name}")

            # Also update the "latest" checkpoint
            agent.save(f"models/{save_path}")

            print(f"\n💾 Checkpoint saved at episode {episode + 1}")
            print(f"   Versioned: checkpoints/{checkpoint_name}")
            print(f"   Latest: models/{save_path}")

            # Save training state for resume capability
            training_state = {
                "episode": episode + 1,
                "wins": wins,
                "avg_rewards": avg_rewards,
                "episode_lengths": episode_lengths,
                "win_rate_history": win_rate_history,
                "opponent_stats": opponent_stats,
                "best_win_rate": best_win_rate,
                "best_episode": best_episode,
            }
            with open(f"checkpoints/training_state_ep{episode + 1}.pkl", "wb") as f:
                pickle.dump(training_state, f)

            # Generate updated plot with both regular and timestamped versions
            plot_save_name = save_path.replace(".pth", "")

            # Save latest plot (overwrites)
            plot_training_curves(
                win_rate_history,
                episode_lengths,
                avg_rewards,
                plot_save_name,
                output_dir="plots",
            )

            # Save timestamped plot for history
            timestamped_name = f"{plot_save_name}_ep{episode + 1}_{timestamp}"
            plot_training_curves(
                win_rate_history,
                episode_lengths,
                avg_rewards,
                timestamped_name,
                output_dir="plots",
            )

            print(f"   Plot updated: plots/{plot_save_name}_*.png")

            # Cleanup old checkpoints (keep last 3)
            cleanup_old_checkpoints(save_path, keep_last=3)
            print()

    # Save final model
    agent.save(f"models/{save_path}")
    print(f"\n💾 Final model saved to: models/{save_path}")

    # Final plot
    plot_training_curves(
        win_rate_history,
        episode_lengths,
        avg_rewards,
        save_path.replace(".pth", ""),
        output_dir="plots",
    )

    # Final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!" if not interrupted else "TRAINING INTERRUPTED!")
    print("=" * 60)

    if len(wins) >= eval_interval:
        final_win_rate = sum(wins[-eval_interval:]) / eval_interval
    else:
        final_win_rate = sum(wins) / len(wins) if wins else 0

    print(f"Final win rate: {final_win_rate:.2%}")
    print(f"Best win rate: {best_win_rate:.2%} (episode {best_episode})")

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
    print(f"\nBest model saved to: models/best_{save_path}")
    print(f"Latest model saved to: models/{save_path}")
    print(f"Training log saved to: {log_file}")
    print("=" * 60 + "\n")

    return agent


if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Train DQN agent with improved architecture and mixed opponents
        # Set to run indefinitely until keyboard interrupt (Ctrl+C)
        agent = train_dqn(
            n_episodes=10000000,  # Run until interrupted (essentially infinite)
            opponent="mixed",  # Use mixed opponents (random + heuristic + self-play)
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            learning_rate=0.001,
            gamma=0.99,
            batch_size=64,
            buffer_size=100000,
            target_update_freq=1000,
            eval_interval=5000,  # Quick stats every 5k episodes
            save_interval=50000,  # Full checkpoint + plot every 50k episodes
            snapshot_interval=10000,  # Self-play snapshots every 10k
            csv_log_interval=1000,  # Log metrics to CSV every 1k episodes
            load_checkpoint=False,  # Set to True to resume from checkpoint
            checkpoint_path=None,  # Specify checkpoint file to load
            use_prioritized_replay=True,
            per_alpha=0.6,
            per_beta=0.4,
        )
    except KeyboardInterrupt:
        print("\n\n✅ Training interrupted gracefully!")
        if "agent" in locals():
            agent.save("models/interrupted_save.pth")
            print("💾 Emergency checkpoint saved to: models/interrupted_save.pth")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        if "agent" in locals():
            agent.save("models/emergency_save.pth")
            print("💾 Emergency checkpoint saved to: models/emergency_save.pth")
