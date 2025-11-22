from policy_dqn import Policy_DQN
from tqdm import tqdm
import numpy as np
import os
import csv
import pickle
import signal
import torch
import traceback
from datetime import datetime
import multiprocessing as mp

from misc import plot_training_curves
from utils import ensure_directories, cleanup_old_checkpoints
from opponent import OpponentManager
from parallel_worker import rollout_worker

# Global flag for graceful shutdown
interrupted = False


def train_dqn(
    n_episodes=50000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.98,
    learning_rate=0.005,
    gamma=0.99,
    batch_size=2048,
    buffer_size=1000000,
    target_update_freq=1000,
    save_path="policy_dqn.pth",
    eval_interval=5000,
    save_interval=50000,
    snapshot_interval=10000,
    log_interval=1000,
    load_checkpoint=False,
    checkpoint_path=None,
    use_prioritized_replay=True,
    per_alpha=0.6,
    per_beta=0.4,
    num_workers=None,
    episodes_per_batch=10,
    learning_steps_per_batch=50,
):
    global interrupted

    ensure_directories()

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)  # Leave 2 cores for system

    print("\n🚀 Parallel Training Configuration:")
    print(f"   CPU Cores Available: {mp.cpu_count()}")
    print(f"   Workers: {num_workers}")
    print(f"   Episodes per batch: {num_workers * episodes_per_batch}")
    print(f"   Learning steps per batch: {learning_steps_per_batch}")
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   GPU: {gpu_device}")
    print()

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

    start_episode = 0

    if load_checkpoint:
        checkpoint_file = checkpoint_path if checkpoint_path else f"models/{save_path}"

        if os.path.exists(checkpoint_file):
            agent.load(checkpoint_file)

            print(f"✅ Loaded checkpoint from {checkpoint_file}")

    opponent_manager = OpponentManager(agent, max_snapshots=10)

    # Snapshots for workers
    opponent_snapshots = []

    MAX_HISTORY = 100000  # Prevent memory overflow on long runs

    wins = []
    avg_rewards = []
    episode_lengths = []
    win_rate_history = []

    opponent_stats = {
        "Random": {"wins": 0, "games": 0, "rewards": []},
        "Heuristic": {"wins": 0, "games": 0, "rewards": []},
        "Self-Play": {"wins": 0, "games": 0, "rewards": []},
    }

    best_win_rate = 0.0
    best_episode = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "win_rate_overall",
                "win_rate_vs_random",
                "win_rate_vs_heuristic",
                "win_rate_vs_self_play",
                "avg_reward",
                "avg_length",
                "epsilon",
                "buffer_size",
                "steps_done",
                "timestamp",
            ]
        )

    print("=" * 60)
    print("STARTING PARALLEL TRAINING")
    print("=" * 60 + "\n")

    # Convert infinite episodes to very large number for range
    n_episodes_actual = int(n_episodes) if n_episodes != float("inf") else 10**9

    # Calculate number of batches
    total_episodes_per_iteration = num_workers * episodes_per_batch
    n_iterations = n_episodes_actual // total_episodes_per_iteration

    total_episodes = 0

    # Create multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        for iteration in tqdm(
            range(n_iterations),
            desc="Training Batches",
            initial=start_episode // total_episodes_per_iteration,
        ):
            if interrupted:
                print("\n🛑 Stopping training due to keyboard interrupt...")
                break

            # Get current weights for workers
            weights = agent.get_weights()

            # Prepare opponent snapshots for self-play
            if len(opponent_manager.snapshots) > 0:
                # Get recent snapshots
                recent_snapshots = opponent_manager.snapshots[-5:]

                opponent_snapshots = [
                    {
                        "policy_net": snapshot[1],
                        "epsilon": agent.epsilon * 0.5,
                    }
                    for snapshot in recent_snapshots
                ]
            else:
                opponent_snapshots = []

            # Determine agent player positions for each episode
            agent_player_positions = [
                (total_episodes + i) % 2 for i in range(total_episodes_per_iteration)
            ]

            # Split positions across workers
            positions_per_worker = [
                agent_player_positions[
                    i * episodes_per_batch : (i + 1) * episodes_per_batch
                ]
                for i in range(num_workers)
            ]

            # Collect episodes in parallel
            results = pool.starmap(
                rollout_worker,
                [
                    (
                        worker_id,
                        episodes_per_batch,
                        weights,
                        positions_per_worker[worker_id],
                        opponent_snapshots,
                    )
                    for worker_id in range(num_workers)
                ],
            )

            # Process results from all workers
            batch_wins = []
            batch_rewards = []
            batch_lengths = []

            for worker_trajectories in results:
                for episode_data in worker_trajectories:
                    trajectory = episode_data["trajectory"]
                    opponent_type = episode_data["opponent_type"]
                    agent_won = episode_data["agent_won"]
                    episode_length = episode_data["episode_length"]
                    final_reward = episode_data["final_reward"]

                    if len(trajectory) > 0:
                        # Add trajectory to replay buffer
                        agent.add_trajectory_to_buffer(trajectory)

                        # Track metrics
                        batch_wins.append(1 if agent_won else 0)
                        batch_rewards.append(final_reward)
                        batch_lengths.append(episode_length)

                        # Track opponent-specific stats
                        opponent_name_map = {
                            "random": "Random",
                            "heuristic": "Heuristic",
                            "self_play": "Self-Play",
                        }
                        opponent_name = opponent_name_map.get(opponent_type, "Unknown")

                        if opponent_name not in opponent_stats:
                            opponent_stats[opponent_name] = {
                                "wins": 0,
                                "games": 0,
                                "rewards": [],
                            }
                        opponent_stats[opponent_name]["games"] += 1
                        opponent_stats[opponent_name]["rewards"].append(final_reward)
                        if agent_won:
                            opponent_stats[opponent_name]["wins"] += 1

            # Add to history
            wins.extend(batch_wins)
            avg_rewards.extend(batch_rewards)
            episode_lengths.extend(batch_lengths)

            # Memory management
            if len(wins) > MAX_HISTORY:
                wins = wins[-MAX_HISTORY:]
                avg_rewards = avg_rewards[-MAX_HISTORY:]
                episode_lengths = episode_lengths[-MAX_HISTORY:]

            # Perform learning updates
            if len(agent.replay_buffer) >= batch_size:
                for _ in range(learning_steps_per_batch):
                    agent._learn()
                    agent.steps_done += 1

                    # Update target network
                    if agent.steps_done % target_update_freq == 0:
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # Decay epsilon
            agent.decay_epsilon()

            # Update episode count
            total_episodes += total_episodes_per_iteration
            agent.episode_count = total_episodes

            # Save snapshots for self-play
            if total_episodes % snapshot_interval < total_episodes_per_iteration:
                opponent_manager.save_snapshot(total_episodes)

            # Logging
            if total_episodes % log_interval < total_episodes_per_iteration:
                recent_win_rate = sum(wins[-log_interval:]) / min(
                    log_interval, len(wins)
                )

                recent_avg_length = (
                    np.mean(episode_lengths[-log_interval:]) if episode_lengths else 0
                )

                recent_avg_reward = (
                    np.mean(avg_rewards[-log_interval:]) if avg_rewards else 0
                )

                # Calculate per-opponent win rates from recent games
                def calculate_recent_opponent_winrate(opponent_name):
                    if (
                        opponent_name not in opponent_stats
                        or opponent_stats[opponent_name]["games"] == 0
                    ):
                        return 0.0
                    return (
                        opponent_stats[opponent_name]["wins"]
                        / opponent_stats[opponent_name]["games"]
                    )

                win_rate_random = calculate_recent_opponent_winrate("Random")
                win_rate_heuristic = calculate_recent_opponent_winrate("Heuristic")
                win_rate_self_play = calculate_recent_opponent_winrate("Self-Play")

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            total_episodes,
                            recent_win_rate,
                            win_rate_random,
                            win_rate_heuristic,
                            win_rate_self_play,
                            recent_avg_reward,
                            recent_avg_length,
                            agent.epsilon,
                            len(agent.replay_buffer),
                            agent.steps_done,
                            datetime.now().isoformat(),
                        ]
                    )

            # Evaluation
            if total_episodes % eval_interval < total_episodes_per_iteration:
                win_rate = sum(wins[-eval_interval:]) / min(eval_interval, len(wins))
                win_rate_history.append((total_episodes, win_rate))
                avg_length = (
                    np.mean(episode_lengths[-eval_interval:]) if episode_lengths else 0
                )
                avg_reward = np.mean(avg_rewards[-eval_interval:]) if avg_rewards else 0

                print(f"\n\nEpisode {total_episodes}/{n_episodes_actual}")
                print(f"  Overall Win Rate (last {eval_interval}): {win_rate:.2%}")
                print(f"  Avg Reward (last {eval_interval}): {avg_reward:.3f}")
                print(f"  Avg Episode Length: {avg_length:.1f}")
                print(f"  Replay Buffer Size: {len(agent.replay_buffer)}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Total Steps: {agent.steps_done}")

                print("\n  Win Rates by Opponent (all-time):")
                for opp_name in ["Random", "Heuristic", "Self-Play"]:
                    if (
                        opp_name in opponent_stats
                        and opponent_stats[opp_name]["games"] > 0
                    ):
                        stats = opponent_stats[opp_name]
                        opp_wr = stats["wins"] / stats["games"]
                        avg_opp_reward = (
                            np.mean(stats["rewards"]) if stats["rewards"] else 0
                        )
                        print(
                            f"    vs {opp_name:12s}: {opp_wr:.2%} (W:{stats['wins']}/G:{stats['games']}) | Avg Reward: {avg_opp_reward:.3f}"
                        )
                print(f"  Snapshots saved: {len(opponent_manager.snapshots)}")

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_episode = total_episodes

                    agent.save(f"models/best_{save_path}")

                    print(f"\n  🏆 New best win rate: {win_rate:.2%}")

                print()

            # Save checkpoint periodically
            if total_episodes % save_interval < total_episodes_per_iteration:
                checkpoint_name = (
                    f"{save_path.replace('.pth', '')}_ep{total_episodes}.pth"
                )

                agent.save(f"checkpoints/{checkpoint_name}")

                agent.save(f"models/{save_path}")

                print(f"\n💾 Checkpoint saved at episode {total_episodes}")
                print(f"   Versioned: checkpoints/{checkpoint_name}")
                print(f"   Latest: models/{save_path}")

                training_state = {
                    "episode": total_episodes,
                    "wins": wins,
                    "avg_rewards": avg_rewards,
                    "episode_lengths": episode_lengths,
                    "win_rate_history": win_rate_history,
                    "opponent_stats": opponent_stats,
                    "best_win_rate": best_win_rate,
                    "best_episode": best_episode,
                }

                with open(
                    f"checkpoints/training_state_ep{total_episodes}.pkl", "wb"
                ) as f:
                    pickle.dump(training_state, f)

                plot_save_name = save_path.replace(".pth", "")

                plot_training_curves(
                    win_rate_history,
                    episode_lengths,
                    avg_rewards,
                    plot_save_name,
                    output_dir="plots",
                )

                timestamped_name = f"{plot_save_name}_ep{total_episodes}_{timestamp}"

                plot_training_curves(
                    win_rate_history,
                    episode_lengths,
                    avg_rewards,
                    timestamped_name,
                    output_dir="plots",
                )

                print(f"   Plot updated: plots/{plot_save_name}_*.png")

                cleanup_old_checkpoints(save_path, keep_last=3)

                print()

    agent.save(f"models/{save_path}")

    print(f"\n💾 Final model saved to: models/{save_path}")

    plot_training_curves(
        win_rate_history,
        episode_lengths,
        avg_rewards,
        save_path.replace(".pth", ""),
        output_dir="plots",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!" if not interrupted else "TRAINING INTERRUPTED!")
    print("=" * 60)

    if len(wins) >= eval_interval:
        final_win_rate = sum(wins[-eval_interval:]) / eval_interval
    else:
        final_win_rate = sum(wins) / len(wins) if wins else 0

    print(f"Final win rate: {final_win_rate:.2%}")
    print(f"Best win rate: {best_win_rate:.2%} (episode {best_episode})")

    print("\nFinal Win Rates by Opponent Type:")
    for opp_name, stats in opponent_stats.items():
        if stats["games"] > 0:
            opp_wr = stats["wins"] / stats["games"]
            print(f"  vs {opp_name}: {opp_wr:.2%} ({stats['wins']}/{stats['games']})")

    print(f"\nReplay buffer size: {len(agent.replay_buffer)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total episodes: {agent.episode_count}")
    print(f"Total steps: {agent.steps_done}")
    print(f"\nBest model saved to: models/best_{save_path}")
    print(f"Latest model saved to: models/{save_path}")
    print(f"Training log saved to: {log_file}")
    print("=" * 60 + "\n")

    return agent


def local_signal_handler(sig, frame):
    global interrupted
    print("\n\n⚠️  Keyboard interrupt received!")
    print("Finishing current batch and saving...")
    interrupted = True


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    signal.signal(signal.SIGINT, local_signal_handler)

    try:
        # Run until interrupted (essentially infinite)
        agent = train_dqn(n_episodes=100000, use_prioritized_replay=False)

    except KeyboardInterrupt:
        print("\n\n✅ Training interrupted gracefully!")

        if "agent" in locals():
            agent.save("models/interrupted_save.pth")
            print("💾 Emergency checkpoint saved to: models/interrupted_save.pth")

    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")

        traceback.print_exc()

        if "agent" in locals():
            agent.save("models/emergency_save.pth")
            print("💾 Emergency checkpoint saved to: models/emergency_save.pth")
