import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(
    win_rate_history, episode_lengths, avg_rewards, save_path, output_dir="plots"
):
    """
    Plot training curves and save to file

    Args:
        win_rate_history: List of (episode, win_rate) tuples
        episode_lengths: List of episode lengths
        avg_rewards: List of average rewards
        save_path: Base name for the plot file (without extension)
        output_dir: Directory to save plots (default: 'plots')
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Win rate over time
    if win_rate_history:
        episodes, win_rates = zip(*win_rate_history)
        axes[0, 0].plot(episodes, win_rates, linewidth=2, color="blue")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Win Rate")
        axes[0, 0].set_title("Win Rate Over Training (DQN)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.5, color="r", linestyle="--", label="50% baseline")
        axes[0, 0].legend()

    # Episode length over time (moving average)
    window = 100
    if len(episode_lengths) > window:
        ma_lengths = np.convolve(
            episode_lengths, np.ones(window) / window, mode="valid"
        )
        axes[0, 1].plot(ma_lengths, linewidth=2, color="green")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Episode Length")
        axes[0, 1].set_title(f"Episode Length (MA-{window})")
        axes[0, 1].grid(True, alpha=0.3)

    # Reward distribution
    if avg_rewards:
        axes[1, 0].hist(
            avg_rewards, bins=50, edgecolor="black", alpha=0.7, color="orange"
        )
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].grid(True, alpha=0.3)

    # Cumulative wins
    if avg_rewards:
        cumulative_wins = np.cumsum([1 if r > 0 else 0 for r in avg_rewards])
        episodes_range = np.arange(1, len(cumulative_wins) + 1)

        axes[1, 1].plot(
            episodes_range, cumulative_wins, linewidth=2, label="Wins", color="purple"
        )
        axes[1, 1].plot(
            episodes_range, episodes_range * 0.5, "r--", label="50% baseline"
        )
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Cumulative Wins")
        axes[1, 1].set_title("Cumulative Wins Over Training")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    plt.tight_layout()

    # Save with proper path
    output_file = f"{output_dir}/{save_path}_plots.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Training plots saved to {output_file}")
