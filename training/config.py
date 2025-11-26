"""
Configuration file for PPO training on Ludo environment.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""

    # Environment settings
    encoding_type: str = "handcrafted"  # "handcrafted" or "onehot"
    n_envs: int = 16  # Number of parallel environments
    agent_player: int = 0  # Which player is the agent (0 or 1, randomly chosen if None)

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per environment before update
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10  # Number of epochs when optimizing the surrogate loss
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clipping parameter
    clip_range_vf: Optional[float] = None  # Clipping for value function
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Network architecture
    policy_kwargs: dict = None  # Will be set in __post_init__

    # Training settings
    total_timesteps_stage1: int = 5_000_000  # Stage 1: vs random
    total_timesteps_stage2: int = 10_000_000  # Stage 2: vs heuristic
    total_timesteps_stage3: int = 20_000_000  # Stage 3: self-play

    # Curriculum learning
    stage1_threshold: float = 0.75  # Win rate to advance from stage 1
    stage2_threshold: float = 0.75  # Win rate to advance from stage 2
    eval_freq: int = 50_000  # Evaluate every N timesteps
    n_eval_episodes: int = 100  # Number of episodes for evaluation

    # Self-play settings
    self_play_opponent_update_freq: int = 100_000  # Update opponent every N timesteps

    # Logging and saving
    log_dir: str = "./logs"
    save_dir: str = "./models"
    tensorboard_log: str = "./tensorboard_logs"
    save_freq: int = 100_000  # Save model every N timesteps

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42

    # Verbose
    verbose: int = 1

    def __post_init__(self):
        """Set policy kwargs based on encoding type."""
        if self.policy_kwargs is None:
            if self.encoding_type == "handcrafted":
                # Smaller network for handcrafted features (70-dim)
                self.policy_kwargs = {
                    "net_arch": [{"pi": [128, 128], "vf": [128, 128]}],
                    "activation_fn": "torch.nn.ReLU",
                }
            elif self.encoding_type == "onehot":
                # Larger network for one-hot encoding (946-dim)
                self.policy_kwargs = {
                    "net_arch": [{"pi": [256, 256, 128], "vf": [256, 256, 128]}],
                    "activation_fn": "torch.nn.ReLU",
                }
            else:
                raise ValueError(f"Unknown encoding_type: {self.encoding_type}")


# Predefined configurations
HANDCRAFTED_CONFIG = TrainingConfig(
    encoding_type="handcrafted",
    n_envs=16,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

ONEHOT_CONFIG = TrainingConfig(
    encoding_type="onehot",
    n_envs=16,
    learning_rate=2e-4,  # Slightly lower LR for larger network
    n_steps=2048,
    batch_size=128,  # Larger batch size for more stable training
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


def get_config(encoding_type: str = "handcrafted") -> TrainingConfig:
    """
    Get training configuration for a specific encoding type.

    Args:
        encoding_type: "handcrafted" or "onehot"

    Returns:
        TrainingConfig object
    """
    if encoding_type == "handcrafted":
        return HANDCRAFTED_CONFIG
    elif encoding_type == "onehot":
        return ONEHOT_CONFIG
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}")


# Multi-GPU configuration
@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""

    n_gpus: int = 6
    envs_per_gpu: int = 16

    # GPU device IDs
    gpu_ids: list = None

    def __post_init__(self):
        """Set GPU IDs."""
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.n_gpus))

    @property
    def total_envs(self) -> int:
        """Total number of environments across all GPUs."""
        return self.n_gpus * self.envs_per_gpu


MULTI_GPU_CONFIG = MultiGPUConfig(
    n_gpus=6,
    envs_per_gpu=16,  # 96 total environments
)
