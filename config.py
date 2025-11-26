from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    encoding_type: str = "handcrafted"  # "handcrafted" or "onehot"
    n_envs: int = 48
    agent_player: Optional[int] = None

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = (
        1024  # Steps per environment before update (reduced for faster updates)
    )
    batch_size: int = 512  # Minibatch size (reduced to match n_steps better)
    n_epochs: int = 5  # Number of epochs when optimizing the surrogate loss (reduced to avoid overfitting)
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clipping parameter
    clip_range_vf: Optional[float] = None  # Clipping for value function
    ent_coef: float = 0.1  # Entropy coefficient (increased for more exploration)
    vf_coef: float = (
        1.0  # Value function coefficient (increased for better value learning)
    )
    max_grad_norm: float = (
        1.0  # Gradient clipping (increased to allow larger gradients)
    )

    # Network architecture
    policy_kwargs: dict = None  # Will be set in __post_init__

    # Training settings
    total_timesteps_stage1: int = 100_000_000
    total_timesteps_stage2: int = 100_000_000
    total_timesteps_stage3: int = 100_000_000

    # Curriculum learning
    stage1_threshold: float = (
        0.60  # Win rate to advance from stage 1 (lowered from 0.75)
    )
    stage2_threshold: float = (
        0.60  # Win rate to advance from stage 2 (lowered from 0.75)
    )
    eval_freq: int = 500_000  # Evaluate every N timesteps
    n_eval_episodes: int = (
        1000  # Number of episodes for evaluation (reduced for faster iteration)
    )

    self_play_opponent_update_freq: int = 500_000  # Update opponent every N timesteps

    # Logging and saving
    log_dir: str = "./logs"
    save_dir: str = "./models"
    tensorboard_log: str = "./logs/tensorboard"
    save_freq: int = 500_000  # Save model every N timesteps

    # Behavior Cloning settings
    use_bc_loss: bool = False  # Enable behavior cloning loss
    bc_loss_coef: float = 1.0  # Initial coefficient for BC loss
    bc_loss_decay_rate: float = 1e-6  # Decay rate for BC loss (per timestep)
    bc_loss_min_coef: float = 0.0  # Minimum BC loss coefficient

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42

    # Verbose
    verbose: int = 1

    def __post_init__(self):
        if self.policy_kwargs is None:
            if self.encoding_type == "handcrafted":
                self.policy_kwargs = {
                    "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
                    "activation_fn": "torch.nn.ReLU",
                }
            elif self.encoding_type == "onehot":
                self.policy_kwargs = {
                    "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
                    "activation_fn": "torch.nn.ReLU",
                }
            else:
                raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
