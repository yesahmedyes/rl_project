from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    encoding_type: str = "handcrafted"  # "handcrafted" or "onehot"
    n_envs: int = 48
    agent_player: Optional[int] = None

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 1024  # Steps per environment before update
    batch_size: int = 512  # Minibatch size
    n_epochs: int = 5  # Number of epochs when optimizing the surrogate loss
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clipping parameter
    clip_range_vf: Optional[float] = None  # Clipping for value function
    ent_coef: float = 0.10  # Entropy coefficient
    vf_coef: float = 1.0  # Value function coefficient
    max_grad_norm: float = 1.0  # Gradient clipping

    # Network architecture
    net_arch: list = None  # Network architecture (will be applied to both pi and vf)
    policy_kwargs: dict = None  # Will be set in __post_init__

    # Training settings
    total_timesteps_stage1: int = 0
    total_timesteps_stage2: int = 0
    total_timesteps_stage3: int = 100_000_000

    # Curriculum learning
    stage1_threshold: float = 0.75
    stage2_threshold: float = 0.75
    eval_freq: int = 500_000  # Evaluate every N timesteps
    n_eval_episodes: int = 1000  # Number of episodes for evaluation

    self_play_opponent_update_freq: int = 500_000  # Update opponent every N timesteps

    # Logging and saving
    log_dir: str = "./logs"
    save_dir: str = "./models"
    tensorboard_log: str = "./logs/tensorboard"
    save_freq: int = 500_000  # Save model every N timesteps

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42

    # Verbose
    verbose: int = 1

    def __post_init__(self):
        # Set default net_arch if not provided
        if self.net_arch is None:
            self.net_arch = [512, 512, 256]

        if self.policy_kwargs is None:
            if self.encoding_type == "handcrafted":
                self.policy_kwargs = {
                    "net_arch": {"pi": self.net_arch, "vf": self.net_arch},
                    "activation_fn": "torch.nn.ReLU",
                }
            elif self.encoding_type == "onehot":
                self.policy_kwargs = {
                    "net_arch": {"pi": self.net_arch, "vf": self.net_arch},
                    "activation_fn": "torch.nn.ReLU",
                }
            else:
                raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

    def get_model_name(self, prefix: str = "", stage: Optional[int] = None) -> str:
        parts = []

        if prefix:
            parts.append(prefix)

        if stage is not None:
            parts.append(f"stage{stage}")

        parts.append(self.encoding_type)
        parts.append(f"lr{self.learning_rate:.0e}")
        parts.append(f"ent{self.ent_coef}")
        parts.append(f"arch{self.net_arch[0]}")

        return "_".join(parts) + ".zip"
