import os
import argparse
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from config import TrainingConfig
from env.vec_env_factory import make_vec_env, BatchedOpponentVecEnv
from evaluate import evaluate_alternating_players
from misc.curriculum_callback import CurriculumCallback
from misc.checkpoint_callback import CheckpointCallback
from misc.update_config import update_model_hyperparameters

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pygame.pkgdata",
)

warnings.filterwarnings(
    "ignore",
    message="env.get_action_space to get variables from other wrappers is deprecated and will be removed",
    category=UserWarning,
    module="gymnasium.core",
)


def train_stage(
    config: TrainingConfig,
    stage: int,
    opponent_type: str,
    total_timesteps: int,
    model=None,
    gpu_id: int = 0,
    reload_configs: bool = False,
):
    print(f"\n{'#' * 60}")
    print(f"Starting Stage {stage}: Training vs {opponent_type}")
    print(f"Encoding: {config.encoding_type}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"{'#' * 60}\n")

    device = (
        f"cuda:{gpu_id}"
        if torch.cuda.is_available() and config.device != "cpu"
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)

    # Create log directory for this stage
    stage_log_dir = os.path.join(config.log_dir, f"stage{stage}_{config.encoding_type}")

    # Create vectorized environment
    if stage == 3:
        # Self-play environment with batched GPU opponent
        vec_env = BatchedOpponentVecEnv(
            n_envs=config.n_envs,
            encoding_type=config.encoding_type,
            agent_player=config.agent_player,
            log_dir=stage_log_dir,
            seed=config.seed,
            opponent_model_path=config.self_play_opponent_model_path,
            opponent_device=config.self_play_opponent_device,
        )
    else:
        vec_env = make_vec_env(
            n_envs=config.n_envs,
            encoding_type=config.encoding_type,
            opponent_type=opponent_type,
            agent_player=config.agent_player,
            log_dir=stage_log_dir,
            seed=config.seed,
            use_subprocess=True,
        )

    if model is None:
        print("Creating new MaskablePPO model...")

        policy_kwargs = config.policy_kwargs.copy()

        if "activation_fn" in policy_kwargs:
            if policy_kwargs["activation_fn"] == "torch.nn.ReLU":
                policy_kwargs["activation_fn"] = torch.nn.ReLU
            elif policy_kwargs["activation_fn"] == "torch.nn.Tanh":
                policy_kwargs["activation_fn"] = torch.nn.Tanh

        model_kwargs = {
            "policy": MaskableActorCriticPolicy,
            "env": vec_env,
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "clip_range_vf": config.clip_range_vf,
            "ent_coef": config.ent_coef,
            "vf_coef": config.vf_coef,
            "max_grad_norm": config.max_grad_norm,
            "normalize_advantage": True,
            "policy_kwargs": policy_kwargs,
            "tensorboard_log": config.tensorboard_log,
            "device": device,
            "verbose": config.verbose,
            "seed": config.seed,
        }

        model = MaskablePPO(**model_kwargs)
    else:
        print("Continuing training with existing model...")
        model.set_env(vec_env)

        # Reload hyperparameters from config if requested
        if reload_configs:
            update_model_hyperparameters(model, config)

    # Create callbacks
    callbacks = []

    # Checkpoint callback for periodic saving (overwrites same file)
    latest_checkpoint_name = config.get_model_name(
        prefix="latest_checkpoint", stage=stage
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.save_dir,
        model_name=latest_checkpoint_name,
        verbose=config.verbose,
    )
    callbacks.append(checkpoint_callback)

    # Curriculum callback for evaluation (also saves best models)
    curriculum_callback = CurriculumCallback(
        config=config,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        stage1_threshold=config.stage1_threshold,
        stage2_threshold=config.stage2_threshold,
        verbose=config.verbose,
    )
    curriculum_callback.current_stage = stage
    callbacks.append(curriculum_callback)

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"stage{stage}_{config.encoding_type}",
        reset_num_timesteps=False,  # Continue counting timesteps
    )

    # Save latest model
    latest_model_name = config.get_model_name(prefix="latest", stage=stage)
    latest_model_path = os.path.join(config.save_dir, latest_model_name)
    model.save(latest_model_path)
    print(f"\nSaved latest model to {latest_model_path}")

    # Close environment
    vec_env.close()

    return model


def train_curriculum(
    encoding_type: str = None,
    gpu_id: int = 0,
    resume_from: str = None,
    learning_rate: float = None,
    ent_coef: float = None,
    net_arch: list = None,
    batch_size: int = None,
    reload_configs: bool = False,
    self_play_opponent_path: str = None,
    self_play_opponent_device: str = None,
):
    config_kwargs = {}

    if encoding_type is not None:
        config_kwargs["encoding_type"] = encoding_type
    if learning_rate is not None:
        config_kwargs["learning_rate"] = learning_rate
    if ent_coef is not None:
        config_kwargs["ent_coef"] = ent_coef
    if net_arch is not None:
        config_kwargs["net_arch"] = net_arch
    if batch_size is not None:
        config_kwargs["batch_size"] = batch_size
    if self_play_opponent_path is not None:
        config_kwargs["self_play_opponent_model_path"] = self_play_opponent_path
    if self_play_opponent_device is not None:
        config_kwargs["self_play_opponent_device"] = self_play_opponent_device

    config = TrainingConfig(**config_kwargs)

    print("\n" + "=" * 60)
    print("Starting Curriculum Training")
    print(f"Encoding Type: {encoding_type}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Entropy Coefficient: {ent_coef}")
    print(f"Network Architecture: {config.net_arch}")
    print(f"Batch Size: {config.batch_size}")
    print(f"GPU ID: {gpu_id}")
    if config.self_play_opponent_model_path:
        print(f"Self-Play Opponent: {config.self_play_opponent_model_path}")
        print(f"Self-Play Opponent Device: {config.self_play_opponent_device}")
    print("=" * 60 + "\n")

    # Load existing model if resuming
    model = None
    if resume_from is not None:
        print(f"Loading model from {resume_from}...")
        model = MaskablePPO.load(resume_from)

        # Reload hyperparameters from config if requested
        if reload_configs:
            update_model_hyperparameters(model, config)

    # Stage 1: Train vs Random
    if config.total_timesteps_stage1 > 0:
        model = train_stage(
            config=config,
            stage=1,
            opponent_type="random",
            total_timesteps=config.total_timesteps_stage1,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
        )

    # Stage 2: Train vs Heuristic
    if config.total_timesteps_stage2 > 0:
        model = train_stage(
            config=config,
            stage=2,
            opponent_type="heuristic",
            total_timesteps=config.total_timesteps_stage2,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
        )

    # Stage 3: Self-Play
    model = train_stage(
        config=config,
        stage=3,
        opponent_type="self",
        total_timesteps=config.total_timesteps_stage3,
        model=model,
        gpu_id=gpu_id,
        reload_configs=reload_configs,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60 + "\n")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Ludo")
    parser.add_argument(
        "--encoding",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "onehot"],
        help="State encoding type",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume from",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=None,
        help="Learning rate for the optimizer (default: 3e-4)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Entropy coefficient for exploration (default: 0.1)",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=None,
        help="Network architecture as space-separated integers (e.g., 512 512 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: 512)",
    )
    parser.add_argument(
        "--reload-configs",
        action="store_true",
        help="Reload all hyperparameters from config when resuming (except net_arch)",
    )
    parser.add_argument(
        "--self-play-opponent",
        type=str,
        default=None,
        help="Path to opponent model checkpoint for self-play (stage 3)",
    )
    parser.add_argument(
        "--self-play-opponent-device",
        type=str,
        default=None,
        help="Device for opponent model (e.g., 'cpu', 'cuda:1'). Default: cpu",
    )

    args = parser.parse_args()

    # Train
    train_curriculum(
        encoding_type=args.encoding,
        gpu_id=args.gpu,
        resume_from=args.resume,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        net_arch=args.net_arch,
        batch_size=args.batch_size,
        reload_configs=args.reload_configs,
        self_play_opponent_path=args.self_play_opponent,
        self_play_opponent_device=args.self_play_opponent_device,
    )


if __name__ == "__main__":
    main()
