import os
import argparse
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize

from config import TrainingConfig
from env.vec_env_factory import make_vec_env, BatchedOpponentVecEnv
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
    opponent_distribution: list,
    total_timesteps: int,
    model=None,
    gpu_id: int = 0,
    reload_configs: bool = False,
    use_self_play_env: bool = False,
):
    # Format opponent info for display
    if use_self_play_env:
        opponent_info = "Self-Play"
    else:
        opponent_names = ["Random", "Heuristic", "Milestone2"]
        opponent_info = ", ".join(
            [
                f"{name}: {dist * 100:.0f}%"
                for name, dist in zip(opponent_names, opponent_distribution)
                if dist > 0
            ]
        )

    print(f"\n{'#' * 60}")
    print(f"Starting Stage {stage}: Training vs {opponent_info}")
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
    if use_self_play_env:
        # Self-play environment with batched GPU opponent
        vec_env = BatchedOpponentVecEnv(
            n_envs=config.n_envs,
            encoding_type=config.encoding_type,
            agent_player=config.agent_player,
            log_dir=stage_log_dir,
            seed=config.seed,
            opponent_model_path=config.self_play_opponent_model_path,
            opponent_device=config.self_play_opponent_device,
            use_dense_reward=config.use_dense_reward,
        )
    else:
        vec_env = make_vec_env(
            n_envs=config.n_envs,
            encoding_type=config.encoding_type,
            opponent_distribution=opponent_distribution,
            agent_player=config.agent_player,
            log_dir=stage_log_dir,
            seed=config.seed,
            use_subprocess=True,
            use_dense_reward=config.use_dense_reward,
        )

    # Wrap with VecNormalize for reward normalization if enabled
    if config.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,  # Don't normalize observations
            norm_reward=True,  # Normalize rewards
            clip_obs=10.0,
            clip_reward=10.0,
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


def train_self_play(
    config: TrainingConfig,
    gpu_id: int = 0,
    resume_from: str = None,
    reload_configs: bool = False,
):
    print("\n" + "=" * 60)
    print("Starting Self-Play Training")
    print(f"Encoding Type: {config.encoding_type}")
    print(f"Reward Type: {'Dense' if config.use_dense_reward else 'Sparse'}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Entropy Coefficient: {config.ent_coef}")
    print(f"Network Architecture: {config.net_arch}")
    print(f"Batch Size: {config.batch_size}")
    print(f"N Envs: {config.n_envs}")
    print(f"N Steps: {config.n_steps}")
    print(f"Gamma: {config.gamma}")
    print(f"GPU ID: {gpu_id}")

    if config.self_play_opponent_model_path:
        print(f"Self-Play Opponent: {config.self_play_opponent_model_path}")
        print(f"Self-Play Opponent Device: {config.self_play_opponent_device}")

    print("=" * 60 + "\n")

    model = None

    if resume_from is not None:
        print(f"Loading model from {resume_from}...")
        model = MaskablePPO.load(resume_from)

        if reload_configs:
            update_model_hyperparameters(model, config)

    # Train with self-play
    model = train_stage(
        config=config,
        stage=0,  # Use stage 0 for self-play
        opponent_distribution=[0.0, 0.0, 0.0],  # Not used for self-play
        total_timesteps=config.total_timesteps,
        model=model,
        gpu_id=gpu_id,
        reload_configs=reload_configs,
        use_self_play_env=True,
    )

    print("\n" + "=" * 60)
    print("Self-Play Training Complete!")
    print("=" * 60 + "\n")

    return model


def train_curriculum(
    config: TrainingConfig,
    gpu_id: int = 0,
    resume_from: str = None,
    reload_configs: bool = False,
):
    """Train agent using 5-stage curriculum learning with automatic transitions."""
    print("\n" + "=" * 60)
    print("Starting 5-Stage Curriculum Training")
    print(f"Encoding Type: {config.encoding_type}")
    print(f"Reward Type: {'Dense' if config.use_dense_reward else 'Sparse'}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Entropy Coefficient: {config.ent_coef}")
    print(f"Network Architecture: {config.net_arch}")
    print(f"Batch Size: {config.batch_size}")
    print(f"N Envs: {config.n_envs}")
    print(f"N Steps: {config.n_steps}")
    print(f"Gamma: {config.gamma}")
    print(f"GPU ID: {gpu_id}")
    print("\nCurriculum Stages:")
    print(
        f"  Stage 0: 100% Random (threshold: {config.stage0_threshold * 100:.0f}% vs random)"
    )
    print(
        f"  Stage 1: 80% Random, 10% Heuristic, 10% Milestone2 (threshold: {config.stage1_threshold * 100:.0f}% vs random)"
    )
    print(
        f"  Stage 2: 60% Random, 20% Heuristic, 20% Milestone2 (threshold: {config.stage2_threshold * 100:.0f}% vs random)"
    )
    print(
        f"  Stage 3: 40% Random, 30% Heuristic, 30% Milestone2 (threshold: {config.stage3_threshold * 100:.0f}% vs milestone2)"
    )
    print("  Stage 4: 20% Random, 40% Heuristic, 40% Milestone2 (final optimization)")
    print("=" * 60 + "\n")

    # Load existing model if resuming
    model = None

    if resume_from is not None:
        print(f"Loading model from {resume_from}...")
        model = MaskablePPO.load(resume_from)

        # Reload hyperparameters from config if requested
        if reload_configs:
            update_model_hyperparameters(model, config)

    # Stage 0: 100% Random
    if config.total_timesteps_stage0 > 0:
        model = train_stage(
            config=config,
            stage=0,
            opponent_distribution=config.opponent_distribution_stage0,
            total_timesteps=config.total_timesteps_stage0,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
            use_self_play_env=False,
        )

    # Stage 1: 80% Random, 10% Heuristic, 10% Milestone2
    if config.total_timesteps_stage1 > 0:
        model = train_stage(
            config=config,
            stage=1,
            opponent_distribution=config.opponent_distribution_stage1,
            total_timesteps=config.total_timesteps_stage1,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
            use_self_play_env=False,
        )

    # Stage 2: 60% Random, 20% Heuristic, 20% Milestone2
    if config.total_timesteps_stage2 > 0:
        model = train_stage(
            config=config,
            stage=2,
            opponent_distribution=config.opponent_distribution_stage2,
            total_timesteps=config.total_timesteps_stage2,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
            use_self_play_env=False,
        )

    # Stage 3: 40% Random, 30% Heuristic, 30% Milestone2
    if config.total_timesteps_stage3 > 0:
        model = train_stage(
            config=config,
            stage=3,
            opponent_distribution=config.opponent_distribution_stage3,
            total_timesteps=config.total_timesteps_stage3,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
            use_self_play_env=False,
        )

    # Stage 4: 20% Random, 40% Heuristic, 40% Milestone2
    if config.total_timesteps_stage4 > 0:
        model = train_stage(
            config=config,
            stage=4,
            opponent_distribution=config.opponent_distribution_stage4,
            total_timesteps=config.total_timesteps_stage4,
            model=model,
            gpu_id=gpu_id,
            reload_configs=reload_configs,
            use_self_play_env=False,
        )

    print("\n" + "=" * 60)
    print("5-Stage Curriculum Training Complete!")
    print("=" * 60 + "\n")

    return model


def train(
    config: TrainingConfig,
    gpu_id: int = 0,
    resume_from: str = None,
    reload_configs: bool = False,
):
    if config.use_self_play:
        return train_self_play(
            config=config,
            gpu_id=gpu_id,
            resume_from=resume_from,
            reload_configs=reload_configs,
        )
    else:
        return train_curriculum(
            config=config,
            gpu_id=gpu_id,
            resume_from=resume_from,
            reload_configs=reload_configs,
        )


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
    parser.add_argument(
        "--use-self-play",
        action="store_true",
        help="Use self-play training instead of curriculum learning",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Number of epochs when optimizing the surrogate loss (default: 5)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total timesteps for self-play training (default: 100_000_000)",
    )
    parser.add_argument(
        "--total-timesteps-stage0",
        type=int,
        default=None,
        help="Total timesteps for stage 0 (curriculum only) (default: 50_000_000)",
    )
    parser.add_argument(
        "--total-timesteps-stage1",
        type=int,
        default=None,
        help="Total timesteps for stage 1 (curriculum only) (default: 50_000_000)",
    )
    parser.add_argument(
        "--total-timesteps-stage2",
        type=int,
        default=None,
        help="Total timesteps for stage 2 (curriculum only) (default: 50_000_000)",
    )
    parser.add_argument(
        "--total-timesteps-stage3",
        type=int,
        default=None,
        help="Total timesteps for stage 3 (curriculum only) (default: 100_000_000)",
    )
    parser.add_argument(
        "--total-timesteps-stage4",
        type=int,
        default=None,
        help="Total timesteps for stage 4 (curriculum only) (default: 100_000_000)",
    )
    parser.add_argument(
        "--opponent-dist-stage0",
        type=float,
        nargs=3,
        default=None,
        metavar=("RANDOM", "HEURISTIC", "MILESTONE2"),
        help="Opponent distribution for stage 0 (must sum to 1.0). Default: [1, 0, 0]",
    )
    parser.add_argument(
        "--opponent-dist-stage1",
        type=float,
        nargs=3,
        default=None,
        metavar=("RANDOM", "HEURISTIC", "MILESTONE2"),
        help="Opponent distribution for stage 1 (must sum to 1.0). Default: [0.8, 0.1, 0.1]",
    )
    parser.add_argument(
        "--opponent-dist-stage2",
        type=float,
        nargs=3,
        default=None,
        metavar=("RANDOM", "HEURISTIC", "MILESTONE2"),
        help="Opponent distribution for stage 2 (must sum to 1.0). Default: [0.6, 0.2, 0.2]",
    )
    parser.add_argument(
        "--opponent-dist-stage3",
        type=float,
        nargs=3,
        default=None,
        metavar=("RANDOM", "HEURISTIC", "MILESTONE2"),
        help="Opponent distribution for stage 3 (must sum to 1.0). Default: [0.4, 0.3, 0.3]",
    )
    parser.add_argument(
        "--opponent-dist-stage4",
        type=float,
        nargs=3,
        default=None,
        metavar=("RANDOM", "HEURISTIC", "MILESTONE2"),
        help="Opponent distribution for stage 4 (must sum to 1.0). Default: [0.2, 0.4, 0.4]",
    )
    parser.add_argument(
        "--dense-reward",
        action="store_true",
        help="Use dense reward shaping (default: sparse rewards)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: 16)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of steps per environment before update (default: 512)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default: 0.995)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="GAE lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=None,
        help="PPO clipping parameter (default: 0.2)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Gradient clipping threshold (default: 0.5)",
    )

    args = parser.parse_args()

    # Build config kwargs from command-line arguments
    config_kwargs = {}

    if args.encoding is not None:
        config_kwargs["encoding_type"] = args.encoding
    if args.learning_rate is not None:
        config_kwargs["learning_rate"] = args.learning_rate
    if args.ent_coef is not None:
        config_kwargs["ent_coef"] = args.ent_coef
    if args.net_arch is not None:
        config_kwargs["net_arch"] = args.net_arch
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.self_play_opponent is not None:
        config_kwargs["self_play_opponent_model_path"] = args.self_play_opponent
    if args.self_play_opponent_device is not None:
        config_kwargs["self_play_opponent_device"] = args.self_play_opponent_device
    if args.use_self_play:
        config_kwargs["use_self_play"] = True
    if args.n_epochs is not None:
        config_kwargs["n_epochs"] = args.n_epochs
    if args.total_timesteps is not None:
        config_kwargs["total_timesteps"] = args.total_timesteps
    if args.total_timesteps_stage0 is not None:
        config_kwargs["total_timesteps_stage0"] = args.total_timesteps_stage0
    if args.total_timesteps_stage1 is not None:
        config_kwargs["total_timesteps_stage1"] = args.total_timesteps_stage1
    if args.total_timesteps_stage2 is not None:
        config_kwargs["total_timesteps_stage2"] = args.total_timesteps_stage2
    if args.total_timesteps_stage3 is not None:
        config_kwargs["total_timesteps_stage3"] = args.total_timesteps_stage3
    if args.total_timesteps_stage4 is not None:
        config_kwargs["total_timesteps_stage4"] = args.total_timesteps_stage4
    if args.opponent_dist_stage0 is not None:
        config_kwargs["opponent_distribution_stage0"] = args.opponent_dist_stage0
    if args.opponent_dist_stage1 is not None:
        config_kwargs["opponent_distribution_stage1"] = args.opponent_dist_stage1
    if args.opponent_dist_stage2 is not None:
        config_kwargs["opponent_distribution_stage2"] = args.opponent_dist_stage2
    if args.opponent_dist_stage3 is not None:
        config_kwargs["opponent_distribution_stage3"] = args.opponent_dist_stage3
    if args.opponent_dist_stage4 is not None:
        config_kwargs["opponent_distribution_stage4"] = args.opponent_dist_stage4
    if args.dense_reward:
        config_kwargs["use_dense_reward"] = True
    if args.n_envs is not None:
        config_kwargs["n_envs"] = args.n_envs
    if args.n_steps is not None:
        config_kwargs["n_steps"] = args.n_steps
    if args.gamma is not None:
        config_kwargs["gamma"] = args.gamma
    if args.gae_lambda is not None:
        config_kwargs["gae_lambda"] = args.gae_lambda
    if args.clip_range is not None:
        config_kwargs["clip_range"] = args.clip_range
    if args.max_grad_norm is not None:
        config_kwargs["max_grad_norm"] = args.max_grad_norm

    # Create config
    config = TrainingConfig(**config_kwargs)

    # Train
    train(
        config=config,
        gpu_id=args.gpu,
        resume_from=args.resume,
        reload_configs=args.reload_configs,
    )


if __name__ == "__main__":
    main()
