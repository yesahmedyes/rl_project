import os
import argparse
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from config import TrainingConfig
from env.vec_env_factory import make_vec_env, SelfPlayVecEnv
from evaluate import evaluate_alternating_players
from misc.curriculum_callback import CurriculumCallback
from misc.selfplay_callback import SelfPlayCallback
from bc_ppo import BCMaskablePPO


def train_stage(
    config: TrainingConfig,
    stage: int,
    opponent_type: str,
    total_timesteps: int,
    model=None,
    gpu_id: int = 0,
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
        # Self-play environment
        vec_env = SelfPlayVecEnv(
            n_envs=config.n_envs,
            encoding_type=config.encoding_type,
            agent_player=config.agent_player,
            log_dir=stage_log_dir,
            seed=config.seed,
            use_subprocess=True,
        )

        # Initialize opponent with current model if available
        if model is not None:
            vec_env.update_opponent_policy(model.policy)
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

        # Use BCMaskablePPO if behavior cloning is enabled
        model_class = BCMaskablePPO if config.use_bc_loss else MaskablePPO

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
            "policy_kwargs": policy_kwargs,
            "tensorboard_log": config.tensorboard_log,
            "device": device,
            "verbose": config.verbose,
            "seed": config.seed,
        }

        # Add BC-specific parameters if using BCMaskablePPO
        if config.use_bc_loss:
            model_kwargs["use_bc_loss"] = config.use_bc_loss
            model_kwargs["bc_loss_coef"] = config.bc_loss_coef
            model_kwargs["bc_loss_decay_rate"] = config.bc_loss_decay_rate
            model_kwargs["bc_loss_min_coef"] = config.bc_loss_min_coef

        model = model_class(**model_kwargs)
    else:
        print("Continuing training with existing model...")
        model.set_env(vec_env)

    # Create callbacks
    callbacks = []

    # Curriculum callback for evaluation
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

    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq // config.n_envs,
        save_path=config.save_dir,
        name_prefix=f"stage{stage}_{config.encoding_type}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    # Self-play callback (only for stage 3)
    if stage == 3:
        self_play_callback = SelfPlayCallback(
            update_freq=config.self_play_opponent_update_freq,
            vec_env=vec_env,
            verbose=config.verbose,
        )
        callbacks.append(self_play_callback)

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"stage{stage}_{config.encoding_type}",
        reset_num_timesteps=False,  # Continue counting timesteps
    )

    # Save final model
    final_model_path = os.path.join(
        config.save_dir, f"final_stage{stage}_{config.encoding_type}.zip"
    )
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")

    # Close environment
    vec_env.close()

    return model


def train_curriculum(
    encoding_type: str = "handcrafted",
    gpu_id: int = 0,
    resume_from: str = None,
    use_bc_loss: bool = False,
):
    config = TrainingConfig(encoding_type=encoding_type, use_bc_loss=use_bc_loss)

    print("\n" + "=" * 60)
    print("Starting Curriculum Training")
    print(f"Encoding Type: {encoding_type}")
    print(f"GPU ID: {gpu_id}")
    print("=" * 60 + "\n")

    # Load existing model if resuming
    model = None
    if resume_from is not None:
        print(f"Loading model from {resume_from}...")
        # Load with appropriate class based on config
        model_class = BCMaskablePPO if config.use_bc_loss else MaskablePPO
        model = model_class.load(resume_from)

    # Stage 1: Train vs Random
    model = train_stage(
        config=config,
        stage=1,
        opponent_type="random",
        total_timesteps=config.total_timesteps_stage1,
        model=model,
        gpu_id=gpu_id,
    )

    # Stage 2: Train vs Heuristic
    model = train_stage(
        config=config,
        stage=2,
        opponent_type="heuristic",
        total_timesteps=config.total_timesteps_stage2,
        model=model,
        gpu_id=gpu_id,
    )

    # Stage 3: Self-Play
    model = train_stage(
        config=config,
        stage=3,
        opponent_type="self",
        total_timesteps=config.total_timesteps_stage3,
        model=model,
        gpu_id=gpu_id,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete! Running Final Evaluation...")
    print("=" * 60 + "\n")

    for opponent in ["random", "heuristic"]:
        print(f"\nEvaluating vs {opponent}:")
        evaluate_alternating_players(
            model=model,
            opponent_type=opponent,
            n_eval_episodes=10000,
            encoding_type=encoding_type,
            use_deterministic=True,
            verbose=True,
        )

    print("\n" + "=" * 60)
    print("All training stages completed!")
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
    parser.add_argument(
        "--use-bc-loss",
        default=False,
        action="store_true",
        help="Enable behavior cloning loss to imitate heuristic policy",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume from",
    )

    args = parser.parse_args()

    # Train
    train_curriculum(
        encoding_type=args.encoding,
        gpu_id=args.gpu,
        resume_from=args.resume,
        use_bc_loss=args.use_bc_loss,
    )


if __name__ == "__main__":
    main()
