"""
Main PPO training script with curriculum learning for Ludo environment.

Implements three-stage curriculum:
1. Stage 1: Train vs random policy until >75% win rate
2. Stage 2: Train vs heuristic policy until >75% win rate
3. Stage 3: Self-play training
"""

import os
import argparse
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from training.config import get_config, TrainingConfig
from env.vec_env_factory import make_vec_env, SelfPlayVecEnv
from training.evaluate import quick_eval, evaluate_alternating_players


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        config: TrainingConfig,
        eval_freq: int,
        n_eval_episodes: int,
        stage1_threshold: float,
        stage2_threshold: float,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.config = config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold

        self.current_stage = 1
        self.stage_advanced = False
        self.best_win_rate = 0.0
        self.evaluations = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_maybe_advance()

        return True

    def _evaluate_and_maybe_advance(self):
        if self.current_stage == 1:
            opponent_type = "random"
        elif self.current_stage == 2:
            opponent_type = "heuristic"
        else:
            opponent_type = "random"  # In self-play, evaluate vs random for monitoring

        # Evaluate
        if self.verbose > 0:
            print(f"\n{'=' * 60}")
            print(f"Evaluation at step {self.num_timesteps}")
            print(f"Current Stage: {self.current_stage}")
            print(f"{'=' * 60}")

        win_rate = quick_eval(
            model=self.model,
            opponent_type=opponent_type,
            n_episodes=self.n_eval_episodes,
            encoding_type=self.config.encoding_type,
        )

        # Log to tensorboard
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/stage", self.current_stage)

        # Store evaluation
        self.evaluations.append(
            {
                "timestep": self.num_timesteps,
                "stage": self.current_stage,
                "opponent": opponent_type,
                "win_rate": win_rate,
            }
        )

        if self.verbose > 0:
            print(f"Win Rate vs {opponent_type}: {win_rate * 100:.2f}%")
            print(f"Best Win Rate: {self.best_win_rate * 100:.2f}%")

        # Update best win rate
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate

            # Save best model
            model_path = os.path.join(
                self.config.save_dir,
                f"best_model_stage{self.current_stage}_{self.config.encoding_type}.zip",
            )
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved new best model to {model_path}")

        # Check if we should advance to next stage
        if self.current_stage == 1 and win_rate >= self.stage1_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print(
                    f"Stage 1 complete! Win rate {win_rate * 100:.2f}% >= {self.stage1_threshold * 100}%"
                )
                print("Advancing to Stage 2: Training vs Heuristic")
                print("!" * 60 + "\n")
            self.current_stage = 2
            self.stage_advanced = True
            self.best_win_rate = 0.0  # Reset for new stage

        elif self.current_stage == 2 and win_rate >= self.stage2_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print(
                    f"Stage 2 complete! Win rate {win_rate * 100:.2f}% >= {self.stage2_threshold * 100}%"
                )
                print("Advancing to Stage 3: Self-Play Training")
                print("!" * 60 + "\n")
            self.current_stage = 3
            self.stage_advanced = True
            self.best_win_rate = 0.0  # Reset for new stage

        if self.verbose > 0:
            print("=" * 60 + "\n")


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play that periodically updates the opponent policy.
    """

    def __init__(
        self,
        update_freq: int,
        vec_env: SelfPlayVecEnv,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.vec_env = vec_env
        self.last_update = 0

    def _on_step(self) -> bool:
        """
        Called at every step.

        Returns:
            bool: If False, training stops
        """
        # Check if it's time to update opponent
        if self.num_timesteps - self.last_update >= self.update_freq:
            self._update_opponent()
            self.last_update = self.num_timesteps

        return True

    def _update_opponent(self):
        """Update opponent policy with current model."""
        if self.verbose > 0:
            print("\n" + "~" * 60)
            print(f"Updating opponent policy at step {self.num_timesteps}")
            print("~" * 60 + "\n")

        # Update opponent in vectorized environment
        # Note: This creates a copy of the current policy for the opponent
        self.vec_env.update_opponent_policy(self.model.policy)

        # Log to tensorboard
        self.logger.record(
            "self_play/opponent_updates", self.num_timesteps // self.update_freq
        )


def train_stage(
    config: TrainingConfig,
    stage: int,
    opponent_type: str,
    total_timesteps: int,
    model=None,
    gpu_id: int = 0,
):
    """
    Train one stage of the curriculum.

    Args:
        config: Training configuration
        stage: Current stage number (1, 2, or 3)
        opponent_type: Type of opponent ("random", "heuristic", or "self")
        total_timesteps: Total timesteps to train in this stage
        model: Existing model to continue training (if any)
        gpu_id: GPU device ID

    Returns:
        Trained model
    """
    print(f"\n{'#' * 60}")
    print(f"Starting Stage {stage}: Training vs {opponent_type}")
    print(f"Encoding: {config.encoding_type}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"{'#' * 60}\n")

    # Set device
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

    # Create or load model
    if model is None:
        print("Creating new MaskablePPO model...")

        # Fix policy_kwargs format for SB3
        policy_kwargs = config.policy_kwargs.copy()
        if "activation_fn" in policy_kwargs:
            # Convert string to actual torch module
            if policy_kwargs["activation_fn"] == "torch.nn.ReLU":
                policy_kwargs["activation_fn"] = torch.nn.ReLU
            elif policy_kwargs["activation_fn"] == "torch.nn.Tanh":
                policy_kwargs["activation_fn"] = torch.nn.Tanh

        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range_vf,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.tensorboard_log,
            device=device,
            verbose=config.verbose,
            seed=config.seed,
        )
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

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq // config.n_envs,  # Adjust for vectorized env
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
):
    """
    Train using curriculum learning across all three stages.

    Args:
        encoding_type: "handcrafted" or "onehot"
        gpu_id: GPU device ID
        resume_from: Path to model checkpoint to resume from
    """
    # Get configuration
    config = get_config(encoding_type)

    print("\n" + "=" * 60)
    print("Starting Curriculum Training")
    print(f"Encoding Type: {encoding_type}")
    print(f"GPU ID: {gpu_id}")
    print("=" * 60 + "\n")

    # Load existing model if resuming
    model = None
    if resume_from is not None:
        print(f"Loading model from {resume_from}...")
        model = MaskablePPO.load(resume_from)

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
            n_eval_episodes=200,
            encoding_type=encoding_type,
            use_deterministic=True,
            verbose=True,
        )

    print("\n" + "=" * 60)
    print("All training stages completed!")
    print("=" * 60 + "\n")

    return model


def main():
    """Main entry point."""
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

    args = parser.parse_args()

    # Train
    train_curriculum(
        encoding_type=args.encoding,
        gpu_id=args.gpu,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
