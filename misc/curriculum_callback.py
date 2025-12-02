from stable_baselines3.common.callbacks import BaseCallback
from config import TrainingConfig
from evaluate import quick_eval
import os


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        config: TrainingConfig,
        eval_freq: int,
        n_eval_episodes: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.config = config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.current_stage = 0  # Start from stage 0
        self.stage_advanced = False
        self.best_win_rate = 0.0  # For stages with single opponent evaluation
        self.best_win_rates = {}  # For stages with multiple opponent evaluation
        self.evaluations = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_and_maybe_advance()

        return True

    def _evaluate_and_maybe_advance(self):
        best_model_opponents = ["random", "heuristic"]

        if self.verbose > 0:
            print(f"\n{'=' * 60}")
            print(f"Evaluation at step {self.num_timesteps}")
            print(f"Current Stage: {self.current_stage}")
            print(
                "Best Model Selection: 200 episodes vs Random + 200 episodes vs Heuristic"
            )
            print(f"{'=' * 60}")

        win_rates = {}

        for opponent_type in best_model_opponents:
            win_rate = quick_eval(
                model=self.model,
                opponent_type=opponent_type,
                n_episodes=self.config.n_eval_episodes // 2,
                encoding_type=self.config.encoding_type,
                use_dense_reward=self.config.use_dense_reward,
            )

            win_rates[opponent_type] = win_rate

            # Log win rate for each opponent
            self.logger.record(f"eval/win_rate_{opponent_type}", win_rate)

            if self.verbose > 0:
                print(f"Win Rate vs {opponent_type}: {win_rate * 100:.2f}%")

        # Calculate average win rate for best model selection
        avg_win_rate = sum(win_rates.values()) / len(win_rates)

        self.logger.record("eval/avg_win_rate", avg_win_rate)

        if self.verbose > 0:
            print(f"Average Win Rate: {avg_win_rate * 100:.2f}%")
            print(f"Best Average Win Rate: {self.best_win_rate * 100:.2f}%")

        # Save model if average win rate improved
        if avg_win_rate > self.best_win_rate:
            self.best_win_rate = avg_win_rate

            model_name = self.config.get_model_name(
                prefix="best", stage=self.current_stage
            )
            model_path = os.path.join(self.config.save_dir, model_name)
            self.model.save(model_path)

            if self.verbose > 0:
                print(f"🎉 New best model! Saved to {model_path}")

        # Now evaluate for stage advancement (stage-specific)
        if self.verbose > 0:
            print("\n--- Stage Advancement Evaluation ---")

        # Determine which opponent(s) to evaluate for stage transition
        if self.current_stage == 0:
            stage_eval_opponents = self.config.stage0_eval_opponent
        elif self.current_stage == 1:
            stage_eval_opponents = self.config.stage1_eval_opponent
        elif self.current_stage == 2:
            stage_eval_opponents = self.config.stage2_eval_opponents
        elif self.current_stage == 3:
            stage_eval_opponents = self.config.stage3_eval_opponents
        else:  # stage 4
            stage_eval_opponents = None  # No advancement from final stage

        win_rate_stage = None

        if stage_eval_opponents is not None and stage_eval_opponents in win_rates:
            win_rate_stage = win_rates[stage_eval_opponents]

        # Store evaluation results
        eval_result = {
            "timestep": self.num_timesteps,
            "stage": self.current_stage,
            "win_rates": win_rates,
            "avg_win_rate": avg_win_rate,
        }

        self.evaluations.append(eval_result)

        # Check for stage advancement
        should_advance = False

        if self.current_stage == 0 and win_rate_stage >= self.config.stage0_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print("Advancing to Stage 1: 80% Random, 10% Heuristic, 10% Milestone2")
                print("!" * 60 + "\n")

            should_advance = True

        elif self.current_stage == 1 and win_rate_stage >= self.config.stage1_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print("Advancing to Stage 2: 60% Random, 20% Heuristic, 20% Milestone2")
                print("!" * 60 + "\n")

            should_advance = True

        elif self.current_stage == 2 and win_rate_stage >= self.config.stage2_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print("Advancing to Stage 3: 40% Random, 30% Heuristic, 30% Milestone2")
                print("!" * 60 + "\n")

            should_advance = True

        elif self.current_stage == 3 and win_rate_stage >= self.config.stage3_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print("Advancing to Stage 4: 20% Random, 40% Heuristic, 40% Milestone2")
                print("!" * 60 + "\n")

            should_advance = True

        if should_advance:
            self.current_stage += 1
            self.stage_advanced = True

        if self.verbose > 0:
            print("=" * 60 + "\n")
