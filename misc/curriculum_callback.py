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
        else:
            opponent_type = "heuristic"

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

        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/stage", self.current_stage)

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

        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate

            model_path = os.path.join(
                self.config.save_dir,
                f"best_model_stage{self.current_stage}_{self.config.encoding_type}.zip",
            )

            self.model.save(model_path)

            if self.verbose > 0:
                print(f"Saved new best model to {model_path}")

        if self.current_stage == 1 and win_rate >= self.stage1_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print("Advancing to Stage 2: Training vs Heuristic")
                print("!" * 60 + "\n")

            self.current_stage = 2
            self.stage_advanced = True
            self.best_win_rate = 0.0  # Reset for new stage

        elif self.current_stage == 2 and win_rate >= self.stage2_threshold:
            if self.verbose > 0:
                print("\n" + "!" * 60)
                print("Advancing to Stage 3: Self-Play Training")
                print("!" * 60 + "\n")

            self.current_stage = 3
            self.stage_advanced = True
            self.best_win_rate = 0.0  # Reset for new stage

        if self.verbose > 0:
            print("=" * 60 + "\n")
