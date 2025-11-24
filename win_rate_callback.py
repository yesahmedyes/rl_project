import csv
import os
from typing import Dict, List
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from ludo_maskable_env import LudoMaskableEnv


def mask_fn(env: LudoMaskableEnv):
    return env.get_action_mask()


class WinRateCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int = 100_000,
        n_eval_games: int = 100,
        opponents: List[str] = None,
        dense_rewards: bool = True,
        log_path: str = "logs/win_rates.csv",
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.opponents = opponents or ["random", "heuristic"]
        self.dense_rewards = dense_rewards
        self.log_path = log_path
        self.win_rates_history: List[Dict] = []
        self.last_eval_timestep = 0

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestep"] + [f"win_rate_vs_{opp}" for opp in self.opponents]
                )

    def _evaluate_model(self, model: MaskablePPO) -> Dict[str, float]:
        win_rates = {}

        # Set model to eval mode for evaluation
        model.policy.set_training_mode(False)

        for opponent in self.opponents:
            env = ActionMasker(
                LudoMaskableEnv(
                    opponents=(opponent,),
                    dense_rewards=self.dense_rewards,
                    alternate_start=True,
                    seed=42,
                ),
                mask_fn,
            )

            wins = 0

            for _ in range(self.n_eval_games):
                obs, info = env.reset()

                done = False

                while not done:
                    action, _ = model.predict(
                        obs, action_masks=info.get("action_mask"), deterministic=True
                    )
                    obs, reward, done, _, info = env.step(action)

                if env.unwrapped.agent_won():
                    wins += 1

            win_rate = wins / self.n_eval_games
            win_rates[opponent] = win_rate

            if self.verbose > 0:
                print(f"  Win rate vs {opponent}: {win_rate:.2%}")

        # Restore training mode
        model.policy.set_training_mode(True)

        return win_rates

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            if self.verbose > 0:
                print(f"\n📊 Evaluating win rates at timestep {self.num_timesteps}...")

            win_rates = self._evaluate_model(self.model)

            # Log to CSV
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                row = [self.num_timesteps] + [win_rates[opp] for opp in self.opponents]
                writer.writerow(row)

            # Log to TensorBoard
            for opponent, win_rate in win_rates.items():
                self.logger.record(f"eval/win_rate_vs_{opponent}", win_rate)
            self.logger.dump(self.num_timesteps)

            win_rates["timestep"] = self.num_timesteps
            self.win_rates_history.append(win_rates)
            self.last_eval_timestep = self.num_timesteps

            if self.verbose > 0:
                print(f"✅ Win rates saved to {self.log_path}\n")

        return True
