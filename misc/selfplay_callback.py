from stable_baselines3.common.callbacks import BaseCallback
from env.vec_env_factory import SelfPlayVecEnv


class SelfPlayCallback(BaseCallback):
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
        if self.num_timesteps - self.last_update >= self.update_freq:
            self._update_opponent()
            self.last_update = self.num_timesteps

        return True

    def _update_opponent(self):
        if self.verbose > 0:
            print("\n" + "~" * 60)
            print(f"Updating opponent policy at step {self.num_timesteps}")
            print("~" * 60 + "\n")

        self.vec_env.update_opponent_policy(self.model.policy)

        self.logger.record(
            "self_play/opponent_updates", self.num_timesteps // self.update_freq
        )
