from stable_baselines3.common.callbacks import BaseCallback
import os


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        model_name: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name
        self.last_save_timestep = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, self.model_name)
            self.model.save(model_path)

            if self.verbose > 0:
                print(
                    f"Saved checkpoint to {model_path} at {self.num_timesteps} timesteps"
                )

            self.last_save_timestep = self.num_timesteps

        return True
