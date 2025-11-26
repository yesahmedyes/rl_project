from typing import Optional, Callable, Any
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
import os


def make_ludo_env(
    rank: int,
    encoding_type: str = "handcrafted",
    opponent_type: str = "random",
    opponent_policy: Optional[Any] = None,
    agent_player: int = 0,
    log_dir: Optional[str] = None,
    seed: int = 0,
) -> Callable:
    def _init():
        from env.ludo_gym_env import LudoGymEnv

        env = LudoGymEnv(
            encoding_type=encoding_type,
            opponent_policy=opponent_policy,
            opponent_type=opponent_type,
            agent_player=agent_player,
        )

        # Set seed for reproducibility
        env.reset(seed=seed + rank)

        # Wrap with ActionMasker for MaskablePPO
        def mask_fn(env):
            # Return the action mask from the current state
            return env.unwrapped._get_action_mask()

        env = ActionMasker(env, mask_fn)

        # Wrap with Monitor for logging
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))

        return env

    return _init


def make_vec_env(
    n_envs: int,
    encoding_type: str = "handcrafted",
    opponent_type: str = "random",
    opponent_policy: Optional[Any] = None,
    agent_player: int = 0,
    log_dir: Optional[str] = None,
    seed: int = 0,
    use_subprocess: bool = True,
) -> SubprocVecEnv:
    env_fns = [
        make_ludo_env(
            rank=i,
            encoding_type=encoding_type,
            opponent_type=opponent_type,
            opponent_policy=opponent_policy,
            agent_player=agent_player,
            log_dir=log_dir,
            seed=seed,
        )
        for i in range(n_envs)
    ]

    # Create vectorized environment
    if use_subprocess:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env


def make_eval_env(
    encoding_type: str = "handcrafted",
    opponent_type: str = "random",
    opponent_policy: Optional[Any] = None,
    agent_player: int = 0,
    seed: int = 0,
):
    from env.ludo_gym_env import LudoGymEnv

    env = LudoGymEnv(
        encoding_type=encoding_type,
        opponent_policy=opponent_policy,
        opponent_type=opponent_type,
        agent_player=agent_player,
    )

    env.reset(seed=seed)

    # Wrap with ActionMasker for MaskablePPO
    def mask_fn(env):
        return env.unwrapped._get_action_mask()

    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    return env


class SelfPlayVecEnv:
    def __init__(
        self,
        n_envs: int,
        encoding_type: str = "handcrafted",
        agent_player: int = 0,
        log_dir: Optional[str] = None,
        seed: int = 0,
        use_subprocess: bool = True,
    ):
        self.n_envs = n_envs
        self.encoding_type = encoding_type
        self.agent_player = agent_player
        self.log_dir = log_dir
        self.seed = seed
        self.use_subprocess = use_subprocess

        self.vec_env = make_vec_env(
            n_envs=n_envs,
            encoding_type=encoding_type,
            opponent_type="random",
            agent_player=agent_player,
            log_dir=log_dir,
            seed=seed,
            use_subprocess=use_subprocess,
        )

    def update_opponent_policy(self, policy):
        self.vec_env.close()

        self.vec_env = make_vec_env(
            n_envs=self.n_envs,
            encoding_type=self.encoding_type,
            opponent_type="self",
            opponent_policy=policy,
            agent_player=self.agent_player,
            log_dir=self.log_dir,
            seed=self.seed,
            use_subprocess=self.use_subprocess,
        )

    def __getattr__(self, name):
        return getattr(self.vec_env, name)

    def close(self):
        self.vec_env.close()
