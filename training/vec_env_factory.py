"""
Factory functions for creating vectorized Ludo environments.
"""

from typing import Optional, Callable, Any
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
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
    """
    Create a function that returns a Ludo environment.

    Args:
        rank: Unique ID for the environment
        encoding_type: "handcrafted" or "onehot"
        opponent_type: "random", "heuristic", or "self"
        opponent_policy: Custom opponent policy object
        agent_player: Which player is the agent (0 or 1)
        log_dir: Directory to save monitor logs
        seed: Random seed

    Returns:
        Callable that creates the environment
    """

    def _init() -> Monitor:
        from env.ludo_gym_env import LudoGymEnv

        env = LudoGymEnv(
            encoding_type=encoding_type,
            opponent_policy=opponent_policy,
            opponent_type=opponent_type,
            agent_player=agent_player,
        )

        # Set seed for reproducibility
        env.reset(seed=seed + rank)

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
    """
    Create a vectorized environment with multiple parallel Ludo environments.

    Args:
        n_envs: Number of parallel environments
        encoding_type: "handcrafted" or "onehot"
        opponent_type: "random", "heuristic", or "self"
        opponent_policy: Custom opponent policy object
        agent_player: Which player is the agent (0 or 1)
        log_dir: Directory to save monitor logs
        seed: Random seed
        use_subprocess: If True, use SubprocVecEnv (parallel), else DummyVecEnv (sequential)

    Returns:
        Vectorized environment
    """
    # Create list of environment creation functions
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
) -> Monitor:
    """
    Create a single evaluation environment.

    Args:
        encoding_type: "handcrafted" or "onehot"
        opponent_type: "random", "heuristic", or "self"
        opponent_policy: Custom opponent policy object
        agent_player: Which player is the agent (0 or 1)
        seed: Random seed

    Returns:
        Monitored environment
    """
    from env.ludo_gym_env import LudoGymEnv

    env = LudoGymEnv(
        encoding_type=encoding_type,
        opponent_policy=opponent_policy,
        opponent_type=opponent_type,
        agent_player=agent_player,
    )

    env.reset(seed=seed)
    env = Monitor(env)

    return env


class SelfPlayVecEnv:
    """
    Wrapper for vectorized environments that handles self-play by updating opponent policies.
    """

    def __init__(
        self,
        n_envs: int,
        encoding_type: str = "handcrafted",
        agent_player: int = 0,
        log_dir: Optional[str] = None,
        seed: int = 0,
        use_subprocess: bool = True,
    ):
        """
        Initialize self-play vectorized environment.

        Args:
            n_envs: Number of parallel environments
            encoding_type: "handcrafted" or "onehot"
            agent_player: Which player is the agent (0 or 1)
            log_dir: Directory to save monitor logs
            seed: Random seed
            use_subprocess: If True, use SubprocVecEnv (parallel), else DummyVecEnv (sequential)
        """
        self.n_envs = n_envs
        self.encoding_type = encoding_type
        self.agent_player = agent_player
        self.log_dir = log_dir
        self.seed = seed
        self.use_subprocess = use_subprocess

        # Create initial vec env with random opponent
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
        """
        Update the opponent policy for self-play.

        This requires recreating the vectorized environment with the new policy.

        Args:
            policy: New opponent policy
        """
        # Close existing environments
        self.vec_env.close()

        # Create new vec env with updated opponent policy
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
        """Delegate attribute access to the underlying vec_env."""
        return getattr(self.vec_env, name)

    def close(self):
        """Close the vectorized environment."""
        self.vec_env.close()
