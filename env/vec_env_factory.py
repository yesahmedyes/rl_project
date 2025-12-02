from typing import Optional, Callable, Any, List
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
import os
import numpy as np


def _get_action_mask(env):
    return env.unwrapped._get_action_mask()


def _distribute_opponents(
    n_envs: int, opponent_distribution: List[float], seed: int = 0
) -> List[str]:
    """
    Distribute opponent types across environments based on the distribution.

    Args:
        n_envs: Number of parallel environments
        opponent_distribution: List of [random, heuristic, milestone2] ratios (must sum to 1.0)
        seed: Random seed for shuffling

    Returns:
        List of opponent types for each environment
    """
    opponent_types = ["random", "heuristic", "milestone2"]

    # Validate distribution
    if len(opponent_distribution) != 3:
        raise ValueError(
            "opponent_distribution must have exactly 3 elements [random, heuristic, milestone2]"
        )

    if not np.isclose(sum(opponent_distribution), 1.0):
        raise ValueError(
            f"opponent_distribution must sum to 1.0, got {sum(opponent_distribution)}"
        )

    # Calculate number of envs for each opponent type
    counts = []
    remaining = n_envs

    for i, ratio in enumerate(opponent_distribution[:-1]):
        count = int(np.round(ratio * n_envs))
        counts.append(count)
        remaining -= count

    # Last one gets the remaining to ensure we have exactly n_envs
    counts.append(remaining)

    # Create list of opponent types
    opponent_list = []
    for opponent_type, count in zip(opponent_types, counts):
        opponent_list.extend([opponent_type] * count)

    # Shuffle to mix opponent types (with seed for reproducibility)
    rng = np.random.RandomState(seed)
    rng.shuffle(opponent_list)

    return opponent_list


def make_ludo_env(
    rank: int,
    encoding_type: str = "handcrafted",
    opponent_type: str = "random",  # "random", "heuristic", "milestone2", "self"
    opponent_policy: Optional[Any] = None,
    agent_player: int = 0,
    log_dir: Optional[str] = None,
    seed: int = 0,
    use_dense_reward: bool = True,
) -> Callable:
    def _init():
        from env.ludo_gym_env import LudoGymEnv

        env = LudoGymEnv(
            encoding_type=encoding_type,
            opponent_policy=opponent_policy,
            opponent_type=opponent_type,
            agent_player=agent_player,
            use_dense_reward=use_dense_reward,
        )

        # Set seed for reproducibility
        env.reset(seed=seed + rank)

        # Wrap with ActionMasker for MaskablePPO
        env = ActionMasker(env, _get_action_mask)

        # Wrap with Monitor for logging
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))

        return env

    return _init


def make_vec_env(
    n_envs: int,
    encoding_type: str = "handcrafted",
    opponent_type: Optional[str] = None,
    opponent_distribution: Optional[List[float]] = None,
    opponent_policy: Optional[Any] = None,
    agent_player: int = 0,
    log_dir: Optional[str] = None,
    seed: int = 0,
    use_subprocess: bool = True,
    use_dense_reward: bool = True,
) -> SubprocVecEnv:
    """
    Create vectorized Ludo environments.

    Args:
        n_envs: Number of parallel environments
        encoding_type: State encoding type ("handcrafted" or "onehot")
        opponent_type: Single opponent type for all envs (use this OR opponent_distribution)
        opponent_distribution: Distribution of opponents [random, heuristic, milestone2] ratios (must sum to 1.0)
                              Example: [1, 0, 0] = all random, [0.5, 0.5, 0] = 50% random + 50% heuristic
        opponent_policy: Custom opponent policy (overrides opponent_type/opponent_distribution)
        agent_player: Which player is the agent (0 or 1)
        log_dir: Directory for logging
        seed: Random seed
        use_subprocess: Whether to use subprocesses
        use_dense_reward: Whether to use dense reward shaping

    Returns:
        Vectorized environment
    """
    if opponent_distribution is not None:
        opponent_types = _distribute_opponents(n_envs, opponent_distribution, seed)
    elif opponent_type is not None:
        opponent_types = [opponent_type] * n_envs
    else:
        raise ValueError("opponent_type or opponent_distribution must be provided")

    env_fns = [
        make_ludo_env(
            rank=i,
            encoding_type=encoding_type,
            opponent_type=opponent_types[i],
            opponent_policy=opponent_policy,
            agent_player=agent_player,
            log_dir=log_dir,
            seed=seed,
            use_dense_reward=use_dense_reward,
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
    use_dense_reward: bool = True,
):
    from env.ludo_gym_env import LudoGymEnv

    env = LudoGymEnv(
        encoding_type=encoding_type,
        opponent_policy=opponent_policy,
        opponent_type=opponent_type,
        agent_player=agent_player,
        use_dense_reward=use_dense_reward,
    )

    env.reset(seed=seed)

    # Wrap with ActionMasker for MaskablePPO
    env = ActionMasker(env, _get_action_mask)
    env = Monitor(env)

    return env


class SelfPlayVecEnv(VecEnv):
    def __init__(
        self,
        n_envs: int,
        encoding_type: str = "handcrafted",
        agent_player: int = 0,
        log_dir: Optional[str] = None,
        seed: int = 0,
        use_subprocess: bool = True,
        opponent_model_path: Optional[str] = None,
        opponent_device: str = "cpu",
        use_dense_reward: bool = True,
    ):
        self.n_envs = n_envs
        self.encoding_type = encoding_type
        self.agent_player = agent_player
        self.log_dir = log_dir
        self.seed = seed
        self.use_subprocess = use_subprocess
        self.opponent_model_path = opponent_model_path
        self.opponent_device = opponent_device
        self.use_dense_reward = use_dense_reward

        # Load opponent model if path is provided
        opponent_policy = None
        opponent_type = "random"

        if opponent_model_path is not None:
            from policies.policy_snakes import Policy_Snakes

            print(f"Loading opponent model from {opponent_model_path}...")
            print(f"Opponent will run on device: {opponent_device}")

            # Create Policy_Snakes with specified device
            # Note: This will be recreated in each subprocess
            opponent_policy = Policy_Snakes(
                checkpoint_path=opponent_model_path,
                encoding_type=self.encoding_type,
                device=opponent_device,
            )
            opponent_type = "self"
            print("Opponent model loaded successfully")

        self.vec_env = make_vec_env(
            n_envs=n_envs,
            encoding_type=encoding_type,
            opponent_type=opponent_type,
            opponent_policy=opponent_policy,
            agent_player=agent_player,
            log_dir=log_dir,
            seed=seed,
            use_subprocess=use_subprocess,
            use_dense_reward=self.use_dense_reward,
        )

        # Initialize VecEnv base class
        super().__init__(
            num_envs=self.vec_env.num_envs,
            observation_space=self.vec_env.observation_space,
            action_space=self.vec_env.action_space,
        )

    # Implement required VecEnv methods
    def reset(self):
        return self.vec_env.reset()

    def step_async(self, actions):
        return self.vec_env.step_async(actions)

    def step_wait(self):
        return self.vec_env.step_wait()

    def close(self):
        return self.vec_env.close()

    def get_attr(self, attr_name, indices=None):
        return self.vec_env.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.vec_env.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.vec_env.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.vec_env.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed=None):
        return self.vec_env.seed(seed)

    def __getattr__(self, name):
        # Fallback for any other attributes not explicitly defined
        return getattr(self.vec_env, name)


class BatchedOpponentVecEnv(VecEnv):
    def __init__(
        self,
        n_envs: int,
        encoding_type: str = "handcrafted",
        agent_player: int = 0,
        log_dir: Optional[str] = None,
        seed: int = 0,
        opponent_model_path: Optional[str] = None,
        opponent_device: str = "cpu",
        use_dense_reward: bool = True,
    ):
        self.n_envs = n_envs
        self.encoding_type = encoding_type
        self.agent_player = agent_player
        self.log_dir = log_dir
        self.seed = seed
        self.opponent_device = opponent_device
        self.use_dense_reward = use_dense_reward

        # Load opponent model ONCE in main process
        self.opponent_policy = None

        if opponent_model_path is not None:
            from policies.policy_snakes import Policy_Snakes

            print(f"Loading opponent model from {opponent_model_path}...")
            print(f"Opponent will run on device: {opponent_device}")

            self.opponent_policy = Policy_Snakes(
                checkpoint_path=opponent_model_path,
                encoding_type=encoding_type,
                device=opponent_device,
            )
            print("Opponent model loaded successfully (batched inference enabled)")

        # Create environments WITHOUT opponent policy (we'll handle it centrally)
        # Use DummyVecEnv (single process) to avoid GPU conflicts
        env_fns = [
            make_ludo_env(
                rank=i,
                encoding_type=encoding_type,
                opponent_type="self",  # We'll handle opponent actions ourselves
                opponent_policy=None,  # Don't give env the opponent policy
                agent_player=agent_player,
                log_dir=log_dir,
                seed=seed,
                use_dense_reward=self.use_dense_reward,
            )
            for i in range(n_envs)
        ]

        self.vec_env = DummyVecEnv(env_fns)

        # Initialize VecEnv base class
        super().__init__(
            num_envs=self.vec_env.num_envs,
            observation_space=self.vec_env.observation_space,
            action_space=self.vec_env.action_space,
        )

        # Override opponent policy in each environment
        if self.opponent_policy is not None:
            for env in self.vec_env.envs:
                env.unwrapped.unwrapped.set_opponent_policy(self.opponent_policy)

    # Implement required VecEnv methods - delegate to vec_env
    def reset(self):
        return self.vec_env.reset()

    def step_async(self, actions):
        return self.vec_env.step_async(actions)

    def step_wait(self):
        return self.vec_env.step_wait()

    def close(self):
        return self.vec_env.close()

    def get_attr(self, attr_name, indices=None):
        return self.vec_env.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.vec_env.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.vec_env.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.vec_env.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed=None):
        return self.vec_env.seed(seed)

    def __getattr__(self, name):
        # Fallback for any other attributes not explicitly defined
        return getattr(self.vec_env, name)
