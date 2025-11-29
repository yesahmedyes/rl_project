from typing import Optional, Callable, Any
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
import os


def _get_action_mask(env):
    return env.unwrapped._get_action_mask()


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
    ):
        self.n_envs = n_envs
        self.encoding_type = encoding_type
        self.agent_player = agent_player
        self.log_dir = log_dir
        self.seed = seed
        self.use_subprocess = use_subprocess
        self.opponent_model_path = opponent_model_path

        # Load opponent model if path is provided
        opponent_policy = None
        opponent_type = "random"

        if opponent_model_path is not None:
            from sb3_contrib import MaskablePPO

            print(f"Loading opponent model from {opponent_model_path}...")

            opponent_model = MaskablePPO.load(opponent_model_path, device="cpu")

            # Set opponent policy to eval mode and ensure no gradients
            opponent_model.policy.eval()

            for param in opponent_model.policy.parameters():
                param.requires_grad = False

            # Wrap the neural network policy
            opponent_policy = NeuralNetworkPolicyWrapper(
                opponent_model.policy, self.encoding_type, device="cpu"
            )

            opponent_type = "self"

            print("Opponent model loaded successfully (CPU, eval mode, no gradients)")

        self.vec_env = make_vec_env(
            n_envs=n_envs,
            encoding_type=encoding_type,
            opponent_type=opponent_type,
            opponent_policy=opponent_policy,
            agent_player=agent_player,
            log_dir=log_dir,
            seed=seed,
            use_subprocess=use_subprocess,
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


class NeuralNetworkPolicyWrapper:
    def __init__(self, policy, encoding_type: str = "handcrafted", device: str = "cpu"):
        self.policy = policy
        self.encoding_type = encoding_type
        self.device = device

        # Ensure policy is on the correct device
        self.policy.to(self.device)
        # Set to eval mode
        self.policy.eval()

    def get_action(self, state, action_space):
        if not action_space:
            return None

        # Import here to avoid circular imports
        from misc.state_encoding import encode_handcrafted_state, encode_onehot_state
        import torch
        import numpy as np

        # Encode state
        if self.encoding_type == "handcrafted":
            _, _, obs = encode_handcrafted_state(state)
        else:
            _, _, obs = encode_onehot_state(state)

        # Create action mask
        max_actions = 12
        mask = np.zeros(max_actions, dtype=np.int8)
        for dice_idx, goti_idx in action_space:
            action_int = dice_idx * 4 + goti_idx
            if action_int < max_actions:
                mask[action_int] = 1

        # Get action from policy (with no gradient computation)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Get action distribution
            dist = self.policy.get_distribution(obs_tensor)
            # Apply mask
            action_logits = dist.distribution.logits.clone()
            action_logits[0, mask == 0] = float("-inf")
            action = action_logits.argmax().item()

        # Convert back to tuple
        dice_idx = action // 4
        goti_idx = action % 4

        return (dice_idx, goti_idx)
