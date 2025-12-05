import numpy as np
import torch
import ray
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.bc import BCConfig
from misc.state_encoding import encode_handcrafted_state, encode_onehot_state


class Policy_BC:
    def __init__(
        self,
        checkpoint_path: str,
        encoding_type: str = "handcrafted",
        device: str = "cpu",
    ):
        self.encoding_type = encoding_type
        self.max_actions = 12
        self.device = torch.device("cpu" if device == "auto" else device)

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Provide a valid path from train_bc.py outputs."
            )

        if not ray.is_initialized():
            ray.init(
                include_dashboard=False,
                ignore_reinit_error=True,
                log_to_driver=False,
            )

        # RLlib v0 checkpoints need manual config + restore; newer checkpoints can
        # use Algorithm.from_checkpoint. We support both.
        ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent

        try:
            # Try the modern loader first
            self.algo = Algorithm.from_checkpoint(str(ckpt))
        except ValueError:
            # Build a compatible BC config (mirrors train_bc.py)
            config = BCConfig()
            config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )

            if self.encoding_type == "handcrafted":
                obs_space_shape = (70,)
            elif self.encoding_type == "onehot":
                obs_space_shape = (946,)
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")

            config.environment(
                observation_space=(
                    "Box",
                    {"shape": obs_space_shape, "low": -np.inf, "high": np.inf},
                ),
                action_space=("Discrete", {"n": 12}),
            )

            # Minimal training settings (not used in inference but required)
            config.training(
                lr=1e-4,
                train_batch_size_per_learner=64,
            )

            # No offline data needed at inference time
            config.offline_data(
                input_=None,
                dataset_num_iters_per_learner=1,
            )

            self.algo = config.build()
            self.algo.restore(str(ckpt_dir))

        self.policy = self.algo.get_policy()

        # Ensure model sits on the requested device
        if hasattr(self.policy, "model") and hasattr(self.policy.model, "to"):
            self.policy.model.to(self.device)
            self.policy.model.eval()

    def _encode_state(self, state):
        if self.encoding_type == "handcrafted":
            _, _, obs = encode_handcrafted_state(state)
        elif self.encoding_type == "onehot":
            _, _, obs = encode_onehot_state(state)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        return obs

    def _compute_logits(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.policy.model({"obs": obs_tensor}, [], None)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        return logits.squeeze(0).detach().cpu().numpy()

    def _mask_logits(self, logits: np.ndarray, action_space) -> np.ndarray:
        masked = np.full(self.max_actions, -np.inf, dtype=np.float32)

        for dice_idx, goti_idx in action_space:
            action_int = dice_idx * 4 + goti_idx
            if 0 <= action_int < self.max_actions:
                masked[action_int] = logits[action_int]

        return masked

    def get_action(self, state, action_space):
        if not action_space:
            return None

        obs = self._encode_state(state)

        try:
            logits = self._compute_logits(obs)
            masked_logits = self._mask_logits(logits, action_space)
            action_int = int(np.argmax(masked_logits))
        except Exception:
            # Fallback to RLlib's helper if direct logits computation fails
            action_int, _, _ = self.algo.compute_single_action(obs, explore=False)

        dice_idx, goti_idx = divmod(action_int, 4)
        candidate = (dice_idx, goti_idx)

        if candidate not in action_space:
            # Safety fallback to first valid action
            return action_space[0]

        return candidate
