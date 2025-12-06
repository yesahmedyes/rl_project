import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.cql import CQLConfig

from misc.state_encoding import encode_handcrafted_state, encode_onehot_state


class Policy_CQL:
    def __init__(
        self,
        checkpoint_path: str,
        encoding_type: str = "handcrafted",
        device: str = "cpu",
    ):
        self.encoding_type = encoding_type
        self.max_actions = 12
        self.device = torch.device("cpu" if device == "auto" else device)

        ckpt = self._normalize_checkpoint_path(checkpoint_path)

        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Provide a valid path from train_cql.py outputs."
            )

        if not ray.is_initialized():
            ray.init(
                include_dashboard=False,
                ignore_reinit_error=True,
                log_to_driver=False,
            )

        ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent

        try:
            self.algo = Algorithm.from_checkpoint(str(ckpt_dir))
        except ValueError:
            config = CQLConfig()
            config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )

            if self.encoding_type == "handcrafted":
                observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(70,),
                    dtype=np.float32,
                )
            elif self.encoding_type == "onehot":
                observation_space = gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(946,),
                    dtype=np.float32,
                )
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")

            action_space = gym.spaces.Discrete(12)

            config.environment(
                observation_space=observation_space,
                action_space=action_space,
            )

            config.rollouts(num_rollout_workers=0)

            config.training(
                lr=3e-4,
                train_batch_size=256,
                min_q_weight=5.0,
                bc_iters=20000,
            )

            config.offline_data(
                input_="sampler",
                dataset_num_iters_per_learner=1,
            )

            self.algo = config.build()

            try:
                self.algo.restore(str(ckpt_dir))
            except ValueError:
                try:
                    policy_state = torch.load(str(ckpt), map_location=self.device)
                except Exception:
                    policy_state = torch.load(
                        str(ckpt), map_location=self.device, weights_only=False
                    )

                policy = self.algo.get_policy()

                if hasattr(policy, "import_state"):
                    policy.import_state(policy_state)
                elif hasattr(policy, "set_state"):
                    policy.set_state(policy_state)
                else:
                    raise RuntimeError(
                        "Policy checkpoint could not be loaded: no import_state/set_state"
                    )

        self.policy = self.algo.get_policy()

        if hasattr(self.policy, "model") and hasattr(self.policy.model, "to"):
            self.policy.model.to(self.device)
            self.policy.model.eval()

    @staticmethod
    def _find_latest_checkpoint(base_dir: Path):
        if not base_dir.exists() or not base_dir.is_dir():
            return None

        candidates = sorted(
            base_dir.glob("checkpoint*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return candidates[0] if candidates else None

    @staticmethod
    def _normalize_checkpoint_path(path_str: str) -> Path:
        """Handle raw paths and Ray result reprs like TrainingResult(checkpoint=...)."""
        ckpt = Path(path_str)

        if ckpt.exists():
            return ckpt

        match = re.search(r"path=([^,\\s)]+)", path_str)
        if match:
            candidate = Path(match.group(1))

            if candidate.exists():
                return candidate

            latest = Policy_CQL._find_latest_checkpoint(candidate)
            if latest:
                return latest

        if ckpt.is_dir():
            latest = Policy_CQL._find_latest_checkpoint(ckpt)
            if latest:
                return latest

        return ckpt

    def _encode_state(self, state):
        if self.encoding_type == "handcrafted":
            _, _, obs = encode_handcrafted_state(state)
        elif self.encoding_type == "onehot":
            _, _, obs = encode_onehot_state(state)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        return obs

    def _compute_action_scores(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs, _ = self.policy.model({"obs": obs_tensor}, [], None)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        return outputs.squeeze(0).detach().cpu().numpy()

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
            logits = self._compute_action_scores(obs)
            masked_logits = self._mask_logits(logits, action_space)
            action_int = int(np.argmax(masked_logits))
        except Exception:
            action_int, _, _ = self.algo.compute_single_action(obs, explore=False)

        dice_idx, goti_idx = divmod(action_int, 4)
        candidate = (dice_idx, goti_idx)

        if candidate not in action_space:
            return action_space[0]

        return candidate
