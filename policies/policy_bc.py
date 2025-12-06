import numpy as np
import torch
from pathlib import Path
from torch import nn
from typing import Sequence

from misc.state_encoding import encode_handcrafted_state, encode_onehot_state


class BCPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_layers: Sequence[int], action_dim: int):
        super().__init__()
        layers = []
        last = obs_dim

        for width in hidden_layers:
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            last = width

        layers.append(nn.Linear(last, action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


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

        ckpt = self._resolve_checkpoint_path(checkpoint_path)

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

        checkpoint = torch.load(str(ckpt), map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict containing model weights.")

        metadata = checkpoint.get("metadata", {})
        state_dict = checkpoint.get("state_dict")

        if state_dict is None:
            modules = checkpoint.get("modules", {})
            state_dict = modules.get("model")

        obs_dim = metadata.get("obs_dim")
        hidden_layers = metadata.get("hidden_layers", [256, 256])

        if obs_dim is None:
            obs_dim = 70 if encoding_type == "handcrafted" else 946

        if state_dict is None:
            raise ValueError("Checkpoint missing model weights.")

        self.model = BCPolicyNet(
            obs_dim=obs_dim,
            hidden_layers=hidden_layers,
            action_dim=self.max_actions,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_checkpoint_path(path_str: str) -> Path:
        ckpt_path = Path(path_str)

        if ckpt_path.is_dir():
            candidates = sorted(
                ckpt_path.glob("model_epoch_*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            if candidates:
                return candidates[0]

        return ckpt_path

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
            logits = self.model(obs_tensor)

        return logits.squeeze(0).detach().cpu().numpy()

    def _mask_logits(self, logits: np.ndarray, action_space) -> np.ndarray:
        masked = np.full(self.max_actions, -np.inf, dtype=np.float32)

        for dice_idx, goti_idx in action_space:
            action_int = dice_idx * 4 + goti_idx
            if 0 <= action_int < self.max_actions:
                masked[action_int] = logits[action_int]

        return masked

    def get_action_from_ted_step(self, ted_step: dict, action_space):
        if ted_step is None or "observation" not in ted_step:
            raise ValueError("TED step must contain an 'observation' entry.")

        obs = np.asarray(ted_step["observation"], dtype=np.float32)
        logits = self._compute_logits(obs)
        masked_logits = self._mask_logits(logits, action_space)
        action_int = int(np.argmax(masked_logits))

        dice_idx, goti_idx = divmod(action_int, 4)
        candidate = (dice_idx, goti_idx)

        if candidate not in action_space:
            return action_space[0]

        return candidate

    def get_action(self, state, action_space):
        if not action_space:
            return None

        obs = self._encode_state(state)
        logits = self._compute_logits(obs)
        masked_logits = self._mask_logits(logits, action_space)
        action_int = int(np.argmax(masked_logits))

        dice_idx, goti_idx = divmod(action_int, 4)
        candidate = (dice_idx, goti_idx)

        if candidate not in action_space:
            return action_space[0]

        return candidate
