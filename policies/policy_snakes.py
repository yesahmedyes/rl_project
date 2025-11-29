import os
import torch
import numpy as np
from sb3_contrib import MaskablePPO
from misc.state_encoding import encode_handcrafted_state, encode_onehot_state
from config import TrainingConfig


class Policy_Snakes:
    def __init__(
        self,
        checkpoint_path: str = None,
        encoding_type: str = "handcrafted",
        device: str = "auto",
    ):
        self.encoding_type = encoding_type
        self.device = device

        if checkpoint_path is None:
            config = TrainingConfig(encoding_type=encoding_type)
            model_name = config.get_model_name(prefix="latest")
            checkpoint_path = os.path.join(config.save_dir, model_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Please train a model first or specify a valid checkpoint path."
            )

        print(f"Loading trained model from {checkpoint_path} on device {device}...")

        self.model = MaskablePPO.load(checkpoint_path, device=device)
        self.policy = self.model.policy
        self.policy.eval()

        # Disable gradients for inference-only use
        for param in self.policy.parameters():
            param.requires_grad = False

        print(f"Model loaded successfully on {device}!")

    def get_action(self, state, action_space):
        if not action_space:
            return None

        if self.encoding_type == "handcrafted":
            _, _, obs = encode_handcrafted_state(state)
        elif self.encoding_type == "onehot":
            _, _, obs = encode_onehot_state(state)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        max_actions = 12

        mask = np.zeros(max_actions, dtype=np.int8)

        for dice_idx, goti_idx in action_space:
            action_int = dice_idx * 4 + goti_idx

            if action_int < max_actions:
                mask[action_int] = 1

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

        selected_action = (dice_idx, goti_idx)

        if selected_action not in action_space:
            # Fallback: return first valid action if something went wrong
            return action_space[0]

        return (dice_idx, goti_idx)

    def get_actions_batch(self, states, action_spaces):
        """
        Batch version of get_action for efficient GPU inference.

        Args:
            states: List of game states
            action_spaces: List of action spaces (list of valid (dice_idx, goti_idx) tuples)

        Returns:
            List of actions as tuples (dice_idx, goti_idx)
        """
        if not states:
            return []

        batch_size = len(states)

        # Encode all states
        obs_list = []
        for state in states:
            if self.encoding_type == "handcrafted":
                _, _, obs = encode_handcrafted_state(state)
            elif self.encoding_type == "onehot":
                _, _, obs = encode_onehot_state(state)
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            obs_list.append(obs)

        # Create batch tensor
        obs_batch = torch.as_tensor(np.stack(obs_list)).to(self.device)

        # Create masks for all states
        max_actions = 12
        masks = np.zeros((batch_size, max_actions), dtype=np.int8)
        for i, action_space in enumerate(action_spaces):
            for dice_idx, goti_idx in action_space:
                action_int = dice_idx * 4 + goti_idx
                if action_int < max_actions:
                    masks[i, action_int] = 1

        # Batch forward pass
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_batch)
            action_logits = dist.distribution.logits.clone()

            # Apply masks
            mask_tensor = torch.as_tensor(masks).to(self.device)
            action_logits[mask_tensor == 0] = float("-inf")
            actions = action_logits.argmax(dim=1).cpu().numpy()

        # Convert back to tuples
        results = []
        for i, action_int in enumerate(actions):
            dice_idx = action_int // 4
            goti_idx = action_int % 4
            selected_action = (dice_idx, goti_idx)

            # Validate action is in action space
            if selected_action not in action_spaces[i] and action_spaces[i]:
                # Fallback to first valid action
                selected_action = action_spaces[i][0]

            results.append(selected_action)

        return results
