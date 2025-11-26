import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any
from sb3_contrib import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
import warnings


class BCRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.raw_states = None
        self.heuristic_actions = None

    def reset(self) -> None:
        super().reset()

        self.raw_states = []
        self.heuristic_actions = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
        raw_state: Optional[Any] = None,
        heuristic_action: Optional[int] = None,
    ) -> None:
        """Add with optional raw state and heuristic action."""
        # Call parent add method
        if hasattr(super(), "add"):
            # For MaskableRolloutBuffer
            super().add(
                obs, action, reward, episode_start, value, log_prob, action_masks
            )
        else:
            super().add(obs, action, reward, episode_start, value, log_prob)

        # Store raw state and heuristic action if provided
        if raw_state is not None:
            self.raw_states.append(raw_state)
        if heuristic_action is not None:
            self.heuristic_actions.append(heuristic_action)


class BCMaskablePPO(MaskablePPO):
    """
    MaskablePPO with optional Behavior Cloning loss.

    Adds a cross-entropy loss term that encourages the policy to match
    a heuristic policy's action choices. The BC loss coefficient can decay
    over time to gradually transition from imitation to pure RL.
    """

    def __init__(
        self,
        *args,
        use_bc_loss: bool = False,
        bc_loss_coef: float = 1.0,
        bc_loss_decay_rate: float = 0.0,
        bc_loss_min_coef: float = 0.0,
        **kwargs,
    ):
        """
        Initialize BCMaskablePPO.

        Args:
            use_bc_loss: Enable behavior cloning loss
            bc_loss_coef: Initial coefficient for BC loss
            bc_loss_decay_rate: Decay rate for BC loss coefficient (per update)
            bc_loss_min_coef: Minimum BC loss coefficient (decay stops here)
        """
        super().__init__(*args, **kwargs)

        self.use_bc_loss = use_bc_loss
        self.bc_loss_coef_initial = bc_loss_coef
        self.bc_loss_coef_current = bc_loss_coef
        self.bc_loss_decay_rate = bc_loss_decay_rate
        self.bc_loss_min_coef = bc_loss_min_coef

        # Storage for heuristic actions during rollout collection
        self.current_heuristic_actions = {}

        # Initialize heuristic policy if BC loss is enabled
        if self.use_bc_loss:
            from policies.policy_heuristic import Policy_Heuristic

            self.heuristic_policy = Policy_Heuristic()
            print(
                f"Behavior Cloning enabled with initial coef={bc_loss_coef}, "
                f"decay={bc_loss_decay_rate}, min={bc_loss_min_coef}"
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a RolloutBuffer.

        Extended to compute and store heuristic actions for BC loss.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        # Clear heuristic actions storage
        self.current_heuristic_actions = {}

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # Get actions and values from policy
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                actions_cpu = actions.cpu().numpy()

            # Compute heuristic actions if BC loss is enabled
            heuristic_actions_batch = None
            if self.use_bc_loss and self.bc_loss_coef_current > 0:
                heuristic_actions_batch = []

                # Get raw game states from environments
                for env_idx in range(env.num_envs):
                    try:
                        # Try to get raw state from environment
                        raw_state = None
                        if hasattr(env, "get_attr"):
                            raw_state = env.get_attr(
                                "current_state", indices=[env_idx]
                            )[0]

                        if raw_state is not None:
                            # Get action space from environment
                            action_space_raw = env.get_attr("env", indices=[env_idx])[
                                0
                            ].get_action_space()

                            # Get heuristic action
                            heuristic_action_tuple = self.heuristic_policy.get_action(
                                raw_state, action_space_raw
                            )

                            # Convert to discrete action
                            if heuristic_action_tuple is not None:
                                dice_idx, goti_idx = heuristic_action_tuple
                                heuristic_action = dice_idx * 4 + goti_idx
                            else:
                                heuristic_action = None
                        else:
                            heuristic_action = None
                    except Exception:
                        # If we can't get heuristic action, skip
                        heuristic_action = None

                    heuristic_actions_batch.append(heuristic_action)

                # Store for this step
                self.current_heuristic_actions[n_steps] = heuristic_actions_batch

            # Rescale and perform action
            clipped_actions = actions_cpu

            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, torch.Tensor):
                clipped_actions = np.clip(
                    actions_cpu, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, torch.Tensor):
                actions = torch.from_numpy(actions_cpu).to(self.device)

            # Handle timeout (episode truncation)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # Add to buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Compute returns and advantages
        with torch.no_grad():
            values = self.policy.predict_values(
                torch.as_tensor(new_obs).to(self.device)
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Overrides the base train() method to add behavior cloning loss.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current values and log probs from the rollout buffer
        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        bc_losses = []

        continue_training = True

        # Prepare heuristic actions tensor if BC is enabled
        heuristic_actions_tensor = None
        if (
            self.use_bc_loss
            and self.bc_loss_coef_current > 0
            and self.current_heuristic_actions
        ):
            # Flatten heuristic actions from dict to array
            all_heuristic_actions = []
            for step_idx in sorted(self.current_heuristic_actions.keys()):
                all_heuristic_actions.extend(self.current_heuristic_actions[step_idx])

            # Convert to tensor, handling None values
            heuristic_actions_array = np.array(all_heuristic_actions)
            valid_mask = np.array([a is not None for a in all_heuristic_actions])

            if valid_mask.sum() > 0:
                heuristic_actions_tensor = torch.as_tensor(heuristic_actions_array).to(
                    self.device
                )
                valid_mask_tensor = torch.as_tensor(valid_mask).to(self.device)
            else:
                heuristic_actions_tensor = None

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_idx, rollout_data in enumerate(
                self.rollout_buffer.get(self.batch_size)
            ):
                actions = rollout_data.actions

                if isinstance(self.action_space, torch.Tensor):
                    actions = actions.long().flatten()

                # Get action masks if available
                action_masks = None
                if hasattr(rollout_data, "action_masks"):
                    action_masks = rollout_data.action_masks

                # Evaluate actions with current policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=action_masks,
                )
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Ratio between old and new policy
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Add Behavior Cloning loss if enabled
                if (
                    self.use_bc_loss
                    and heuristic_actions_tensor is not None
                    and self.bc_loss_coef_current > 0
                ):
                    # Compute BC loss for this batch
                    bc_loss = self._compute_bc_loss_for_batch(
                        rollout_data.observations,
                        heuristic_actions_tensor,
                        valid_mask_tensor,
                        action_masks,
                        rollout_idx,
                    )

                    if bc_loss is not None:
                        loss = loss + self.bc_loss_coef_current * bc_loss
                        bc_losses.append(bc_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs

        # Decay BC loss coefficient
        if self.use_bc_loss and self.bc_loss_decay_rate > 0:
            self.bc_loss_coef_current = max(
                self.bc_loss_min_coef,
                self.bc_loss_coef_current * (1 - self.bc_loss_decay_rate),
            )

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        if self.use_bc_loss:
            if bc_losses:
                self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_coef", self.bc_loss_coef_current)

        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

    def _compute_bc_loss_for_batch(
        self,
        observations: torch.Tensor,
        heuristic_actions: torch.Tensor,
        valid_mask: torch.Tensor,
        action_masks: Optional[torch.Tensor],
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Compute behavior cloning loss for a batch.

        Args:
            observations: Batch of observations
            heuristic_actions: Heuristic policy's actions
            valid_mask: Mask indicating which samples have valid heuristic actions
            action_masks: Valid action masks
            batch_idx: Current batch index

        Returns:
            BC loss, or None if not computable
        """
        try:
            # Get policy distribution
            distribution = self.policy.get_distribution(
                observations, action_masks=action_masks
            )

            # Get action log probabilities
            action_log_probs = distribution.distribution.logits

            # Compute cross-entropy loss with heuristic actions
            # Only for samples with valid heuristic actions
            batch_size = observations.shape[0]

            # Get corresponding heuristic actions for this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            if end_idx <= len(heuristic_actions):
                batch_heuristic = heuristic_actions[start_idx:end_idx]
                batch_valid = valid_mask[start_idx:end_idx]

                if batch_valid.sum() > 0:
                    # Compute cross-entropy only for valid samples
                    ce_loss = F.cross_entropy(
                        action_log_probs[batch_valid],
                        batch_heuristic[batch_valid].long(),
                        reduction="mean",
                    )
                    return ce_loss

            return None

        except Exception as e:
            # Silently skip BC loss if there's an error
            warnings.warn(f"BC loss computation failed: {e}")
            return None

    def get_bc_loss_coef(self) -> float:
        """Get the current BC loss coefficient."""
        return self.bc_loss_coef_current if self.use_bc_loss else 0.0
