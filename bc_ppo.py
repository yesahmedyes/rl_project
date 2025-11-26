import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv


class BCMaskablePPO(MaskablePPO):
    def __init__(
        self,
        *args,
        use_bc_loss: bool = False,
        bc_loss_coef: float = 1.0,
        bc_loss_decay_rate: float = 1e-3,
        bc_loss_min_coef: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_bc_loss = use_bc_loss
        self.bc_loss_coef_initial = bc_loss_coef
        self.bc_loss_coef_current = bc_loss_coef
        self.bc_loss_decay_rate = bc_loss_decay_rate
        self.bc_loss_min_coef = bc_loss_min_coef

        # Storage for BC data collected during rollout collection
        self.bc_observations = []
        self.bc_actions = []

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
        use_masking: bool = True,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        # Clear BC storage
        self.bc_observations = []
        self.bc_actions = []

        if use_masking:
            action_masks_data = env.env_method("action_masks")
            self._last_action_masks = np.array(action_masks_data)
        else:
            self._last_action_masks = None

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # Get actions and values from policy
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=self._last_action_masks
                )
                actions_cpu = actions.cpu().numpy()

            # Compute heuristic actions and store BC data if BC loss is enabled
            if self.use_bc_loss and self.bc_loss_coef_current > 0:
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

                    # Store BC sample if we have a valid heuristic action
                    if heuristic_action is not None:
                        self.bc_observations.append(
                            np.array(self._last_obs[env_idx], copy=True)
                        )
                        self.bc_actions.append(int(heuristic_action))

            # Perform action in the environment
            clipped_actions = actions_cpu

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Get action masks for next step (from new observations)
            if use_masking:
                action_masks = np.array(
                    [
                        info.get(
                            "action_mask", np.ones(self.action_space.n, dtype=np.int8)
                        )
                        for info in infos
                    ]
                )
            else:
                action_masks = None

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())

            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

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

            # Add to buffer (use actions_cpu which is numpy, not tensor)
            if use_masking and self._last_action_masks is not None:
                rollout_buffer.add(
                    self._last_obs,
                    actions_cpu,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    action_masks=self._last_action_masks,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,
                    actions_cpu,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                )

            self._last_obs = new_obs
            self._last_episode_starts = dones

            if use_masking:
                self._last_action_masks = action_masks

        # Compute returns and advantages
        with torch.no_grad():
            values = self.policy.predict_values(
                torch.as_tensor(new_obs).to(self.device)
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
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

        # Prepare BC data tensors if BC is enabled
        bc_obs_tensor = None
        bc_actions_tensor = None
        bc_indices = None
        bc_start_idx = 0
        bc_batch_size = 0
        bc_num_samples = 0

        if (
            self.use_bc_loss
            and self.bc_loss_coef_current > 0
            and hasattr(self, "bc_observations")
            and len(self.bc_observations) > 0
        ):
            bc_obs_tensor = torch.as_tensor(
                np.array(self.bc_observations), device=self.device
            )
            bc_actions_tensor = torch.as_tensor(
                np.array(self.bc_actions), device=self.device, dtype=torch.long
            )
            bc_num_samples = bc_actions_tensor.shape[0]
            bc_indices = np.random.permutation(bc_num_samples)
            bc_batch_size = min(self.batch_size, bc_num_samples)

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_idx, rollout_data in enumerate(
                self.rollout_buffer.get(self.batch_size)
            ):
                actions = rollout_data.actions
                # Ensure correct action shape/dtype
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                else:
                    actions = actions.reshape((-1,) + self.action_space.shape)

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
                    and bc_obs_tensor is not None
                    and bc_actions_tensor is not None
                    and self.bc_loss_coef_current > 0
                    and bc_num_samples > 0
                ):
                    # Sample a BC mini-batch
                    if bc_start_idx + bc_batch_size > bc_num_samples:
                        bc_indices = np.random.permutation(bc_num_samples)
                        bc_start_idx = 0

                    batch_bc_indices = bc_indices[
                        bc_start_idx : bc_start_idx + bc_batch_size
                    ]
                    bc_start_idx += bc_batch_size

                    bc_obs_batch = bc_obs_tensor[batch_bc_indices]
                    bc_actions_batch = bc_actions_tensor[batch_bc_indices]

                    # Get policy distribution and compute cross-entropy loss
                    bc_distribution = self.policy.get_distribution(bc_obs_batch)
                    bc_logits = bc_distribution.distribution.logits

                    bc_loss = F.cross_entropy(
                        bc_logits,
                        bc_actions_batch,
                        reduction="mean",
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

    def get_bc_loss_coef(self) -> float:
        return self.bc_loss_coef_current if self.use_bc_loss else 0.0
