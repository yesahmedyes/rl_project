def update_model_hyperparameters(model, config):
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

    # Update learning rate
    if hasattr(config, "learning_rate"):
        old_lr = model.learning_rate if hasattr(model, "learning_rate") else None
        new_lr = config.learning_rate

        lr_changed = False

        if callable(old_lr) or old_lr != new_lr:
            lr_changed = True

        if lr_changed:
            model.learning_rate = new_lr

            if callable(new_lr):
                model.lr_schedule = new_lr
            else:
                # Create a constant schedule that always returns the new learning rate
                model.lr_schedule = lambda progress_remaining: new_lr

            # Update the optimizer's learning rate directly
            if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                for param_group in model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr if not callable(new_lr) else new_lr(1.0)

            print(f"  Learning rate: {old_lr} -> {new_lr}")

    # Check if n_steps or n_envs changed (requires buffer recreation)
    n_steps_changed = (
        hasattr(model, "n_steps")
        and hasattr(config, "n_steps")
        and model.n_steps != config.n_steps
    )
    n_envs_changed = (
        hasattr(model, "env")
        and model.env is not None
        and hasattr(config, "n_envs")
        and model.env.num_envs != config.n_envs
    )

    # Update other hyperparameters
    hyperparams_to_update = [
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ]

    for param_name in hyperparams_to_update:
        if hasattr(model, param_name) and hasattr(config, param_name):
            old_value = getattr(model, param_name)
            new_value = getattr(config, param_name)

            if old_value != new_value:
                setattr(model, param_name, new_value)
                print(f"  {param_name}: {old_value} -> {new_value}")

    # Recreate rollout buffer if n_steps or n_envs changed
    # Only recreate if env is set; otherwise it will be recreated when set_env is called
    if (n_steps_changed or n_envs_changed) and model.env is not None:
        print("  Recreating rollout buffer due to changed n_steps or n_envs...")

        # Get buffer parameters from existing buffer or model
        buffer_size = model.n_steps
        n_envs = model.env.num_envs

        # Create new buffer with updated size
        model.rollout_buffer = MaskableRolloutBuffer(
            buffer_size=buffer_size,
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            gae_lambda=model.gae_lambda,
            gamma=model.gamma,
            n_envs=n_envs,
        )
        print(
            f"  New buffer size: {buffer_size} steps x {n_envs} envs = {buffer_size * n_envs} total"
        )
    elif n_steps_changed and model.env is None:
        print("  n_steps changed; buffer will be recreated when environment is set")
