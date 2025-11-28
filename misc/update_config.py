from stable_baselines3.common.schedules import ConstantSchedule


def update_model_hyperparameters(model, config):
    # Update learning rate
    if hasattr(model, "learning_rate") and config.learning_rate != model.learning_rate:
        old_lr = model.learning_rate

        model.learning_rate = config.learning_rate

        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = config.learning_rate

        print(f"  Learning rate: {old_lr} -> {config.learning_rate}")

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

    if hasattr(model, "clip_range") and hasattr(config, "clip_range"):
        old_clip_range = model.clip_range

        # Convert float to ConstantSchedule if needed
        if isinstance(config.clip_range, float):
            new_clip_range = ConstantSchedule(config.clip_range)
        else:
            new_clip_range = config.clip_range

        if old_clip_range != new_clip_range:
            model.clip_range = new_clip_range
            print(f"  clip_range: {old_clip_range} -> {new_clip_range}")

    if hasattr(model, "clip_range_vf") and hasattr(config, "clip_range_vf"):
        if config.clip_range_vf is not None:
            old_clip_range_vf = model.clip_range_vf
            # Convert float to ConstantSchedule if needed
            if isinstance(config.clip_range_vf, float):
                new_clip_range_vf = ConstantSchedule(config.clip_range_vf)
            else:
                new_clip_range_vf = config.clip_range_vf

            if old_clip_range_vf != new_clip_range_vf:
                model.clip_range_vf = new_clip_range_vf
                print(f"  clip_range_vf: {old_clip_range_vf} -> {new_clip_range_vf}")
