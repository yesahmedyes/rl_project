import argparse

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from ludo_maskable_env import LudoMaskableEnv


def mask_fn(env: LudoMaskableEnv):
    return env.get_action_mask()


def evaluate(
    model_path: str,
    n_games: int,
    opponent: str,
    dense_rewards: bool,
    seed: int,
    wrap: bool = False,
):
    env = ActionMasker(
        LudoMaskableEnv(
            opponents=(opponent,),
            dense_rewards=dense_rewards,
            alternate_start=True,
            seed=seed,
        ),
        mask_fn,
    )

    if wrap:
        policy = MaskableActorCriticPolicy.load(model_path, device="auto")

        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="auto",
        )

        model.policy.load_state_dict(policy.state_dict())
    else:
        model = MaskablePPO.load(model_path, env=env)

    # Set model to eval mode for evaluation
    model.policy.set_training_mode(False)

    wins = 0

    for _ in range(n_games):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(
                obs, action_masks=info.get("action_mask"), deterministic=True
            )
            obs, reward, done, _, info = env.step(action)

        if env.unwrapped.agent_won():
            wins += 1

        model.policy.set_training_mode(True)

    return wins / max(1, n_games)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MaskablePPO vs scripted opponents"
    )
    parser.add_argument("--model-path", type=str, default="models/maskable_ppo.zip")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument(
        "--dense", action="store_true", help="use dense rewards (default: sparse)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="wrap policy file in MaskablePPO model (for BC policy files)",
    )

    args = parser.parse_args()

    # Test against all opponents
    opponents = ["random", "heuristic", "milestone2"]

    print(f"Evaluating model: {args.model_path}")
    print(f"Games per opponent: {args.games}")
    print(f"Reward type: {'dense' if args.dense else 'sparse'}")
    print("-" * 50)

    for opponent in opponents:
        win_rate = evaluate(
            model_path=args.model_path,
            n_games=args.games,
            opponent=opponent,
            dense_rewards=args.dense,
            seed=args.seed,
            wrap=args.wrap,
        )
        print(f"Win rate vs {opponent}: {win_rate:.2%}")

    print("-" * 50)


if __name__ == "__main__":
    main()
