import argparse

from sb3_contrib.common.maskable.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from ludo_maskable_env import LudoMaskableEnv


def mask_fn(env: LudoMaskableEnv):
    return env.get_action_mask()


def evaluate(model_path: str, n_games: int, opponent: str, dense_rewards: bool, seed: int):
    env = ActionMasker(
        LudoMaskableEnv(
            opponents=(opponent,),
            dense_rewards=dense_rewards,
            alternate_start=True,
            seed=seed,
        ),
        mask_fn,
    )
    model = MaskablePPO.load(model_path, env=env)

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

    return wins / max(1, n_games)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO vs scripted opponents")
    parser.add_argument("--model-path", type=str, default="models/maskable_ppo.zip")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--opponent", type=str, default="heuristic")
    parser.add_argument("--sparse", action="store_true", help="use sparse win/loss rewards")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    win_rate = evaluate(
        model_path=args.model_path,
        n_games=args.games,
        opponent=args.opponent,
        dense_rewards=not args.sparse,
        seed=args.seed,
    )
    print(f"Win rate vs {args.opponent}: {win_rate:.2%}")


if __name__ == "__main__":
    main()
