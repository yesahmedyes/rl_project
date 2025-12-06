from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic
from policies.milestone2 import Policy_Milestone2
from policies.policy_bc import Policy_BC
from env.ludo import Ludo
import argparse
import os


def get_win_percentages(num_games, policy1, policy2):
    env = Ludo()
    wins = [0, 0]
    policies = [policy1, policy2]

    for i in range(2):
        for _ in range(num_games // 2):
            state = env.reset()
            terminated = False
            player_turn = 0

            while not terminated:
                action_space = env.get_action_space()
                action = policies[player_turn].get_action(state, action_space)

                state = env.step(action)
                terminated, player_turn = state[3], state[4]

            wins[player_turn - i] += 1

        policies[0], policies[1] = policies[1], policies[0]

    win_percentages = [(win / num_games) * 100 for win in wins]

    return win_percentages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy against baseline opponents"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (BC/RLlib)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on"
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="handcrafted",
        choices=["onehot", "handcrafted"],
        help="Encoding type used by the model",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1000,
        help="Number of self-play games per matchup (split between seats)",
    )

    args = parser.parse_args()

    # Extract model name from path
    model_name = os.path.basename(args.model_path).replace(".zip", "")

    # Define opponents
    opponents = [
        ("Random", Policy_Random()),
        ("Heuristic", Policy_Heuristic()),
        ("Milestone2", Policy_Milestone2()),
    ]

    # Test the specified model against all opponents
    print(f"Testing model: {model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Encoding type: {args.encoding_type}")
    print("=" * 50)

    # Build the BC policy under test once to avoid reloading per matchup
    test_policy = Policy_BC(
        checkpoint_path=args.model_path,
        encoding_type=args.encoding_type,
        device=args.device,
    )

    for opponent_name, opponent_policy in opponents:
        print(f"\nTesting {model_name} vs {opponent_name}:")

        results = get_win_percentages(
            args.num_games,
            opponent_policy,
            test_policy,
        )

        print(f"  {model_name} win rate vs {opponent_name}: {results[1]:.2f}%")

    print(f"\nCompleted testing for {model_name}")
