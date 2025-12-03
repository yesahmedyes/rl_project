from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic
from policies.milestone2 import Policy_Milestone2
from policies.policy_snakes import Policy_Snakes
from env.ludo import Ludo
import argparse
import os


def get_win_percentages(n, policy1, policy2):
    env = Ludo()
    wins = [0, 0]
    policies = [policy1, policy2]

    for i in range(2):
        for _ in range(n // 2):
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

    win_percentages = [(win / n) * 100 for win in wins]

    return win_percentages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a single snake policy against opponents"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the snake model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on"
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="onehot",
        choices=["onehot", "handcrafted"],
        help="Encoding type used by the model",
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

    for opponent_name, opponent_policy in opponents:
        print(f"\nTesting {model_name} vs {opponent_name}:")

        results = get_win_percentages(
            10000,
            opponent_policy,
            Policy_Snakes(
                encoding_type=args.encoding_type,
                checkpoint_path=args.model_path,
                device=args.device,
            ),
        )

        print(f"  {model_name} win rate vs {opponent_name}: {results[1]:.2f}%")

    print(f"\nCompleted testing for {model_name}")
