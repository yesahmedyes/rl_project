from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic
from policies.milestone2 import Policy_Milestone2
from policies.policy_snakes import Policy_Snakes
from env.ludo import Ludo
from tqdm import tqdm


def get_win_percentages(n, policy1, policy2):
    env = Ludo()
    wins = [0, 0]
    policies = [policy1, policy2]

    for i in range(2):
        for _ in tqdm(range(n // 2)):
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


print(
    get_win_percentages(
        1000,
        # Policy_Heuristic(),
        Policy_Milestone2(),
        Policy_Snakes(checkpoint_path="./models/snakes/latest_handcrafted.zip"),
    )
)
