import random


class Policy_Random:
    def get_action(self, state, action_space):
        if action_space:
            return random.choice(action_space)
        return None
