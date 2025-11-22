from policy_dqn import Policy_DQN
from policy_random import Policy_Random
from policy_heuristic import Policy_Heuristic
from milestone2 import Policy_Milestone2

import numpy as np
import copy


class OpponentManager:
    def __init__(self, agent, max_snapshots=10):
        self.agent = agent
        self.max_snapshots = max_snapshots
        self.snapshots = []  # List of (episode, state_dict) tuples

        self.random_policy = Policy_Random()
        self.heuristic_policy = Policy_Heuristic()
        self.milestone2_policy = Policy_Milestone2()

        self.self_play_agent = Policy_DQN(
            training_mode=False,
            device=agent.device,
            use_prioritized_replay=agent.use_prioritized_replay,
            policy_path="dummy_path_no_load.pth",  # Dummy path that doesn't exist
        )

    def save_snapshot(self, episode):
        snapshot = copy.deepcopy(self.agent.policy_net.state_dict())

        self.snapshots.append((episode, snapshot))

        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def select_opponent(self):
        opponent_type = np.random.choice(
            ["random", "heuristic", "milestone2", "self_play"],
            p=[0.25, 0.25, 0.25, 0.25],
        )

        if opponent_type == "random":
            return self.random_policy, "Random"

        elif opponent_type == "heuristic":
            return self.heuristic_policy, "Heuristic"

        elif opponent_type == "milestone2":
            return self.milestone2_policy, "Milestone2"

        else:
            if self.snapshots:
                recent_snapshots = self.snapshots[-5:]

                snapshot_episode, snapshot_state = recent_snapshots[
                    np.random.randint(len(recent_snapshots))
                ]

                # Load snapshot into self-play agent
                self.self_play_agent.policy_net.load_state_dict(snapshot_state)
                self.self_play_agent.target_net.load_state_dict(snapshot_state)

                return self.self_play_agent, "Self-Play"
            else:
                # No snapshots yet, use random agent
                return self.random_policy, "Random"
