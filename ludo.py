import random
import build_coordinates
import copy

STARTING = -1
DESTINATION = 56
SAFE_SQUARES = [0, 8, 13, 21, 25, 26, 34, 39, 47, 51, 52, 53, 54, 55, 56]


class Ludo:
    def __init__(self, render_mode="", video=False, video_name="Video"):
        self.all_gotis = [Gotis("red"), Gotis("yellow")]
        self.dice = Dice()
        self.terminated = False
        self.player_turn = 0
        self.roll = self.dice.roll()
        self.render_mode = render_mode
        self.build_video = video
        self.video_name = video_name

        if self.render_mode == "human":
            self.states = []

    def __repr__(self):
        gotis_repr = "\n\n".join([repr(g) for g in self.all_gotis])
        return (
            f"{gotis_repr}\n\n"
            f"Dice Roll: {self.roll}\n"
            f"Terminated: {self.terminated}\n"
            f"Player Turn: {self.player_turn}"
        )

    def step(self, action=None):
        if self.terminated:
            return self._get_state()

        if self._no_action_possible(action):
            return self._change_turn()

        if not self._is_valid_input(action):
            return self._handle_invalid_action()

        return self._perform_move(action)

    def reset(self):
        self.__init__(self.render_mode, self.build_video, self.video_name)
        return self._get_state()

    def get_action_space(self):
        gotis = (
            self.all_gotis[0].gotis
            if self.player_turn == 0
            else self.all_gotis[1].gotis
        )

        action_space = [
            (dice_index, goti_index)
            for dice_index, dice in enumerate(self.roll)
            for goti_index, goti in enumerate(gotis)
            if self._is_valid_move(goti, dice)
        ]

        return action_space

    def check_win(self, gotis):
        gotis = gotis.gotis
        return all(goti.position == 56 for goti in gotis)

    def _perform_move(self, action):
        dice_index, goti_number = action
        dice = self.roll[dice_index]
        self.roll.pop(dice_index)

        self.all_gotis[self.player_turn].move_goti(goti_number, dice)

        current_goti = self.all_gotis[self.player_turn].gotis[goti_number]
        current_goti_position_opponent_view = (
            current_goti.convert_into_opponent_position()
        )

        if self.all_gotis[not self.player_turn].kill_goti(
            current_goti_position_opponent_view
        ):
            return self._get_extra_turn()

        if self.check_win(self.all_gotis[self.player_turn]):
            return self._handle_win()

        if current_goti.position == DESTINATION:
            return self._get_extra_turn()

        if len(self.roll) >= 1:
            return self._get_state()

        return self._change_turn()

    def _no_action_possible(self, action):
        return action is None and not self.get_action_space()

    def _is_valid_move(self, goti, dice):
        if goti.position == STARTING:
            return dice == 6
        return goti.position + dice <= DESTINATION

    def _is_valid_input(self, action):
        return action in self.get_action_space()

    def _change_turn(self):
        self.player_turn = not self.player_turn
        self.roll = self.dice.roll()
        return self._get_state()

    def _handle_invalid_action(self):
        self.terminated = True

        if self.render_mode == "human":
            self.render()

        self.player_turn = not self.player_turn
        return self._get_state()

    def _handle_win(self):
        self.terminated = True

        if self.render_mode == "human":
            self.render()

        return self._get_state()

    def _get_extra_turn(self):
        new_roll = self.dice.roll()
        self.roll += new_roll
        return self._get_state()

    def _get_state(self):
        state = (
            self.all_gotis[0],
            self.all_gotis[1],
            self.roll,
            self.terminated,
            self.player_turn,
        )

        if self.render_mode == "human":
            self.states.append(copy.deepcopy(state))

        return state

    def render(self):
        build_coordinates.build_input_list(self.states)
        import renderer

        frames = renderer.main()

        if self.build_video:
            renderer.build_video(self.video_name, frames)


class Gotis:
    def __init__(self, color: str):
        self.color = color.capitalize()
        self.gotis = [Goti() for _ in range(4)]

    def __repr__(self):
        goti_positions = "\n".join(
            [f" Goti {i + 1}: {goti.position}" for i, goti in enumerate(self.gotis)]
        )
        return f"{self.color} Gotis' Distance from starting point:\n{goti_positions}"

    def move_goti(self, goti_number, dice):
        self.gotis[goti_number].move(dice)

    def kill_goti(self, position):
        for i in range(4):
            if self.gotis[i].position == position:
                if self.gotis[i].kill_goti():
                    return True
        return False


class Goti:
    def __init__(self, position=STARTING):
        self.position = position
        assert STARTING <= self.position <= DESTINATION

    def __repr__(self):
        return f"Goti's Distance from starting point: {self.position}"

    def move(self, dice):
        if self.position == STARTING:
            if dice == 6:
                self.position = 0
            return

        if self.position + dice <= DESTINATION:
            self.position += dice

    def convert_into_opponent_position(self):
        if STARTING >= self.position or self.position > 50 or self.position == 25:
            return -2  # Position cannot be converted

        if self.position <= 24:
            return self.position + 26

        return self.position - 26

    def kill_goti(self):
        if self.position not in SAFE_SQUARES:
            self.position = -1
            return True
        return False


class Dice:
    def roll(self):
        rolls = []

        for _ in range(3):
            roll = self.simulate_one_dice_roll()
            rolls.append(roll)

            if roll != 6:
                break
            elif len(rolls) == 3 and roll == 6:
                return []

        return rolls

    def simulate_one_dice_roll(self):
        return random.randint(1, 6)


# # Sample Use

# from policy_random import *

# env = Ludo(render_mode="human", video=True, video_name="Test")
# wins = [0, 0]
# policies = [Policy_Random(), Policy_Random()]


# state = env.reset()
# terminated = False
# player_turn = 0

# while not terminated:
#     action_space = env.get_action_space()
#     action = policies[player_turn].get_action(state, action_space)

#     state = env.step(action)
#     terminated = state[3]
