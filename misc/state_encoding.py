import numpy as np
from typing import Tuple
from ludo import DESTINATION, SAFE_SQUARES, STARTING


def encode_handcrafted_state(state) -> Tuple[int, int, np.ndarray]:
    STATE_DIM = 70
    MAX_ACTIONS = 12

    gotis_red, gotis_yellow, dice_roll, _, player_turn = state

    if player_turn == 0:
        my_gotis = gotis_red.gotis
        opp_gotis = gotis_yellow.gotis
    else:
        my_gotis = gotis_yellow.gotis
        opp_gotis = gotis_red.gotis

    def normalize_position(pos: int) -> float:
        # STARTING (-1) -> 0, DESTINATION (56) -> 1
        return (pos + 1) / 57.0

    def convert_to_my_frame(opp_goti):
        return opp_goti.convert_into_opponent_position()

    def goti_features(my_goti, opponent_gotis) -> list[float]:
        my_pos = my_goti.position

        norm_pos = normalize_position(my_pos)

        at_home = float(my_pos == STARTING)
        in_play = float(0 <= my_pos < DESTINATION)
        reached_goal = float(my_pos == DESTINATION)

        danger_short = 0.0
        danger_long = 0.0
        opp_short = 0.0
        opp_long = 0.0

        for opp in opponent_gotis:
            if not (0 <= opp.position < DESTINATION):
                continue

            opp_in_my_frame = opp.convert_into_opponent_position()

            if opp_in_my_frame < 0:
                continue

            delta = my_pos - opp_in_my_frame

            if 1 <= delta <= 6:
                normalized = 1 - (delta / 6.0)
                danger_short = max(danger_short, normalized)
            elif 7 <= delta <= 12:
                normalized = 1 - ((delta - 6) / 6.0)
                danger_long = max(danger_long, normalized)
            elif -6 <= delta <= -1 and opp.position not in SAFE_SQUARES:
                normalized = 1 - (abs(delta) / 6.0)
                opp_short = max(opp_short, normalized)
            elif -12 <= delta <= -7 and opp.position not in SAFE_SQUARES:
                normalized = 1 - ((abs(delta) - 6) / 6.0)
                opp_long = max(opp_long, normalized)

        return [
            norm_pos,
            at_home,
            in_play,
            reached_goal,
            danger_short,
            danger_long,
            opp_short,
            opp_long,
        ]

    my_features = []
    opp_features = []

    for goti in my_gotis:
        my_features.extend(goti_features(goti, opp_gotis))

    for goti in opp_gotis:
        opp_features.extend(goti_features(goti, my_gotis))

    dice_normalized = [0.0, 0.0, 0.0]
    dice_is_six = [0.0, 0.0, 0.0]

    if dice_roll:
        for i, d in enumerate(dice_roll[:3]):
            dice_normalized[i] = d / 6.0
            dice_is_six[i] = 1.0 if d == 6 else 0.0

    state_vector = my_features + opp_features + dice_normalized + dice_is_six

    return STATE_DIM, MAX_ACTIONS, np.array(state_vector, dtype=np.float32)


def encode_onehot_state(state) -> Tuple[int, int, np.ndarray]:
    STATE_DIM = 946
    MAX_ACTIONS = 12

    gotis_red, gotis_yellow, dice_roll, _, player_turn = state

    if player_turn == 0:
        my_gotis = gotis_red.gotis
        opp_gotis = gotis_yellow.gotis
    else:
        my_gotis = gotis_yellow.gotis
        opp_gotis = gotis_red.gotis

    # Position encoding: -1 (STARTING) to 56 (DESTINATION) = 58 positions
    # Position -1 maps to index 0, position 0 maps to index 1, ..., position 56 maps to index 57
    # Position -2 (can't convert) maps to all zeros

    def position_to_index(pos: int) -> int:
        if pos == -2:
            return -1
        return pos + 1

    def onehot_position(pos: int, num_positions: int = 58) -> np.ndarray:
        encoded = np.zeros(num_positions, dtype=np.float32)
        idx = position_to_index(pos)

        if idx >= 0:
            encoded[idx] = 1.0

        return encoded

    my_gotis_encoded = []

    for goti in my_gotis:
        my_gotis_encoded.extend(onehot_position(goti.position))

    opp_gotis_encoded = []

    for goti in opp_gotis:
        opp_gotis_encoded.extend(onehot_position(goti.position))

    my_gotis_opp_view_encoded = []

    for goti in my_gotis:
        opp_view_pos = goti.convert_into_opponent_position()
        my_gotis_opp_view_encoded.extend(onehot_position(opp_view_pos))

    opp_gotis_my_view_encoded = []

    for goti in opp_gotis:
        my_view_pos = goti.convert_into_opponent_position()
        opp_gotis_my_view_encoded.extend(onehot_position(my_view_pos))

    dice_encoded = []

    for i in range(3):
        if i < len(dice_roll) and dice_roll[i] is not None:
            dice_onehot = np.zeros(6, dtype=np.float32)
            dice_onehot[dice_roll[i] - 1] = 1.0
            dice_encoded.extend(dice_onehot)
        else:
            dice_encoded.extend(np.zeros(6, dtype=np.float32))

    state_vector = (
        my_gotis_encoded
        + opp_gotis_encoded
        + my_gotis_opp_view_encoded
        + opp_gotis_my_view_encoded
        + dice_encoded
    )

    return STATE_DIM, MAX_ACTIONS, np.array(state_vector, dtype=np.float32)
