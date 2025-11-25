import numpy as np

from ludo import DESTINATION, SAFE_SQUARES, STARTING

STATE_DIM = 28
MAX_ACTIONS = 12


def encode_ludo_state(state) -> np.ndarray:
    gotis_red, gotis_yellow, dice_roll, _, player_turn = state

    if player_turn == 0:
        my_gotis = [g.position for g in gotis_red.gotis]
        opp_gotis = [g.position for g in gotis_yellow.gotis]
    else:
        my_gotis = [g.position for g in gotis_yellow.gotis]
        opp_gotis = [g.position for g in gotis_red.gotis]

    def normalize_position(pos: int) -> float:
        # STARTING (-1) -> 0, DESTINATION (56) -> 1
        return (pos + 1) / 57.0

    my_positions = [normalize_position(pos) for pos in my_gotis]
    opp_positions = [normalize_position(pos) for pos in opp_gotis]

    dice_normalized = [0.0, 0.0, 0.0]
    if dice_roll:
        for i, d in enumerate(dice_roll[:3]):
            dice_normalized[i] = d / 6.0

    my_distances = [(DESTINATION - pos) / 57.0 if pos >= 0 else 1.0 for pos in my_gotis]

    my_at_home = sum(1 for pos in my_gotis if pos == STARTING) / 4.0
    opp_at_home = sum(1 for pos in opp_gotis if pos == STARTING) / 4.0

    my_at_dest = sum(1 for pos in my_gotis if pos == DESTINATION) / 4.0
    opp_at_dest = sum(1 for pos in opp_gotis if pos == DESTINATION) / 4.0

    my_on_safe = sum(1 for pos in my_gotis if pos in SAFE_SQUARES) / 4.0
    opp_on_safe = sum(1 for pos in opp_gotis if pos in SAFE_SQUARES) / 4.0

    def calc_progress(positions):
        active = [p for p in positions if p >= 0]
        if not active:
            return 0.0
        return sum(active) / (len(active) * DESTINATION)

    my_avg_progress = calc_progress(my_gotis)
    opp_avg_progress = calc_progress(opp_gotis)

    pieces_in_danger = 0
    for my_pos in my_gotis:
        if my_pos >= 0 and my_pos not in SAFE_SQUARES:
            for opp_pos in opp_gotis:
                if opp_pos >= 0 and 1 <= (my_pos - opp_pos) <= 6:
                    pieces_in_danger += 1
                    break
    pieces_in_danger_norm = pieces_in_danger / 4.0

    capture_opportunities = 0
    for my_pos in my_gotis:
        if my_pos >= 0:
            for opp_pos in opp_gotis:
                if opp_pos >= 0 and opp_pos not in SAFE_SQUARES:
                    if 1 <= (opp_pos - my_pos) <= 6:
                        capture_opportunities += 1
                        break
    capture_opportunities_norm = capture_opportunities / 4.0

    has_six = 1.0 if (dice_roll and 6 in dice_roll) else 0.0
    num_dice = len(dice_roll) / 3.0 if dice_roll else 0.0

    state_vector = (
        my_positions
        + opp_positions
        + dice_normalized
        + [float(player_turn)]
        + my_distances
        + [my_at_home, opp_at_home]
        + [my_at_dest, opp_at_dest]
        + [my_on_safe, opp_on_safe]
        + [my_avg_progress, opp_avg_progress]
        + [pieces_in_danger_norm]
        + [capture_opportunities_norm]
        + [has_six]
        + [num_dice]
    )

    return np.array(state_vector, dtype=np.float32)


def calculate_dense_reward(
    state_before,
    state_after,
    terminated: bool,
    agent_won: bool,
    agent_player: int,
) -> float:
    if terminated:
        return 1.0 if agent_won else -1.0

    reward = -0.0001

    gotis_red_before, gotis_yellow_before, _, _, _ = state_before
    gotis_red_after, gotis_yellow_after, _, _, _ = state_after

    if agent_player == 0:
        my_gotis_before = gotis_red_before.gotis
        my_gotis_after = gotis_red_after.gotis
        opp_gotis_before = gotis_yellow_before.gotis
        opp_gotis_after = gotis_yellow_after.gotis
    else:
        my_gotis_before = gotis_yellow_before.gotis
        my_gotis_after = gotis_yellow_after.gotis
        opp_gotis_before = gotis_red_before.gotis
        opp_gotis_after = gotis_red_after.gotis

    total_progress_before = sum(max(0, g.position) for g in my_gotis_before)
    total_progress_after = sum(max(0, g.position) for g in my_gotis_after)
    progress_delta = total_progress_after - total_progress_before
    if progress_delta > 0:
        reward += progress_delta * 0.001

    pieces_home_before = sum(1 for g in my_gotis_before if g.position == DESTINATION)
    pieces_home_after = sum(1 for g in my_gotis_after if g.position == DESTINATION)
    if pieces_home_after > pieces_home_before:
        reward += 0.05

    opp_at_start_before = sum(1 for g in opp_gotis_before if g.position == STARTING)
    opp_at_start_after = sum(1 for g in opp_gotis_after if g.position == STARTING)
    if opp_at_start_after > opp_at_start_before:
        reward += 0.03 * (opp_at_start_after - opp_at_start_before)

    my_at_start_before = sum(1 for g in my_gotis_before if g.position == STARTING)
    my_at_start_after = sum(1 for g in my_gotis_after if g.position == STARTING)
    if my_at_start_after > my_at_start_before:
        reward -= 0.03 * (my_at_start_after - my_at_start_before)

    my_in_play_before = sum(1 for g in my_gotis_before if 0 <= g.position < DESTINATION)
    my_in_play_after = sum(1 for g in my_gotis_after if 0 <= g.position < DESTINATION)
    if my_in_play_after > my_in_play_before:
        reward += 0.01 * (my_in_play_after - my_in_play_before)

    return reward
