from ludo import DESTINATION, SAFE_SQUARES, STARTING


def calculate_dense_reward(
    state_before,
    state_after,
    terminated: bool,
    agent_won: bool,
    agent_player: int,
) -> float:
    if terminated:
        return 100.0 if agent_won else -100.0

    reward = -0.05

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

    # Progress towards destination
    total_progress_before = sum(max(0, g.position) for g in my_gotis_before)
    total_progress_after = sum(max(0, g.position) for g in my_gotis_after)
    progress_delta = total_progress_after - total_progress_before
    if progress_delta > 0:
        reward += progress_delta * 0.05

    # Leaving base
    base_before = sum(1 for g in my_gotis_before if g.position == STARTING)
    base_after = sum(1 for g in my_gotis_after if g.position == STARTING)
    if base_after < base_before:
        reward += 5.0 * (base_before - base_after)

    # Entering safe squares
    safe_squares_before = sum(1 for g in my_gotis_before if g.position in SAFE_SQUARES)
    safe_squares_after = sum(1 for g in my_gotis_after if g.position in SAFE_SQUARES)
    if safe_squares_after > safe_squares_before:
        reward += 2.0 * (safe_squares_after - safe_squares_before)

    # Reaching destination
    home_before = sum(1 for g in my_gotis_before if g.position == DESTINATION)
    home_after = sum(1 for g in my_gotis_after if g.position == DESTINATION)
    if home_after > home_before:
        reward += 20.0 * (home_after - home_before)

    # Capturing opponents
    captured_opponents = sum(
        1
        for before, after in zip(opp_gotis_before, opp_gotis_after)
        if before.position != STARTING and after.position == STARTING
    )

    if captured_opponents > 0:
        reward += 15.0 * captured_opponents

    # Getting captured
    got_captured = sum(
        1
        for before, after in zip(my_gotis_before, my_gotis_after)
        if before.position != STARTING and after.position == STARTING
    )

    if got_captured > 0:
        reward -= 15.0 * got_captured

    return reward / 100
