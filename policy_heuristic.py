from ludo import STARTING, DESTINATION, SAFE_SQUARES


class Policy_Heuristic:
    def get_action(self, state, action_space):
        if not action_space:
            return None

        # Single action case
        if len(action_space) == 1:
            return action_space[0]

        best_action = None
        best_score = float("-inf")

        for action in action_space:
            score = self.evaluate_heuristic(state, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def evaluate_heuristic(self, state, action):
        def get_detailed_state_info(state):
            gotis_red, gotis_yellow, dice_roll, _, player_turn = state

            if player_turn == 0:
                my_gotis = [g.position for g in gotis_red.gotis]
                opp_gotis = [g.position for g in gotis_yellow.gotis]
            else:
                my_gotis = [g.position for g in gotis_yellow.gotis]
                opp_gotis = [g.position for g in gotis_red.gotis]

            return my_gotis, opp_gotis, dice_roll, player_turn

        def convert_to_opponent_position(position):
            """
            Same logic as Goti.convert_into_opponent_position in the env.
            Convert *our* board position into how the opponent sees it.
            """
            if position <= STARTING or position > 50 or position == 25:
                return -2  # cannot be converted

            if position <= 24:
                return position + 26

            return position - 26

        def calculate_danger(position, opp_gotis):
            """
            Estimate how dangerous a given position is based on opponent pieces.
            Higher value = more likely to be killed within 1–6 steps.
            """
            # Off-board or home or safe squares are not "dangerous"
            if position in SAFE_SQUARES:
                return 0.0

            if position < 0 or position >= DESTINATION:
                return 0.0

            opp_view_pos = convert_to_opponent_position(position)
            if opp_view_pos == -2:
                return 0.0

            danger = 0.0

            for opp_pos in opp_gotis:
                # Ignore opponents that are in base or already in home stretch
                if opp_pos < 0 or opp_pos >= 51:
                    continue

                # Distance along opponent’s loop (0..50)
                if opp_view_pos >= opp_pos:
                    distance = opp_view_pos - opp_pos
                else:
                    distance = opp_view_pos + 51 - opp_pos

                if 0 < distance <= 6:
                    danger += 0.8 / distance  # closer opponent = more danger

            return danger

        my_gotis, opp_gotis, dice_roll, player_turn = get_detailed_state_info(state)

        dice_index, goti_index = action
        dice_value = dice_roll[dice_index]
        current_pos = my_gotis[goti_index]

        # Compute new position according to game rules approximation
        if current_pos == STARTING:
            # Action space guarantees dice_value == 6 if this action is valid
            new_pos = 0
        else:
            new_pos = current_pos + dice_value

        score = 0.0

        # --- 0. DANGER BEFORE / AFTER (used later for escape bonus) ---
        danger_before = calculate_danger(current_pos, opp_gotis)
        danger_after = calculate_danger(new_pos, opp_gotis)

        # --- 1. BRINGING GOTI OUT OF BASE ---
        if current_pos == STARTING and dice_value == 6:
            # Bringing a new piece into play is valuable
            if 0 not in opp_gotis:
                score += 1.0  # safe(ish) to bring out
            else:
                score += 0.5  # still valuable but opponent is sitting on your spawn

        # --- 2. CAPTURING OPPONENT ---
        if new_pos != DESTINATION and new_pos >= 0:
            opp_view_pos = convert_to_opponent_position(new_pos)
            # In the env, killing happens when opponent.position == opp_view_pos
            if (
                opp_view_pos != -2
                and opp_view_pos in opp_gotis
                and opp_view_pos not in SAFE_SQUARES
            ):
                score += 2.0  # capturing is very valuable

        # --- 3. REACHING HOME ---
        if new_pos == DESTINATION:
            score += 3.0  # finishing a piece is highest value

        # --- 4. PROGRESS TOWARDS HOME ---
        if current_pos >= 0 and new_pos <= DESTINATION:
            # Reward actual distance progressed, scaled by how advanced the piece already is.
            progress = new_pos - current_pos
            # Piece further along gets slightly more value for progress
            progress_factor = 0.5 + (current_pos / DESTINATION) * 0.5  # 0.5..1.0
            score += progress * progress_factor * 0.05

        # --- 5. MOVING TO SAFETY / LEAVING SAFETY ---
        # Entering a safe square
        if new_pos in SAFE_SQUARES and current_pos not in SAFE_SQUARES:
            score += 0.4

        # Leaving a safe square into a non-safe one (unless we are starting)
        if (
            current_pos in SAFE_SQUARES
            and new_pos not in SAFE_SQUARES
            and current_pos != STARTING
        ):
            score -= 0.35

        # --- 6. DANGER ADJUSTMENT (AVOID / ESCAPE DANGER) ---
        # Penalize final danger at new position
        score -= danger_after

        # Bonus for escaping danger (danger_after < danger_before)
        if danger_after < danger_before:
            score += (danger_before - danger_after) * 0.5

        # --- 7. "WASTE OF SIX" LOGIC ---
        # If there are gotis in base, we prefer not to spend a 6 on a very advanced piece
        any_in_base = any(pos == STARTING for pos in my_gotis)
        if (
            dice_value == 6
            and current_pos > 40
            and new_pos < DESTINATION
            and any_in_base
        ):
            score -= 0.3

        # --- 8. SPREAD STRATEGY (DISCOURAGE CLUSTERING) ---
        future_positions = my_gotis.copy()
        future_positions[goti_index] = new_pos

        active_positions = [p for p in future_positions if 0 <= p < DESTINATION]
        duplicates = len(active_positions) - len(set(active_positions))
        if duplicates > 0:
            score -= duplicates * 0.1

        # --- 9. ENDGAME STRATEGY ---
        gotis_at_home = sum(1 for pos in my_gotis if pos == DESTINATION)
        if gotis_at_home >= 2 and 0 <= current_pos < DESTINATION:
            # In endgame, prioritize pieces closer to home more strongly
            score += (current_pos / DESTINATION) * 0.5

        # --- 10. MULTI-DICE MICRO-STRATEGY ---
        # Slightly prefer using smaller dice for low-impact moves when all else is equal.
        # This is a small adjustment so it doesn't override big strategic decisions.
        score -= dice_value * 0.02

        return score
