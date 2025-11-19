from ludo import STARTING, DESTINATION, SAFE_SQUARES


class Policy_Heuristic:
    def get_action(self, state, action_space):
        if not action_space:
            return None

        # Single action case
        if len(action_space) == 1:
            return action_space[0]

        # Evaluate all actions and pick the best one
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

        def calculate_danger(position, opp_gotis):
            if position in SAFE_SQUARES:
                return 0

            if position < 0 or position >= 56:
                return 0

            danger = 0
            opp_view_pos = convert_to_opponent_position(position)

            for opp_pos in opp_gotis:
                if opp_pos < 0 or opp_pos >= 51:
                    continue

                # Calculate distance opponent needs to reach this position
                distance = (
                    opp_view_pos - opp_pos
                    if opp_view_pos >= opp_pos
                    else opp_view_pos + 51 - opp_pos
                )

                if distance <= 6 and distance > 0:
                    danger += 0.8 / distance

            return danger

        def convert_to_opponent_position(position):
            if position < 0 or position > 50 or position == 25:
                return -2

            if position <= 24:
                return position + 26

            return position - 26

        my_gotis, opp_gotis, dice_roll, player_turn = get_detailed_state_info(state)

        dice_index, goti_index = action
        dice_value = dice_roll[dice_index]
        current_pos = my_gotis[goti_index]

        new_pos = current_pos + dice_value if current_pos != -1 else 0

        score = 0.0

        # 1. BRINGING GOTI OUT OF BASE
        if current_pos == STARTING and dice_value == 6:
            if 0 not in opp_gotis:
                score += 1.0  # Safe to bring out
            else:
                score += 0.5  # Still valuable but risky

        # 2. CAPTURING OPPONENT
        if new_pos != DESTINATION and new_pos >= 0:
            # Convert to opponent's perspective
            opp_view_pos = convert_to_opponent_position(new_pos)

            if opp_view_pos in opp_gotis and opp_view_pos not in SAFE_SQUARES:
                score += 2.0  # Capturing is very valuable

        # 3. REACHING HOME
        if new_pos == DESTINATION:
            score += 3.0

        # 4. PROGRESS TOWARDS HOME
        if current_pos >= 0 and new_pos < DESTINATION:
            # Reward progress, especially for gotis closer to home
            progress_bonus = dice_value * (1 + current_pos / 56)
            score += progress_bonus * 0.1

        # 5. MOVING TO SAFETY
        if new_pos in SAFE_SQUARES and current_pos not in SAFE_SQUARES:
            score += 0.4

        # 6. AVOIDING DANGER
        danger_score = calculate_danger(new_pos, opp_gotis)
        score -= danger_score

        # 7. WASTE OF SIX
        if dice_value == 6 and current_pos > 40 and new_pos < DESTINATION:
            score -= 0.3  # Don't waste 6 on advanced gotis unless necessary

        # 8. BLOCKING STRATEGY
        # Prefer positions that block opponent progress
        if new_pos in SAFE_SQUARES:
            for opp_pos in opp_gotis:
                if opp_pos >= 0 and opp_pos < new_pos and new_pos - opp_pos < 10:
                    score += 0.15

        # 9. SPREAD STRATEGY
        # Prefer spreading gotis rather than clustering
        unique_positions = len(set([pos for pos in my_gotis if pos >= 0 and pos < 56]))
        if current_pos == -1:
            score += unique_positions * 0.05

        # 10. ENDGAME STRATEGY
        gotis_at_home = sum(1 for pos in my_gotis if pos == 56)
        if gotis_at_home >= 2:
            # In endgame, prioritize gotis closer to home
            if current_pos > 30:
                score += 0.5

        return score
