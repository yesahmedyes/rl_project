import pickle

position_to_coordinate_mapping_red = {
    0: (7, 14),
    1: (7, 13),
    2: (7, 12),
    3: (7, 11),
    4: (7, 10),
    5: (6, 9),
    6: (5, 9),
    7: (4, 9),
    8: (3, 9),
    9: (2, 9),
    10: (1, 9),
    11: (1, 8),
    12: (1, 7),
    13: (2, 7),
    14: (3, 7),
    15: (4, 7),
    16: (5, 7),
    17: (6, 7),
    18: (7, 6),
    19: (7, 5),
    20: (7, 4),
    21: (7, 3),
    22: (7, 2),
    23: (7, 1),
    24: (8, 1),
    25: (9, 1),
    26: (9, 2),
    27: (9, 3),
    28: (9, 4),
    29: (9, 5),
    30: (9, 6),
    31: (10, 7),
    32: (11, 7),
    33: (12, 7),
    34: (13, 7),
    35: (14, 7),
    36: (15, 7),
    37: (15, 8),
    38: (15, 9),
    39: (14, 9),
    40: (13, 9),
    41: (12, 9),
    42: (11, 9),
    43: (10, 9),
    44: (9, 10),
    45: (9, 11),
    46: (9, 12),
    47: (9, 13),
    48: (8, 14),
    49: (9, 15),
    50: (8, 15),
    51: (8, 14),
    52: (8, 13),
    53: (8, 12),
    54: (8, 11),
    55: (8, 10),
    56: (8, 9),
}


position_to_coordinate_mapping_yellow = {
    0: (9, 2),
    1: (9, 3),
    2: (9, 4),
    3: (9, 5),
    4: (9, 6),
    5: (10, 7),
    6: (11, 7),
    7: (12, 7),
    8: (13, 7),
    9: (14, 7),
    10: (15, 7),
    11: (15, 8),
    12: (15, 9),
    13: (14, 9),
    14: (13, 9),
    15: (12, 9),
    16: (11, 9),
    17: (10, 9),
    18: (9, 10),
    19: (9, 11),
    20: (9, 12),
    21: (9, 13),
    22: (9, 14),
    23: (9, 15),
    24: (8, 15),
    25: (7, 15),
    26: (7, 14),
    27: (7, 13),
    28: (7, 12),
    29: (7, 11),
    30: (7, 10),
    31: (6, 9),
    32: (5, 9),
    33: (4, 9),
    34: (3, 9),
    35: (2, 9),
    36: (1, 9),
    37: (1, 8),
    38: (1, 7),
    39: (2, 7),
    40: (3, 7),
    41: (4, 7),
    42: (5, 7),
    43: (6, 7),
    44: (7, 6),
    45: (7, 5),
    46: (7, 4),
    47: (7, 3),
    48: (7, 2),
    49: (7, 1),
    50: (8, 1),
    51: (8, 2),
    52: (8, 3),
    53: (8, 4),
    54: (8, 5),
    55: (8, 6),
    56: (8, 7),
}


def build_item_in_input_list(state):
    gotis_red, gotis_yellow, roll, _, player_turn = state

    home_positions_red = [(3, 12), (4, 12), (3, 13), (4, 13)]
    home_positions_yellow = [(12, 3), (13, 3), (12, 4), (13, 4)]

    coordinates_red = []
    coordinates_yellow = []

    for i, goti in enumerate(gotis_red.gotis):
        if goti.position == -1:
            coordinates_red.append(home_positions_red[i])
        else:
            coordinates_red.append(position_to_coordinate_mapping_red[goti.position])

    for i, goti in enumerate(gotis_yellow.gotis):
        if goti.position == -1:
            coordinates_yellow.append(home_positions_yellow[i])
        else:
            coordinates_yellow.append(
                position_to_coordinate_mapping_yellow[goti.position]
            )

    return coordinates_red, coordinates_yellow, roll, player_turn


def build_input_list(states):
    input_lists = [build_item_in_input_list(state) for state in states]
    with open("input_lists.pkl", "wb") as f:
        pickle.dump(input_lists, f)
