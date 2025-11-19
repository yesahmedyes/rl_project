import pygame
import time
import imageio
import ludo_positions
import os
import pickle

input_lists = []

with open("input_lists.pkl", "rb") as f:
    input_lists = pickle.load(f)


GRID_SIZE = 17
TILE_SIZE = 47
WINDOW_SIZE = GRID_SIZE * TILE_SIZE
FPS = 10

dictionary_images = {
    "neutral_domain": "neutral_domain.png",
    "player_2_home": "player_2_home.png",
    "grey_1_home": "grey_home.png",
    "grey_2_home": "grey_home.png",
    "player_1_home": "player_1_home.png",
    "neutral_home": "neutral_home.png",
    "player_2_border": "player_2_border.png",
    "player_1_border": "player_1_border.png",
    "player_1_domain": "player_1_domain.png",
    "player_2_domain": "player_2_domain.png",
    "grey_domain": "grey_domain.png",
    "player_2_gotti_0": "player_2_gotti.png",
    "player_2_gotti_1": "player_2_gotti.png",
    "player_2_gotti_2": "player_2_gotti.png",
    "player_2_gotti_3": "player_2_gotti.png",
    "player_1_gotti_0": "player_1_gotti.png",
    "player_1_gotti_1": "player_1_gotti.png",
    "player_1_gotti_2": "player_1_gotti.png",
    "player_1_gotti_3": "player_1_gotti.png",
    "dice_1": "4_dice_1.png",
    "dice_2": "4_dice_2.png",
    "dice_3": "4_dice_3.png",
    "dice_4": "4_dice_4.png",
    "dice_5": "4_dice_5.png",
    "dice_6": "4_dice_6.png",
}


def start_game():
    for ls in input_lists:
        current_move = ls
        draw_gotis(current_move)
        time.sleep(0.1)


pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("15x15 LUDO DISPLAY")
clock = pygame.time.Clock()


def load_images():
    image_dictionary = dict()
    for names, images in dictionary_images.items():
        image_dictionary[names] = pygame.image.load(
            os.path.join("ludo_images", images)
        ).convert()

    return image_dictionary


image_dictionary = load_images()


def sized_pixel(lst):
    return [(x * TILE_SIZE, y * TILE_SIZE) for x, y in lst]


def extract_dice_numbers(input_list):
    dice_numbers = input_list[2]
    return dice_numbers


def return_relevant_dice_img(num):
    if num == 1:
        dice_img = "dice_1"
    elif num == 2:
        dice_img = "dice_2"
    elif num == 3:
        dice_img = "dice_3"
    elif num == 4:
        dice_img = "dice_4"
    elif num == 5:
        dice_img = "dice_5"
    elif num == 6:
        dice_img = "dice_6"
    return dice_img


def sliced_img_list(dice_img_list):
    if len(dice_img_list) > 7:
        dice_img_list = dice_img_list[:8]
        return dice_img_list
    else:
        return dice_img_list


def sliced_position_tuple(dice_img_list, tile_positions):
    len_of_dice_img_list = len(dice_img_list)
    sliced_tile_positions = tile_positions[:len_of_dice_img_list]
    return sliced_tile_positions


def return_img_to_tile_pos(dice_img_list):
    tile_positions_list = [
        (329, 376),
        (376, 376),
        (423, 376),
        (470, 376),
        (517, 376),
        (564, 376),
        (611, 376),
    ]
    tile_positions_list = sliced_position_tuple(dice_img_list, tile_positions_list)

    img_to_position_tuple = list(zip(dice_img_list, tile_positions_list))
    return img_to_position_tuple


def blit_list(img_to_tile_pos):
    for img, positions in img_to_tile_pos:
        screen.blit(image_dictionary[img], positions)


def draw_dice(input_list):
    dice_numbers_list = extract_dice_numbers(input_list)
    dice_img_list = [return_relevant_dice_img(num) for num in dice_numbers_list]
    dice_img_list = sliced_img_list(dice_img_list)
    img_to_tile_pos_dict = return_img_to_tile_pos(dice_img_list)
    blit_list(img_to_tile_pos_dict)


def extract_gotti_positions(input_list):
    goti_input_positions = input_list[:2]
    return goti_input_positions


def return_single_cord_list(input_list):
    double_gotti_coord = extract_gotti_positions(input_list)
    blue_coord, purple_coord = double_gotti_coord
    gotti_coord = blue_coord + purple_coord
    return gotti_coord


def get_img_to_pos_list(gotti_position):
    gotti_img_to_pos_list = [
        (gotti_position[0], "player_2_gotti_0"),
        (gotti_position[1], "player_2_gotti_1"),
        (gotti_position[2], "player_2_gotti_2"),
        (gotti_position[3], "player_2_gotti_3"),
        (gotti_position[4], "player_1_gotti_0"),
        (gotti_position[5], "player_1_gotti_1"),
        (gotti_position[6], "player_1_gotti_2"),
        (gotti_position[7], "player_1_gotti_3"),
    ]
    return gotti_img_to_pos_list


def blit_gotti_list(gotti_img_to_pos_list):
    for positions, image in gotti_img_to_pos_list:
        screen.blit(image_dictionary[image], positions)


def draw_gotis(input_list):
    gotti_coord = return_single_cord_list(input_list)
    gotti_position = sized_pixel(gotti_coord)
    gotti_img_to_pos_list = get_img_to_pos_list(gotti_position)
    blit_gotti_list(gotti_img_to_pos_list)


def blit_grid(dict_x):
    for positions, images in dict_x.items():
        screen.blit(image_dictionary[images], positions)


def draw_grid(border_color):
    if border_color == True:
        blit_grid(ludo_positions.grid_yellow_border)
    else:
        blit_grid(ludo_positions.grid_red_border)


def extract_color(input_list):
    border_color = input_list[3]
    return border_color


def main_grid(input_list):
    border_color = extract_color(input_list)
    draw_grid(border_color)


def main():
    frames = []
    running = True
    i = 0
    while running:
        for input_list in input_lists:
            main_grid(input_list)
            draw_dice(input_list)
            draw_gotis(input_list)
            pygame.display.flip()

            frame = pygame.surfarray.array3d(screen)
            frame = frame.swapaxes(0, 1)
            frames.append(frame)

            clock.tick(FPS)
            time.sleep(1)

        running = False
        pygame.quit()

    return frames


def build_video(name, frames):
    return imageio.mimsave(name + ".mp4", frames, fps=2)
