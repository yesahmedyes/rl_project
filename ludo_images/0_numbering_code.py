import pygame
import time


GRID_SIZE = 15         
TILE_SIZE = 47        
WINDOW_SIZE = GRID_SIZE * TILE_SIZE
FPS = 60

ROWS = 15
COLS = 15
CELL_SIZE = 47
BG_COLOR = (240, 240, 245)     # board background (soft)
LINE_COLOR = (200, 200, 205)   # grid lines
TEXT_COLOR = (40, 40, 40)
MARGIN = 1



ball_path = [(0, 0), (0, 1)]  


pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("15x15 LUDO DISPLAY")
clock = pygame.time.Clock()




def draw_grid_1(surface, font):
    """Draws the grid and writes 'row,column' in the center of each tile."""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y = col * TILE_SIZE
            x = row * TILE_SIZE

            # Draw tile background (optional - alternating faint shades could be used)
            # pygame.draw.rect(surface, (245, 245, 250) if (row+col)%2 else BG_COLOR, (x, y, TILE_SIZE, TILE_SIZE))

            # Draw grid cell border
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(surface, LINE_COLOR, rect, MARGIN)

            # Prepare and draw the text "row,col" centered
            label = f"{row},{col}"
            text_surf = font.render(label, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=rect.center)
            surface.blit(text_surf, text_rect)

    # Draw grid
    for r in range(COLS):
        for c in range(ROWS):
            pygame.draw.rect(
                screen,
                (180, 180, 180),  # light gray box
                (c * CELL_SIZE + 40, r * CELL_SIZE + 40, CELL_SIZE, CELL_SIZE),
                1  # line width
            )


def load_images():
    dictionary_images = {
    "background_image" : "1_silver.jpg",
    "home_blue": "2_blue.jpg",
    "home_grey_1": "2_grey.jpg",
    "home_grey_2": "2_grey.jpg",
    "home_purple": "2_purple.jpg",
    "home_silver": "2_silver.jpg",
    "domain_purple": "1_purple.jpg",
    "domain_blue": "1_blue.jpg",
    "domain_grey": "1_grey.jpg",
}
    image_dictionary = dict()

    for names, images in dictionary_images.items():
        image_dictionary[names] = pygame.image.load(images).convert() 

    return image_dictionary 


def draw_row_col_labels(surface, rows, cols, cell_size, font):
    label_color = (0, 0, 0)  # black text

    # Draw row numbers on left
    for r in range(rows):
        row_label = font.render(str(r), True, label_color)
        surface.blit(row_label, (5, r * cell_size + cell_size // 4))

    # Draw column numbers on top
    for c in range(cols):
        col_label = font.render(str(c), True, label_color)
        surface.blit(col_label, (c * cell_size + cell_size // 3, 5))



def formulated_grid_positions(raw_coordinates):
    grid_positions = []
    row_start, row_end, column_start, column_end = raw_coordinates
    for row in range(row_start, row_end):
        for col in range(column_start, column_end):
            x = col * TILE_SIZE
            y = row * TILE_SIZE
            grid_positions.append((x, y))
    return grid_positions


image_dictionary = load_images() 


def draw_background():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * TILE_SIZE, row * TILE_SIZE
            screen.blit(image_dictionary["background_image"], (x, y))



def draw_homes():
    home_images_coord = {"home_blue": [9,15,0,6],
                   "home_grey_1": [0,6,0,6],
                   "home_grey_2": [9,15,9,15],
                   "home_purple": [0,6,9,15], 
                   "home_silver": [6,9,6,9],}

    home_imag_pix_pos = {home_name: formulated_grid_positions(raw_coord) for home_name, 
                           raw_coord in home_images_coord.items()}    

    for image_name, image_positions in home_imag_pix_pos.items():
        for pos in image_positions:
            screen.blit(image_dictionary[image_name], pos)




def sized_pixel(lst):
    return [(x * TILE_SIZE, y * TILE_SIZE) for x, y in lst]


def draw_inside_positions():
    purple_positions = [(7, 1), (7, 2),  (7, 3),  (7, 4),  (7, 5), (6, 2), (8, 1)]
    blue_positions =   [(7, 9), (7, 10),  (7, 11),  (7, 12),  (7, 13), (6, 13), (8, 12)]
    grey_positions =   [(9, 7), (10, 7), (11, 7), (12, 7), (13, 7),(13, 8), (12, 6), 
                        (1, 7), (2, 7),  (3, 7),  (4, 7),  (5, 7), (2, 8), (1, 6)]
    
    position_list = [(purple_positions, "domain_purple"), 
                     (blue_positions, "domain_blue"), 
                      (grey_positions, "domain_grey")]


    for color_positions, color_name in position_list:
        for positions in sized_pixel(color_positions):
            screen.blit(image_dictionary[color_name], positions)



def draw_grid():
    draw_background()
    draw_homes()
    draw_inside_positions()


def main():
    running = True
    i = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        

        draw_grid() 
        font_size = int(TILE_SIZE * 0.5)
        font = pygame.font.SysFont(None, font_size)

        
        draw_grid_1(screen, font)
        
        #first one 
        pygame.display.flip()
        clock.tick(FPS)
        #second one 

    pygame.quit()

main()