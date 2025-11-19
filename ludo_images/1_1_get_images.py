#### domain 
#### home 
#### gotti 
#### border_image 



from PIL import Image, ImageDraw

#### 1. HOME 

def darker(color, factor=0.6):
    """Return a darker shade of the given RGB color."""
    return tuple(max(0, min(255, int(c * factor))) for c in color)

def create_home_image(image_size=200, tile_color=(255, 253, 208), pattern_width=10):
    # Create a new image
    img = Image.new("RGB", (image_size, image_size), tile_color)
    # draw = ImageDraw.Draw(img)
    
    # # Compute darker color
    # dark_color = darker(tile_color)

    # # Draw repeating grid pattern
    # step = pattern_width * 3
    # for x in range(0, image_size, step):
    #     for y in range(0, image_size, step):
    #         # Draw a small dark square in each grid cell
    #         draw.rectangle(
    #             [x, y, x + pattern_width, y + pattern_width],
    #             fill=dark_color
    #         )

    return img






#### 2. DOMAIN 

def create_domain_image(image_size=200, tile_color=(255, 255, 204), border_width=6):
    # Create image
    img = Image.new("RGB", (image_size, image_size), tile_color)
    draw = ImageDraw.Draw(img)

    # Darker border color
    border_color = darker(tile_color)

    # Draw the border (rectangle outline)
    for i in range(border_width):
        draw.rectangle(
            [i, i, image_size - i - 1, image_size - i - 1],
            outline=border_color
        )

    return img






### 3. GOTTI 

def darker_color_gotti(color, factor=0.7):
    """Return a darker shade of the input RGB color."""
    r, g, b = color
    return (int(r * factor), int(g * factor), int(b * factor))

def create_gotti_image(
    image_size=47,
    square_size=47,
    fill_color=(140,200,255),
    border_color=(0,0,0),
    border_width=2,
    save_path=None,
    line_colors=None
):
    """
    Creates a square image with a Ludo gotti (triangle body + smaller circle head).
    """
    img = Image.new("RGB", (image_size, image_size), fill_color)
    draw = ImageDraw.Draw(img)

    # Draw border around square
    draw.rectangle([0, 0, square_size-1, square_size-1],
                   outline=border_color, width=border_width)

    gotti_color = darker_color_gotti(fill_color)
    padding = 6

    # Smaller circle head
    circle_radius = square_size // 6      # slightly smaller than before
    circle_center = (square_size//2, padding + circle_radius)
    draw.ellipse(
        [
            (circle_center[0]-circle_radius, circle_center[1]-circle_radius),
            (circle_center[0]+circle_radius, circle_center[1]+circle_radius)
        ],
        fill=gotti_color,
        outline="black",
        width=2
    )

    # Bigger triangle body
    triangle_top = circle_center[1] + circle_radius // 2
    base_y = square_size - padding
    base_left = padding
    base_right = square_size - padding
    top_point = (square_size // 2, triangle_top)
    bottom_left = (base_left, base_y)
    bottom_right = (base_right, base_y)

    draw.polygon([top_point, bottom_left, bottom_right],
                 fill=gotti_color, outline="black", width=2)

    # Optional internal lines
    if line_colors:
        n = len(line_colors)
        step = square_size // (n + 1)
        for i, color in enumerate(line_colors, start=1):
            x = step * i
            draw.line([(x, 0), (x, square_size)], fill=color, width=2)
            y = step * i
            draw.line([(0, y), (square_size, y)], fill=color, width=2)

    if save_path:
        img.save(save_path)

    return img





from PIL import Image, ImageDraw

def lighter(color, factor=3):
    """Return a lighter shade of the given RGB color."""
    return tuple(min(255, int(c * factor)) for c in color)

def create_border_image(image_size=200, tile_color=(255, 255, 204), pattern_width=10):
    img = Image.new("RGB", (image_size, image_size), tile_color)
    # draw = ImageDraw.Draw(img)

    # light_color = lighter(tile_color)

    # offset = pattern_width // 2
    # pw = pattern_width

    # # Function to draw a square with top half light color, bottom half white
    # def half_colored_square(x1, y1):
    #     x2 = x1 + pw
    #     y2 = y1 + pw
    #     mid_y = y1 + pw // 2
    #     # top half
    #     draw.rectangle([x1, y1, x2, mid_y], fill=light_color)
    #     # bottom half
    #     draw.rectangle([x1, mid_y, x2, y2], fill=(255, 255, 255))

    # # top-left
    # half_colored_square(offset, offset)
    # # top-right
    # half_colored_square(image_size - pw - offset, offset)
    # # bottom-left
    # half_colored_square(offset, image_size - pw - offset)
    # # bottom-right
    # half_colored_square(image_size - pw - offset, image_size - pw - offset)

    return img




# def lighter(color, factor=1):
#     """Return a lighter shade of the given RGB color."""
#     return tuple(min(255, int(c * factor)) for c in color)

# def create_border_image(image_size, tile_color):
#     img = Image.new("RGB", (image_size, image_size), (255, 255, 255))  # start with white
#     draw = ImageDraw.Draw(img)

#     # Draw top half with the given color
#     draw.rectangle([0, 0, image_size, image_size // 2], fill=lighter(tile_color))

#     return img






if __name__ == "__main__":

    tile_size = 47

    input_x = [((255, 255, 204), "player_2"), ((255, 190, 190), "player_1")]

    for i in input_x:

        tile = create_home_image(image_size=tile_size, tile_color=i[0], pattern_width=12)
        tile.save(f"{i[1]}_home.png")
        tile = create_domain_image(image_size=tile_size, tile_color=i[0], border_width=3)
        tile.save(f"{i[1]}_domain.png")
        gotti_img = create_gotti_image(image_size=tile_size, square_size=47, fill_color=i[0], border_color=(0,0,0), 
                                        border_width=2, save_path=f"{i[1]}_gotti.png")
        # tile = create_border_image(image_size=47, tile_color=(255, 255, 204), pattern_width=12)
        # tile.save("yellow_border.png")
        tile = create_border_image(image_size=tile_size, tile_color=i[0])
        tile.save(f"{i[1]}_border.png")

    


    input_y = [((220, 220, 220), "grey"), ((255, 255, 255), "neutral")]
    
    for i in input_y:

        tile = create_home_image(image_size=tile_size, tile_color=i[0], pattern_width=12)
        tile.save(f"{i[1]}_home.png")
        tile = create_domain_image(image_size=tile_size, tile_color=i[0], border_width=3)
        tile.save(f"{i[1]}_domain.png")










