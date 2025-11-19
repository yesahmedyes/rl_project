from PIL import Image

def resize_image(filename, size):
    img = Image.open(filename)
    resized_img = img.resize((47, 47))
    resized_img.save("resized.png")


resize_image("4_dice_1.", 47)