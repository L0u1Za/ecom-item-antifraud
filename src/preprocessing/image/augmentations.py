import random
from PIL import Image, ImageEnhance, ImageFilter

def random_flip(image: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def random_rotation(image: Image.Image) -> Image.Image:
    angle = random.randint(-30, 30)
    return image.rotate(angle)

def random_brightness(image: Image.Image, factor_range=(0.5, 1.5)) -> Image.Image:
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def random_blur(image: Image.Image, radius_range=(0, 2)) -> Image.Image:
    radius = random.uniform(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius))

def augment_image(image: Image.Image) -> Image.Image:
    image = random_flip(image)
    image = random_rotation(image)
    image = random_brightness(image)
    image = random_blur(image)
    return image