from io import BytesIO
from PIL import Image, ImageFile
from typing import Any

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_pil_image(image: Any):
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, bytes):
        pil_image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        pil_image = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_image.convert("RGB")
