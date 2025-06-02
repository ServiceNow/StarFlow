from PIL import Image
import base64
import io


def get_base64_image(image: Image.Image, format: str):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()
