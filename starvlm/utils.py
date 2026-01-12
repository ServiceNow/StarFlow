from base64 import b64encode
from importlib import import_module
from io import BytesIO
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import numpy as np
import requests


def divide_path(
    path: str, separator: str, divide_at: int = 1, from_left: bool = False
) -> tuple[str, str]:
    if not path:
        raise ValueError("path cannot be empty")
    if not separator:
        raise ValueError("separator cannot be empty")
    if divide_at <= 0:
        raise ValueError(f"divide_at must be positive, got {divide_at}")
    parts = path.split(separator)
    if len(parts) <= divide_at:
        raise ValueError(
            f"path '{path}' must contain at least {divide_at} instance(s) of separator '{separator}', got {len(parts) - 1}"
        )
    if not all(parts):
        raise ValueError(
            f"path '{path}' cannot be split by separator '{separator}' without producing empty part(s)"
        )
    if from_left:
        head = separator.join(parts[:divide_at])
        tail = separator.join(parts[divide_at:])
    else:
        head = separator.join(parts[:-divide_at])
        tail = separator.join(parts[-divide_at:])
    return head, tail


def get_config(config_path: str) -> dict[str, Any]:
    config_path = config_path.strip()
    if not config_path:
        raise ValueError("config_path cannot be empty")
    try:
        group_path, config_name = divide_path(config_path, "/")
    except ValueError as e:
        raise ValueError(
            f"config_path '{config_path}' cannot be divided into group_path and config_name"
        ) from e
    config_file = (
        Path(__file__)
        .resolve()
        .parent.joinpath("config", group_path, f"{config_name}.yaml")
    )
    if not config_file.is_file():
        raise ValueError(f"config_file '{config_file}' is not valid file")
    try:
        config = OmegaConf.load(str(config_file))
    except Exception as e:
        raise RuntimeError(f"failed to load config from '{config_file}'") from e
    try:
        config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    except Exception as e:
        raise RuntimeError("failed to convert config to container") from e
    if not isinstance(config, dict):
        raise TypeError(f"type of config must be dict, got {type(config).__name__}")
    return config


def get_class_instance(class_path: str, **kwargs) -> Any:
    class_path = class_path.strip()
    if not class_path:
        raise ValueError("class_path cannot be empty")
    try:
        module_path, class_name = divide_path(class_path, ".")
    except ValueError as e:
        raise ValueError(
            f"class_path '{class_path}' cannot be divided into module_path and class_name"
        ) from e
    try:
        module = import_module(module_path)
    except Exception as e:
        raise RuntimeError(f"failed to import module '{module_path}'") from e
    try:
        class_handle = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"module '{module_path}' does not contain class '{class_name}'"
        ) from e
    if not isinstance(class_handle, type):
        raise TypeError(
            f"type of class '{class_name}' in module '{module_path}' must be type, got {type(class_handle).__name__}"
        )
    try:
        class_instance = class_handle(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"failed to instantiate class '{class_name}' in module '{module_path}' with kwargs '{kwargs}'"
        ) from e
    return class_instance


def get_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, np.ndarray):
        try:
            pil_image = Image.fromarray(image)
        except Exception as e:
            raise RuntimeError("failed to convert NumPy array to PIL image") from e
    elif isinstance(image, bytes):
        try:
            pil_image = Image.open(BytesIO(image))
            pil_image.load()
        except Exception as e:
            raise RuntimeError("failed to convert bytes to PIL image") from e
    elif isinstance(image, (str, Path)):
        if isinstance(image, str) and urlparse(image).scheme in ("http", "https"):
            try:
                response = requests.get(image, timeout=(3, 10))
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise ValueError(f"image '{image}' is not accessible URL") from e
            try:
                pil_image = Image.open(BytesIO(response.content))
                pil_image.load()
            except Exception as e:
                raise RuntimeError(f"failed to load image from '{image}'") from e
        else:
            if isinstance(image, str):
                image = Path(image).expanduser()
            if not image.is_file():
                raise ValueError(f"image '{image}' is not valid file")
            try:
                pil_image = Image.open(str(image))
                pil_image.load()
            except Exception as e:
                raise RuntimeError(f"failed to load image from '{image}'") from e
    else:
        raise TypeError(f"type of image cannot be {type(image).__name__}")
    pil_image = pil_image.convert("RGB")
    return pil_image


def get_image_url(image: Any) -> str:
    try:
        pil_image = get_pil_image(image)
    except Exception as e:
        raise RuntimeError("failed to get PIL image") from e
    buffer = BytesIO()
    try:
        pil_image.save(buffer, format="PNG")
    except Exception as e:
        raise RuntimeError("failed to convert PIL image to bytes") from e
    base64_image = b64encode(buffer.getvalue()).decode()
    image_url = f"data:image/png;base64,{base64_image}"
    return image_url
