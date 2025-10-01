from base64 import b64encode
from glob import glob
from importlib import import_module
from io import BytesIO
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageFile
from typing import Any
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset_config(dataset_name: str) -> DictConfig:
    dataset_configs = []
    for file in glob(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "dataset", "*.yaml"
        )
    ):
        dataset_config = OmegaConf.load(file)
        if dataset_config.dataset.name == dataset_name:
            dataset_configs.append(dataset_config)
    if len(dataset_configs) == 0:
        raise ValueError(f"There is no dataset named {dataset_name}")
    if len(dataset_configs) > 1:
        raise ValueError(f"There are multiple datasets named {dataset_name}")
    return dataset_configs[0]


def get_model_config(model_name: str) -> DictConfig:
    model_configs = []
    for file in glob(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "model", "*.yaml"
        )
    ):
        model_config = OmegaConf.load(file)
        if model_config.model.name == model_name:
            model_configs.append(model_config)
    if len(model_configs) == 0:
        raise ValueError(f"There is no model named {model_name}")
    if len(model_configs) > 1:
        raise ValueError(f"There are multiple models named {model_name}")
    return model_configs[0]


def get_pipeline_config(pipeline_name: str) -> DictConfig:
    pipeline_configs = []
    for file in glob(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "pipeline", "*.yaml"
        )
    ):
        pipeline_config = OmegaConf.load(file)
        if pipeline_config.pipeline.name == pipeline_name:
            pipeline_configs.append(pipeline_config)
    if len(pipeline_configs) == 0:
        raise ValueError(f"There is no pipeline named {pipeline_name}")
    if len(pipeline_configs) > 1:
        raise ValueError(f"There are multiple pipelines named {pipeline_name}")
    return pipeline_configs[0]


def get_vl_object(class_path: str, **kwargs) -> Any:
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)(**kwargs)


def get_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    elif isinstance(image, bytes):
        pil_image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        pil_image = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_image.convert("RGB")


def get_image_url(image: Any) -> str:
    pil_image = get_pil_image(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{image_base64}"
