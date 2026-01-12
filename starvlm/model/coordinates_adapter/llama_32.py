from PIL import Image
from transformers import AutoProcessor
from transformers.models.mllama.image_processing_mllama import (
    get_all_supported_aspect_ratios,
)
from starvlm.model.coordinates_adapter.base import VLCoordinatesAdapter


class LlamaCoordinatesAdapter(VLCoordinatesAdapter):
    def __init__(self, **kwargs) -> None:
        model_path = kwargs["model_path"]
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(
        self,
        x: int,
        y: int,
        original_width: int,
        original_height: int,
        to_original: bool,
    ) -> tuple[int, int]:
        feature = self.processor(
            images=[Image.new("RGB", (original_width, original_height))],
            text="",
            return_tensors="pt",
        )
        _, _, _, _, tile_height, tile_width = feature["pixel_values"].shape
        num_tiles_height, num_tiles_width = get_all_supported_aspect_ratios(
            self.processor.image_processor.max_image_tiles
        )[feature["aspect_ratio_ids"].item() - 1]
        resized_width = tile_width * num_tiles_width
        resized_height = tile_height * num_tiles_height
        if to_original:
            x = int(x / resized_width * original_width)
            y = int(y / resized_height * original_height)
        else:
            x = int(x / original_width * resized_width)
            y = int(y / original_height * resized_height)
        return x, y
