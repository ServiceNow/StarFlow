from PIL import Image
from transformers import AutoProcessor
from starvlm.model.coordinates_adapter.base import VLCoordinatesAdapter


class QwenCoordinatesAdapter(VLCoordinatesAdapter):
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
        _, num_patches_height, num_patches_width = feature["image_grid_thw"].tolist()[0]
        resized_width = self.processor.image_processor.patch_size * num_patches_width
        resized_height = self.processor.image_processor.patch_size * num_patches_height
        if to_original:
            x = int(x / resized_width * original_width)
            y = int(y / resized_height * original_height)
        else:
            x = int(x / original_width * resized_width)
            y = int(y / original_height * resized_height)
        return x, y
