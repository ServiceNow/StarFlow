from PIL import Image
from transformers import AutoProcessor
from starvlm.model.coordinates_adapter.base import VLCoordinatesAdapter


class GemmaCoordinatesAdapter(VLCoordinatesAdapter):
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
            text=self.processor.tokenizer.boi_token,
            return_tensors="pt",
        )
        _, _, resized_height, resized_width = feature["pixel_values"].shape
        if to_original:
            x = int(x / resized_width * original_width)
            y = int(y / resized_height * original_height)
        else:
            x = int(x / original_width * resized_width)
            y = int(y / original_height * resized_height)
        return x, y
