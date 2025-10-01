from PIL import Image
from typing import Any
from starflow.dataset.base import VLDataset
from starflow.utils import get_pil_image


class BigDocsDataset(VLDataset):
    def post_init(self, **kwargs):
        pass

    def get_identifier(self, example: dict[str, Any]) -> str:
        return str(example["identifier"])

    def get_images(self, example: dict[str, Any]) -> list[Image.Image]:
        return [get_pil_image(image) for image in example["images"]]

    def get_queries(self, example: dict[str, Any]) -> list[str]:
        return [str(query) for query in example["queries"]]

    def get_annotations(self, example: dict[str, Any]) -> list[str]:
        return [str(annotation) for annotation in example["annotations"]]

    def get_task(self, example: dict[str, Any]) -> str:
        return str(example["task"])

    def get_source(self, example: dict[str, Any]) -> str:
        return str(example["source"])
