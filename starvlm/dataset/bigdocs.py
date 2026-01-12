from PIL import Image
from typing import Any
from starvlm.dataset.base import VLDataset
from starvlm.utils import get_pil_image


class BigDocsDataset(VLDataset):
    def post_init(self, **kwargs) -> None:
        pass

    def get_identifier(self, example: dict[str, Any]) -> str:
        identifier = str(example["identifier"])
        return identifier

    def get_images(self, example: dict[str, Any]) -> list[Image.Image]:
        images = [get_pil_image(image) for image in example["images"]]
        return images

    def get_queries(self, example: dict[str, Any]) -> list[str]:
        queries = [str(query) for query in example["queries"]]
        return queries

    def get_annotations(self, example: dict[str, Any]) -> list[str]:
        annotations = [str(annotation) for annotation in example["annotations"]]
        return annotations
