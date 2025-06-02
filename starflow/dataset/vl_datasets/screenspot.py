from starflow.dataset.utils import get_pil_image
from starflow.dataset.vl_dataset import VLDataset
import hashlib


class ScreenSpotDataset(VLDataset):
    def post_init(self, **kwargs):
        use_pixel_coordinates = kwargs["use_pixel_coordinates"]
        self.use_pixel_coordinates = use_pixel_coordinates

    def get_identifier(self, example: dict):
        buffer = (example["file_name"] + example["instruction"]).encode()
        return hashlib.md5(buffer).hexdigest()

    def get_images(self, example: dict):
        return [get_pil_image(example["image"])]

    def get_queries(self, example: dict):
        return [
            f"""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {example["description"]}"""
        ]

    def get_annotations(self, example: dict):
        x1, y1, x2, y2 = example["bbox"]
        if self.use_pixel_coordinates:
            width, height = example["image"].size
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        else:
            x1 = int(x1 * 1000)
            y1 = int(y1 * 1000)
            x2 = int(x2 * 1000)
            y2 = int(y2 * 1000)
        return [f"[{x1}, {y1}, {x2}, {y2}]"]

    def get_task(self, example: dict):
        return str(example["data_type"])

    def get_source(self, example: dict):
        return str(example["data_source"])
