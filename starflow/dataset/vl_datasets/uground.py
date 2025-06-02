from starflow.dataset.utils import get_pil_image
from starflow.dataset.vl_dataset import VLDataset
import ast
import hashlib
import json
import re


class UGroundDataset(VLDataset):
    def post_init(self, **kwargs):
        use_pixel_coordinates = kwargs["use_pixel_coordinates"]
        self.use_pixel_coordinates = use_pixel_coordinates

    def get_identifier(self, example: dict):
        buffer = example["conversations"].encode()
        return hashlib.md5(buffer).hexdigest()

    def get_images(self, example: dict):
        return [get_pil_image(example["image"])]

    def get_queries(self, example: dict):
        try:
            query = json.loads(example["conversations"])[0]["value"]
            match = re.search("<image>(.*)Answer:", query, re.DOTALL)
            queries = [match.group(1).strip()]
        except:
            queries = []
        return queries

    def get_annotations(self, example: dict):
        try:
            annotation = json.loads(example["conversations"])[1]["value"]
            x, y = ast.literal_eval(annotation)
            if self.use_pixel_coordinates:
                x = int((x / 1000) * example["width"])
                y = int((y / 1000) * example["height"])
            annotations = [f"({x}, {y})"]
        except:
            annotations = []
        return annotations

    def get_task(self, example: dict):
        return "UGround"

    def get_source(self, example: dict):
        return "UGround"
