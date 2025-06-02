from starflow.dataset.utils import get_pil_image
from starflow.dataset.vl_dataset import VLDataset


class BigDocsDataset(VLDataset):
    def post_init(self, **kwargs):
        pass

    def get_identifier(self, example: dict):
        return str(example["identifier"])

    def get_images(self, example: dict):
        return [get_pil_image(image) for image in example["images"]]

    def get_queries(self, example: dict):
        return [str(query) for query in example["queries"]]

    def get_annotations(self, example: dict):
        return [str(annotation) for annotation in example["annotations"]]

    def get_task(self, example: dict):
        return str(example["task"])

    def get_source(self, example: dict):
        return str(example["source"])
