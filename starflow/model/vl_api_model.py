from abc import ABC, abstractmethod
from typing import Any
from starflow.dataset.vl_dataset import VLExample


class VLAPIModel(ABC):
    def __init__(self, **kwargs):
        client = self.get_client(**kwargs)
        self.client = client
        self.post_init(**kwargs)

    @abstractmethod
    def get_client(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def post_init(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, vl_example: VLExample):
        raise NotImplementedError

    def generate(self, vl_examples: list[VLExample], **kwargs):
        vl_api_inputs = []
        for vl_example in vl_examples:
            vl_api_input = self.preprocess(vl_example)
            vl_api_inputs.append(vl_api_input)
        return self.generate_inner(vl_api_inputs, **kwargs)

    @abstractmethod
    def generate_inner(self, vl_api_inputs: list[Any], **kwargs):
        raise NotImplementedError
