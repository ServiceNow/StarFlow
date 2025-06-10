from abc import ABC, abstractmethod
from torch.nn import Module
from starflow.dataset.vl_dataset import VLExample
import torch


class VLInput(ABC):
    @abstractmethod
    def pin_memory(self):
        raise NotImplementedError

    @abstractmethod
    def to(self, *args, **kwargs):
        raise NotImplementedError


class VLModel(ABC, Module):
    def __init__(self, **kwargs):
        super().__init__()
        model, processor = self.get_model_and_processor(**kwargs)
        self.model = model
        self.processor = processor
        self.post_init(**kwargs)

    @abstractmethod
    def get_model_and_processor(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def post_init(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_layer_classes(self):
        raise NotImplementedError

    @abstractmethod
    def requires_grad(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, vl_example: VLExample, for_generate: bool):
        raise NotImplementedError

    def collate(self, vl_examples: list[VLExample]):
        vl_inputs = []
        for vl_example in vl_examples:
            vl_input = self.preprocess(vl_example, False)
            vl_inputs.append(vl_input)
        return self.collate_inner(vl_inputs)

    @abstractmethod
    def collate_inner(self, vl_inputs: list[VLInput]):
        raise NotImplementedError

    @abstractmethod
    def forward(self, vl_input: VLInput):
        raise NotImplementedError

    @torch.inference_mode()
    def generate(self, vl_example: VLExample, **kwargs):
        vl_input = self.preprocess(vl_example, True)
        vl_input = vl_input.to(self.model.device)
        return self.generate_inner(vl_input, **kwargs)

    @abstractmethod
    def generate_inner(self, vl_input: VLInput, **kwargs):
        raise NotImplementedError
