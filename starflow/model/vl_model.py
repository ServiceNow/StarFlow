from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.nn import Module
from starflow.dataset.vl_dataset import VLExample
import torch


@dataclass(kw_only=True)
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

    def collate_fn(self, vl_examples: list[VLExample]):
        vl_inputs = []
        for vl_example in vl_examples:
            vl_input = self.preprocess(vl_example, False)
            vl_inputs.append(vl_input)
        return self.collate_fn_inner(vl_inputs)

    @torch.inference_mode()
    def generate(self, vl_example: VLExample, **kwargs):
        vl_input = self.preprocess(vl_example, True)
        vl_input = vl_input.to(self.model.device)
        return self.generate_inner(vl_input, **kwargs)

    @abstractmethod
    def get_model_and_processor(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def post_init(self, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def layer_classes(self):
        raise NotImplementedError

    @abstractmethod
    def freeze_model(
        self,
        train_language_model: bool,
        train_vision_model: bool,
        train_connector: bool,
    ):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, vl_example: VLExample, for_generate: bool):
        raise NotImplementedError

    @abstractmethod
    def collate_fn_inner(self, vl_inputs: list[VLInput]):
        raise NotImplementedError

    @abstractmethod
    def forward(self, vl_input: VLInput):
        raise NotImplementedError

    @abstractmethod
    def generate_inner(self, vl_input: VLInput, **kwargs):
        raise NotImplementedError
