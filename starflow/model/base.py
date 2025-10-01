from abc import ABC, abstractmethod
from copy import deepcopy
from torch import inference_mode
from torch.nn import Module
from typing import Any, Iterator
from starflow.dataset.base import VLExample
from starflow.utils import get_image_url


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
    def get_model_and_processor(self, **kwargs) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def post_init(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_layer_classes(self) -> set[type]:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, vl_example: VLExample, for_generate: bool) -> VLInput:
        raise NotImplementedError

    def collate(self, vl_examples: list[VLExample]) -> VLInput:
        vl_inputs = []
        for vl_example in vl_examples:
            vl_input = self.preprocess(vl_example, False)
            vl_inputs.append(vl_input)
        return self.collate_inner(vl_inputs)

    @abstractmethod
    def collate_inner(self, vl_inputs: list[VLInput]) -> VLInput:
        raise NotImplementedError

    @abstractmethod
    def forward(self, vl_input: VLInput) -> Any:
        raise NotImplementedError

    @inference_mode()
    def generate(self, vl_example: VLExample) -> str:
        vl_input = self.preprocess(vl_example, True)
        vl_input = vl_input.to(self.model.device)
        return self.generate_inner(vl_input)

    @abstractmethod
    def generate_inner(self, vl_input: VLInput) -> str:
        raise NotImplementedError


class VLMessage(dict):
    def __init__(self, role: str):
        super().__init__()
        self["role"] = role
        self["content"] = []

    def add_text(self, text: str):
        if len(self["content"]) != 0:
            text = "\n" + text
        if len(self["content"]) != 0 and self["content"][-1]["type"] == "text":
            self["content"][-1]["text"] += text
        else:
            self["content"].append({"type": "text", "text": text})

    def add_image(self, image: Any):
        if len(self["content"]) != 0 and self["content"][-1]["type"] == "text":
            self["content"][-1]["text"] += "\n"
        self["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": get_image_url(image)},
            }
        )

    def extend_content(self, content: list[dict]):
        for item in content:
            if item["type"] == "text":
                self.add_text(item["text"].strip())
            else:
                if len(self["content"]) != 0 and self["content"][-1]["type"] == "text":
                    self["content"][-1]["text"] += "\n"
                self["content"].append(deepcopy(item))

    def copy(self) -> "VLMessage":
        vl_message = self.__class__(self["role"])
        for item in self["content"]:
            vl_message["content"].append(deepcopy(item))
        return vl_message


class VLConversation:
    def __init__(self):
        self.vl_messages = []

    def __getitem__(self, index: int) -> VLMessage:
        return self.vl_messages[index]

    def __iter__(self) -> Iterator:
        return iter(self.vl_messages)

    def __len__(self) -> int:
        return len(self.vl_messages)

    def add_message(self, vl_message: VLMessage):
        if (
            len(self.vl_messages) != 0
            and self.vl_messages[-1]["role"] == vl_message["role"]
        ):
            self.vl_messages[-1].extend_content(vl_message["content"])
        else:
            self.vl_messages.append(vl_message.copy())

    def copy(self) -> "VLConversation":
        vl_conversation = self.__class__()
        for vl_message in self.vl_messages:
            vl_conversation.vl_messages.append(vl_message.copy())
        return vl_conversation


class VLAPIModel(ABC):
    def __init__(self, **kwargs):
        client = self.get_client(**kwargs)
        self.client = client
        self.post_init(**kwargs)

    @abstractmethod
    def get_client(self, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def post_init(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, vl_example: VLExample) -> VLConversation:
        raise NotImplementedError

    def generate(self, vl_example: VLExample) -> str:
        vl_conversation = self.preprocess(vl_example)
        return self.generate_inner(vl_conversation)

    @abstractmethod
    def generate_inner(self, vl_conversation: VLConversation) -> str:
        raise NotImplementedError
