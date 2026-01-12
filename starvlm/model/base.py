from abc import ABC, abstractmethod
from collections.abc import Iterator
from copy import deepcopy
from torch import inference_mode, Tensor
from transformers import PreTrainedModel, ProcessorMixin
from typing import Any
from starvlm.dataset.base import VLExample
from starvlm.utils import get_image_url


class VLLocalInput(ABC):
    @abstractmethod
    def pin_memory(self) -> "VLLocalInput":
        raise NotImplementedError

    @abstractmethod
    def to(self, *args, **kwargs) -> "VLLocalInput":
        raise NotImplementedError


class VLLocalModel(ABC):
    def __init__(self, **kwargs) -> None:
        model, processor = self.get_model_and_processor(**kwargs)
        self._model = model
        self._processor = processor
        self.post_init(**kwargs)

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @model.setter
    def model(self, model: PreTrainedModel) -> None:
        self._model = model

    @property
    def processor(self) -> ProcessorMixin:
        return self._processor

    @processor.setter
    def processor(self, processor: ProcessorMixin) -> None:
        self._processor = processor

    @abstractmethod
    def get_model_and_processor(
        self, **kwargs
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        raise NotImplementedError

    @abstractmethod
    def post_init(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_layer_classes(self) -> set[type]:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, vl_example: VLExample, for_generate: bool) -> VLLocalInput:
        raise NotImplementedError

    def collate(self, vl_examples: list[VLExample]) -> VLLocalInput:
        if not vl_examples:
            raise ValueError("vl_examples cannot be empty")
        vl_local_input = self.collate_inner(
            [self.preprocess(vl_example, False) for vl_example in vl_examples]
        )
        return vl_local_input

    @abstractmethod
    def collate_inner(self, vl_local_inputs: list[VLLocalInput]) -> VLLocalInput:
        raise NotImplementedError

    @abstractmethod
    def forward(self, vl_local_input: VLLocalInput) -> Tensor:
        raise NotImplementedError

    @inference_mode()
    def generate(self, vl_example: VLExample) -> str:
        vl_local_input = self.preprocess(vl_example, True)
        vl_local_input = vl_local_input.to(self.model.device)
        output = self.generate_inner(vl_local_input)
        return output

    @abstractmethod
    def generate_inner(self, vl_local_input: VLLocalInput) -> str:
        raise NotImplementedError


class VLAPIMessage(dict):
    def __init__(self, role: str) -> None:
        super().__init__()
        self["role"] = role
        self["content"] = []

    def add_text(self, text: str) -> None:
        if len(self["content"]) != 0:
            text = "\n" + text
        if len(self["content"]) != 0 and self["content"][-1]["type"] == "text":
            self["content"][-1]["text"] += text
        else:
            self["content"].append({"type": "text", "text": text})

    def add_image(self, image: Any) -> None:
        image_url = get_image_url(image)
        if len(self["content"]) != 0 and self["content"][-1]["type"] == "text":
            self["content"][-1]["text"] += "\n"
        self["content"].append({"type": "image_url", "image_url": {"url": image_url}})

    def extend_content(self, content: list[dict[str, Any]]) -> None:
        for item in content:
            if item["type"] == "text":
                self.add_text(item["text"].strip())
            else:
                if len(self["content"]) != 0 and self["content"][-1]["type"] == "text":
                    self["content"][-1]["text"] += "\n"
                self["content"].append(deepcopy(item))

    def copy(self) -> "VLAPIMessage":
        vl_api_message = self.__class__(self["role"])
        for item in self["content"]:
            vl_api_message["content"].append(deepcopy(item))
        return vl_api_message


class VLAPIConversation:
    def __init__(self) -> None:
        self.vl_api_messages = []

    def __getitem__(self, index: int) -> VLAPIMessage:
        return self.vl_api_messages[index]

    def __iter__(self) -> Iterator[VLAPIMessage]:
        return iter(self.vl_api_messages)

    def __len__(self) -> int:
        return len(self.vl_api_messages)

    def add_message(self, vl_api_message: VLAPIMessage) -> None:
        if (
            len(self.vl_api_messages) == 0
            or self.vl_api_messages[-1]["role"] != vl_api_message["role"]
        ):
            self.vl_api_messages.append(VLAPIMessage(vl_api_message["role"]))
        self.vl_api_messages[-1].extend_content(vl_api_message["content"])

    def copy(self) -> "VLAPIConversation":
        vl_api_conversation = self.__class__()
        for vl_api_message in self.vl_api_messages:
            vl_api_conversation.vl_api_messages.append(vl_api_message.copy())
        return vl_api_conversation


class VLAPIModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError

    def __call__(self, vl_api_conversation: VLAPIConversation) -> VLAPIMessage:
        output = self.generate_inner(vl_api_conversation)
        vl_api_message = VLAPIMessage("assistant")
        vl_api_message.add_text(output)
        return vl_api_message

    def generate(self, vl_example: VLExample) -> str:
        vl_api_conversation = VLAPIConversation()
        for index, (query, annotation) in enumerate(
            zip(vl_example.queries, vl_example.annotations)
        ):
            vl_api_message = VLAPIMessage("user")
            if index == 0:
                for image in vl_example.images:
                    vl_api_message.add_image(image)
            vl_api_message.add_text(query)
            vl_api_conversation.add_message(vl_api_message)
            if index < len(vl_example.queries) - 1:
                vl_api_message = VLAPIMessage("assistant")
                vl_api_message.add_text(annotation)
                vl_api_conversation.add_message(vl_api_message)
        output = self.generate_inner(vl_api_conversation)
        return output

    @abstractmethod
    def generate_inner(self, vl_api_conversation: VLAPIConversation) -> str:
        raise NotImplementedError
