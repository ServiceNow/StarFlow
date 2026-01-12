from dataclasses import dataclass
from torch import Tensor
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
    Qwen3VLForConditionalGeneration,
)
from typing import Optional
from starvlm.dataset.base import VLExample
from starvlm.model.base import VLLocalInput, VLLocalModel
import torch


@dataclass
class QwenInput(VLLocalInput):
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor
    labels: Optional[Tensor]

    def pin_memory(self) -> "QwenInput":
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.pixel_values = self.pixel_values.pin_memory()
        self.image_grid_thw = self.image_grid_thw.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs) -> "QwenInput":
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.pixel_values = self.pixel_values.to(*args, **kwargs)
        self.image_grid_thw = self.image_grid_thw.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class QwenModel(VLLocalModel):
    def get_model_and_processor(
        self, **kwargs
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        model_path = kwargs["model_path"]
        dtype = kwargs["dtype"]
        attn_implementation = kwargs["attn_implementation"]
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor

    def post_init(self, **kwargs) -> None:
        ignore_index = kwargs["ignore_index"]
        max_length = kwargs["max_length"]
        requires_grad_kwargs = kwargs["requires_grad_kwargs"]
        generate_kwargs = kwargs["generate_kwargs"]
        self.ignore_index = ignore_index
        self.max_length = max_length
        for param in self.model.parameters():
            param.requires_grad = requires_grad_kwargs["language_model_requires_grad"]
        for param in self.model.model.visual.parameters():
            param.requires_grad = requires_grad_kwargs["vision_model_requires_grad"]
        for param in self.model.model.visual.deepstack_merger_list.parameters():
            param.requires_grad = requires_grad_kwargs["connector_requires_grad"]
        for param in self.model.model.visual.merger.parameters():
            param.requires_grad = requires_grad_kwargs["connector_requires_grad"]
        self.generate_kwargs = generate_kwargs

    def get_layer_classes(self) -> set[type]:
        layer_classes = set()
        for layer in self.model.model.language_model.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.model.visual.blocks:
            layer_classes.add(layer.__class__)
        for layer in self.model.model.visual.deepstack_merger_list:
            layer_classes.add(layer.__class__)
        layer_classes.add(self.model.model.visual.merger.__class__)
        return layer_classes

    def preprocess(self, vl_example: VLExample, for_generate: bool) -> QwenInput:
        input_ids = []
        pixel_values = None
        image_grid_thw = None
        labels = None if for_generate else []
        messages = []
        length = 0
        for index, (query, annotation) in enumerate(
            zip(vl_example.queries, vl_example.annotations)
        ):
            is_first_turn = index == 0
            is_last_turn = index == len(vl_example.queries) - 1
            if is_first_turn:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "image"} for _ in vl_example.images]
                        + [{"type": "text", "text": query}],
                    }
                )
            else:
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": query}]}
                )
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )[length:]
            length += len(prompt)
            if is_first_turn:
                feature = self.processor(
                    images=vl_example.images,
                    text=prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                prompt_ids = feature["input_ids"].to(torch.long)
                pixel_values = feature["pixel_values"].to(torch.float)
                image_grid_thw = feature["image_grid_thw"].to(torch.long)
            else:
                prompt_ids = self.processor.tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(torch.long)
            input_ids.append(prompt_ids)
            if not for_generate:
                labels.append(torch.full_like(prompt_ids, fill_value=self.ignore_index))
            if not for_generate or not is_last_turn:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": annotation}],
                    }
                )
                response = self.processor.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                )[length:]
                length += len(response)
                response_ids = self.processor.tokenizer(
                    response, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(torch.long)
                input_ids.append(response_ids)
                if not for_generate:
                    labels.append(response_ids)
        input_ids = torch.cat(input_ids, dim=1)[:, : self.max_length]
        attention_mask = torch.full_like(input_ids, fill_value=True, dtype=torch.bool)
        if not for_generate:
            labels = torch.cat(labels, dim=1)[:, : self.max_length]
            if torch.all(labels == self.ignore_index).item():
                labels[0, -1] = self.processor.tokenizer.eos_token_id
        qwen_input = QwenInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        return qwen_input

    def collate_inner(self, qwen_inputs: list[QwenInput]) -> QwenInput:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [qwen_input.input_ids.squeeze(0) for qwen_input in qwen_inputs],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [qwen_input.attention_mask.squeeze(0) for qwen_input in qwen_inputs],
            batch_first=True,
            padding_value=False,
        )
        pixel_values = torch.cat(
            [qwen_input.pixel_values for qwen_input in qwen_inputs]
        )
        image_grid_thw = torch.cat(
            [qwen_input.image_grid_thw for qwen_input in qwen_inputs]
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [qwen_input.labels.squeeze(0) for qwen_input in qwen_inputs],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        qwen_input = QwenInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        return qwen_input

    def forward(self, qwen_input: QwenInput) -> Tensor:
        loss = self.model(
            input_ids=qwen_input.input_ids,
            attention_mask=qwen_input.attention_mask,
            pixel_values=qwen_input.pixel_values,
            image_grid_thw=qwen_input.image_grid_thw,
            labels=qwen_input.labels,
            return_dict=True,
            use_cache=False,
        ).loss
        return loss

    def generate_inner(self, qwen_input: QwenInput) -> str:
        output_ids = self.model.generate(
            input_ids=qwen_input.input_ids,
            attention_mask=qwen_input.attention_mask,
            pixel_values=qwen_input.pixel_values,
            image_grid_thw=qwen_input.image_grid_thw,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            **self.generate_kwargs,
        )
        output = self.processor.tokenizer.batch_decode(
            output_ids[:, qwen_input.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output
