from dataclasses import dataclass
from torch import Tensor
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    PreTrainedModel,
    ProcessorMixin,
)
from typing import Optional
from starvlm.dataset.base import VLExample
from starvlm.model.base import VLLocalInput, VLLocalModel
import torch


@dataclass
class LlamaInput(VLLocalInput):
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    aspect_ratio_ids: Tensor
    aspect_ratio_mask: Tensor
    cross_attention_mask: Tensor
    labels: Optional[Tensor]

    def pin_memory(self) -> "LlamaInput":
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.pixel_values = self.pixel_values.pin_memory()
        self.aspect_ratio_ids = self.aspect_ratio_ids.pin_memory()
        self.aspect_ratio_mask = self.aspect_ratio_mask.pin_memory()
        self.cross_attention_mask = self.cross_attention_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs) -> "LlamaInput":
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.pixel_values = self.pixel_values.to(*args, **kwargs)
        self.aspect_ratio_ids = self.aspect_ratio_ids.to(*args, **kwargs)
        self.aspect_ratio_mask = self.aspect_ratio_mask.to(*args, **kwargs)
        self.cross_attention_mask = self.cross_attention_mask.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class LlamaModel(VLLocalModel):
    def get_model_and_processor(
        self, **kwargs
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        model_path = kwargs["model_path"]
        dtype = kwargs["dtype"]
        attn_implementation = kwargs["attn_implementation"]
        model = MllamaForConditionalGeneration.from_pretrained(
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
        for param in self.model.model.vision_model.parameters():
            param.requires_grad = requires_grad_kwargs["vision_model_requires_grad"]
        for param in self.model.model.multi_modal_projector.parameters():
            param.requires_grad = requires_grad_kwargs["connector_requires_grad"]
        self.generate_kwargs = generate_kwargs

    def get_layer_classes(self) -> set[type]:
        layer_classes = set()
        for layer in self.model.model.language_model.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.model.vision_model.transformer.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.model.vision_model.global_transformer.layers:
            layer_classes.add(layer.__class__)
        return layer_classes

    def preprocess(self, vl_example: VLExample, for_generate: bool) -> LlamaInput:
        input_ids = []
        pixel_values = None
        aspect_ratio_ids = None
        aspect_ratio_mask = None
        cross_attention_mask = None
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
                aspect_ratio_ids = feature["aspect_ratio_ids"].to(torch.long)
                aspect_ratio_mask = feature["aspect_ratio_mask"].to(torch.bool)
                cross_attention_mask = feature["cross_attention_mask"].to(torch.long)
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
        cross_attention_mask = torch.cat(
            [
                cross_attention_mask,
                torch.tile(
                    cross_attention_mask[:, -1:, :, :],
                    [
                        1,
                        max(input_ids.shape[1] - cross_attention_mask.shape[1], 0),
                        1,
                        1,
                    ],
                ),
            ],
            dim=1,
        )[:, : self.max_length]
        if not for_generate:
            labels = torch.cat(labels, dim=1)[:, : self.max_length]
            if torch.all(labels == self.ignore_index).item():
                labels[0, -1] = self.processor.tokenizer.eos_token_id
        llama_input = LlamaInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
            labels=labels,
        )
        return llama_input

    def collate_inner(self, llama_inputs: list[LlamaInput]) -> LlamaInput:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [llama_input.input_ids.squeeze(0) for llama_input in llama_inputs],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [llama_input.attention_mask.squeeze(0) for llama_input in llama_inputs],
            batch_first=True,
            padding_value=False,
        )
        pixel_values = torch.nn.utils.rnn.pad_sequence(
            [llama_input.pixel_values.squeeze(0) for llama_input in llama_inputs],
            batch_first=True,
            padding_value=0.0,
        )
        aspect_ratio_ids = torch.nn.utils.rnn.pad_sequence(
            [llama_input.aspect_ratio_ids.squeeze(0) for llama_input in llama_inputs],
            batch_first=True,
            padding_value=0,
        )
        aspect_ratio_mask = torch.nn.utils.rnn.pad_sequence(
            [llama_input.aspect_ratio_mask.squeeze(0) for llama_input in llama_inputs],
            batch_first=True,
            padding_value=False,
        )
        cross_attention_masks = []
        for llama_input in llama_inputs:
            cross_attention_masks.append(
                torch.nn.functional.pad(
                    llama_input.cross_attention_mask,
                    pad=[
                        0,
                        0,
                        0,
                        pixel_values.shape[1]
                        - llama_input.cross_attention_mask.shape[2],
                        0,
                        input_ids.shape[1] - llama_input.cross_attention_mask.shape[1],
                    ],
                    value=0,
                )
            )
        cross_attention_mask = torch.cat(cross_attention_masks)
        labels = torch.nn.utils.rnn.pad_sequence(
            [llama_input.labels.squeeze(0) for llama_input in llama_inputs],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        llama_input = LlamaInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
            labels=labels,
        )
        return llama_input

    def forward(self, llama_input: LlamaInput) -> Tensor:
        loss = self.model(
            input_ids=llama_input.input_ids,
            attention_mask=llama_input.attention_mask,
            pixel_values=llama_input.pixel_values,
            aspect_ratio_ids=llama_input.aspect_ratio_ids,
            aspect_ratio_mask=llama_input.aspect_ratio_mask,
            cross_attention_mask=llama_input.cross_attention_mask,
            labels=llama_input.labels,
            return_dict=True,
            use_cache=False,
        ).loss
        return loss

    def generate_inner(self, llama_input: LlamaInput) -> str:
        output_ids = self.model.generate(
            input_ids=llama_input.input_ids,
            attention_mask=llama_input.attention_mask,
            pixel_values=llama_input.pixel_values,
            aspect_ratio_ids=llama_input.aspect_ratio_ids,
            aspect_ratio_mask=llama_input.aspect_ratio_mask,
            cross_attention_mask=llama_input.cross_attention_mask,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            **self.generate_kwargs,
        )
        output = self.processor.tokenizer.batch_decode(
            output_ids[:, llama_input.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output
