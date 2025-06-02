from dataclasses import dataclass
from torch import Tensor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import Optional
from starflow.dataset.vl_dataset import VLExample
from starflow.model.vl_model import VLInput, VLModel
import torch


@dataclass(kw_only=True)
class QwenInput(VLInput):
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor
    labels: Optional[Tensor]

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.pixel_values = self.pixel_values.pin_memory()
        self.image_grid_thw = self.image_grid_thw.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.pixel_values = self.pixel_values.to(*args, **kwargs)
        self.image_grid_thw = self.image_grid_thw.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class QwenModel(VLModel):
    def get_model_and_processor(self, **kwargs):
        pretrained_model_path = kwargs["pretrained_model_path"]
        torch_dtype = kwargs["torch_dtype"]
        attn_implementation = kwargs["attn_implementation"]
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(pretrained_model_path)
        return model, processor

    def post_init(self, **kwargs):
        ignore_index = kwargs["ignore_index"]
        max_length = kwargs["max_length"]
        max_new_tokens = kwargs["max_new_tokens"]
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    @property
    def layer_classes(self):
        layer_classes = set()
        for layer in self.model.model.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.visual.blocks:
            layer_classes.add(layer.__class__)
        return layer_classes

    def freeze_model(
        self,
        train_language_model: bool,
        train_vision_model: bool,
        train_connector: bool,
    ):
        for param in self.model.parameters():
            param.requires_grad = train_language_model
        for param in self.model.visual.parameters():
            param.requires_grad = train_vision_model
        for param in self.model.visual.merger.parameters():
            param.requires_grad = train_connector

    def preprocess(self, vl_example: VLExample, for_generate: bool):
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
        truncation_length = (
            self.max_length if for_generate else self.max_length + self.max_new_tokens
        )
        input_ids = torch.cat(input_ids, dim=1)[:, :truncation_length]
        attention_mask = torch.full_like(input_ids, fill_value=True, dtype=torch.bool)
        if not for_generate:
            labels = torch.cat(labels, dim=1)[:, :truncation_length]
            if torch.all(labels == self.ignore_index).item():
                labels[0, -1] = self.processor.tokenizer.eos_token_id
        return QwenInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

    def collate_fn_inner(self, vl_inputs: list[VLInput]):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [vl_input.input_ids.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [vl_input.attention_mask.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=False,
        )
        pixel_values = torch.cat([vl_input.pixel_values for vl_input in vl_inputs])
        image_grid_thw = torch.cat([vl_input.image_grid_thw for vl_input in vl_inputs])
        labels = torch.nn.utils.rnn.pad_sequence(
            [vl_input.labels.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        return QwenInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

    def forward(self, vl_input: VLInput):
        return self.model(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            pixel_values=vl_input.pixel_values,
            image_grid_thw=vl_input.image_grid_thw,
            labels=vl_input.labels,
            return_dict=True,
            use_cache=False,
        ).loss

    def generate_inner(self, vl_input: VLInput, **kwargs):
        output_ids = self.model.generate(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            pixel_values=vl_input.pixel_values,
            image_grid_thw=vl_input.image_grid_thw,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            **kwargs,
        )
        return self.processor.tokenizer.batch_decode(
            output_ids[:, vl_input.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
