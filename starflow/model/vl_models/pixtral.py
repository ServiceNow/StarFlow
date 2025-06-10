from dataclasses import dataclass
from torch import Tensor
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Optional
from starflow.dataset.vl_dataset import VLExample
from starflow.model.vl_model import VLInput, VLModel
import torch


@dataclass
class PixtralInput(VLInput):
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_sizes: Tensor
    labels: Optional[Tensor]

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.pixel_values = self.pixel_values.pin_memory()
        self.image_sizes = self.image_sizes.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.pixel_values = self.pixel_values.to(*args, **kwargs)
        self.image_sizes = self.image_sizes.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class PixtralModel(VLModel):
    def get_model_and_processor(self, **kwargs):
        pretrained_model_path = kwargs["pretrained_model_path"]
        torch_dtype = kwargs["torch_dtype"]
        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_path, torch_dtype=torch_dtype
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

    def get_layer_classes(self):
        layer_classes = set()
        for layer in self.model.model.language_model.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.model.vision_tower.transformer.layers:
            layer_classes.add(layer.__class__)
        return layer_classes

    def requires_grad(self, **kwargs):
        language_model_requires_grad = kwargs["language_model_requires_grad"]
        vision_model_requires_grad = kwargs["vision_model_requires_grad"]
        connector_requires_grad = kwargs["connector_requires_grad"]
        for param in self.model.parameters():
            param.requires_grad = language_model_requires_grad
        for param in self.model.model.vision_tower.parameters():
            param.requires_grad = vision_model_requires_grad
        for param in self.model.model.multi_modal_projector.parameters():
            param.requires_grad = connector_requires_grad

    def preprocess(self, vl_example: VLExample, for_generate: bool):
        input_ids = []
        pixel_values = None
        image_sizes = None
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
                        + [{"type": "text", "content": query}],
                    }
                )
            else:
                messages.append(
                    {"role": "user", "content": [{"type": "text", "content": query}]}
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
                image_sizes = feature["image_sizes"].to(torch.long)
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
                        "content": [{"type": "text", "content": annotation}],
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
        return PixtralInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=labels,
        )

    def collate_inner(self, vl_inputs: list[PixtralInput]):
        assert len(vl_inputs) == 1
        return vl_inputs[0]

    def forward(self, vl_input: PixtralInput):
        return self.model(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            pixel_values=vl_input.pixel_values,
            image_sizes=vl_input.image_sizes,
            labels=vl_input.labels,
            return_dict=True,
            use_cache=False,
        ).loss

    def generate_inner(self, vl_input: PixtralInput, **kwargs):
        output_ids = self.model.generate(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            pixel_values=vl_input.pixel_values,
            image_sizes=vl_input.image_sizes,
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
