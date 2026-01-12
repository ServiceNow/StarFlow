from dataclasses import dataclass
from torch import Tensor
from transformers import PreTrainedModel, ProcessorMixin
from typing import Optional
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from starvlm.dataset.base import VLExample
from starvlm.model.base import VLLocalInput, VLLocalModel
import torch


@dataclass
class DeepseekInput(VLLocalInput):
    input_ids: Tensor
    attention_mask: Tensor
    images: Tensor
    images_seq_mask: Tensor
    images_spatial_crop: Tensor
    labels: Optional[Tensor]

    def pin_memory(self) -> "DeepseekInput":
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.images = self.images.pin_memory()
        self.images_seq_mask = self.images_seq_mask.pin_memory()
        self.images_spatial_crop = self.images_spatial_crop.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs) -> "DeepseekInput":
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.images = self.images.to(*args, **kwargs)
        self.images_seq_mask = self.images_seq_mask.to(*args, **kwargs)
        self.images_spatial_crop = self.images_spatial_crop.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class DeepseekModel(VLLocalModel):
    def get_model_and_processor(
        self, **kwargs
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        model_path = kwargs["model_path"]
        torch_dtype = kwargs["torch_dtype"]
        model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, torch_dtype)
        )
        processor = DeepseekVLV2Processor.from_pretrained(model_path)
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
        for param in self.model.vision.parameters():
            param.requires_grad = requires_grad_kwargs["vision_model_requires_grad"]
        for param in self.model.projector.parameters():
            param.requires_grad = requires_grad_kwargs["connector_requires_grad"]
        self.generate_kwargs = generate_kwargs

    def get_layer_classes(self) -> set[type]:
        layer_classes = set()
        for layer in self.model.language.model.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.vision.blocks:
            layer_classes.add(layer.__class__)
        layer_classes.add(self.model.vision.attn_pool.__class__)
        layer_classes.add(self.model.projector.__class__)
        return layer_classes

    def preprocess(self, vl_example: VLExample, for_generate: bool) -> DeepseekInput:
        input_ids = []
        images = None
        images_seq_mask = None
        images_spatial_crop = None
        labels = None if for_generate else []
        messages = []
        length = 0
        for index, (query, annotation) in enumerate(
            zip(vl_example.queries, vl_example.annotations)
        ):
            is_first_turn = index == 0
            is_last_turn = index == len(vl_example.queries) - 1
            if is_first_turn:
                query = "\n".join(
                    [self.processor.image_token for _ in vl_example.images] + [query]
                )
            messages.append({"role": "<|User|>", "content": query})
            messages.append({"role": "<|Assistant|>", "content": ""})
            prompt = self.processor.format_messages(messages)[length:]
            length += len(prompt)
            if is_first_turn:
                feature = self.processor(
                    prompt=prompt,
                    images=vl_example.images,
                    apply_sft_format=False,
                    force_batchify=True,
                    inference_mode=True,
                )
                prompt_ids = feature["input_ids"].to(torch.long)
                images = feature["images"].to(torch.float)
                images_seq_mask = feature["images_seq_mask"].to(torch.bool)
                images_spatial_crop = feature["images_spatial_crop"].to(torch.long)
            else:
                prompt_ids = self.processor.tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(torch.long)
            input_ids.append(prompt_ids)
            if not for_generate:
                labels.append(torch.full_like(prompt_ids, fill_value=self.ignore_index))
            if not for_generate or not is_last_turn:
                messages[-1]["content"] = annotation
                response = self.processor.format_messages(messages)[length:]
                length += len(response)
                response_ids = self.processor.tokenizer(
                    response, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(torch.long)
                input_ids.append(response_ids)
                if not for_generate:
                    labels.append(response_ids)
        input_ids = torch.cat(input_ids, dim=1)[:, : self.max_length]
        attention_mask = torch.full_like(input_ids, fill_value=True, dtype=torch.bool)
        images_seq_mask = torch.cat(
            [
                images_seq_mask,
                torch.full(
                    [1, max(input_ids.shape[1] - images_seq_mask.shape[1], 0)],
                    fill_value=False,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )[:, : self.max_length]
        if not for_generate:
            labels = torch.cat(labels, dim=1)[:, : self.max_length]
            if torch.all(labels == self.ignore_index).item():
                labels[0, -1] = self.processor.tokenizer.eos_token_id
        deepseek_input = DeepseekInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
        )
        return deepseek_input

    def collate_inner(self, deepseek_inputs: list[DeepseekInput]) -> DeepseekInput:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [deepseek_input.input_ids.squeeze(0) for deepseek_input in deepseek_inputs],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                deepseek_input.attention_mask.squeeze(0)
                for deepseek_input in deepseek_inputs
            ],
            batch_first=True,
            padding_value=False,
        )
        images = torch.nn.utils.rnn.pad_sequence(
            [deepseek_input.images.squeeze(0) for deepseek_input in deepseek_inputs],
            batch_first=True,
            padding_value=0.0,
        )
        images_seq_mask = torch.nn.utils.rnn.pad_sequence(
            [
                deepseek_input.images_seq_mask.squeeze(0)
                for deepseek_input in deepseek_inputs
            ],
            batch_first=True,
            padding_value=False,
        )
        images_spatial_crop = torch.nn.utils.rnn.pad_sequence(
            [
                deepseek_input.images_spatial_crop.squeeze(0)
                for deepseek_input in deepseek_inputs
            ],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [deepseek_input.labels.squeeze(0) for deepseek_input in deepseek_inputs],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        deepseek_input = DeepseekInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
        )
        return deepseek_input

    def forward(self, deepseek_input: DeepseekInput) -> Tensor:
        loss = self.model(
            input_ids=deepseek_input.input_ids,
            attention_mask=deepseek_input.attention_mask,
            images=deepseek_input.images,
            images_seq_mask=deepseek_input.images_seq_mask,
            images_spatial_crop=deepseek_input.images_spatial_crop,
            labels=deepseek_input.labels,
            return_dict=True,
            use_cache=False,
        ).loss
        return loss

    def generate_inner(self, deepseek_input: DeepseekInput) -> str:
        inputs_embeds = self.model.prepare_inputs_embeds(
            input_ids=deepseek_input.input_ids,
            images=deepseek_input.images,
            images_seq_mask=deepseek_input.images_seq_mask,
            images_spatial_crop=deepseek_input.images_spatial_crop,
        )
        output_ids = self.model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=deepseek_input.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            **self.generate_kwargs,
        )
        output = self.processor.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output
