from dataclasses import dataclass
from torch import Tensor
from typing import Optional
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from starflow.dataset.vl_dataset import VLExample
from starflow.model.vl_model import VLInput, VLModel
import torch


@dataclass
class DeepseekInput(VLInput):
    input_ids: Tensor
    attention_mask: Tensor
    images: Tensor
    images_seq_mask: Tensor
    images_spatial_crop: Tensor
    labels: Optional[Tensor]

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.images = self.images.pin_memory()
        self.images_seq_mask = self.images_seq_mask.pin_memory()
        self.images_spatial_crop = self.images_spatial_crop.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.images = self.images.to(*args, **kwargs)
        self.images_seq_mask = self.images_seq_mask.to(*args, **kwargs)
        self.images_spatial_crop = self.images_spatial_crop.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class DeepseekModel(VLModel):
    def get_model_and_processor(self, **kwargs):
        pretrained_model_path = kwargs["pretrained_model_path"]
        torch_dtype = kwargs["torch_dtype"]
        model = DeepseekVLV2ForCausalLM.from_pretrained(
            pretrained_model_path, torch_dtype=getattr(torch, torch_dtype)
        )
        processor = DeepseekVLV2Processor.from_pretrained(pretrained_model_path)
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
        for layer in self.model.language.model.layers:
            layer_classes.add(layer.__class__)
        for layer in self.model.vision.blocks:
            layer_classes.add(layer.__class__)
        return layer_classes

    def requires_grad(self, **kwargs):
        language_model_requires_grad = kwargs["language_model_requires_grad"]
        vision_model_requires_grad = kwargs["vision_model_requires_grad"]
        connector_requires_grad = kwargs["connector_requires_grad"]
        for param in self.model.parameters():
            param.requires_grad = language_model_requires_grad
        for param in self.model.vision.parameters():
            param.requires_grad = vision_model_requires_grad
        for param in self.model.projector.parameters():
            param.requires_grad = connector_requires_grad

    def preprocess(self, vl_example: VLExample, for_generate: bool):
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
        truncation_length = (
            self.max_length if for_generate else self.max_length + self.max_new_tokens
        )
        input_ids = torch.cat(input_ids, dim=1)[:, :truncation_length]
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
        )[:, :truncation_length]
        if not for_generate:
            labels = torch.cat(labels, dim=1)[:, :truncation_length]
            if torch.all(labels == self.ignore_index).item():
                labels[0, -1] = self.processor.tokenizer.eos_token_id
        return DeepseekInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
        )

    def collate_inner(self, vl_inputs: list[DeepseekInput]):
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
        images = torch.nn.utils.rnn.pad_sequence(
            [vl_input.images.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=0.0,
        )
        images_seq_mask = torch.nn.utils.rnn.pad_sequence(
            [vl_input.images_seq_mask.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=False,
        )
        images_spatial_crop = torch.nn.utils.rnn.pad_sequence(
            [vl_input.images_spatial_crop.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [vl_input.labels.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        return DeepseekInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
        )

    def forward(self, vl_input: DeepseekInput):
        return self.model(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            images=vl_input.images,
            images_seq_mask=vl_input.images_seq_mask,
            images_spatial_crop=vl_input.images_spatial_crop,
            labels=vl_input.labels,
            return_dict=True,
            use_cache=False,
        ).loss

    def generate_inner(self, vl_input: DeepseekInput, **kwargs):
        inputs_embeds = self.model.prepare_inputs_embeds(
            input_ids=vl_input.input_ids,
            images=vl_input.images,
            images_seq_mask=vl_input.images_seq_mask,
            images_spatial_crop=vl_input.images_spatial_crop,
        )
        output_ids = self.model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=vl_input.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            **kwargs,
        )
        return self.processor.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
