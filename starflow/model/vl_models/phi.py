from dataclasses import dataclass
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import Optional
from starflow.dataset.vl_dataset import VLExample
from starflow.model.vl_model import VLInput, VLModel
import torch


@dataclass(kw_only=True)
class PhiInput(VLInput):
    input_ids: Tensor
    attention_mask: Tensor
    input_image_embeds: Tensor
    image_sizes: Tensor
    image_attention_mask: Tensor
    labels: Optional[Tensor]

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.input_image_embeds = self.input_image_embeds.pin_memory()
        self.image_sizes = self.image_sizes.pin_memory()
        self.image_attention_mask = self.image_attention_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.input_image_embeds = self.input_image_embeds.to(*args, **kwargs)
        self.image_sizes = self.image_sizes.to(*args, **kwargs)
        self.image_attention_mask = self.image_attention_mask.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self


class PhiModel(VLModel):
    def get_model_and_processor(self, **kwargs):
        pretrained_model_path = kwargs["pretrained_model_path"]
        torch_dtype = kwargs["torch_dtype"]
        attn_implementation = kwargs["attn_implementation"]
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
            _attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        model.gradient_checkpointing_disable()
        del model.model.embed_tokens_extend.audio_embed
        for layer in model.model.layers:
            del layer.mlp.down_proj.lora_A.speech
            del layer.mlp.down_proj.lora_B.speech
            del layer.mlp.gate_up_proj.lora_A.speech
            del layer.mlp.gate_up_proj.lora_B.speech
            del layer.self_attn.o_proj.lora_A.speech
            del layer.self_attn.o_proj.lora_B.speech
            del layer.self_attn.qkv_proj.lora_A.speech
            del layer.self_attn.qkv_proj.lora_B.speech
        processor = AutoProcessor.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        return model, processor

    def post_init(self, **kwargs):
        pretrained_model_path = kwargs["pretrained_model_path"]
        ignore_index = kwargs["ignore_index"]
        max_length = kwargs["max_length"]
        max_new_tokens = kwargs["max_new_tokens"]
        generation_config = GenerationConfig.from_pretrained(pretrained_model_path)
        self.generation_config = generation_config
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    @property
    def layer_classes(self):
        layer_classes = set()
        for layer in self.model.model.layers:
            layer_classes.add(layer.__class__)
        for (
            layer
        ) in (
            self.model.model.embed_tokens_extend.image_embed.img_processor.encoder.layers
        ):
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
        for param in self.model.model.embed_tokens_extend.image_embed.parameters():
            param.requires_grad = train_vision_model
        for (
            param
        ) in (
            self.model.model.embed_tokens_extend.image_embed.img_projection.parameters()
        ):
            param.requires_grad = train_connector

    def preprocess(self, vl_example: VLExample, for_generate: bool):
        input_ids = []
        input_image_embeds = None
        image_sizes = None
        image_attention_mask = None
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
                    [
                        f"<|image_{index}|>"
                        for index in range(1, len(vl_example.images) + 1)
                    ]
                    + [query]
                )
            messages.append({"role": "user", "content": query})
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )[length:]
            length += len(prompt)
            if is_first_turn:
                feature = self.processor(
                    images=vl_example.images, text=prompt, return_tensors="pt"
                )
                prompt_ids = feature["input_ids"].to(torch.long)
                input_image_embeds = feature["input_image_embeds"].to(torch.float)
                image_sizes = feature["image_sizes"].to(torch.long)
                image_attention_mask = feature["image_attention_mask"].to(torch.bool)
            else:
                prompt_ids = self.processor.tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(torch.long)
            input_ids.append(prompt_ids)
            if not for_generate:
                labels.append(torch.full_like(prompt_ids, fill_value=self.ignore_index))
            if not for_generate or not is_last_turn:
                messages.append({"role": "assistant", "content": annotation})
                response = self.processor.tokenizer.apply_chat_template(
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

        return PhiInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
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
        input_image_embeds = torch.nn.utils.rnn.pad_sequence(
            [vl_input.input_image_embeds.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=0.0,
        )
        image_sizes = torch.cat([vl_input.image_sizes for vl_input in vl_inputs])
        image_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [vl_input.image_attention_mask.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=False,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [vl_input.labels.squeeze(0) for vl_input in vl_inputs],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        return PhiInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            labels=labels,
        )

    def forward(self, vl_input: VLInput):
        return self.model(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            input_image_embeds=vl_input.input_image_embeds,
            image_sizes=vl_input.image_sizes,
            image_attention_mask=vl_input.image_attention_mask,
            labels=vl_input.labels,
            input_mode=1,
            return_dict=True,
            use_cache=False,
        ).loss

    def generate_inner(self, vl_input: VLInput, **kwargs):
        output_ids = self.model.generate(
            input_ids=vl_input.input_ids,
            attention_mask=vl_input.attention_mask,
            input_image_embeds=vl_input.input_image_embeds,
            image_sizes=vl_input.image_sizes,
            image_attention_mask=vl_input.image_attention_mask,
            input_mode=1,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            use_cache=True,
            **kwargs,
        )
        return self.processor.tokenizer.batch_decode(
            output_ids[:, vl_input.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
