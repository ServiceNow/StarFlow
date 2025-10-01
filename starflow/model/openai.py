from backoff import expo, on_exception
from openai import OpenAI, RateLimitError
from typing import Any
from starflow.dataset.base import VLExample
from starflow.model.base import VLAPIModel, VLConversation, VLMessage
import os


class OpenAIModel(VLAPIModel):
    def get_client(self, **kwargs) -> Any:
        base_url = kwargs["base_url"]
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI(base_url=base_url, api_key=api_key)

    def post_init(self, **kwargs):
        model_path = kwargs["model_path"]
        generate_kwargs = kwargs["generate_kwargs"]
        self.model_path = model_path
        self.generate_kwargs = generate_kwargs

    def preprocess(self, vl_example: VLExample) -> VLConversation:
        vl_conversation = VLConversation()
        for index, (query, annotation) in enumerate(
            zip(vl_example.queries, vl_example.annotations)
        ):
            vl_message = VLMessage("user")
            if index == 0:
                for image in vl_example.images:
                    vl_message.add_image(image)
            vl_message.add_text(query)
            vl_conversation.add_message(vl_message)
            if index < len(vl_example.queries) - 1:
                vl_message = VLMessage("assistant")
                vl_message.add_text(annotation)
                vl_conversation.add_message(vl_message)
        return vl_conversation

    def generate_inner(self, vl_conversation: VLConversation) -> str:
        @on_exception(expo, RateLimitError)
        def get_output(vl_conversation: VLConversation):
            completion = self.client.chat.completions.create(
                model=self.model_path, messages=vl_conversation, **self.generate_kwargs
            )
            try:
                output = completion.choices[0].message.content
            except:
                output = ""
            return output

        return get_output(vl_conversation)
