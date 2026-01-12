from backoff import expo, on_exception
from openai import OpenAI, RateLimitError
from starvlm.model.base import VLAPIConversation, VLAPIModel
import os


class OpenAIModel(VLAPIModel):
    def __init__(self, **kwargs) -> None:
        base_url = kwargs["base_url"]
        model_path = kwargs["model_path"]
        generate_kwargs = kwargs["generate_kwargs"]
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY"))
        self.model_path = model_path
        self.generate_kwargs = generate_kwargs

    def generate_inner(self, vl_api_conversation: VLAPIConversation) -> str:
        @on_exception(expo, RateLimitError)
        def get_output(vl_api_conversation: VLAPIConversation) -> str:
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=vl_api_conversation,
                **self.generate_kwargs,
            )
            try:
                output = completion.choices[0].message.content
            except:
                output = ""
            return output

        output = get_output(vl_api_conversation)
        return output
