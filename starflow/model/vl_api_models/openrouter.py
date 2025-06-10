from openai import AsyncOpenAI, RateLimitError
from typing import Any
from starflow.dataset.vl_dataset import VLExample
from starflow.model.utils import get_base64_image
from starflow.model.vl_api_model import VLAPIModel
import asyncio
import backoff
import os


class OpenRouterAPIModel(VLAPIModel):
    def get_client(self, **kwargs):
        base_url = kwargs["base_url"]
        api_key = os.getenv("OPENROUTER_API_KEY")
        return AsyncOpenAI(base_url=base_url, api_key=api_key)

    def post_init(self, **kwargs):
        model_id = kwargs["model_id"]
        self.model_id = model_id

    def preprocess(self, vl_example: VLExample):
        vl_api_input = []
        for index, (query, annotation) in enumerate(
            zip(vl_example.queries, vl_example.annotations)
        ):
            if index == 0:
                base64_images = [
                    get_base64_image(image, "PNG") for image in vl_example.images
                ]
                vl_api_input.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            }
                            for base64_image in base64_images
                        ]
                        + [{"type": "text", "text": query}],
                    }
                )
            else:
                vl_api_input.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": query}],
                    }
                )
            if index < len(vl_example.queries) - 1:
                vl_api_input.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": annotation}],
                    }
                )
        return vl_api_input

    def generate_inner(self, vl_api_inputs: list[Any], **kwargs):
        @backoff.on_exception(backoff.expo, RateLimitError)
        async def get_prediction(vl_api_input: Any, **kwargs):
            response = await self.client.chat.completions.create(
                model=self.model_id, messages=vl_api_input, **kwargs
            )
            try:
                prediction = response.choices[0].message.content
            except:
                prediction = ""
            return prediction

        async def get_predictions(vl_api_inputs: list[Any], **kwargs):
            predictions = []
            for vl_api_input in vl_api_inputs:
                prediction = get_prediction(vl_api_input, **kwargs)
                predictions.append(prediction)
            return await asyncio.gather(*predictions)

        return asyncio.run(get_predictions(vl_api_inputs, **kwargs))
