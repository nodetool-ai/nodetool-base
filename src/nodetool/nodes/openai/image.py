from base64 import b64decode
import PIL.Image
from io import BytesIO
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef, Provider
from nodetool.workflows.base_node import BaseNode
from pydantic import Field
from enum import Enum

from openai.types.images_response import ImagesResponse
from nodetool.chat.providers.openai_prediction import run_openai


class CreateImage(BaseNode):
    """
    Generates images from textual descriptions.
    image, t2i, tti, text-to-image, create, generate, picture, photo, art, drawing, illustration

    Use cases:
    1. Create custom illustrations for articles or presentations
    2. Generate concept art for creative projects
    3. Produce visual aids for educational content
    4. Design unique marketing visuals or product mockups
    5. Explore artistic ideas and styles programmatically
    """

    class Size(str, Enum):
        _1024x1024 = "1024x1024"
        _1536x1024 = "1536x1024"
        _1024x1536 = "1024x1536"

    class Background(str, Enum):
        transparent = "transparent"
        opaque = "opaque"
        auto = "auto"

    class Quality(str, Enum):
        high = "high"
        medium = "medium"
        low = "low"

    class Model(str, Enum):
        GPT_IMAGE_1 = "gpt-image-1"

    prompt: str = Field(default="", description="The prompt to use.")
    model: Model = Field(
        default=Model.GPT_IMAGE_1, description="The model to use for image generation."
    )
    size: Size = Field(
        default=Size._1024x1024, description="The size of the image to generate."
    )
    background: Background = Field(
        default=Background.auto, description="The background of the image to generate."
    )
    quality: Quality = Field(
        default=Quality.high, description="The quality of the image to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        params = {
            "prompt": self.prompt,
            "n": 1,
            "size": self.size.value,
            "quality": self.quality.value,
            "background": self.background.value,
        }

        response = await context.run_prediction(
            node_id=self._id,
            provider=Provider.OpenAI,
            model=self.model.value,
            run_prediction_function=run_openai,
            params=params,
        )

        pil_image = PIL.Image.open(BytesIO(response))
        return await context.image_from_pil(pil_image)
