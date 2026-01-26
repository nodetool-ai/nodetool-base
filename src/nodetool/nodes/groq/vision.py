from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class VisionModel(str, Enum):
    """Available Groq vision models."""

    LLAMA_3_2_11B_VISION = "llama-3.2-11b-vision-preview"
    LLAMA_3_2_90B_VISION = "llama-3.2-90b-vision-preview"


class ImageToText(BaseNode):
    """
    Analyze images and generate text descriptions using Groq's vision models.
    groq, vision, image, analysis, multimodal, llama, fast

    Uses Groq's LPU inference with Llama vision models for extremely fast
    image understanding and description. Supports questions about images.
    Requires a Groq API key.

    Use cases:
    - Fast image description and captioning
    - Visual question answering
    - Image content analysis
    - Accessibility descriptions
    - Visual data extraction
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to analyze",
    )

    prompt: str = Field(
        default="Describe this image in detail.",
        description="The prompt/question about the image",
    )

    model: VisionModel = Field(
        default=VisionModel.LLAMA_3_2_11B_VISION,
        description="The vision model to use",
    )

    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation",
    )

    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Analyze an image and generate a text response.

        Args:
            context: The processing context.

        Returns:
            str: The generated text describing or analyzing the image.
        """
        if not self.image.is_set():
            raise ValueError("Image is required")

        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        api_key = await context.get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not configured")

        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)

        # Convert image to base64 data URL
        image_url = await context.image_to_base64_url(self.image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        response = await client.chat.completions.create(
            model=self.model.value,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response or not response.choices:
            raise ValueError("No response received from Groq API")

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "prompt"]
