from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class VisionModel(str, Enum):
    """Available Mistral AI vision models."""

    PIXTRAL_LARGE = "pixtral-large-latest"
    PIXTRAL_12B = "pixtral-12b-2409"


class ImageToText(BaseNode):
    """
    Analyze images and generate text descriptions using Mistral AI's Pixtral models.
    mistral, pixtral, vision, image, ocr, analysis, multimodal

    Uses Mistral AI's Pixtral vision models to understand and describe images.
    Can perform OCR, image analysis, and answer questions about images.
    Requires a Mistral API key.

    Use cases:
    - Extract text from images (OCR)
    - Describe image contents
    - Answer questions about images
    - Analyze charts and diagrams
    - Document understanding
    """

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["MISTRAL_API_KEY"]

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to analyze",
    )

    prompt: str = Field(
        default="Describe this image in detail.",
        description="The prompt/question about the image",
    )

    model: VisionModel = Field(
        default=VisionModel.PIXTRAL_LARGE,
        description="The Pixtral model to use for vision tasks",
    )

    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for response generation",
    )

    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=16384,
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

        api_key = await context.get_secret("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not configured")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": await context.image_ref_to_data_uri(self.image)},
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        response = await client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response or not response.choices:
            raise ValueError("No response received from Mistral API")

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "prompt"]


class OCR(BaseNode):
    """
    Extract text from images using Mistral AI's Pixtral models.
    mistral, pixtral, ocr, text extraction, document, image

    Specialized node for optical character recognition (OCR) using Pixtral.
    Optimized for extracting text content from documents, screenshots, and images.
    Requires a Mistral API key.

    Use cases:
    - Extract text from scanned documents
    - Read text from screenshots
    - Digitize printed materials
    - Extract data from forms and receipts
    """

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["MISTRAL_API_KEY"]

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to extract text from",
    )

    model: VisionModel = Field(
        default=VisionModel.PIXTRAL_LARGE,
        description="The Pixtral model to use for OCR",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Extract text from an image.

        Args:
            context: The processing context.

        Returns:
            str: The extracted text from the image.
        """
        if not self.image.is_set():
            raise ValueError("Image is required")

        api_key = await context.get_secret("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not configured")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": await context.image_ref_to_data_uri(self.image)},
                    },
                    {
                        "type": "text",
                        "text": "Extract and return all text visible in this image. "
                        "Preserve the original formatting and structure as much as possible. "
                        "Return only the extracted text without any additional commentary.",
                    },
                ],
            }
        ]

        response = await client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            temperature=0.0,  # Use low temperature for accurate extraction
            max_tokens=8192,
        )

        if not response or not response.choices:
            raise ValueError("No response received from Mistral API")

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image"]
