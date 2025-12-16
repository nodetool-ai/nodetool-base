from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef, Provider
from nodetool.providers.gemini_provider import GeminiProvider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

log = get_logger(__name__)


class ImageGenerationModel(str, Enum):
    GEMINI_2_0_FLASH_PREVIEW = "gemini-2.0-flash-preview-image-generation"
    GEMINI_2_5_FLASH_IMAGE_PREVIEW = "gemini-2.5-flash-image-preview"
    IMAGEN_3_0_GENERATE_001 = "imagen-3.0-generate-001"
    IMAGEN_3_0_GENERATE_002 = "imagen-3.0-generate-002"
    IMAGEN_4_0_GENERATE_PREVIEW = "imagen-4.0-generate-preview-06-06"
    IMAGEN_4_0_ULTRA_GENERATE_PREVIEW = "imagen-4.0-ultra-generate-preview-06-06"


class ImageGeneration(BaseNode):
    """
    Generate an image using Google's Imagen model via the Gemini API.
    google, image generation, ai, imagen

    Use cases:
    - Create images from text descriptions
    - Generate assets for creative projects
    - Explore AI-powered image synthesis
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="", description="The text prompt describing the image to generate."
    )

    model: ImageGenerationModel = Field(
        default=ImageGenerationModel.IMAGEN_3_0_GENERATE_002,
        description="The image generation model to use",
    )

    image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a base for the generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        from google.genai.types import (
            FinishReason,
            GenerateContentConfig,
            GenerateImagesConfig,
        )

        if not self.prompt:
            raise ValueError("The input prompt cannot be empty.")

        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]

        # If a Gemini image-capable model is selected, use the IMAGE+TEXT API
        if self.model.value.startswith("gemini-"):
            log.info(f"Using Gemini image-capable model: {self.model.value}")
            # Build contents with optional image if provided
            contents: list = [self.prompt]
            if isinstance(self.image, ImageRef) and self.image.is_set():
                # Convert ImageRef to PIL for the Gemini client
                pil_image = await context.image_to_pil(self.image)
                log.info(f"Image converted to PIL: {pil_image}")
                contents.append(pil_image)

            response = await client.models.generate_content(
                model=self.model.value,
                contents=contents if len(contents) > 1 else contents[0],
                config=GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            log.debug(f"Gemini API response: {response}")

            # Extract first inline image from response parts
            if not response or not response.candidates:
                log.error("No response received from Gemini API")
                raise ValueError("No response received from Gemini API")

            candidate = response.candidates[0]

            if candidate.finish_reason == FinishReason.PROHIBITED_CONTENT:
                log.error("Prohibited content in the input prompt")
                raise ValueError("Prohibited content in the input prompt")

            if not candidate or not candidate.content or not candidate.content.parts:
                log.error("Invalid response format from Gemini API")
                raise ValueError("Invalid response format from Gemini API")

            image_bytes = None
            for part in candidate.content.parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    image_bytes = inline_data.data
                    break

            assert image_bytes, "No image bytes returned in response"
            return await context.image_from_bytes(image_bytes)

        # Otherwise, fallback to the images generation API (Imagen models)
        response = await client.models.generate_images(
            model=self.model.value,
            prompt=self.prompt,
            config=GenerateImagesConfig(
                number_of_images=1,
            ),
        )
        assert response.generated_images, "No images generated"
        image = response.generated_images[0].image
        assert image, "No image"
        assert image.image_bytes, "No image bytes"
        return await context.image_from_bytes(image.image_bytes)
