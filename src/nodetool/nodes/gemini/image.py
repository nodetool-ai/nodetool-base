from pydantic import Field
from enum import Enum
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from google.genai.client import AsyncClient
from google.genai.types import GenerateImagesConfig
from nodetool.config.environment import Environment
from google.genai import Client


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    if not api_key:
        raise ApiKeyMissingError(
            "GEMINI_API_KEY is not configured in the nodetool settings"
        )
    return Client(api_key=api_key).aio


class ImageGenerationModel(str, Enum):
    GEMINI_2_0_FLASH_PREVIEW = "gemini-2.0-flash-preview-image-generation"
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

    _expose_as_tool: bool = True

    prompt: str = Field(
        default="", description="The text prompt describing the image to generate."
    )
    
    model: ImageGenerationModel = Field(
        default=ImageGenerationModel.IMAGEN_3_0_GENERATE_002,
        description="The image generation model to use"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if not self.prompt:
            raise ValueError("The input prompt cannot be empty.")

        response = await get_genai_client().models.generate_images(
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
