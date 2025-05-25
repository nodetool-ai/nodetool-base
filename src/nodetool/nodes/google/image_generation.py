from pydantic import Field
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from google.genai.client import AsyncClient
from google.genai.types import GenerateImagesConfig
from nodetool.common.environment import Environment
from google.genai import Client


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY is not set"
    return Client(api_key=api_key).aio


class ImageGeneration(BaseNode):
    """
    Generate an image using Google's Imagen model via the Gemini API.
    google, image generation, ai, imagen

    Use cases:
    - Create images from text descriptions
    - Generate assets for creative projects
    - Explore AI-powered image synthesis
    """

    prompt: str = Field(
        default="", description="The text prompt describing the image to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if not self.prompt:
            raise ValueError("The input prompt cannot be empty.")

        response = await get_genai_client().models.generate_images(
            model="imagen-3.0-generate-002",
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
