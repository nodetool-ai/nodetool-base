from pydantic import Field
from nodetool.dsl.graph import GraphNode


class ImageGeneration(GraphNode):
    """
    Generate an image using Google's Imagen model via the Gemini API.
    google, image generation, ai, imagen

    Use cases:
    - Create images from text descriptions
    - Generate assets for creative projects
    - Explore AI-powered image synthesis
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text prompt describing the image to generate."
    )

    @classmethod
    def get_node_type(cls):
        return "google.image_generation.ImageGeneration"
