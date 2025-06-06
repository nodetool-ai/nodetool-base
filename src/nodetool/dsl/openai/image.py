from pydantic import Field
import typing
import nodetool.metadata.types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.openai.image


class CreateImage(GraphNode):
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

    Model: typing.ClassVar[type] = nodetool.nodes.openai.image.CreateImage.Model
    Size: typing.ClassVar[type] = nodetool.nodes.openai.image.CreateImage.Size
    Background: typing.ClassVar[type] = (
        nodetool.nodes.openai.image.CreateImage.Background
    )
    Quality: typing.ClassVar[type] = nodetool.nodes.openai.image.CreateImage.Quality
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt to use."
    )
    model: nodetool.nodes.openai.image.CreateImage.Model = Field(
        default=nodetool.nodes.openai.image.CreateImage.Model.GPT_IMAGE_1,
        description="The model to use for image generation.",
    )
    size: nodetool.nodes.openai.image.CreateImage.Size = Field(
        default=nodetool.nodes.openai.image.CreateImage.Size._1024x1024,
        description="The size of the image to generate.",
    )
    background: nodetool.nodes.openai.image.CreateImage.Background = Field(
        default=nodetool.nodes.openai.image.CreateImage.Background.auto,
        description="The background of the image to generate.",
    )
    quality: nodetool.nodes.openai.image.CreateImage.Quality = Field(
        default=nodetool.nodes.openai.image.CreateImage.Quality.high,
        description="The quality of the image to generate.",
    )

    @classmethod
    def get_node_type(cls):
        return "openai.image.CreateImage"
