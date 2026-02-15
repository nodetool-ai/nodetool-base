from io import BytesIO
from enum import Enum
from typing import ClassVar

from nodetool.metadata.types import ImageRef, Provider
from nodetool.providers.openai_prediction import run_openai
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class ImageModel(str, Enum):
    """Available OpenAI image models."""

    GPT_IMAGE_1 = "gpt-image-1"


class ImageSize(str, Enum):
    """Available image sizes."""

    _1024x1024 = "1024x1024"
    _1536x1024 = "1536x1024"
    _1024x1536 = "1024x1536"


class ImageBackground(str, Enum):
    """Available background options."""

    transparent = "transparent"
    opaque = "opaque"
    auto = "auto"


class ImageQuality(str, Enum):
    """Available quality levels."""

    high = "high"
    medium = "medium"
    low = "low"


class CreateImage(BaseNode):
    """
    Generates images from textual descriptions.
    image, t2i, tti, text-to-image, create, generate, picture, photo, art, drawing, illustration
    """

    prompt: str = Field(default="", description="The prompt to use.")
    model: ImageModel = Field(
        default=ImageModel.GPT_IMAGE_1,
        description="The model to use for image generation.",
    )
    size: ImageSize = Field(
        default=ImageSize._1024x1024, description="The size of the image to generate."
    )
    background: ImageBackground = Field(
        default=ImageBackground.auto,
        description="The background of the image to generate.",
    )
    quality: ImageQuality = Field(
        default=ImageQuality.high, description="The quality of the image to generate."
    )

    _auto_save_asset: ClassVar[bool] = True
    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["OPENAI_API_KEY"]

    async def process(self, context: ProcessingContext) -> ImageRef:
        import PIL.Image

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


class EditImage(BaseNode):
    """
    Edit images using OpenAI's gpt-image-1 model.
    image, edit, modify, transform, inpaint, outpaint, variation

    Takes an input image and a text prompt to generate a modified version.
    Can be used for inpainting, outpainting, style transfer, and image modification.
    Optionally accepts a mask to specify which areas to edit.
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to edit.",
    )
    mask: ImageRef = Field(
        default=ImageRef(),
        description="Optional mask image. White areas will be edited, black areas preserved.",
    )
    prompt: str = Field(
        default="",
        description="The prompt describing the desired edit.",
    )
    model: ImageModel = Field(
        default=ImageModel.GPT_IMAGE_1,
        description="The model to use for image editing.",
    )
    size: ImageSize = Field(
        default=ImageSize._1024x1024,
        description="The size of the output image.",
    )
    quality: ImageQuality = Field(
        default=ImageQuality.high,
        description="The quality of the generated image.",
    )

    _auto_save_asset: ClassVar[bool] = True
    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["OPENAI_API_KEY"]

    async def process(self, context: ProcessingContext) -> ImageRef:
        import PIL.Image

        from nodetool.metadata.types import ImageModel as ImageModelType
        from nodetool.providers.openai_provider import OpenAIProvider
        from nodetool.providers.types import ImageToImageParams

        if not self.prompt:
            raise ValueError("Edit prompt cannot be empty")

        if not self.image.is_set():
            raise ValueError("Input image is required")

        provider = await context.get_provider(Provider.OpenAI)
        assert isinstance(provider, OpenAIProvider)

        image_bytes = await context.asset_to_bytes(self.image)
        mask_bytes = (
            await context.asset_to_bytes(self.mask) if self.mask.is_set() else None
        )

        params = ImageToImageParams(
            model=ImageModelType(provider=Provider.OpenAI, id=self.model.value),
            prompt=self.prompt,
            target_width=int(self.size.value.split("x")[0]),
            target_height=int(self.size.value.split("x")[1]),
        )

        result_bytes = await provider.image_to_image(
            image=image_bytes,
            params=params,
            mask=mask_bytes,
        )

        pil_image = PIL.Image.open(BytesIO(result_bytes))
        return await context.image_from_pil(pil_image)
