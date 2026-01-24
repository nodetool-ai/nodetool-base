from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import Field

from nodetool.metadata.types import ColorRef, FontRef, ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    pass


class Background(BaseNode):
    """
    The Background Node creates a blank background.
    image, background, blank, base, layer
    This node is mainly used for generating a base layer for image processing tasks. It produces a uniform image, having a user-specified width, height and color. The color is given in a hexadecimal format, defaulting to white if not specified.

    #### Applications
    - As a base layer for creating composite images.
    - As a starting point for generating patterns or graphics.
    - When blank backgrounds of specific colors are required for visualization tasks.
    """
    _auto_save_asset: ClassVar[bool] = True

    width: int = Field(default=512, ge=1, le=4096)
    height: int = Field(default=512, ge=1, le=4096)
    color: ColorRef = Field(default=ColorRef(value="#FFFFFF"))

    async def process(self, context: ProcessingContext) -> ImageRef:
        import PIL.Image

        img = PIL.Image.new("RGB", (self.width, self.height), self.color.value)
        return await context.image_from_pil(img)


class RenderText(BaseNode):
    """
    This node allows you to add text to images using system fonts or web fonts.
    text, font, label, title, watermark, caption, image, overlay, google fonts

    This node takes text, font updates, coordinates (where to place the text), and an image to work with.
    A user can use the Render Text Node to add a label or title to an image, watermark an image,
    or place a caption directly on an image.

    The Render Text Node offers customizable options, including the ability to choose the text's font,
    size, color, and alignment (left, center, or right). Text placement can also be defined,
    providing flexibility to place the text wherever you see fit.

    ### Font Sources

    The node supports three font sources:

    1. **System Fonts** (default): Use fonts installed on the system
       - `FontRef(name="Arial")` - Uses local Arial font

    2. **Google Fonts**: Automatically download and cache fonts from Google Fonts
       - `FontRef(name="Roboto", source=FontSource.GOOGLE_FONTS)`
       - `FontRef(name="Open Sans", source=FontSource.GOOGLE_FONTS, weight="bold")`
       - Supports 50+ popular fonts including Roboto, Open Sans, Lato, Montserrat, Poppins, etc.

    3. **Custom URL**: Download fonts from any URL
       - `FontRef(name="CustomFont", source=FontSource.URL, url="https://example.com/font.ttf")`

    #### Applications
    - Labeling images in an image gallery or database.
    - Watermarking images for copyright protection.
    - Adding custom captions to photographs.
    - Creating instructional images to guide the reader's view.
    - Using premium Google Fonts for professional typography.
    """
    _auto_save_asset: ClassVar[bool] = True

    class TextAlignment(str, Enum):
        LEFT = "left"
        CENTER = "center"
        RIGHT = "right"

    text: str = Field(default="", description="The text to render.")
    font: FontRef = Field(
        default=FontRef(name="DejaVuSans"),
        description="The font to use. Supports system fonts, Google Fonts, and custom URLs.",
    )
    x: int = Field(default=0, description="The x coordinate.")
    y: int = Field(default=0, description="The y coordinate.")
    size: int = Field(default=12, ge=1, le=512, description="The font size.")
    color: ColorRef = Field(
        default=ColorRef(value="#000000"), description="The font color."
    )
    align: TextAlignment = TextAlignment.LEFT
    image: ImageRef = Field(default=ImageRef(), description="The image to render on.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        import PIL.ImageDraw
        import PIL.ImageFont

        image = await context.image_to_pil(self.image)
        draw = PIL.ImageDraw.Draw(image)

        # Use get_font_path which handles system fonts, Google Fonts, and URL fonts
        font_path = context.get_font_path(self.font)
        font = PIL.ImageFont.truetype(font_path, self.size)

        draw.text(
            (self.x, self.y),
            self.text,
            font=font,
            fill=self.color.value,
            align=self.align.value,
        )
        return await context.image_from_pil(image)


class GaussianNoise(BaseNode):
    """
    This node creates and adds Gaussian noise to an image.
    image, noise, gaussian, distortion, artifact

    The Gaussian Noise Node is designed to simulate realistic distortions that can occur in a photographic image. It generates a noise-filled image using the Gaussian (normal) distribution. The noise level can be adjusted using the mean and standard deviation parameters.

    #### Applications
    - Simulating sensor noise in synthetic data.
    - Testing image-processing algorithms' resilience to noise.
    - Creating artistic effects in images.
    """
    _auto_save_asset: ClassVar[bool] = True

    mean: float = Field(default=0.0)
    stddev: float = Field(default=1.0)
    width: int = Field(default=512, ge=1, le=1024)
    height: int = Field(default=512, ge=1, le=1024)

    async def process(self, context: ProcessingContext) -> ImageRef:
        import PIL.Image

        image = np.random.normal(self.mean, self.stddev, (self.height, self.width, 3))
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        image = PIL.Image.fromarray(image)
        return await context.image_from_pil(image)
