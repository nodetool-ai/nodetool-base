"""
Image Enhance DSL Example

Improve image quality with basic enhancement tools like sharpening, contrast and color adjustment.

Workflow:
1. **Image Input** - Load the source image
2. **Sharpen** - Apply sharpening filter to enhance edges and details
3. **Auto Contrast** - Automatically adjust contrast for optimal visibility
4. **Image Output** - Save the enhanced result
"""

from nodetool.dsl.graph import graph_result
from nodetool.workflows.processing_context import AssetOutputMode
from nodetool.dsl.nodetool.input import ImageInput
from nodetool.dsl.lib.pillow.enhance import Sharpen, AutoContrast
from nodetool.dsl.nodetool.output import ImageOutput
from nodetool.metadata.types import ImageRef


async def example():
    """
    Load an image and enhance it with sharpening and contrast adjustment.
    """
    # Load image from URL
    image_input = ImageInput(
        name="image",
        description="",
        value=ImageRef(
            uri="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Blurry_photo.jpg/1600px-Blurry_photo.jpg?20220511161035",
            type="image",
        ),
    )

    # Apply enhancements in sequence: Sharpen → AutoContrast
    enhanced = AutoContrast(
        image=Sharpen(image=image_input),
        cutoff=108,
    )

    # Output the enhanced image
    output = ImageOutput(
        name="enhanced",
        description="",
        value=enhanced,
    )

    result = await graph_result(output, asset_output_mode=AssetOutputMode.WORKSPACE)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Enhanced image saved: {result}")
