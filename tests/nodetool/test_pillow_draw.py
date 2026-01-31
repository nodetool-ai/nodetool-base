import pytest
from io import BytesIO
from PIL import Image
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef, ColorRef
from nodetool.nodes.lib.pillow.draw import (
    Background,
    RenderText,
    GaussianNoise,
)


# Create a dummy ImageRef for testing
buffer = BytesIO()
Image.new("RGB", (100, 100), color="red").save(buffer, format="PNG")
dummy_image = ImageRef(data=buffer.getvalue())


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_background_default(context: ProcessingContext):
    """Test Background node with default settings."""
    node = Background()
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    # Verify the image was created
    img = await context.image_to_pil(result)
    assert img.size == (512, 512)  # default size
    # Check that the image is white (default color)
    assert img.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.asyncio
async def test_background_custom_size(context: ProcessingContext):
    """Test Background node with custom size."""
    node = Background(width=256, height=128)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (256, 128)


@pytest.mark.asyncio
async def test_background_custom_color(context: ProcessingContext):
    """Test Background node with custom color."""
    node = Background(width=100, height=100, color=ColorRef(value="#FF0000"))
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    # Check that the image is red
    assert img.getpixel((50, 50)) == (255, 0, 0)


@pytest.mark.asyncio
async def test_render_text_basic(context: ProcessingContext):
    """Test RenderText node with basic text."""
    node = RenderText(
        text="Hello",
        image=dummy_image,
        x=10,
        y=10,
        size=12,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)  # Same as input image


@pytest.mark.asyncio
async def test_render_text_with_color(context: ProcessingContext):
    """Test RenderText node with custom color."""
    node = RenderText(
        text="Test",
        image=dummy_image,
        x=5,
        y=5,
        size=20,
        color=ColorRef(value="#0000FF"),
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_render_text_alignment(context: ProcessingContext):
    """Test RenderText node with different alignments."""
    for align in [RenderText.TextAlignment.LEFT, RenderText.TextAlignment.CENTER, RenderText.TextAlignment.RIGHT]:
        node = RenderText(
            text="Aligned Text",
            image=dummy_image,
            x=50,
            y=50,
            size=10,
            align=align,
        )
        result = await node.process(context)
        assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_gaussian_noise_default(context: ProcessingContext):
    """Test GaussianNoise node with default settings."""
    node = GaussianNoise()
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (512, 512)  # default size


@pytest.mark.asyncio
async def test_gaussian_noise_custom_size(context: ProcessingContext):
    """Test GaussianNoise node with custom size."""
    node = GaussianNoise(width=256, height=128)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (256, 128)


@pytest.mark.asyncio
async def test_gaussian_noise_parameters(context: ProcessingContext):
    """Test GaussianNoise node with different mean and stddev."""
    node = GaussianNoise(mean=0.5, stddev=0.2, width=100, height=100)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)
