import pytest
from io import BytesIO
from PIL import Image
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.nodes.lib.pillow.filter import (
    Invert,
    Solarize,
    Posterize,
    Expand,
    Blur,
    Contour,
    Emboss,
    FindEdges,
    Smooth,
    ConvertToGrayscale,
    GetChannel,
)


# Create a dummy ImageRef for testing
buffer = BytesIO()
Image.new("RGB", (100, 100), color=(128, 64, 192)).save(buffer, format="PNG")
dummy_image = ImageRef(data=buffer.getvalue())


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_invert(context: ProcessingContext):
    """Test Invert filter node."""
    node = Invert(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)
    # Inverted pixel should be (255-128, 255-64, 255-192) = (127, 191, 63)
    pixel = img.getpixel((50, 50))
    assert pixel == (127, 191, 63)


@pytest.mark.asyncio
async def test_solarize_default(context: ProcessingContext):
    """Test Solarize filter with default threshold."""
    node = Solarize(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_solarize_custom_threshold(context: ProcessingContext):
    """Test Solarize filter with custom threshold."""
    node = Solarize(image=dummy_image, threshold=200)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_posterize_default(context: ProcessingContext):
    """Test Posterize filter with default bits."""
    node = Posterize(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_posterize_custom_bits(context: ProcessingContext):
    """Test Posterize filter with custom bits."""
    node = Posterize(image=dummy_image, bits=2)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_expand_default(context: ProcessingContext):
    """Test Expand filter with default settings (no border)."""
    node = Expand(image=dummy_image, border=0)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)  # No expansion


@pytest.mark.asyncio
async def test_expand_with_border(context: ProcessingContext):
    """Test Expand filter with border."""
    node = Expand(image=dummy_image, border=10, fill=255)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (120, 120)  # 100 + 10*2


@pytest.mark.asyncio
async def test_blur(context: ProcessingContext):
    """Test Blur filter."""
    node = Blur(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_contour(context: ProcessingContext):
    """Test Contour filter."""
    node = Contour(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_emboss(context: ProcessingContext):
    """Test Emboss filter."""
    node = Emboss(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_find_edges(context: ProcessingContext):
    """Test FindEdges filter."""
    node = FindEdges(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_smooth(context: ProcessingContext):
    """Test Smooth filter."""
    node = Smooth(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_convert_to_grayscale(context: ProcessingContext):
    """Test ConvertToGrayscale filter."""
    node = ConvertToGrayscale(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)
    # Result should be grayscale (mode L) or converted back to RGB
    assert img.mode in ("L", "RGB")


@pytest.mark.asyncio
async def test_get_channel_red(context: ProcessingContext):
    """Test GetChannel filter for red channel."""
    node = GetChannel(image=dummy_image, channel=GetChannel.ChannelEnum.RED)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_get_channel_green(context: ProcessingContext):
    """Test GetChannel filter for green channel."""
    node = GetChannel(image=dummy_image, channel=GetChannel.ChannelEnum.GREEN)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_get_channel_blue(context: ProcessingContext):
    """Test GetChannel filter for blue channel."""
    node = GetChannel(image=dummy_image, channel=GetChannel.ChannelEnum.BLUE)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
