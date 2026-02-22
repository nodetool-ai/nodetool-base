import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.nodes.lib.pillow.enhance import (
    AutoContrast,
    Sharpness,
    Equalize,
    Contrast,
    EdgeEnhance,
    Sharpen,
    RankFilter,
    UnsharpMask,
    Brightness,
    Color,
    Detail,
    AdaptiveContrast,
)


# Create a dummy ImageRef for testing
buffer = BytesIO()
Image.new("RGB", (100, 100), color=(128, 64, 192)).save(buffer, format="PNG")
dummy_image = ImageRef(data=buffer.getvalue())


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_auto_contrast_default(context: ProcessingContext):
    """Test AutoContrast with default settings."""
    node = AutoContrast(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_auto_contrast_with_cutoff(context: ProcessingContext):
    """Test AutoContrast with non-zero cutoff."""
    node = AutoContrast(image=dummy_image, cutoff=5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_sharpness_default(context: ProcessingContext):
    """Test Sharpness with default factor (no change)."""
    node = Sharpness(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_sharpness_increase(context: ProcessingContext):
    """Test Sharpness with factor > 1 (sharper)."""
    node = Sharpness(image=dummy_image, factor=2.0)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_sharpness_decrease(context: ProcessingContext):
    """Test Sharpness with factor < 1 (blurrier)."""
    node = Sharpness(image=dummy_image, factor=0.5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_equalize(context: ProcessingContext):
    """Test Equalize node."""
    node = Equalize(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_contrast_default(context: ProcessingContext):
    """Test Contrast with default factor (no change)."""
    node = Contrast(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_contrast_increase(context: ProcessingContext):
    """Test Contrast with factor > 1."""
    node = Contrast(image=dummy_image, factor=1.5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_contrast_decrease(context: ProcessingContext):
    """Test Contrast with factor < 1."""
    node = Contrast(image=dummy_image, factor=0.5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_edge_enhance(context: ProcessingContext):
    """Test EdgeEnhance node."""
    node = EdgeEnhance(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_sharpen(context: ProcessingContext):
    """Test Sharpen node."""
    node = Sharpen(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_rank_filter_default(context: ProcessingContext):
    """Test RankFilter with default settings."""
    node = RankFilter(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_rank_filter_custom(context: ProcessingContext):
    """Test RankFilter with custom size and rank."""
    node = RankFilter(image=dummy_image, size=5, rank=10)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_unsharp_mask_default(context: ProcessingContext):
    """Test UnsharpMask with default settings."""
    node = UnsharpMask(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_unsharp_mask_custom(context: ProcessingContext):
    """Test UnsharpMask with custom settings."""
    node = UnsharpMask(image=dummy_image, radius=4, percent=200, threshold=5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_brightness_default(context: ProcessingContext):
    """Test Brightness with default factor (no change)."""
    node = Brightness(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_brightness_increase(context: ProcessingContext):
    """Test Brightness with factor > 1 (brighter)."""
    node = Brightness(image=dummy_image, factor=1.5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_brightness_decrease(context: ProcessingContext):
    """Test Brightness with factor < 1 (darker)."""
    node = Brightness(image=dummy_image, factor=0.5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_color_default(context: ProcessingContext):
    """Test Color with default factor (no change)."""
    node = Color(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_color_increase(context: ProcessingContext):
    """Test Color with factor > 1 (more vivid)."""
    node = Color(image=dummy_image, factor=1.5)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_color_decrease(context: ProcessingContext):
    """Test Color with factor < 1 (desaturated)."""
    node = Color(image=dummy_image, factor=0.0)
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_detail(context: ProcessingContext):
    """Test Detail node."""
    node = Detail(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_adaptive_contrast_default(context: ProcessingContext):
    """Test AdaptiveContrast with default settings using mocked cv2."""
    # Build a mock that mimics cv2 behavior
    mock_cv2 = MagicMock()
    arr = np.array(Image.new("RGB", (100, 100), color=(128, 64, 192)))
    mock_cv2.cvtColor.return_value = arr
    mock_cv2.split.return_value = (arr[:, :, 0], arr[:, :, 1], arr[:, :, 2])
    clahe_mock = MagicMock()
    clahe_mock.apply.return_value = arr[:, :, 0]
    mock_cv2.createCLAHE.return_value = clahe_mock
    mock_cv2.merge.return_value = arr

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        node = AdaptiveContrast(image=dummy_image)
        result = await node.process(context)
        assert isinstance(result, ImageRef)
        img = await context.image_to_pil(result)
        assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_adaptive_contrast_custom_params(context: ProcessingContext):
    """Test AdaptiveContrast with custom clip_limit and grid_size."""
    mock_cv2 = MagicMock()
    arr = np.array(Image.new("RGB", (100, 100), color=(100, 100, 100)))
    mock_cv2.cvtColor.return_value = arr
    mock_cv2.split.return_value = (arr[:, :, 0], arr[:, :, 1], arr[:, :, 2])
    clahe_mock = MagicMock()
    clahe_mock.apply.return_value = arr[:, :, 0]
    mock_cv2.createCLAHE.return_value = clahe_mock
    mock_cv2.merge.return_value = arr

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        node = AdaptiveContrast(image=dummy_image, clip_limit=4.0, grid_size=16)
        result = await node.process(context)
        assert isinstance(result, ImageRef)
        mock_cv2.createCLAHE.assert_called_once_with(
            clipLimit=4.0, tileGridSize=(16, 16)
        )
