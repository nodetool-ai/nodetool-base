import pytest
from unittest.mock import MagicMock, AsyncMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.nodes.nodetool.compare import CompareImages


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_compare_images_get_title():
    """Test CompareImages get_title class method."""
    assert CompareImages.get_title() == "Compare Images"


@pytest.mark.asyncio
async def test_compare_images_is_cacheable():
    """Test CompareImages is_cacheable class method."""
    assert CompareImages.is_cacheable() is False


@pytest.mark.asyncio
async def test_compare_images_get_basic_fields():
    """Test CompareImages get_basic_fields class method."""
    fields = CompareImages.get_basic_fields()
    assert "image_a" in fields
    assert "image_b" in fields


@pytest.mark.asyncio
async def test_compare_images_empty_image_a(context: ProcessingContext):
    """Test CompareImages returns early when image_a is empty."""
    node = CompareImages(
        image_a=ImageRef(),
        image_b=ImageRef(uri="http://example.com/image.png"),
    )
    result = await node.process(context)
    assert result is None


@pytest.mark.asyncio
async def test_compare_images_empty_image_b(context: ProcessingContext):
    """Test CompareImages returns early when image_b is empty."""
    node = CompareImages(
        image_a=ImageRef(uri="http://example.com/image.png"),
        image_b=ImageRef(),
    )
    result = await node.process(context)
    assert result is None


@pytest.mark.asyncio
async def test_compare_images_both_empty(context: ProcessingContext):
    """Test CompareImages returns early when both images are empty."""
    node = CompareImages(
        image_a=ImageRef(),
        image_b=ImageRef(),
    )
    result = await node.process(context)
    assert result is None


@pytest.mark.asyncio
async def test_compare_images_with_valid_images(context: ProcessingContext):
    """Test CompareImages sends preview update with valid images."""
    # Create a mock for context methods
    mock_context = MagicMock(spec=ProcessingContext)
    mock_context.normalize_output_value = AsyncMock(side_effect=lambda x: x)
    mock_context.post_message = MagicMock()

    node = CompareImages(
        image_a=ImageRef(uri="http://example.com/image_a.png"),
        image_b=ImageRef(uri="http://example.com/image_b.png"),
        label_a="Before",
        label_b="After",
    )
    node._id = "test-node-id"  # Set node ID

    await node.process(mock_context)

    # Verify post_message was called
    mock_context.post_message.assert_called_once()

    # Verify the preview update content
    call_args = mock_context.post_message.call_args[0][0]
    assert call_args.node_id == "test-node-id"
    assert call_args.value["type"] == "image_comparison"
    assert call_args.value["label_a"] == "Before"
    assert call_args.value["label_b"] == "After"


@pytest.mark.asyncio
async def test_compare_images_default_labels(context: ProcessingContext):
    """Test CompareImages uses default labels."""
    node = CompareImages(
        image_a=ImageRef(uri="http://example.com/image_a.png"),
        image_b=ImageRef(uri="http://example.com/image_b.png"),
    )
    assert node.label_a == "A"
    assert node.label_b == "B"
