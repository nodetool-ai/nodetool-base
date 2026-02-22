"""
Tests for the PaddleOCR node.

PaddleOCR is mocked to avoid model downloads during testing.
"""
import sys
import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef, OCRResult
from nodetool.nodes.lib.ocr import PaddleOCRNode, OCRLanguage


# Create a dummy ImageRef for testing
buffer = BytesIO()
Image.new("RGB", (100, 100), color=(255, 255, 255)).save(buffer, format="PNG")
dummy_image = ImageRef(data=buffer.getvalue())


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def _mock_paddleocr_module(ocr_result):
    """
    Return a mock ``paddleocr`` module whose ``PaddleOCR`` class produces
    an instance whose ``.ocr()`` returns *ocr_result*.
    """
    mock_instance = MagicMock()
    mock_instance.ocr.return_value = ocr_result

    mock_class = MagicMock(return_value=mock_instance)

    mock_module = MagicMock()
    mock_module.PaddleOCR = mock_class

    return mock_module, mock_class, mock_instance


@pytest.mark.asyncio
async def test_paddleocr_node_initialize(context: ProcessingContext):
    """Test that initialize creates a PaddleOCR instance."""
    mock_module, mock_class, _ = _mock_paddleocr_module([])

    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        node = PaddleOCRNode(image=dummy_image, language=OCRLanguage.ENGLISH)
        await node.initialize(context)

    mock_class.assert_called_once_with(lang=OCRLanguage.ENGLISH.value)
    assert node._ocr is not None


@pytest.mark.asyncio
async def test_paddleocr_node_no_text(context: ProcessingContext):
    """Test PaddleOCRNode returns empty result when OCR finds nothing."""
    mock_module, _, _ = _mock_paddleocr_module(None)

    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        node = PaddleOCRNode(image=dummy_image)
        await node.initialize(context)
        result = await node.process(context)

    assert result["boxes"] == []
    assert result["text"] == ""


@pytest.mark.asyncio
async def test_paddleocr_node_with_text(context: ProcessingContext):
    """Test PaddleOCRNode extracts text and bounding boxes correctly."""
    box = [[10.0, 20.0], [90.0, 20.0], [90.0, 40.0], [10.0, 40.0]]
    ocr_result = [
        [
            [box, ("Hello World", 0.99)],
        ]
    ]
    mock_module, _, _ = _mock_paddleocr_module(ocr_result)

    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        node = PaddleOCRNode(image=dummy_image)
        await node.initialize(context)
        result = await node.process(context)

    assert len(result["boxes"]) == 1
    ocr_box: OCRResult = result["boxes"][0]
    assert ocr_box.text == "Hello World"
    assert abs(ocr_box.score - 0.99) < 1e-6
    assert ocr_box.top_left == [10.0, 20.0]
    assert ocr_box.top_right == [90.0, 20.0]
    assert ocr_box.bottom_right == [90.0, 40.0]
    assert ocr_box.bottom_left == [10.0, 40.0]
    assert result["text"] == "Hello World"


@pytest.mark.asyncio
async def test_paddleocr_node_multiple_lines(context: ProcessingContext):
    """Test PaddleOCRNode handles multiple text lines."""
    box1 = [[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]]
    box2 = [[0.0, 15.0], [80.0, 15.0], [80.0, 25.0], [0.0, 25.0]]
    ocr_result = [
        [
            [box1, ("Line one", 0.95)],
            [box2, ("Line two", 0.90)],
        ]
    ]
    mock_module, _, _ = _mock_paddleocr_module(ocr_result)

    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        node = PaddleOCRNode(image=dummy_image)
        await node.initialize(context)
        result = await node.process(context)

    assert len(result["boxes"]) == 2
    assert result["text"] == "Line one\nLine two"


@pytest.mark.asyncio
async def test_paddleocr_node_empty_lines_skipped(context: ProcessingContext):
    """Test that None or empty entries in OCR result are skipped."""
    # Mix of None and empty lists which should be skipped
    ocr_result = [[None, []]]
    mock_module, _, _ = _mock_paddleocr_module(ocr_result)

    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        node = PaddleOCRNode(image=dummy_image)
        await node.initialize(context)
        result = await node.process(context)

    assert isinstance(result["boxes"], list)
    assert result["text"] == ""


@pytest.mark.asyncio
async def test_paddleocr_node_language_chinese(context: ProcessingContext):
    """Test PaddleOCRNode initializes with Chinese language."""
    mock_module, mock_class, _ = _mock_paddleocr_module(None)

    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        node = PaddleOCRNode(image=dummy_image, language=OCRLanguage.CHINESE)
        await node.initialize(context)

    mock_class.assert_called_once_with(lang=OCRLanguage.CHINESE.value)


def test_ocr_language_enum_values():
    """Test that key OCRLanguage enum values are correct."""
    assert OCRLanguage.ENGLISH.value == "en"
    assert OCRLanguage.CHINESE.value == "ch"
    assert OCRLanguage.JAPANESE.value == "ja"
    assert OCRLanguage.KOREAN.value == "ko"
    assert OCRLanguage.FRENCH.value == "fr"
    assert OCRLanguage.GERMAN.value == "de"
    assert OCRLanguage.ARABIC.value == "ar"


def test_paddleocr_required_inputs():
    """Test that 'image' is listed as a required input."""
    node = PaddleOCRNode()
    assert "image" in node.required_inputs()

