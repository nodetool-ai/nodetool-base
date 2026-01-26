import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.mistral.text import ChatComplete, CodeComplete, MistralModel
from nodetool.nodes.mistral.embeddings import Embedding
from nodetool.nodes.mistral.vision import ImageToText, OCR, VisionModel
from nodetool.metadata.types import NPArray, ImageRef


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def mock_mistral_client(monkeypatch):
    """Create a mock Mistral client."""
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.Mistral.return_value = mock_client
    monkeypatch.setitem(__import__("sys").modules, "mistralai", mock_module)
    return mock_client, mock_module


# Chat Complete Tests


@pytest.mark.asyncio
async def test_chat_complete_success(context, monkeypatch):
    """Test successful chat completion."""
    node = ChatComplete(prompt="Hello, world!", model=MistralModel.MISTRAL_SMALL)

    # Mock the API key retrieval
    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    # Mock the Mistral client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! How can I help you?"

    mock_client = MagicMock()
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    # Patch the Mistral import
    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.mistral.text.Mistral", mock_mistral.Mistral)

    result = await node.process(context)
    assert result == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_chat_complete_empty_prompt(context):
    """Test that empty prompt raises ValueError."""
    node = ChatComplete(prompt="")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_chat_complete_missing_api_key(context, monkeypatch):
    """Test that missing API key raises ValueError."""
    node = ChatComplete(prompt="Hello")

    async def mock_get_secret(key):
        return None

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    with pytest.raises(ValueError, match="Mistral API key not configured"):
        await node.process(context)


@pytest.mark.asyncio
async def test_chat_complete_with_system_prompt(context, monkeypatch):
    """Test chat completion with system prompt."""
    node = ChatComplete(
        prompt="Write a poem",
        system_prompt="You are a creative poet.",
        model=MistralModel.MISTRAL_LARGE,
    )

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A beautiful poem"

    mock_client = MagicMock()

    async def mock_complete_async(model, messages, temperature, max_tokens):
        # Verify system prompt is included
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a creative poet."
        return mock_response

    mock_client.chat.complete_async = mock_complete_async

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.mistral.text.Mistral", mock_mistral.Mistral)

    result = await node.process(context)
    assert result == "A beautiful poem"


# Code Complete Tests


@pytest.mark.asyncio
async def test_code_complete_success(context, monkeypatch):
    """Test successful code completion."""
    node = CodeComplete(prompt="def fibonacci(n):")

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"

    mock_client = MagicMock()
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.mistral.text.Mistral", mock_mistral.Mistral)

    result = await node.process(context)
    assert "fibonacci" in result


@pytest.mark.asyncio
async def test_code_complete_with_suffix(context, monkeypatch):
    """Test code completion with fill-in-the-middle."""
    node = CodeComplete(
        prompt="def add(a, b):\n    ",
        suffix="\n    return result",
    )

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "result = a + b"

    mock_client = MagicMock()
    mock_client.fim.complete_async = AsyncMock(return_value=mock_response)

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.mistral.text.Mistral", mock_mistral.Mistral)

    result = await node.process(context)
    assert result == "result = a + b"


@pytest.mark.asyncio
async def test_code_complete_empty_prompt(context):
    """Test that empty prompt raises ValueError."""
    node = CodeComplete(prompt="")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await node.process(context)


# Embedding Tests


@pytest.mark.asyncio
async def test_embedding_success(context, monkeypatch):
    """Test successful embedding generation."""
    node = Embedding(input="Hello, world!", chunk_size=4096)

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4]

    mock_client = MagicMock()
    mock_client.embeddings.create_async = AsyncMock(return_value=mock_response)

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr(
        "nodetool.nodes.mistral.embeddings.Mistral", mock_mistral.Mistral
    )

    result = await node.process(context)
    assert isinstance(result, NPArray)
    np.testing.assert_allclose(result.to_numpy(), np.array([0.1, 0.2, 0.3, 0.4]))


@pytest.mark.asyncio
async def test_embedding_chunking(context, monkeypatch):
    """Test embedding with text chunking."""
    node = Embedding(input="hello world", chunk_size=5)

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
        MagicMock(embedding=[0.5, 0.6]),
    ]

    mock_client = MagicMock()

    async def mock_create_async(model, inputs):
        # Verify chunking
        assert inputs == ["hello", " worl", "d"]
        return mock_response

    mock_client.embeddings.create_async = mock_create_async

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr(
        "nodetool.nodes.mistral.embeddings.Mistral", mock_mistral.Mistral
    )

    result = await node.process(context)
    assert isinstance(result, NPArray)
    # Average of [0.1, 0.2], [0.3, 0.4], [0.5, 0.6] = [0.3, 0.4]
    np.testing.assert_allclose(result.to_numpy(), np.array([0.3, 0.4]))


@pytest.mark.asyncio
async def test_embedding_empty_input(context):
    """Test that empty input raises ValueError."""
    node = Embedding(input="")
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        await node.process(context)


# Vision Tests


@pytest.mark.asyncio
async def test_image_to_text_success(context, monkeypatch):
    """Test successful image analysis."""
    # Create a mock ImageRef that is set
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = True

    node = ImageToText(
        image=mock_image,
        prompt="Describe this image",
        model=VisionModel.PIXTRAL_LARGE,
    )

    async def mock_get_secret(key):
        return "test-api-key"

    async def mock_image_to_base64_url(image):
        return "data:image/png;base64,abc123"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)
    monkeypatch.setattr(context, "image_to_base64_url", mock_image_to_base64_url)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A beautiful sunset over the ocean"

    mock_client = MagicMock()
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.mistral.vision.Mistral", mock_mistral.Mistral)

    result = await node.process(context)
    assert result == "A beautiful sunset over the ocean"


@pytest.mark.asyncio
async def test_image_to_text_missing_image(context):
    """Test that missing image raises ValueError."""
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = False

    node = ImageToText(image=mock_image, prompt="Describe this")
    with pytest.raises(ValueError, match="Image is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_image_to_text_empty_prompt(context):
    """Test that empty prompt raises ValueError."""
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = True

    node = ImageToText(image=mock_image, prompt="")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await node.process(context)


# OCR Tests


@pytest.mark.asyncio
async def test_ocr_success(context, monkeypatch):
    """Test successful OCR."""
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = True

    node = OCR(image=mock_image)

    async def mock_get_secret(key):
        return "test-api-key"

    async def mock_image_to_base64_url(image):
        return "data:image/png;base64,abc123"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)
    monkeypatch.setattr(context, "image_to_base64_url", mock_image_to_base64_url)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello World\nLine 2\nLine 3"

    mock_client = MagicMock()
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    mock_mistral = MagicMock()
    mock_mistral.Mistral.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.mistral.vision.Mistral", mock_mistral.Mistral)

    result = await node.process(context)
    assert result == "Hello World\nLine 2\nLine 3"


@pytest.mark.asyncio
async def test_ocr_missing_image(context):
    """Test that missing image raises ValueError."""
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = False

    node = OCR(image=mock_image)
    with pytest.raises(ValueError, match="Image is required"):
        await node.process(context)


# Test basic fields


def test_chat_complete_basic_fields():
    """Test ChatComplete basic fields."""
    assert ChatComplete.get_basic_fields() == ["prompt", "model"]


def test_code_complete_basic_fields():
    """Test CodeComplete basic fields."""
    assert CodeComplete.get_basic_fields() == ["prompt", "suffix"]


def test_embedding_basic_fields():
    """Test Embedding basic fields."""
    assert Embedding.get_basic_fields() == ["input"]


def test_image_to_text_basic_fields():
    """Test ImageToText basic fields."""
    assert ImageToText.get_basic_fields() == ["image", "prompt"]


def test_ocr_basic_fields():
    """Test OCR basic fields."""
    assert OCR.get_basic_fields() == ["image"]
