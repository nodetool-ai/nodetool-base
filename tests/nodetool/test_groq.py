import pytest
from unittest.mock import AsyncMock, MagicMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.groq.text import ChatComplete, GroqModel
from nodetool.nodes.groq.audio import (
    AudioTranscription,
    AudioTranslation,
    WhisperModel,
    get_audio_filename,
)
from nodetool.nodes.groq.vision import ImageToText, VisionModel
from nodetool.metadata.types import AudioRef, ImageRef


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# ChatComplete Tests


@pytest.mark.asyncio
async def test_chat_complete_success(context, monkeypatch):
    """Test successful chat completion."""
    node = ChatComplete(prompt="Hello, world!", model=GroqModel.LLAMA_3_3_70B)

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! How can I help you?"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.text.AsyncGroq", mock_groq)

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

    with pytest.raises(ValueError, match="Groq API key not configured"):
        await node.process(context)


@pytest.mark.asyncio
async def test_chat_complete_with_system_prompt(context, monkeypatch):
    """Test chat completion with system prompt."""
    node = ChatComplete(
        prompt="Write a poem",
        system_prompt="You are a creative poet.",
        model=GroqModel.MIXTRAL_8X7B,
    )

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A beautiful poem"

    mock_client = MagicMock()

    async def mock_create(model, messages, temperature, max_tokens):
        # Verify system prompt is included
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a creative poet."
        return mock_response

    mock_client.chat.completions.create = mock_create

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.text.AsyncGroq", mock_groq)

    result = await node.process(context)
    assert result == "A beautiful poem"


@pytest.mark.asyncio
async def test_chat_complete_no_response(context, monkeypatch):
    """Test handling of empty response."""
    node = ChatComplete(prompt="Hello")

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.choices = []

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.text.AsyncGroq", mock_groq)

    with pytest.raises(ValueError, match="No response received from Groq API"):
        await node.process(context)


# AudioTranscription Tests


@pytest.mark.asyncio
async def test_audio_transcription_success(context, monkeypatch):
    """Test successful audio transcription."""
    mock_audio = MagicMock(spec=AudioRef)
    mock_audio.is_set.return_value = True

    node = AudioTranscription(
        audio=mock_audio,
        model=WhisperModel.WHISPER_LARGE_V3_TURBO,
    )

    async def mock_get_secret(key):
        return "test-api-key"

    async def mock_asset_to_bytes(asset):
        return b"fake audio data"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)
    monkeypatch.setattr(context, "asset_to_bytes", mock_asset_to_bytes)

    # Mock filetype to return a known audio format
    mock_kind = MagicMock()
    mock_kind.mime = "audio/mpeg"
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: mock_kind)

    mock_response = MagicMock()
    mock_response.text = "Hello, this is a transcription."

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.audio.AsyncGroq", mock_groq)

    result = await node.process(context)
    assert result == "Hello, this is a transcription."


@pytest.mark.asyncio
async def test_audio_transcription_missing_audio(context):
    """Test that missing audio raises ValueError."""
    mock_audio = MagicMock(spec=AudioRef)
    mock_audio.is_set.return_value = False

    node = AudioTranscription(audio=mock_audio)
    with pytest.raises(ValueError, match="Audio file is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_audio_transcription_missing_api_key(context, monkeypatch):
    """Test that missing API key raises ValueError."""
    mock_audio = MagicMock(spec=AudioRef)
    mock_audio.is_set.return_value = True

    node = AudioTranscription(audio=mock_audio)

    async def mock_get_secret(key):
        return None

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    with pytest.raises(ValueError, match="Groq API key not configured"):
        await node.process(context)


@pytest.mark.asyncio
async def test_audio_transcription_with_language(context, monkeypatch):
    """Test audio transcription with language specified."""
    mock_audio = MagicMock(spec=AudioRef)
    mock_audio.is_set.return_value = True

    node = AudioTranscription(
        audio=mock_audio,
        language="es",
    )

    async def mock_get_secret(key):
        return "test-api-key"

    async def mock_asset_to_bytes(asset):
        return b"fake audio data"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)
    monkeypatch.setattr(context, "asset_to_bytes", mock_asset_to_bytes)

    # Mock filetype to return a known audio format
    mock_kind = MagicMock()
    mock_kind.mime = "audio/wav"
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: mock_kind)

    mock_response = MagicMock()
    mock_response.text = "Hola, esto es una transcripción."

    mock_client = MagicMock()

    async def mock_create(file, **kwargs):
        assert kwargs.get("language") == "es"
        return mock_response

    mock_client.audio.transcriptions.create = mock_create

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.audio.AsyncGroq", mock_groq)

    result = await node.process(context)
    assert result == "Hola, esto es una transcripción."


# AudioTranslation Tests


@pytest.mark.asyncio
async def test_audio_translation_success(context, monkeypatch):
    """Test successful audio translation."""
    mock_audio = MagicMock(spec=AudioRef)
    mock_audio.is_set.return_value = True

    node = AudioTranslation(
        audio=mock_audio,
        model=WhisperModel.WHISPER_LARGE_V3,
    )

    async def mock_get_secret(key):
        return "test-api-key"

    async def mock_asset_to_bytes(asset):
        return b"fake audio data"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)
    monkeypatch.setattr(context, "asset_to_bytes", mock_asset_to_bytes)

    # Mock filetype to return a known audio format
    mock_kind = MagicMock()
    mock_kind.mime = "audio/ogg"
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: mock_kind)

    mock_response = MagicMock()
    mock_response.text = "Hello, this is translated to English."

    mock_client = MagicMock()
    mock_client.audio.translations.create = AsyncMock(return_value=mock_response)

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.audio.AsyncGroq", mock_groq)

    result = await node.process(context)
    assert result == "Hello, this is translated to English."


@pytest.mark.asyncio
async def test_audio_translation_missing_audio(context):
    """Test that missing audio raises ValueError."""
    mock_audio = MagicMock(spec=AudioRef)
    mock_audio.is_set.return_value = False

    node = AudioTranslation(audio=mock_audio)
    with pytest.raises(ValueError, match="Audio file is required"):
        await node.process(context)


# Vision Tests


@pytest.mark.asyncio
async def test_image_to_text_success(context, monkeypatch):
    """Test successful image analysis."""
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = True

    node = ImageToText(
        image=mock_image,
        prompt="Describe this image",
        model=VisionModel.LLAMA_3_2_11B_VISION,
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
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_groq = MagicMock()
    mock_groq.return_value = mock_client
    monkeypatch.setattr("nodetool.nodes.groq.vision.AsyncGroq", mock_groq)

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


@pytest.mark.asyncio
async def test_image_to_text_missing_api_key(context, monkeypatch):
    """Test that missing API key raises ValueError."""
    mock_image = MagicMock(spec=ImageRef)
    mock_image.is_set.return_value = True

    node = ImageToText(image=mock_image, prompt="Describe this")

    async def mock_get_secret(key):
        return None

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    with pytest.raises(ValueError, match="Groq API key not configured"):
        await node.process(context)


# Test basic fields


def test_chat_complete_basic_fields():
    """Test ChatComplete basic fields."""
    assert ChatComplete.get_basic_fields() == ["prompt", "model"]


def test_audio_transcription_basic_fields():
    """Test AudioTranscription basic fields."""
    assert AudioTranscription.get_basic_fields() == ["audio", "model"]


def test_audio_translation_basic_fields():
    """Test AudioTranslation basic fields."""
    assert AudioTranslation.get_basic_fields() == ["audio", "model"]


def test_image_to_text_basic_fields():
    """Test ImageToText basic fields."""
    assert ImageToText.get_basic_fields() == ["image", "prompt"]


# Test model enums


def test_groq_model_enum():
    """Test GroqModel enum values."""
    assert GroqModel.LLAMA_3_3_70B.value == "llama-3.3-70b-versatile"
    assert GroqModel.MIXTRAL_8X7B.value == "mixtral-8x7b-32768"
    assert GroqModel.GEMMA_2_9B.value == "gemma2-9b-it"


def test_whisper_model_enum():
    """Test WhisperModel enum values."""
    assert WhisperModel.WHISPER_LARGE_V3.value == "whisper-large-v3"
    assert WhisperModel.WHISPER_LARGE_V3_TURBO.value == "whisper-large-v3-turbo"


def test_vision_model_enum():
    """Test VisionModel enum values."""
    assert VisionModel.LLAMA_3_2_11B_VISION.value == "llama-3.2-11b-vision-preview"
    assert VisionModel.LLAMA_3_2_90B_VISION.value == "llama-3.2-90b-vision-preview"


# Test get_audio_filename helper


def test_get_audio_filename_known_format(monkeypatch):
    """Test get_audio_filename with known audio format."""
    mock_kind = MagicMock()
    mock_kind.mime = "audio/wav"
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: mock_kind)

    result = get_audio_filename(b"fake wav data")
    assert result == "audio.wav"


def test_get_audio_filename_mp3(monkeypatch):
    """Test get_audio_filename with MP3 format."""
    mock_kind = MagicMock()
    mock_kind.mime = "audio/mpeg"
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: mock_kind)

    result = get_audio_filename(b"fake mp3 data")
    assert result == "audio.mp3"


def test_get_audio_filename_unknown_format(monkeypatch):
    """Test get_audio_filename with unknown format falls back to mp3."""
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: None)

    result = get_audio_filename(b"unknown data")
    assert result == "audio.mp3"


def test_get_audio_filename_unsupported_mime(monkeypatch):
    """Test get_audio_filename with unsupported MIME type falls back to mp3."""
    mock_kind = MagicMock()
    mock_kind.mime = "audio/unknown"
    monkeypatch.setattr("nodetool.nodes.groq.audio.filetype.guess", lambda x: mock_kind)

    result = get_audio_filename(b"fake data")
    assert result == "audio.mp3"
