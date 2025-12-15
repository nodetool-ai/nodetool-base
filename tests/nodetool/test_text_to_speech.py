"""
Tests for the TextToSpeech node.
"""

import pytest
import numpy as np
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.audio import TextToSpeech
from nodetool.metadata.types import TTSModel, Provider, AudioRef
from unittest.mock import AsyncMock, MagicMock


async def async_generator(items):
    """Helper to create async generator from list."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_text_to_speech_basic():
    """Test basic TextToSpeech node functionality."""

    # Create a mock processing context
    context = MagicMock(spec=ProcessingContext)

    # Mock audio_from_numpy to return an AudioRef
    mock_audio_ref = AudioRef(uri="test://audio.mp3")
    context.audio_from_numpy = AsyncMock(return_value=mock_audio_ref)

    # Create the TextToSpeech node
    node = TextToSpeech(
        model=TTSModel(
            provider=Provider.OpenAI,
            id="tts-1",
            name="TTS 1",
            voices=["alloy", "echo"],
        ),
        text="Hello, world!",
        speed=1.0,
    )

    # Mock the TTS provider - now yields numpy arrays
    mock_provider = MagicMock()
    fake_audio_chunk = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    mock_provider.text_to_speech = MagicMock(
        return_value=async_generator([fake_audio_chunk])
    )

    # Mock context.get_provider to return our mock provider
    context.get_provider = AsyncMock(return_value=mock_provider)

    # Process the node - now uses gen_process
    results = []
    async for result in node.gen_process(context):
        results.append(result)

    # Should have 2 results: chunk output and final audio
    assert len(results) == 2

    # First result is chunk
    assert results[0]["chunk"].content_type == "audio"
    assert results[0]["chunk"].done is False
    assert results[0]["audio"] is None

    # Second result is final audio
    assert results[1]["chunk"].done is True
    assert isinstance(results[1]["audio"], AudioRef)
    assert results[1]["audio"].uri == "test://audio.mp3"

    # Verify the provider was called correctly
    mock_provider.text_to_speech.assert_called_once_with(
        text="Hello, world!",
        model="tts-1",
        voice="alloy",
        speed=1.0,
        context=context,
    )

    # Verify audio_from_numpy was called
    context.audio_from_numpy.assert_called_once()


@pytest.mark.asyncio
async def test_text_to_speech_default_voice():
    """Test TextToSpeech with default voice selection."""

    context = MagicMock(spec=ProcessingContext)
    mock_audio_ref = AudioRef(uri="test://audio.mp3")
    context.audio_from_numpy = AsyncMock(return_value=mock_audio_ref)

    # Create node without specifying voice
    node = TextToSpeech(
        model=TTSModel(
            provider=Provider.OpenAI,
            id="tts-1",
            name="TTS 1",
            voices=["alloy", "echo", "fable"],
        ),
        text="Test text",
        speed=1.0,
    )

    mock_provider = MagicMock()
    fake_audio_chunk = np.array([1, 2, 3], dtype=np.int16)
    mock_provider.text_to_speech = MagicMock(
        return_value=async_generator([fake_audio_chunk])
    )

    # Mock context.get_provider to return our mock provider
    context.get_provider = AsyncMock(return_value=mock_provider)

    results = []
    async for result in node.gen_process(context):
        results.append(result)

    # Should use first voice from model
    mock_provider.text_to_speech.assert_called_once_with(
        text="Test text",
        model="tts-1",
        voice="alloy",  # First voice from the list
        speed=1.0,
        context=context,
    )


@pytest.mark.asyncio
async def test_text_to_speech_different_speed():
    """Test TextToSpeech with different speed settings."""

    context = MagicMock(spec=ProcessingContext)
    mock_audio_ref = AudioRef(uri="test://audio.mp3")
    context.audio_from_numpy = AsyncMock(return_value=mock_audio_ref)

    node = TextToSpeech(
        model=TTSModel(
            provider=Provider.OpenAI,
            id="tts-1-hd",
            name="TTS 1 HD",
            voices=["nova"],
        ),
        text="Fast speech",
        speed=2.0,
    )

    mock_provider = MagicMock()
    fake_audio_chunk = np.array([1, 2, 3], dtype=np.int16)
    mock_provider.text_to_speech = MagicMock(
        return_value=async_generator([fake_audio_chunk])
    )

    # Mock context.get_provider to return our mock provider
    context.get_provider = AsyncMock(return_value=mock_provider)

    results = []
    async for result in node.gen_process(context):
        results.append(result)

    mock_provider.text_to_speech.assert_called_once_with(
        text="Fast speech",
        model="tts-1-hd",
        voice="nova",
        speed=2.0,
        context=context,
    )


def test_text_to_speech_get_basic_fields():
    """Test that get_basic_fields returns expected fields."""
    basic_fields = TextToSpeech.get_basic_fields()
    assert basic_fields == ["model", "text", "voice", "speed"]
