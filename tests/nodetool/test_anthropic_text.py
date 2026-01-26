"""
Tests for Anthropic Claude text generation nodes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.anthropic.text import ChatComplete, ClaudeModel


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# ChatComplete Tests


@pytest.mark.asyncio
async def test_chat_complete_success(context, monkeypatch):
    """Test successful chat completion."""
    node = ChatComplete(prompt="Hello, world!", model=ClaudeModel.CLAUDE_3_5_SONNET)

    # Mock the API key retrieval
    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    # Mock the response
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello! How can I help you?"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    # Patch the anthropic import
    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic.return_value = mock_client
    monkeypatch.setattr(
        "nodetool.nodes.anthropic.text.anthropic", mock_anthropic
    )

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

    with pytest.raises(ValueError, match="Anthropic API key not configured"):
        await node.process(context)


@pytest.mark.asyncio
async def test_chat_complete_with_system_prompt(context, monkeypatch):
    """Test chat completion with system prompt."""
    node = ChatComplete(
        prompt="Write a poem",
        system_prompt="You are a creative poet.",
        model=ClaudeModel.CLAUDE_3_5_HAIKU,
    )

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    # Mock the response
    mock_text_block = MagicMock()
    mock_text_block.text = "A beautiful poem"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    mock_client = MagicMock()

    async def mock_create(**kwargs):
        # Verify system prompt is included
        assert kwargs["system"] == "You are a creative poet."
        return mock_response

    mock_client.messages.create = mock_create

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic.return_value = mock_client
    monkeypatch.setattr(
        "nodetool.nodes.anthropic.text.anthropic", mock_anthropic
    )

    result = await node.process(context)
    assert result == "A beautiful poem"


@pytest.mark.asyncio
async def test_chat_complete_empty_response(context, monkeypatch):
    """Test handling of empty response."""
    node = ChatComplete(prompt="Hello", model=ClaudeModel.CLAUDE_3_5_SONNET)

    async def mock_get_secret(key):
        return "test-api-key"

    monkeypatch.setattr(context, "get_secret", mock_get_secret)

    mock_response = MagicMock()
    mock_response.content = []

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic.return_value = mock_client
    monkeypatch.setattr(
        "nodetool.nodes.anthropic.text.anthropic", mock_anthropic
    )

    result = await node.process(context)
    assert result == ""


# Test basic fields


def test_chat_complete_basic_fields():
    """Test ChatComplete basic fields."""
    assert ChatComplete.get_basic_fields() == ["prompt", "model"]


def test_chat_complete_default_values():
    """Test default field values."""
    node = ChatComplete()
    assert node.model == ClaudeModel.CLAUDE_3_5_SONNET
    assert node.prompt == ""
    assert node.system_prompt == ""
    assert node.temperature == 1.0
    assert node.max_tokens == 1024


def test_claude_model_enum_values():
    """Test that all Claude model enum values are valid strings."""
    for model in ClaudeModel:
        assert isinstance(model.value, str)
        assert "claude" in model.value.lower()
