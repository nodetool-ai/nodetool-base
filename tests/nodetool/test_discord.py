import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.messaging.discord import (
    DiscordBotTrigger,
    DiscordSendMessage,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def test_discord_bot_trigger_instantiation():
    """Test that DiscordBotTrigger can be instantiated."""
    node = DiscordBotTrigger(
        token="test_token",
        channel_id="123456789",
    )
    assert node.token == "test_token"
    assert node.channel_id == "123456789"
    assert node.allow_bot_messages is False


def test_discord_bot_trigger_default_values():
    """Test DiscordBotTrigger default values."""
    node = DiscordBotTrigger()
    assert node.token == ""
    assert node.channel_id is None
    assert node.allow_bot_messages is False


def test_discord_bot_trigger_allows_bot_messages():
    """Test DiscordBotTrigger with allow_bot_messages enabled."""
    node = DiscordBotTrigger(
        token="test_token",
        allow_bot_messages=True,
    )
    assert node.allow_bot_messages is True


def test_discord_send_message_instantiation():
    """Test that DiscordSendMessage can be instantiated."""
    node = DiscordSendMessage(
        token="test_token",
        channel_id="123456789",
        content="Hello Discord!",
    )
    assert node.token == "test_token"
    assert node.channel_id == "123456789"
    assert node.content == "Hello Discord!"


def test_discord_send_message_default_values():
    """Test DiscordSendMessage default values."""
    node = DiscordSendMessage()
    assert node.token == ""
    assert node.channel_id == ""
    assert node.content == ""
    assert node.tts is False
    assert node.embeds == []


def test_discord_send_message_with_tts():
    """Test DiscordSendMessage with TTS enabled."""
    node = DiscordSendMessage(
        token="test_token",
        channel_id="123456789",
        content="Hello Discord!",
        tts=True,
    )
    assert node.tts is True


def test_discord_send_message_with_embeds():
    """Test DiscordSendMessage with embeds."""
    embeds = [
        {"title": "Test Embed", "description": "This is a test embed"},
    ]
    node = DiscordSendMessage(
        token="test_token",
        channel_id="123456789",
        content="Hello Discord!",
        embeds=embeds,
    )
    assert len(node.embeds) == 1
    assert node.embeds[0]["title"] == "Test Embed"


@pytest.mark.asyncio
async def test_discord_send_message_missing_token(context: ProcessingContext):
    """Test DiscordSendMessage raises error without token."""
    node = DiscordSendMessage(
        token="",
        channel_id="123456789",
        content="Hello",
    )
    with pytest.raises(ValueError, match="Discord bot token is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_discord_send_message_missing_channel_id(context: ProcessingContext):
    """Test DiscordSendMessage raises error without channel_id."""
    node = DiscordSendMessage(
        token="test_token",
        channel_id="",
        content="Hello",
    )
    with pytest.raises(ValueError, match="Discord channel ID is required"):
        await node.process(context)
