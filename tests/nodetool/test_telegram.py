import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.messaging.telegram import (
    TelegramBotTrigger,
    TelegramSendMessage,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def test_telegram_bot_trigger_instantiation():
    """Test that TelegramBotTrigger can be instantiated."""
    node = TelegramBotTrigger(
        token="test_token",
        chat_id=12345,
    )
    assert node.token == "test_token"
    assert node.chat_id == 12345
    assert node.allow_bot_messages is False
    assert node.include_edited_messages is False


def test_telegram_bot_trigger_default_values():
    """Test TelegramBotTrigger default values."""
    node = TelegramBotTrigger()
    assert node.token == ""
    assert node.chat_id is None
    assert node.poll_timeout_seconds == 30
    assert node.poll_interval_seconds == 0.2


def test_telegram_bot_trigger_build_event():
    """Test TelegramBotTrigger _build_event method."""
    node = TelegramBotTrigger(token="test_token")
    
    message = {
        "message_id": 123,
        "text": "Hello World",
        "date": 1699999999,
        "chat": {
            "id": 12345,
            "type": "private",
            "username": "testuser",
        },
        "from": {
            "id": 67890,
            "username": "sender",
            "first_name": "Test",
            "last_name": "User",
            "is_bot": False,
        },
    }
    
    event = node._build_event(message, update_id=100, update_type="message")
    
    assert event is not None
    assert event["update_id"] == 100
    assert event["update_type"] == "message"
    assert event["message_id"] == 123
    assert event["text"] == "Hello World"
    assert event["chat"]["id"] == 12345
    assert event["from_user"]["username"] == "sender"
    assert event["source"] == "telegram"


def test_telegram_bot_trigger_build_event_filters_bot():
    """Test TelegramBotTrigger filters bot messages by default."""
    node = TelegramBotTrigger(token="test_token", allow_bot_messages=False)
    
    message = {
        "message_id": 123,
        "text": "Bot message",
        "date": 1699999999,
        "chat": {"id": 12345, "type": "private"},
        "from": {"id": 67890, "is_bot": True},
    }
    
    event = node._build_event(message, update_id=100, update_type="message")
    
    # Should return None because sender is a bot
    assert event is None


def test_telegram_bot_trigger_build_event_allows_bot():
    """Test TelegramBotTrigger can allow bot messages."""
    node = TelegramBotTrigger(token="test_token", allow_bot_messages=True)
    
    message = {
        "message_id": 123,
        "text": "Bot message",
        "date": 1699999999,
        "chat": {"id": 12345, "type": "private"},
        "from": {"id": 67890, "is_bot": True},
    }
    
    event = node._build_event(message, update_id=100, update_type="message")
    
    # Should return event because bot messages are allowed
    assert event is not None
    assert event["from_user"]["is_bot"] is True


def test_telegram_bot_trigger_build_event_filters_chat():
    """Test TelegramBotTrigger filters by chat_id."""
    node = TelegramBotTrigger(token="test_token", chat_id=12345)
    
    # Message from a different chat
    message = {
        "message_id": 123,
        "text": "Wrong chat",
        "date": 1699999999,
        "chat": {"id": 99999, "type": "private"},  # Different chat_id
        "from": {"id": 67890, "is_bot": False},
    }
    
    event = node._build_event(message, update_id=100, update_type="message")
    
    # Should return None because chat_id doesn't match
    assert event is None


def test_telegram_bot_trigger_build_event_with_attachments():
    """Test TelegramBotTrigger builds events with photo attachments."""
    node = TelegramBotTrigger(token="test_token")
    
    message = {
        "message_id": 123,
        "caption": "Photo caption",
        "date": 1699999999,
        "chat": {"id": 12345, "type": "private"},
        "from": {"id": 67890, "is_bot": False},
        "photo": [
            {"file_id": "photo1", "width": 100, "height": 100, "file_size": 1000},
            {"file_id": "photo2", "width": 200, "height": 200, "file_size": 2000},
        ],
    }
    
    event = node._build_event(message, update_id=100, update_type="message")
    
    assert event is not None
    assert event["caption"] == "Photo caption"
    assert len(event["attachments"]) == 2
    assert event["attachments"][0]["type"] == "photo"
    assert event["attachments"][0]["file_id"] == "photo1"


def test_telegram_send_message_instantiation():
    """Test that TelegramSendMessage can be instantiated."""
    node = TelegramSendMessage(
        token="test_token",
        chat_id=12345,
        text="Hello World",
    )
    assert node.token == "test_token"
    assert node.chat_id == 12345
    assert node.text == "Hello World"


def test_telegram_send_message_default_values():
    """Test TelegramSendMessage default values."""
    node = TelegramSendMessage()
    assert node.token == ""
    assert node.text == ""
    assert node.parse_mode == ""
    assert node.disable_web_page_preview is False
    assert node.disable_notification is False
    assert node.reply_to_message_id is None


@pytest.mark.asyncio
async def test_telegram_send_message_missing_token(context: ProcessingContext):
    """Test TelegramSendMessage raises error without token."""
    node = TelegramSendMessage(
        token="",
        chat_id=12345,
        text="Hello",
    )
    with pytest.raises(ValueError, match="Telegram bot token is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_telegram_send_message_success(context: ProcessingContext):
    """Test TelegramSendMessage successful API call."""
    node = TelegramSendMessage(
        token="test_token",
        chat_id=12345,
        text="Hello World",
    )
    
    # Mock the aiohttp session
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={
        "ok": True,
        "result": {
            "message_id": 999,
            "date": 1699999999,
            "chat": {"id": 12345},
        },
    })
    
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.post = MagicMock(return_value=mock_response)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await node.process(context)
    
    assert result["message_id"] == 999
    assert result["date"] == 1699999999
    assert result["chat_id"] == 12345
