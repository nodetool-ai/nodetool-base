import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.telegram.telegram import (
    SendMessage,
    SendPhoto,
    SendDocument,
    GetUpdates,
    GetMe,
)
from nodetool.metadata.types import ImageRef


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def mock_image_ref():
    """Create a mock ImageRef for testing."""
    return ImageRef(asset_id="test_image_id", uri="test://image.jpg")


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message_validation(self, context):
        """Test that SendMessage validates required fields."""
        node = SendMessage(bot_token="", chat_id="", text="")

        with pytest.raises(ValueError, match="Bot token is required"):
            await node.process(context)

        node = SendMessage(bot_token="test_token", chat_id="", text="")
        with pytest.raises(ValueError, match="Chat ID is required"):
            await node.process(context)

        node = SendMessage(bot_token="test_token", chat_id="12345", text="")
        with pytest.raises(ValueError, match="Message text is required"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_send_message_success(self, context):
        """Test successful message sending."""
        node = SendMessage(
            bot_token="test_token",
            chat_id="12345",
            text="Hello, World!",
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "ok": True,
                "result": {
                    "message_id": 1,
                    "chat": {"id": 12345},
                    "text": "Hello, World!",
                },
            }
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await node.process(context)

            assert result["message_id"] == 1
            assert result["text"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_send_message_api_error(self, context):
        """Test handling of Telegram API errors."""
        node = SendMessage(
            bot_token="test_token",
            chat_id="12345",
            text="Hello!",
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": False, "description": "Bad Request: chat not found"}
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(ValueError, match="Telegram API error"):
                await node.process(context)


class TestSendPhoto:
    @pytest.mark.asyncio
    async def test_send_photo_validation(self, context):
        """Test that SendPhoto validates required fields."""
        node = SendPhoto(bot_token="", chat_id="", photo=ImageRef())

        with pytest.raises(ValueError, match="Bot token is required"):
            await node.process(context)

        node = SendPhoto(bot_token="test_token", chat_id="", photo=ImageRef())
        with pytest.raises(ValueError, match="Chat ID is required"):
            await node.process(context)

        node = SendPhoto(bot_token="test_token", chat_id="12345", photo=ImageRef())
        with pytest.raises(ValueError, match="Photo is required"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_send_photo_success(self, context, mock_image_ref):
        """Test successful photo sending."""
        node = SendPhoto(
            bot_token="test_token",
            chat_id="12345",
            photo=mock_image_ref,
            caption="Test photo",
        )

        # Mock context.asset_to_bytes
        context.asset_to_bytes = AsyncMock(return_value=b"fake_image_data")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "ok": True,
                "result": {
                    "message_id": 2,
                    "chat": {"id": 12345},
                    "photo": [{"file_id": "test_file_id"}],
                    "caption": "Test photo",
                },
            }
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await node.process(context)

            assert result["message_id"] == 2
            assert result["caption"] == "Test photo"


class TestSendDocument:
    @pytest.mark.asyncio
    async def test_send_document_validation(self, context):
        """Test that SendDocument validates required fields."""
        node = SendDocument(bot_token="", chat_id="", document_url="")

        with pytest.raises(ValueError, match="Bot token is required"):
            await node.process(context)

        node = SendDocument(bot_token="test_token", chat_id="", document_url="")
        with pytest.raises(ValueError, match="Chat ID is required"):
            await node.process(context)

        node = SendDocument(bot_token="test_token", chat_id="12345", document_url="")
        with pytest.raises(ValueError, match="Document URL is required"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_send_document_success(self, context):
        """Test successful document sending."""
        node = SendDocument(
            bot_token="test_token",
            chat_id="12345",
            document_url="https://example.com/document.pdf",
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "ok": True,
                "result": {
                    "message_id": 3,
                    "chat": {"id": 12345},
                    "document": {"file_id": "test_doc_id"},
                },
            }
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await node.process(context)

            assert result["message_id"] == 3


class TestGetUpdates:
    @pytest.mark.asyncio
    async def test_get_updates_validation(self, context):
        """Test that GetUpdates validates required fields."""
        node = GetUpdates(bot_token="")

        with pytest.raises(ValueError, match="Bot token is required"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_get_updates_success(self, context):
        """Test successful updates retrieval."""
        node = GetUpdates(
            bot_token="test_token",
            limit=10,
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "ok": True,
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "message_id": 1,
                            "text": "Hello bot!",
                        },
                    }
                ],
            }
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await node.process(context)

            assert len(result) == 1
            assert result[0]["update_id"] == 1


class TestGetMe:
    @pytest.mark.asyncio
    async def test_get_me_validation(self, context):
        """Test that GetMe validates required fields."""
        node = GetMe(bot_token="")

        with pytest.raises(ValueError, match="Bot token is required"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_get_me_success(self, context):
        """Test successful bot info retrieval."""
        node = GetMe(bot_token="test_token")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "ok": True,
                "result": {
                    "id": 123456789,
                    "is_bot": True,
                    "first_name": "TestBot",
                    "username": "test_bot",
                },
            }
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            result = await node.process(context)

            assert result["id"] == 123456789
            assert result["username"] == "test_bot"
            assert result["is_bot"] is True
