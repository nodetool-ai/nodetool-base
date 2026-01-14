"""
Telegram Bot API integration nodes for Nodetool.
Provides nodes for sending messages, photos, documents, and managing bot operations.
"""

from typing import Any, ClassVar, Optional
from pydantic import Field
import aiohttp

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef


class SendMessage(BaseNode):
    """
    Send a text message to a Telegram chat.
    telegram, message, send, chat, bot

    Use cases:
    - Send notifications to a Telegram channel
    - Reply to user messages
    - Send automated alerts
    - Broadcast messages to subscribers
    """

    bot_token: str = Field(
        default="",
        description="Telegram Bot API token obtained from @BotFather",
    )
    chat_id: str = Field(
        default="",
        description="Unique identifier for the target chat or username of the target channel",
    )
    text: str = Field(
        default="",
        description="Text of the message to be sent (1-4096 characters)",
    )
    parse_mode: Optional[str] = Field(
        default=None,
        description="Mode for parsing entities in the message text (Markdown, MarkdownV2, or HTML)",
    )
    disable_notification: bool = Field(
        default=False,
        description="Sends the message silently (users will receive a notification with no sound)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """
        Send a text message via Telegram Bot API.

        Returns:
            dict: Response from Telegram API including message_id and other details
        """
        if not self.bot_token:
            raise ValueError("Bot token is required")
        if not self.chat_id:
            raise ValueError("Chat ID is required")
        if not self.text:
            raise ValueError("Message text is required")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        payload = {
            "chat_id": self.chat_id,
            "text": self.text,
            "disable_notification": self.disable_notification,
        }

        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Failed to send message: {response.status} - {error_text}"
                    )
                result = await response.json()
                if not result.get("ok"):
                    raise ValueError(f"Telegram API error: {result.get('description')}")
                return result.get("result", {})


class SendPhoto(BaseNode):
    """
    Send a photo to a Telegram chat.
    telegram, photo, image, send, bot, media

    Use cases:
    - Share generated images with users
    - Send visual reports or charts
    - Distribute photos to a channel
    - Send thumbnails or previews
    """

    bot_token: str = Field(
        default="",
        description="Telegram Bot API token obtained from @BotFather",
    )
    chat_id: str = Field(
        default="",
        description="Unique identifier for the target chat or username of the target channel",
    )
    photo: ImageRef = Field(
        default=ImageRef(),
        description="Photo to send (ImageRef from a previous node)",
    )
    caption: Optional[str] = Field(
        default=None,
        description="Photo caption (0-1024 characters after parsing)",
    )
    parse_mode: Optional[str] = Field(
        default=None,
        description="Mode for parsing entities in the caption (Markdown, MarkdownV2, or HTML)",
    )
    disable_notification: bool = Field(
        default=False,
        description="Sends the message silently",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """
        Send a photo via Telegram Bot API.

        Returns:
            dict: Response from Telegram API including message details
        """
        if not self.bot_token:
            raise ValueError("Bot token is required")
        if not self.chat_id:
            raise ValueError("Chat ID is required")
        if not self.photo or not self.photo.asset_id:
            raise ValueError("Photo is required")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

        # Get photo bytes from context
        photo_bytes = await context.asset_to_bytes(self.photo)

        # Prepare form data
        data = aiohttp.FormData()
        data.add_field("chat_id", self.chat_id)
        data.add_field(
            "photo",
            photo_bytes,
            filename="photo.jpg",
            content_type="image/jpeg",
        )
        data.add_field("disable_notification", str(self.disable_notification).lower())

        if self.caption:
            data.add_field("caption", self.caption)
        if self.parse_mode:
            data.add_field("parse_mode", self.parse_mode)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Failed to send photo: {response.status} - {error_text}"
                    )
                result = await response.json()
                if not result.get("ok"):
                    raise ValueError(f"Telegram API error: {result.get('description')}")
                return result.get("result", {})


class SendDocument(BaseNode):
    """
    Send a document/file to a Telegram chat.
    telegram, document, file, send, bot, media

    Use cases:
    - Send generated reports or PDFs
    - Share data files with users
    - Distribute documents to a channel
    - Send any file type to users
    """

    bot_token: str = Field(
        default="",
        description="Telegram Bot API token obtained from @BotFather",
    )
    chat_id: str = Field(
        default="",
        description="Unique identifier for the target chat or username of the target channel",
    )
    document_url: str = Field(
        default="",
        description="URL of the document to send",
    )
    caption: Optional[str] = Field(
        default=None,
        description="Document caption (0-1024 characters after parsing)",
    )
    parse_mode: Optional[str] = Field(
        default=None,
        description="Mode for parsing entities in the caption",
    )
    disable_notification: bool = Field(
        default=False,
        description="Sends the message silently",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """
        Send a document via Telegram Bot API.

        Returns:
            dict: Response from Telegram API including message details
        """
        if not self.bot_token:
            raise ValueError("Bot token is required")
        if not self.chat_id:
            raise ValueError("Chat ID is required")
        if not self.document_url:
            raise ValueError("Document URL is required")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"

        payload = {
            "chat_id": self.chat_id,
            "document": self.document_url,
            "disable_notification": self.disable_notification,
        }

        if self.caption:
            payload["caption"] = self.caption
        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Failed to send document: {response.status} - {error_text}"
                    )
                result = await response.json()
                if not result.get("ok"):
                    raise ValueError(f"Telegram API error: {result.get('description')}")
                return result.get("result", {})


class GetUpdates(BaseNode):
    """
    Retrieve incoming updates from a Telegram bot.
    telegram, updates, messages, receive, bot, polling

    Use cases:
    - Get new messages sent to the bot
    - Retrieve user commands
    - Monitor bot interactions
    - Implement bot conversation flows
    """

    bot_token: str = Field(
        default="",
        description="Telegram Bot API token obtained from @BotFather",
    )
    offset: Optional[int] = Field(
        default=None,
        description="Identifier of the first update to be returned",
    )
    limit: int = Field(
        default=100,
        description="Limits the number of updates to be retrieved (1-100)",
    )
    timeout: int = Field(
        default=0,
        description="Timeout in seconds for long polling (0 for short polling)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        """
        Retrieve updates from Telegram Bot API.

        Returns:
            list: List of Update objects from Telegram API
        """
        if not self.bot_token:
            raise ValueError("Bot token is required")

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"

        payload = {
            "limit": max(1, min(100, self.limit)),
            "timeout": self.timeout,
        }

        if self.offset is not None:
            payload["offset"] = self.offset

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Failed to get updates: {response.status} - {error_text}"
                    )
                result = await response.json()
                if not result.get("ok"):
                    raise ValueError(f"Telegram API error: {result.get('description')}")
                return result.get("result", [])


class GetMe(BaseNode):
    """
    Get basic information about the bot.
    telegram, bot, info, status, getme

    Use cases:
    - Verify bot credentials
    - Get bot username and ID
    - Check if bot token is valid
    - Display bot information
    """

    bot_token: str = Field(
        default="",
        description="Telegram Bot API token obtained from @BotFather",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """
        Get bot information from Telegram Bot API.

        Returns:
            dict: Bot information including id, username, first_name, etc.
        """
        if not self.bot_token:
            raise ValueError("Bot token is required")

        url = f"https://api.telegram.org/bot{self.bot_token}/getMe"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Failed to get bot info: {response.status} - {error_text}"
                    )
                result = await response.json()
                if not result.get("ok"):
                    raise ValueError(f"Telegram API error: {result.get('description')}")
                return result.get("result", {})
