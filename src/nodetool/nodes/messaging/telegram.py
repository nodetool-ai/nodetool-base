"""
Telegram messaging nodes.

Provides a trigger node to listen for Telegram bot messages via long polling
and a node to send messages to a chat.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, ClassVar, TypedDict

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

from nodetool.nodes.nodetool.triggers import TriggerNode

log = get_logger(__name__)


TelegramBotTriggerOutput = TypedDict(
    "TelegramBotTriggerOutput",
    {
        "update_id": int | None,
        "update_type": str,
        "message_id": int | None,
        "text": str,
        "caption": str,
        "entities": list[dict[str, Any]],
        "chat": dict[str, Any],
        "from_user": dict[str, Any],
        "attachments": list[dict[str, Any]],
        "timestamp": str,
        "source": str,
        "event_type": str,
    },
)


class TelegramBotTrigger(TriggerNode[TelegramBotTriggerOutput]):
    """
    Trigger node that listens for Telegram messages using long polling.

    This trigger connects to Telegram using a bot token and emits events
    for incoming messages.
    """

    token: str = Field(
        default="",
        description="Telegram bot token",
    )
    chat_id: int | None = Field(
        default=None,
        description="Optional chat ID to filter messages",
    )
    allow_bot_messages: bool = Field(
        default=False,
        description="Include messages authored by bots",
    )
    include_edited_messages: bool = Field(
        default=False,
        description="Include edited messages",
    )
    poll_timeout_seconds: int = Field(
        default=30,
        description="Long polling timeout in seconds",
        ge=1,
        le=60,
    )
    poll_interval_seconds: float = Field(
        default=0.2,
        description="Delay between polling requests",
        ge=0,
    )
    OutputType: ClassVar[type] = TelegramBotTriggerOutput

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._poll_task: asyncio.Task | None = None
        self._session = None
        self._offset = 0

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Start polling Telegram for updates."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for TelegramBotTrigger. "
                "Install it with: pip install aiohttp"
            )

        if not self.token:
            raise ValueError("Telegram bot token is required")

        self._session = aiohttp.ClientSession()
        self._poll_task = asyncio.create_task(self._poll_updates())

    async def wait_for_event(
        self, context: ProcessingContext
    ) -> TelegramBotTriggerOutput | None:
        """Wait for the next Telegram message event."""
        return await self.get_event_from_queue()

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Stop polling Telegram for updates."""
        log.info("Cleaning up Telegram bot trigger")

        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.warning(f"Error stopping Telegram polling task: {e}")

        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                log.warning(f"Error closing Telegram session: {e}")

    async def _poll_updates(self) -> None:
        assert self._session is not None

        base_url = f"https://api.telegram.org/bot{self.token}/getUpdates"

        while self._is_running:
            params = {
                "timeout": self.poll_timeout_seconds,
                "offset": self._offset,
                "allowed_updates": ["message", "edited_message"],
            }

            try:
                async with self._session.get(base_url, params=params) as response:
                    if response.status != 200:
                        log.warning(
                            f"Telegram getUpdates returned status {response.status}"
                        )
                        await asyncio.sleep(self.poll_interval_seconds)
                        continue

                    payload = await response.json()

                if not payload.get("ok"):
                    log.warning(f"Telegram getUpdates error: {payload}")
                    await asyncio.sleep(self.poll_interval_seconds)
                    continue

                for update in payload.get("result", []):
                    update_id = update.get("update_id")
                    if update_id is not None:
                        self._offset = update_id + 1

                    message = update.get("message")
                    update_type = "message"
                    if message is None and self.include_edited_messages:
                        message = update.get("edited_message")
                        update_type = "edited_message"

                    if message is None:
                        continue

                    event = self._build_event(message, update_id, update_type)
                    if event is not None:
                        self.push_event(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Telegram polling error: {e}")

            if self.poll_interval_seconds:
                await asyncio.sleep(self.poll_interval_seconds)

    def _build_event(
        self, message: dict[str, Any], update_id: int | None, update_type: str
    ) -> TelegramBotTriggerOutput | None:
        chat = message.get("chat", {})
        sender = message.get("from", {})

        if not self.allow_bot_messages and sender.get("is_bot"):
            return None

        if self.chat_id is not None and chat.get("id") != self.chat_id:
            return None

        attachments = []
        if "photo" in message:
            attachments.extend(
                {
                    "type": "photo",
                    "file_id": photo.get("file_id"),
                    "width": photo.get("width"),
                    "height": photo.get("height"),
                    "file_size": photo.get("file_size"),
                }
                for photo in message.get("photo", [])
            )
        if "document" in message:
            document = message.get("document", {})
            attachments.append(
                {
                    "type": "document",
                    "file_id": document.get("file_id"),
                    "file_name": document.get("file_name"),
                    "mime_type": document.get("mime_type"),
                    "file_size": document.get("file_size"),
                }
            )

        timestamp = message.get("date")
        if timestamp is not None:
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "update_id": update_id,
            "update_type": update_type,
            "message_id": message.get("message_id"),
            "text": message.get("text", ""),
            "caption": message.get("caption", ""),
            "entities": message.get("entities", []),
            "chat": {
                "id": chat.get("id"),
                "type": chat.get("type"),
                "title": chat.get("title"),
                "username": chat.get("username"),
            },
            "from_user": {
                "id": sender.get("id"),
                "username": sender.get("username"),
                "first_name": sender.get("first_name"),
                "last_name": sender.get("last_name"),
                "is_bot": sender.get("is_bot"),
            },
            "attachments": attachments,
            "timestamp": timestamp,
            "source": "telegram",
            "event_type": update_type,
        }


class TelegramSendMessage(BaseNode):
    """
    Node that sends a message to a Telegram chat using a bot token.
    """

    token: str = Field(
        default="",
        description="Telegram bot token",
    )
    chat_id: int = Field(
        default=0,
        description="Target chat ID",
        ge=1,
    )
    text: str = Field(
        default="",
        description="Message text",
    )
    parse_mode: str = Field(
        default="",
        description="Optional parse mode (MarkdownV2 or HTML)",
    )
    disable_web_page_preview: bool = Field(
        default=False,
        description="Disable link previews",
    )
    disable_notification: bool = Field(
        default=False,
        description="Send silently",
    )
    reply_to_message_id: int | None = Field(
        default=None,
        description="Reply to a specific message ID",
        ge=1,
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for TelegramSendMessage. "
                "Install it with: pip install aiohttp"
            )

        if not self.token:
            raise ValueError("Telegram bot token is required")

        if not self.chat_id:
            raise ValueError("Telegram chat ID is required")

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": self.text,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
        }

        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode
        if self.reply_to_message_id is not None:
            payload["reply_to_message_id"] = self.reply_to_message_id

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()

        if not data.get("ok"):
            raise RuntimeError(f"Telegram sendMessage failed: {data}")

        result = data.get("result", {})
        return {
            "message_id": result.get("message_id"),
            "date": result.get("date"),
            "chat_id": result.get("chat", {}).get("id"),
        }


__all__ = [
    "TelegramBotTrigger",
    "TelegramSendMessage",
]
