"""
Discord messaging nodes.

Provides a trigger node to listen for Discord bot messages and a node to send
messages to a channel.

How to create a Discord bot:
1. Go to https://discord.com/developers/applications and create a new application.
2. Open the application, then go to the Bot section and click \"Add Bot\".
3. Copy the bot token and use it in the node's token field.
4. Enable Message Content Intent if you want to receive message text.
5. Invite the bot to your server using the OAuth2 URL generator with bot scopes.
6. Right-click the target channel in Discord, copy its ID, and use it as channel_id.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, TypedDict

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

from nodetool.nodes.nodetool.triggers import TriggerNode

log = get_logger(__name__)


class DiscordBotTriggerOutput(TypedDict):
    message_id: int
    content: str
    author: dict[str, Any]
    channel: dict[str, Any]
    guild: dict[str, Any] | None
    attachments: list[dict[str, Any]]
    timestamp: str
    source: str
    event_type: str


class DiscordBotTrigger(TriggerNode[DiscordBotTriggerOutput]):
    """
    Trigger node that listens for Discord messages from a bot account.

    This trigger connects to Discord using a bot token and emits events
    for incoming messages.
    """

    token: str = Field(
        default="",
        description="Discord bot token",
    )
    channel_id: str | None = Field(
        default=None,
        description="Optional channel ID to filter messages",
    )
    allow_bot_messages: bool = Field(
        default=False,
        description="Include messages authored by bots",
    )
    OutputType: ClassVar[type] = DiscordBotTriggerOutput

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = None
        self._client_task: asyncio.Task | None = None

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Connect the Discord bot and start listening for messages."""
        try:
            import discord  # type: ignore
        except ImportError:
            raise ImportError(
                "discord.py is required for DiscordBotTrigger. "
                "Install it with: pip install discord.py"
            )

        if not self.token:
            raise ValueError("Discord bot token is required")

        # Ensure the event loop is captured for thread-safe event pushing
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            log.debug(f"Captured event loop in setup_trigger: {self._loop}")

        log.debug(f"Discord trigger setup, queue id: {id(self._event_queue)}")

        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.guilds = True

        trigger = self

        class TriggerClient(discord.Client):
            async def on_ready(self) -> None:
                log.info(f"Discord bot connected as {self.user}")

            async def on_message(self, message: discord.Message) -> None:
                log.info(f"Received message from {message.author}: {message.content}")
                if not trigger.allow_bot_messages and message.author.bot:
                    log.info(f"Ignoring bot message from {message.author}")
                    return

                if trigger.channel_id is not None:
                    if str(message.channel.id) != trigger.channel_id:
                        log.info(
                            f"Ignoring message from channel {message.channel.id}, "
                            f"expected {trigger.channel_id}"
                        )
                        return

                attachments = [
                    {
                        "id": attachment.id,
                        "filename": attachment.filename,
                        "url": attachment.url,
                        "content_type": attachment.content_type,
                        "size": attachment.size,
                    }
                    for attachment in message.attachments
                ]

                event = {
                    "message_id": message.id,
                    "content": message.content,
                    "author": {
                        "id": message.author.id,
                        "name": message.author.name,
                        "display_name": message.author.display_name,
                        "bot": message.author.bot,
                    },
                    "channel": {
                        "id": message.channel.id,
                        "name": getattr(message.channel, "name", ""),
                        "type": str(message.channel.type),
                    },
                    "guild": (
                        {
                            "id": message.guild.id,
                            "name": message.guild.name,
                        }
                        if message.guild
                        else None
                    ),
                    "attachments": attachments,
                    "timestamp": message.created_at.isoformat(),
                    "source": "discord",
                    "event_type": "message",
                }

                log.info(
                    f"Pushing Discord event to queue: message_id={event['message_id']}"
                )
                trigger.push_event(event)
                log.info(
                    f"Discord event pushed, queue size: {trigger._event_queue.qsize()}"
                )

        self._client = TriggerClient(intents=intents)
        self._client_task = asyncio.create_task(self._client.start(self.token))

    async def wait_for_event(
        self, context: ProcessingContext
    ) -> DiscordBotTriggerOutput | None:
        """Wait for the next Discord message event."""
        log.info(
            f"Discord trigger waiting for event, queue size: {self._event_queue.qsize()}"
        )
        event = await self.get_event_from_queue()
        if event is not None:
            log.info(
                f"Discord trigger received event: message_id={event.get('message_id')}"
            )
        else:
            log.info("Discord trigger received None event (stopping)")
        return event

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Disconnect the Discord bot."""
        log.info("Cleaning up Discord bot trigger")

        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                log.warning(f"Error closing Discord client: {e}")

        if self._client_task is not None:
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.warning(f"Error stopping Discord client task: {e}")


class DiscordSendMessage(BaseNode):
    """
    Node that sends a message to a Discord channel using a bot token.
    """

    token: str = Field(
        default="",
        description="Discord bot token",
    )
    channel_id: str = Field(
        default="",
        description="Target channel ID",
    )
    content: str = Field(
        default="",
        description="Message content",
    )
    tts: bool = Field(
        default=False,
        description="Send as text-to-speech",
    )
    embeds: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Embeds as Discord embed dictionaries",
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        try:
            import discord  # type: ignore
        except ImportError:
            raise ImportError(
                "discord.py is required for DiscordSendMessage. "
                "Install it with: pip install discord.py"
            )

        if not self.token:
            raise ValueError("Discord bot token is required")

        if not self.channel_id:
            raise ValueError("Discord channel ID is required")

        intents = discord.Intents.default()
        intents.guilds = True

        send_node = self

        class SendClient(discord.Client):
            async def on_ready(self) -> None:
                try:
                    channel_id = int(send_node.channel_id)
                except ValueError:
                    log.error(
                        "Discord channel ID must be a numeric string, got: "
                        f"{send_node.channel_id!r}"
                    )
                    await self.close()
                    return

                channel = self.get_channel(channel_id)
                if channel is None:
                    log.error(
                        "Discord channel not found. "
                        "Ensure the bot has access and the ID is correct."
                    )
                    await self.close()
                    return

                try:
                    embed_objs = [
                        discord.Embed.from_dict(embed) for embed in send_node.embeds
                    ]
                    message = await channel.send(
                        send_node.content,
                        tts=send_node.tts,
                        embeds=embed_objs if embed_objs else None,
                    )
                    send_node._sent_message = message
                except Exception as e:
                    log.error(f"Failed to send Discord message: {e}")
                finally:
                    await self.close()

        client = SendClient(intents=intents)
        self._sent_message = None

        await client.start(self.token)

        message_id = None
        if self._sent_message is not None:
            message_id = self._sent_message.id

        return {
            "message_id": message_id,
        }


__all__ = [
    "DiscordBotTrigger",
    "DiscordSendMessage",
]
