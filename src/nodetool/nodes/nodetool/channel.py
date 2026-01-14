"""Channel nodes for typed streaming communication.

This module provides Send and Receive nodes that use named channels from
ProcessingContext.channels for cross-node communication. Each type has its own
pair of nodes (e.g., SendInteger/ReceiveInteger).

These nodes enable dynamic communication paths without modifying the static
graph topology, useful for:
- Broadcasting progress updates
- Aggregating logs from parallel nodes
- Dynamic cross-node coordination
"""

from typing import Any, AsyncGenerator, TypedDict
import uuid

from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.streaming_channel import ChannelManager


# Module-level channel manager for all channel operations
_channel_manager: ChannelManager | None = None


def get_channel_manager() -> ChannelManager:
    """Get the global channel manager, creating it if necessary."""
    global _channel_manager
    if _channel_manager is None:
        _channel_manager = ChannelManager()
    return _channel_manager


async def reset_channel_manager() -> None:
    """Reset the channel manager, closing all channels. Used for testing."""
    global _channel_manager
    if _channel_manager is not None:
        await _channel_manager.close_all()
        _channel_manager = None


def _generate_subscriber_id(node_id: str) -> str:
    """Generate a unique subscriber ID for a receive node.
    
    Args:
        node_id: The node's ID.
        
    Returns:
        A unique subscriber ID combining the node ID and a random suffix.
    """
    safe_id = node_id if node_id else "unknown"
    return f"{safe_id}_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Send Nodes - Publish typed values to named channels
# =============================================================================


class SendInteger(BaseNode):
    """
    Send an integer value to a named channel for streaming communication.
    send, channel, stream, integer, broadcast, publish
    """

    channel: str = Field(
        default="",
        description="The name of the channel to send to.",
    )
    value: int = Field(
        default=0,
        description="The integer value to send.",
    )

    async def process(self, context: ProcessingContext) -> int:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        await manager.publish(self.channel, self.value)
        return self.value


class SendFloat(BaseNode):
    """
    Send a float value to a named channel for streaming communication.
    send, channel, stream, float, broadcast, publish
    """

    channel: str = Field(
        default="",
        description="The name of the channel to send to.",
    )
    value: float = Field(
        default=0.0,
        description="The float value to send.",
    )

    async def process(self, context: ProcessingContext) -> float:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        await manager.publish(self.channel, self.value)
        return self.value


class SendString(BaseNode):
    """
    Send a string value to a named channel for streaming communication.
    send, channel, stream, string, text, broadcast, publish
    """

    channel: str = Field(
        default="",
        description="The name of the channel to send to.",
    )
    value: str = Field(
        default="",
        description="The string value to send.",
    )

    async def process(self, context: ProcessingContext) -> str:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        await manager.publish(self.channel, self.value)
        return self.value


class SendBoolean(BaseNode):
    """
    Send a boolean value to a named channel for streaming communication.
    send, channel, stream, boolean, broadcast, publish
    """

    channel: str = Field(
        default="",
        description="The name of the channel to send to.",
    )
    value: bool = Field(
        default=False,
        description="The boolean value to send.",
    )

    async def process(self, context: ProcessingContext) -> bool:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        await manager.publish(self.channel, self.value)
        return self.value


class SendAny(BaseNode):
    """
    Send any value to a named channel for streaming communication.
    send, channel, stream, any, broadcast, publish, generic
    """

    channel: str = Field(
        default="",
        description="The name of the channel to send to.",
    )
    value: Any = Field(
        default=None,
        description="The value to send.",
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        await manager.publish(self.channel, self.value)
        return self.value


# =============================================================================
# Receive Nodes - Subscribe to named channels and yield typed values
# =============================================================================


class ReceiveInteger(BaseNode):
    """
    Receive integer values from a named channel as a stream.
    receive, channel, stream, integer, subscribe, listen
    """

    channel: str = Field(
        default="",
        description="The name of the channel to receive from.",
    )

    class OutputType(TypedDict):
        value: int

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        subscriber_id = _generate_subscriber_id(self.id)
        async for item in manager.subscribe(self.channel, subscriber_id):
            if isinstance(item, int):
                yield {"value": item}
            else:
                # Attempt to convert to int
                try:
                    yield {"value": int(item)}
                except (ValueError, TypeError):
                    continue


class ReceiveFloat(BaseNode):
    """
    Receive float values from a named channel as a stream.
    receive, channel, stream, float, subscribe, listen
    """

    channel: str = Field(
        default="",
        description="The name of the channel to receive from.",
    )

    class OutputType(TypedDict):
        value: float

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        subscriber_id = _generate_subscriber_id(self.id)
        async for item in manager.subscribe(self.channel, subscriber_id):
            if isinstance(item, float | int):
                yield {"value": float(item)}
            else:
                # Attempt to convert to float
                try:
                    yield {"value": float(item)}
                except (ValueError, TypeError):
                    continue


class ReceiveString(BaseNode):
    """
    Receive string values from a named channel as a stream.
    receive, channel, stream, string, text, subscribe, listen
    """

    channel: str = Field(
        default="",
        description="The name of the channel to receive from.",
    )

    class OutputType(TypedDict):
        value: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        subscriber_id = _generate_subscriber_id(self.id)
        async for item in manager.subscribe(self.channel, subscriber_id):
            yield {"value": str(item)}


class ReceiveBoolean(BaseNode):
    """
    Receive boolean values from a named channel as a stream.
    receive, channel, stream, boolean, subscribe, listen
    """

    channel: str = Field(
        default="",
        description="The name of the channel to receive from.",
    )

    class OutputType(TypedDict):
        value: bool

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        subscriber_id = _generate_subscriber_id(self.id)
        async for item in manager.subscribe(self.channel, subscriber_id):
            if isinstance(item, bool):
                yield {"value": item}
            else:
                # Attempt to convert to bool
                yield {"value": bool(item)}


class ReceiveAny(BaseNode):
    """
    Receive any values from a named channel as a stream.
    receive, channel, stream, any, subscribe, listen, generic
    """

    channel: str = Field(
        default="",
        description="The name of the channel to receive from.",
    )

    class OutputType(TypedDict):
        value: Any

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        subscriber_id = _generate_subscriber_id(self.id)
        async for item in manager.subscribe(self.channel, subscriber_id):
            yield {"value": item}


# =============================================================================
# Channel Management Nodes
# =============================================================================


class CloseChannel(BaseNode):
    """
    Close a named channel, signaling all subscribers to stop.
    channel, close, cleanup, management
    """

    channel: str = Field(
        default="",
        description="The name of the channel to close.",
    )

    async def process(self, context: ProcessingContext) -> None:
        if not self.channel:
            raise ValueError("Channel name is required")
        manager = get_channel_manager()
        await manager.close_channel(self.channel)
