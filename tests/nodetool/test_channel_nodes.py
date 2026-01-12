"""Tests for channel send/receive nodes."""

import asyncio
import pytest
import pytest_asyncio
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.streaming_channel import Channel, ChannelManager
from nodetool.nodes.nodetool.channel import (
    SendInteger,
    SendFloat,
    SendString,
    SendBoolean,
    SendAny,
    ReceiveInteger,
    ReceiveFloat,
    ReceiveString,
    ReceiveBoolean,
    ReceiveAny,
    CloseChannel,
    get_channel_manager,
    reset_channel_manager,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest_asyncio.fixture(autouse=True)
async def cleanup_channels():
    """Reset channel manager before and after each test."""
    await reset_channel_manager()
    yield
    await reset_channel_manager()


# =============================================================================
# Channel and ChannelManager Unit Tests
# =============================================================================


class TestChannel:
    """Tests for the Channel class."""

    @pytest.mark.asyncio
    async def test_channel_creation(self):
        """Test basic channel creation."""
        channel = Channel("test", buffer_limit=10)
        assert channel.name == "test"
        stats = channel.get_stats()
        assert stats.name == "test"
        assert stats.subscriber_count == 0
        assert stats.is_closed is False

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic publish/subscribe pattern."""
        channel = Channel("test")
        received = []

        async def subscriber():
            async for item in channel.subscribe("sub1"):
                received.append(item)

        # Start subscriber task
        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)  # Let subscriber register

        # Publish items
        await channel.publish(1)
        await channel.publish(2)
        await channel.publish(3)
        await asyncio.sleep(0.01)  # Let items be received

        # Close channel to stop subscriber
        await channel.close()
        await task

        assert received == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers receive same messages."""
        channel = Channel("test")
        received1 = []
        received2 = []

        async def subscriber1():
            async for item in channel.subscribe("sub1"):
                received1.append(item)

        async def subscriber2():
            async for item in channel.subscribe("sub2"):
                received2.append(item)

        task1 = asyncio.create_task(subscriber1())
        task2 = asyncio.create_task(subscriber2())
        await asyncio.sleep(0.01)

        await channel.publish("hello")
        await channel.publish("world")
        await asyncio.sleep(0.01)

        await channel.close()
        await task1
        await task2

        assert received1 == ["hello", "world"]
        assert received2 == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_publish_to_closed_channel(self):
        """Test that publishing to a closed channel raises an error."""
        channel = Channel("test")
        await channel.close()

        with pytest.raises(RuntimeError, match="is closed"):
            await channel.publish("test")

    @pytest.mark.asyncio
    async def test_subscribe_to_closed_channel(self):
        """Test that subscribing to a closed channel yields nothing."""
        channel = Channel("test")
        await channel.close()

        received = []
        async for item in channel.subscribe("sub1"):
            received.append(item)

        assert received == []


class TestChannelManager:
    """Tests for the ChannelManager class."""

    @pytest.mark.asyncio
    async def test_create_channel(self):
        """Test channel creation via manager."""
        manager = ChannelManager()
        channel = await manager.create_channel("test")
        assert channel.name == "test"
        assert manager.get_channel("test") is channel

    @pytest.mark.asyncio
    async def test_get_or_create_channel(self):
        """Test get_or_create returns same channel."""
        manager = ChannelManager()
        channel1 = await manager.get_or_create_channel("test")
        channel2 = await manager.get_or_create_channel("test")
        assert channel1 is channel2

    @pytest.mark.asyncio
    async def test_duplicate_channel_error(self):
        """Test that creating duplicate channel raises error."""
        manager = ChannelManager()
        await manager.create_channel("test")

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_channel("test")

    @pytest.mark.asyncio
    async def test_replace_channel(self):
        """Test replacing an existing channel."""
        manager = ChannelManager()
        channel1 = await manager.create_channel("test")
        channel2 = await manager.create_channel("test", replace=True)

        assert channel1 is not channel2
        assert manager.get_channel("test") is channel2

    @pytest.mark.asyncio
    async def test_publish_creates_channel(self):
        """Test that publish auto-creates channel."""
        manager = ChannelManager()
        await manager.publish("auto", "message")
        assert manager.get_channel("auto") is not None

    @pytest.mark.asyncio
    async def test_list_channels(self):
        """Test listing all channels."""
        manager = ChannelManager()
        await manager.create_channel("channel1")
        await manager.create_channel("channel2")
        await manager.create_channel("channel3")

        channels = manager.list_channels()
        assert set(channels) == {"channel1", "channel2", "channel3"}

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all channels."""
        manager = ChannelManager()
        await manager.create_channel("channel1")
        await manager.create_channel("channel2")

        await manager.close_all()
        assert manager.list_channels() == []


# =============================================================================
# Send Node Tests
# =============================================================================


class TestSendNodes:
    """Tests for Send* nodes."""

    @pytest.mark.asyncio
    async def test_send_integer(self, context):
        """Test SendInteger node."""
        node = SendInteger(id="test", channel="int_channel", value=42)
        result = await node.process(context)
        assert result == 42

    @pytest.mark.asyncio
    async def test_send_float(self, context):
        """Test SendFloat node."""
        node = SendFloat(id="test", channel="float_channel", value=3.14)
        result = await node.process(context)
        assert result == 3.14

    @pytest.mark.asyncio
    async def test_send_string(self, context):
        """Test SendString node."""
        node = SendString(id="test", channel="string_channel", value="hello")
        result = await node.process(context)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_send_boolean(self, context):
        """Test SendBoolean node."""
        node = SendBoolean(id="test", channel="bool_channel", value=True)
        result = await node.process(context)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_any(self, context):
        """Test SendAny node."""
        node = SendAny(id="test", channel="any_channel", value={"key": "value"})
        result = await node.process(context)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_send_requires_channel(self, context):
        """Test that send nodes require a channel name."""
        node = SendInteger(id="test", channel="", value=42)
        with pytest.raises(ValueError, match="Channel name is required"):
            await node.process(context)


# =============================================================================
# Receive Node Tests
# =============================================================================


class TestReceiveNodes:
    """Tests for Receive* nodes."""

    @pytest.mark.asyncio
    async def test_receive_integer(self, context):
        """Test ReceiveInteger node."""
        manager = get_channel_manager()
        node = ReceiveInteger(id="recv_test", channel="test_channel")

        received = []

        async def receive_task():
            async for output in node.gen_process(context):
                received.append(output["value"])
                if len(received) >= 3:
                    break

        task = asyncio.create_task(receive_task())
        await asyncio.sleep(0.01)  # Let receiver register

        await manager.publish("test_channel", 1)
        await manager.publish("test_channel", 2)
        await manager.publish("test_channel", 3)
        await asyncio.sleep(0.01)

        await manager.close_channel("test_channel")
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert received == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_receive_float(self, context):
        """Test ReceiveFloat node."""
        manager = get_channel_manager()
        node = ReceiveFloat(id="recv_test", channel="test_channel")

        received = []

        async def receive_task():
            async for output in node.gen_process(context):
                received.append(output["value"])
                if len(received) >= 2:
                    break

        task = asyncio.create_task(receive_task())
        await asyncio.sleep(0.01)

        await manager.publish("test_channel", 1.5)
        await manager.publish("test_channel", 2)  # int should convert to float
        await asyncio.sleep(0.01)

        await manager.close_channel("test_channel")
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert received == [1.5, 2.0]

    @pytest.mark.asyncio
    async def test_receive_string(self, context):
        """Test ReceiveString node."""
        manager = get_channel_manager()
        node = ReceiveString(id="recv_test", channel="test_channel")

        received = []

        async def receive_task():
            async for output in node.gen_process(context):
                received.append(output["value"])
                if len(received) >= 2:
                    break

        task = asyncio.create_task(receive_task())
        await asyncio.sleep(0.01)

        await manager.publish("test_channel", "hello")
        await manager.publish("test_channel", 123)  # should convert to "123"
        await asyncio.sleep(0.01)

        await manager.close_channel("test_channel")
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert received == ["hello", "123"]

    @pytest.mark.asyncio
    async def test_receive_boolean(self, context):
        """Test ReceiveBoolean node."""
        manager = get_channel_manager()
        node = ReceiveBoolean(id="recv_test", channel="test_channel")

        received = []

        async def receive_task():
            async for output in node.gen_process(context):
                received.append(output["value"])
                if len(received) >= 2:
                    break

        task = asyncio.create_task(receive_task())
        await asyncio.sleep(0.01)

        await manager.publish("test_channel", True)
        await manager.publish("test_channel", False)
        await asyncio.sleep(0.01)

        await manager.close_channel("test_channel")
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert received == [True, False]

    @pytest.mark.asyncio
    async def test_receive_any(self, context):
        """Test ReceiveAny node."""
        manager = get_channel_manager()
        node = ReceiveAny(id="recv_test", channel="test_channel")

        received = []

        async def receive_task():
            async for output in node.gen_process(context):
                received.append(output["value"])
                if len(received) >= 3:
                    break

        task = asyncio.create_task(receive_task())
        await asyncio.sleep(0.01)

        await manager.publish("test_channel", 1)
        await manager.publish("test_channel", "hello")
        await manager.publish("test_channel", {"key": "value"})
        await asyncio.sleep(0.01)

        await manager.close_channel("test_channel")
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert received == [1, "hello", {"key": "value"}]

    @pytest.mark.asyncio
    async def test_receive_requires_channel(self, context):
        """Test that receive nodes require a channel name."""
        node = ReceiveInteger(id="test", channel="")
        with pytest.raises(ValueError, match="Channel name is required"):
            async for _ in node.gen_process(context):
                pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestSendReceiveIntegration:
    """Integration tests for send/receive pattern."""

    @pytest.mark.asyncio
    async def test_send_receive_integer(self, context):
        """Test sending and receiving integers through a channel."""
        send_node = SendInteger(id="send", channel="numbers", value=42)
        receive_node = ReceiveInteger(id="receive", channel="numbers")

        received = []

        async def receiver():
            async for output in receive_node.gen_process(context):
                received.append(output["value"])
                break  # Only receive one item

        # Start receiver
        task = asyncio.create_task(receiver())
        await asyncio.sleep(0.01)

        # Send value
        result = await send_node.process(context)
        assert result == 42
        await asyncio.sleep(0.01)

        # Close channel to stop receiver
        manager = get_channel_manager()
        await manager.close_channel("numbers")

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert received == [42]

    @pytest.mark.asyncio
    async def test_multiple_senders(self, context):
        """Test multiple senders to the same channel."""
        send1 = SendString(id="send1", channel="logs", value="hello")
        send2 = SendString(id="send2", channel="logs", value="world")
        receive = ReceiveString(id="receive", channel="logs")

        received = []

        async def receiver():
            async for output in receive.gen_process(context):
                received.append(output["value"])
                if len(received) >= 2:
                    break

        task = asyncio.create_task(receiver())
        await asyncio.sleep(0.01)

        await send1.process(context)
        await send2.process(context)
        await asyncio.sleep(0.01)

        manager = get_channel_manager()
        await manager.close_channel("logs")

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert set(received) == {"hello", "world"}


# =============================================================================
# Close Channel Node Tests
# =============================================================================


class TestCloseChannelNode:
    """Tests for CloseChannel node."""

    @pytest.mark.asyncio
    async def test_close_channel(self, context):
        """Test CloseChannel node."""
        manager = get_channel_manager()
        await manager.create_channel("to_close")

        assert manager.get_channel("to_close") is not None

        node = CloseChannel(id="close", channel="to_close")
        await node.process(context)

        assert manager.get_channel("to_close") is None

    @pytest.mark.asyncio
    async def test_close_channel_requires_name(self, context):
        """Test that CloseChannel requires a channel name."""
        node = CloseChannel(id="close", channel="")
        with pytest.raises(ValueError, match="Channel name is required"):
            await node.process(context)
