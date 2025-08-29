"""
Tests for streaming input/output behavior in code execution nodes.

These tests validate Unix pipe-like functionality where the output of one
code node can be streamed as input to another code node.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.inbox import NodeInbox
from nodetool.nodes.nodetool.code import ExecutePython, ExecuteBash


class TestCodeNodeStreaming:
    """Test streaming behavior for code execution nodes."""

    @pytest.fixture
    def context(self):
        """Mock processing context."""
        ctx = Mock(spec=ProcessingContext)
        ctx.workspace_dir = "/tmp/test_workspace"
        return ctx

    @pytest.fixture
    def python_node(self):
        """Create a Python execution node."""
        return ExecutePython(
            id="python_node",
            code="import sys\nfor line in sys.stdin:\n    print(f'processed: {line.strip()}')"
        )

    @pytest.fixture
    def bash_node(self):
        """Create a Bash execution node.""" 
        return ExecuteBash(
            id="bash_node",
            code="while IFS= read -r line; do echo \"bash processed: $line\"; done"
        )

    @pytest.mark.asyncio
    async def test_python_node_is_streaming_input(self):
        """Test that Python nodes opt into streaming input."""
        assert ExecutePython.is_streaming_input() == True

    @pytest.mark.asyncio
    async def test_bash_node_is_streaming_input(self):
        """Test that Bash nodes opt into streaming input."""
        assert ExecuteBash.is_streaming_input() == True

    @pytest.mark.asyncio
    async def test_python_node_with_streaming_input(self, python_node, context):
        """Test Python node processing streaming input via STDIN."""
        # Create and configure inbox
        inbox = NodeInbox()
        inbox.add_upstream("input", 1)
        python_node.attach_inbox(inbox)

        # Feed test data to inbox
        test_data = ["hello", "world", "streaming"]
        for data in test_data:
            inbox.put("input", data)
        inbox.mark_source_done("input")

        # Mock the runner to capture stdin_stream
        captured_stdin_data = []
        
        async def mock_stream(*args, stdin_stream=None, **kwargs):
            if stdin_stream:
                async for data in stdin_stream:
                    captured_stdin_data.append(data)
            # Simulate processing output
            for data in captured_stdin_data:
                yield "stdout", f"processed: {data}\n"
        
        # Patch the runner
        python_node.gen_process.__globals__['PythonDockerRunner'] = Mock()
        python_node.gen_process.__globals__['PythonDockerRunner'].return_value.stream = mock_stream

        # Execute and collect results
        results = []
        async for slot, value in python_node.gen_process(context):
            results.append((slot, value))

        # Verify stdin data was captured
        assert captured_stdin_data == ["hello", "world", "streaming"]
        
        # Verify output
        assert len(results) == 3
        assert all(slot == "stdout" for slot, _ in results)
        assert "processed: hello" in results[0][1]
        assert "processed: world" in results[1][1]
        assert "processed: streaming" in results[2][1]

    @pytest.mark.asyncio
    async def test_bash_node_with_streaming_input(self, bash_node, context):
        """Test Bash node processing streaming input via STDIN."""
        # Create and configure inbox
        inbox = NodeInbox()
        inbox.add_upstream("input", 1)
        bash_node.attach_inbox(inbox)

        # Feed test data to inbox
        test_data = ["line1", "line2", "line3"]
        for data in test_data:
            inbox.put("input", data)
        inbox.mark_source_done("input")

        # Mock the runner to capture stdin_stream
        captured_stdin_data = []
        
        async def mock_stream(*args, stdin_stream=None, **kwargs):
            if stdin_stream:
                async for data in stdin_stream:
                    captured_stdin_data.append(data)
            # Simulate processing output
            for data in captured_stdin_data:
                yield "stdout", f"bash processed: {data}\n"
        
        # Patch the runner
        bash_node.gen_process.__globals__['BashDockerRunner'] = Mock()
        bash_node.gen_process.__globals__['BashDockerRunner'].return_value.stream = mock_stream

        # Execute and collect results
        results = []
        async for slot, value in bash_node.gen_process(context):
            results.append((slot, value))

        # Verify stdin data was captured
        assert captured_stdin_data == ["line1", "line2", "line3"]
        
        # Verify output
        assert len(results) == 3
        assert all(slot == "stdout" for slot, _ in results)

    @pytest.mark.asyncio
    async def test_node_without_streaming_input(self, python_node, context):
        """Test node behavior when no streaming input is available."""
        # No inbox attached - should work without streaming input
        
        # Mock the runner
        async def mock_stream(*args, stdin_stream=None, **kwargs):
            assert stdin_stream is None  # Should be None when no streaming input
            yield "stdout", "no input processing\n"
        
        python_node.gen_process.__globals__['PythonDockerRunner'] = Mock()
        python_node.gen_process.__globals__['PythonDockerRunner'].return_value.stream = mock_stream

        # Execute and collect results
        results = []
        async for slot, value in python_node.gen_process(context):
            results.append((slot, value))

        # Should still work without streaming input
        assert len(results) == 1
        assert results[0][0] == "stdout"
        assert "no input processing" in results[0][1]

    @pytest.mark.asyncio
    async def test_multiple_input_handles(self, python_node, context):
        """Test node with multiple streaming input handles."""
        # Create and configure inbox with multiple handles
        inbox = NodeInbox()
        inbox.add_upstream("input1", 1)
        inbox.add_upstream("input2", 1)
        python_node.attach_inbox(inbox)

        # Feed test data to different handles
        inbox.put("input1", "from_input1")
        inbox.put("input2", "from_input2")
        inbox.put("input1", "more_from_input1")
        
        # Mark sources done
        inbox.mark_source_done("input1")
        inbox.mark_source_done("input2")

        # Mock the runner to capture stdin_stream
        captured_stdin_data = []
        
        async def mock_stream(*args, stdin_stream=None, **kwargs):
            if stdin_stream:
                async for data in stdin_stream:
                    captured_stdin_data.append(data)
            # Simulate processing output
            for data in captured_stdin_data:
                yield "stdout", f"processed: {data}\n"
        
        # Patch the runner
        python_node.gen_process.__globals__['PythonDockerRunner'] = Mock()
        python_node.gen_process.__globals__['PythonDockerRunner'].return_value.stream = mock_stream

        # Execute and collect results
        results = []
        async for slot, value in python_node.gen_process(context):
            results.append((slot, value))

        # Verify all data was captured (order may vary due to arrival order)
        assert len(captured_stdin_data) == 3
        assert "from_input1" in captured_stdin_data
        assert "from_input2" in captured_stdin_data
        assert "more_from_input1" in captured_stdin_data

    @pytest.mark.asyncio  
    async def test_stdin_stream_creation(self, context):
        """Test that nodes correctly create stdin streams from inbox data."""
        
        # Create a node with streaming input
        node = ExecutePython(
            id="test_node",
            code="import sys\nprint('processed')"
        )
        
        # Set up inbox with test data
        inbox = NodeInbox()
        inbox.add_upstream("input", 1)
        node.attach_inbox(inbox)
        
        # Add test data
        test_data = ["line1", "line2", "line3"]
        for data in test_data:
            inbox.put("input", data)
        inbox.mark_source_done("input")
        
        # Verify node can detect input
        assert node.has_input()
        
        # Collect data from iter_any_input to simulate what gen_process does
        collected_data = []
        async for handle, value in node.iter_any_input():
            collected_data.append(value)
        
        # Verify all data was collected
        assert collected_data == test_data
        
        # Test stdin stream creation logic (the actual implementation in gen_process)
        # Reset the inbox for this test
        inbox = NodeInbox()
        inbox.add_upstream("input", 1)
        node.attach_inbox(inbox)
        
        for data in test_data:
            inbox.put("input", data)
        inbox.mark_source_done("input")
        
        # Create stdin stream like gen_process does
        stdin_stream = None
        if node._inbox and node._inbox.has_any():
            async def create_stdin_stream():
                async for handle, value in node.iter_any_input():
                    yield str(value) if value is not None else ""
            stdin_stream = create_stdin_stream()
        
        # Verify stdin stream was created and contains expected data
        assert stdin_stream is not None
        
        stdin_data = []
        async for data in stdin_stream:
            stdin_data.append(data)
        
        assert stdin_data == test_data


class TestNodeInboxIntegration:
    """Test NodeInbox integration with code nodes."""

    @pytest.mark.asyncio
    async def test_inbox_has_any_detection(self):
        """Test that nodes can detect when inbox has input."""
        python_node = ExecutePython(id="test", code="print('test')")
        
        # No inbox attached
        assert not python_node.has_input()
        
        # Empty inbox
        inbox = NodeInbox()
        python_node.attach_inbox(inbox)
        assert not python_node.has_input()
        
        # Inbox with data
        inbox.add_upstream("input", 1)
        inbox.put("input", "test_data")
        assert python_node.has_input()

    @pytest.mark.asyncio
    async def test_iter_any_input_functionality(self):
        """Test the iter_any_input helper method."""
        python_node = ExecutePython(id="test", code="print('test')")
        inbox = NodeInbox()
        
        # Add multiple handles and data
        inbox.add_upstream("handle1", 1)
        inbox.add_upstream("handle2", 1) 
        python_node.attach_inbox(inbox)
        
        # Add data to different handles
        inbox.put("handle1", "data1")
        inbox.put("handle2", "data2")
        inbox.put("handle1", "data3")
        
        # Mark sources done
        inbox.mark_source_done("handle1")
        inbox.mark_source_done("handle2")
        
        # Collect all data
        collected = []
        async for handle, value in python_node.iter_any_input():
            collected.append((handle, value))
            
        # Should have received all data
        assert len(collected) == 3
        handles = [item[0] for item in collected]
        values = [item[1] for item in collected]
        
        assert "handle1" in handles
        assert "handle2" in handles
        assert "data1" in values
        assert "data2" in values 
        assert "data3" in values