from nodetool.workflows.run_workflow import WorkflowRunner
import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.nodes.nodetool.code import (
    ExecutePython,
    ExecuteJavaScript,
    ExecuteBash,
    ExecuteRuby,
    ExecuteLua,
    ExecuteCommand,
    RunPythonCommand,
    RunJavaScriptCommand,
    RunBashCommand,
    RunRubyCommand,
    RunLuaCommand,
    RunShellCommand,
    ExecutionMode,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_execute_python_basic_streams_stdout(context: ProcessingContext):
    code = "print('unused in test; stream is patched')"
    node = ExecutePython(code=code)  # type: ignore[call-arg]

    async def fake_stream(
        self,
        user_code,
        env_locals,
        context,
        node,
        allow_dynamic_outputs=True,
        stdin_stream=None,
        **kwargs,
    ):
        yield ("stdout", "10")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)
        collected = outputs.collected()
        assert collected.get("stdout") == "10"


@pytest.mark.asyncio
async def test_execute_python_streams_stderr(context: ProcessingContext):
    node = ExecutePython(code="print('unused')")  # type: ignore[call-arg]

    async def fake_stream(
        self,
        user_code,
        env_locals,
        context,
        node,
        allow_dynamic_outputs=True,
        stdin_stream=None,
        **kwargs,
    ):
        yield ("stderr", "some error message")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)
        collected = outputs.collected()
        assert collected.get("stderr") == "some error message"


@pytest.mark.asyncio
async def test_execute_python_streams_both_channels(context: ProcessingContext):
    node = ExecutePython(code="print('unused')")  # type: ignore[call-arg]

    async def fake_stream(
        self,
        user_code,
        env_locals,
        context,
        node,
        allow_dynamic_outputs=True,
        stdin_stream=None,
        **kwargs,
    ):
        yield ("stdout", "line out")
        yield ("stderr", "line err")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)
        collected = outputs.collected()
        assert collected.get("stdout") == "line out"
        assert collected.get("stderr") == "line err"


@pytest.mark.asyncio
async def test_execute_python_mode_default_is_docker(context: ProcessingContext):
    code = "print('unused in test; mode is inspected')"
    node = ExecutePython(code=code)  # type: ignore[call-arg]

    async def fake_stream(self, *args, **kwargs):
        # Ensure runner is configured for docker by default
        assert getattr(self, "mode", None) == "docker"
        yield ("stdout", "ok")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)


@pytest.mark.asyncio
async def test_execute_python_mode_can_be_subprocess(context: ProcessingContext):
    code = "print('unused in test; mode is inspected')"
    node = ExecutePython(code=code)  # type: ignore[call-arg]
    node.execution_mode = ExecutionMode.SUBPROCESS

    async def fake_stream(self, *args, **kwargs):
        # Ensure runner is configured for subprocess when requested
        assert getattr(self, "mode", None) == "subprocess"
        yield ("stdout", "ok")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)


# ============================================
# Tests for Command nodes (buffered output)
# ============================================


@pytest.mark.asyncio
async def test_run_python_command_is_not_streaming_input():
    """Test that RunPythonCommand does not stream input."""
    assert not RunPythonCommand.is_streaming_input()


@pytest.mark.asyncio
async def test_run_python_command_is_not_streaming_output():
    """Test that RunPythonCommand does not stream output."""
    assert not RunPythonCommand.is_streaming_output()


@pytest.mark.asyncio
async def test_run_javascript_command_is_not_streaming_input():
    """Test that RunJavaScriptCommand does not stream input."""
    assert not RunJavaScriptCommand.is_streaming_input()


@pytest.mark.asyncio
async def test_run_bash_command_is_not_streaming_input():
    """Test that RunBashCommand does not stream input."""
    assert not RunBashCommand.is_streaming_input()


@pytest.mark.asyncio
async def test_run_ruby_command_is_not_streaming_input():
    """Test that RunRubyCommand does not stream input."""
    assert not RunRubyCommand.is_streaming_input()


@pytest.mark.asyncio
async def test_run_lua_command_is_not_streaming_input():
    """Test that RunLuaCommand does not stream input."""
    assert not RunLuaCommand.is_streaming_input()


@pytest.mark.asyncio
async def test_run_shell_command_is_not_streaming_input():
    """Test that RunShellCommand does not stream input."""
    assert not RunShellCommand.is_streaming_input()


@pytest.mark.asyncio
async def test_execute_python_is_streaming_input():
    """Test that ExecutePython opts into streaming input."""
    assert ExecutePython.is_streaming_input()


@pytest.mark.asyncio
async def test_execute_javascript_is_streaming_input():
    """Test that ExecuteJavaScript opts into streaming input."""
    assert ExecuteJavaScript.is_streaming_input()


@pytest.mark.asyncio
async def test_execute_bash_is_streaming_input():
    """Test that ExecuteBash opts into streaming input."""
    assert ExecuteBash.is_streaming_input()


@pytest.mark.asyncio
async def test_execute_ruby_is_streaming_input():
    """Test that ExecuteRuby opts into streaming input."""
    assert ExecuteRuby.is_streaming_input()


@pytest.mark.asyncio
async def test_execute_lua_is_streaming_input():
    """Test that ExecuteLua opts into streaming input."""
    assert ExecuteLua.is_streaming_input()


@pytest.mark.asyncio
async def test_execute_command_is_streaming_input():
    """Test that ExecuteCommand opts into streaming input."""
    assert ExecuteCommand.is_streaming_input()
