from nodetool.workflows.run_workflow import WorkflowRunner
import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.nodes.nodetool.code import (
    ExecutePython,
    ExecutionMode,
    PythonRunner,
    JavaScriptRunner,
    BashRunner,
    RubyRunner,
    LuaRunnerNode,
    ShellRunner,
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
# Tests for Runner nodes (streaming commands)
# ============================================


@pytest.mark.asyncio
async def test_python_runner_is_streaming_input():
    """Test that PythonRunner opts into streaming input."""
    assert PythonRunner.is_streaming_input()


@pytest.mark.asyncio
async def test_python_runner_is_streaming_output():
    """Test that PythonRunner opts into streaming output."""
    assert PythonRunner.is_streaming_output()


@pytest.mark.asyncio
async def test_python_runner_with_static_command(context: ProcessingContext):
    """Test PythonRunner with a static command."""
    node = PythonRunner(commands="print('hello')")  # type: ignore[call-arg]

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
        # Verify that the user_code matches the static command
        assert user_code == "print('hello')"
        yield ("stdout", "hello")

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
        assert collected.get("stdout") == "hello"


@pytest.mark.asyncio
async def test_python_runner_with_streaming_commands(context: ProcessingContext):
    """Test PythonRunner with streaming commands."""
    node = PythonRunner()  # type: ignore[call-arg]

    executed_commands = []

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
        executed_commands.append(user_code)
        yield ("stdout", f"executed: {user_code}")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inbox.add_upstream("commands", 1)
        node.attach_inbox(inbox)

        # Feed streaming commands
        await inbox.put("commands", "print('first')")
        await inbox.put("commands", "print('second')")
        inbox.mark_source_done("commands")

        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)

        # Verify all commands were executed
        assert len(executed_commands) == 2
        assert "print('first')" in executed_commands
        assert "print('second')" in executed_commands


@pytest.mark.asyncio
async def test_bash_runner_is_streaming_input():
    """Test that BashRunner opts into streaming input."""
    assert BashRunner.is_streaming_input()


@pytest.mark.asyncio
async def test_bash_runner_with_streaming_commands(context: ProcessingContext):
    """Test BashRunner with streaming commands."""
    node = BashRunner()  # type: ignore[call-arg]

    executed_commands = []

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
        executed_commands.append(user_code)
        yield ("stdout", f"bash: {user_code}")

    with patch(
        "nodetool.nodes.nodetool.code.BashDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inbox.add_upstream("commands", 1)
        node.attach_inbox(inbox)

        # Feed streaming commands
        await inbox.put("commands", "echo hello")
        await inbox.put("commands", "ls -la")
        inbox.mark_source_done("commands")

        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)

        # Verify all commands were executed
        assert len(executed_commands) == 2
        assert "echo hello" in executed_commands
        assert "ls -la" in executed_commands


@pytest.mark.asyncio
async def test_javascript_runner_is_streaming_input():
    """Test that JavaScriptRunner opts into streaming input."""
    assert JavaScriptRunner.is_streaming_input()


@pytest.mark.asyncio
async def test_ruby_runner_is_streaming_input():
    """Test that RubyRunner opts into streaming input."""
    assert RubyRunner.is_streaming_input()


@pytest.mark.asyncio
async def test_lua_runner_node_is_streaming_input():
    """Test that LuaRunnerNode opts into streaming input."""
    assert LuaRunnerNode.is_streaming_input()


@pytest.mark.asyncio
async def test_shell_runner_is_streaming_input():
    """Test that ShellRunner opts into streaming input."""
    assert ShellRunner.is_streaming_input()


@pytest.mark.asyncio
async def test_shell_runner_with_streaming_commands(context: ProcessingContext):
    """Test ShellRunner with streaming commands."""
    node = ShellRunner()  # type: ignore[call-arg]

    executed_commands = []

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
        executed_commands.append(user_code)
        yield ("stdout", f"shell: {user_code}")

    with patch(
        "nodetool.nodes.nodetool.code.CommandDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inbox.add_upstream("commands", 1)
        node.attach_inbox(inbox)

        # Feed streaming commands
        await inbox.put("commands", "cat file.txt")
        await inbox.put("commands", "grep pattern")
        inbox.mark_source_done("commands")

        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)

        # Verify all commands were executed
        assert len(executed_commands) == 2
        assert "cat file.txt" in executed_commands
        assert "grep pattern" in executed_commands


@pytest.mark.asyncio
async def test_python_runner_mode_default_is_docker():
    """Test that PythonRunner defaults to docker mode."""
    node = PythonRunner()  # type: ignore[call-arg]
    assert node.execution_mode == ExecutionMode.DOCKER


@pytest.mark.asyncio
async def test_bash_runner_mode_default_is_docker():
    """Test that BashRunner defaults to docker mode."""
    node = BashRunner()  # type: ignore[call-arg]
    assert node.execution_mode == ExecutionMode.DOCKER
