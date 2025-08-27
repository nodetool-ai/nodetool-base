import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.code import ExecutePython, EvaluateExpression


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_execute_python_basic_streams_stdout(context: ProcessingContext):
    code = "print('unused in test; stream is patched')"
    node = ExecutePython(code=code)

    async def fake_stream(
        self, user_code, env_locals, context, node, allow_dynamic_outputs=True
    ):
        yield ("stdout", "10")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        results = []
        async for slot, value in node.gen_process(context):
            results.append((slot, value))
        assert ("stdout", "10") in results


@pytest.mark.asyncio
async def test_execute_python_streams_stderr(context: ProcessingContext):
    node = ExecutePython(code="print('unused')")

    async def fake_stream(
        self, user_code, env_locals, context, node, allow_dynamic_outputs=True
    ):
        yield ("stderr", "some error message")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        results = []
        async for slot, value in node.gen_process(context):
            results.append((slot, value))
        assert ("stderr", "some error message") in results


@pytest.mark.asyncio
async def test_execute_python_streams_both_channels(context: ProcessingContext):
    node = ExecutePython(code="print('unused')")

    async def fake_stream(
        self, user_code, env_locals, context, node, allow_dynamic_outputs=True
    ):
        yield ("stdout", "line out")
        yield ("stderr", "line err")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        stdout_lines = []
        stderr_lines = []
        async for slot, value in node.gen_process(context):
            if slot == "stdout":
                stdout_lines.append(value)
            elif slot == "stderr":
                stderr_lines.append(value)
        assert "line out" in stdout_lines
        assert "line err" in stderr_lines


@pytest.mark.asyncio
async def test_evaluate_expression_basic(context: ProcessingContext):
    node = EvaluateExpression(expression="a*b + 1", variables={"a": 3, "b": 4})
    assert await node.process(context) == 13


@pytest.mark.asyncio
async def test_evaluate_expression_allows_whitelisted_calls(context: ProcessingContext):
    node = EvaluateExpression(expression="len([1,2,3])", variables={})
    assert await node.process(context) == 3
