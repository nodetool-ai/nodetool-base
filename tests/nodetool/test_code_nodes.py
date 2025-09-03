import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.nodes.nodetool.code import ExecutePython, EvaluateExpression


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
        **kwargs
    ):
        yield ("stdout", "10")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        outputs = NodeOutputs(
            runner=None, node=node, context=context, capture_only=True
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
        **kwargs
    ):
        yield ("stderr", "some error message")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        outputs = NodeOutputs(
            runner=None, node=node, context=context, capture_only=True
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
        **kwargs
    ):
        yield ("stdout", "line out")
        yield ("stderr", "line err")

    with patch(
        "nodetool.nodes.nodetool.code.PythonDockerRunner.stream", new=fake_stream
    ):
        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        outputs = NodeOutputs(
            runner=None, node=node, context=context, capture_only=True
        )
        await node.run(context, inputs, outputs)
        collected = outputs.collected()
        assert collected.get("stdout") == "line out"
        assert collected.get("stderr") == "line err"


@pytest.mark.asyncio
async def test_evaluate_expression_basic(context: ProcessingContext):
    node = EvaluateExpression(expression="a*b + 1", variables={"a": 3, "b": 4})  # type: ignore[call-arg]
    assert await node.process(context) == 13


@pytest.mark.asyncio
async def test_evaluate_expression_allows_whitelisted_calls(context: ProcessingContext):
    node = EvaluateExpression(expression="len([1,2,3])", variables={})  # type: ignore[call-arg]
    assert await node.process(context) == 3
