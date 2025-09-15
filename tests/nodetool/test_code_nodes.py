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
    node = EvaluateExpression(expression="a*b + 1")
    node._dynamic_properties = {"a": 3, "b": 4}
    assert await node.process(context) == 13


@pytest.mark.asyncio
async def test_evaluate_expression_allows_whitelisted_calls(context: ProcessingContext):
    # Note: Lua doesn't have a len function, use # operator for length
    node = EvaluateExpression(expression="#{1,2,3}")
    node._dynamic_properties = {}
    assert await node.process(context) == 3
