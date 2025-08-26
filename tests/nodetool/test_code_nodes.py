import json
import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.code import ExecutePython, EvaluateExpression


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_execute_python_basic(context: ProcessingContext):
    code = """
def main(env):
    return (env['a'] + env['b']) * 2
"""
    node = ExecutePython(code=code, inputs={"a": 2, "b": 3})

    # Simulate Docker SDK execution returning a magic-tagged success line
    async def fake_to_thread(func, *args, **kwargs):
        payload = {"ok": True, "result": 10}
        return "__NT_RESULT__:" + json.dumps(payload)

    with patch(
        "nodetool.nodes.nodetool.code.Environment.is_production", return_value=False
    ), patch("asyncio.to_thread", side_effect=fake_to_thread):
        assert await node.process(context) == 10


@pytest.mark.asyncio
async def test_execute_python_missing_main_raises(context: ProcessingContext):
    code = """
x = 1
"""
    node = ExecutePython(code=code)

    # Simulate container emitting an error payload due to missing main(env) or result
    async def fake_to_thread(func, *args, **kwargs):
        payload = {
            "ok": False,
            "error": "Provide main(env) or set a global result variable",
        }
        return "__NT_RESULT__:" + json.dumps(payload)

    with patch(
        "nodetool.nodes.nodetool.code.Environment.is_production", return_value=False
    ), patch("asyncio.to_thread", side_effect=fake_to_thread):
        with pytest.raises(RuntimeError):
            await node.process(context)


@pytest.mark.asyncio
async def test_execute_python_legacy_result_style(context: ProcessingContext):
    code = """
a_plus_b = a + b
result = a_plus_b * 2
"""
    node = ExecutePython(code=code, inputs={"a": 2, "b": 3})

    async def fake_to_thread(func, *args, **kwargs):
        payload = {"ok": True, "result": 10}
        return "__NT_RESULT__:" + json.dumps(payload)

    with patch(
        "nodetool.nodes.nodetool.code.Environment.is_production", return_value=False
    ), patch("asyncio.to_thread", side_effect=fake_to_thread):
        assert await node.process(context) == 10


@pytest.mark.asyncio
async def test_evaluate_expression_basic(context: ProcessingContext):
    node = EvaluateExpression(expression="a*b + 1", variables={"a": 3, "b": 4})
    with patch(
        "nodetool.nodes.nodetool.code.Environment.is_production", return_value=False
    ):
        assert await node.process(context) == 13


@pytest.mark.asyncio
async def test_evaluate_expression_allows_whitelisted_calls(context: ProcessingContext):
    node = EvaluateExpression(expression="len([1,2,3])", variables={})
    with patch(
        "nodetool.nodes.nodetool.code.Environment.is_production", return_value=False
    ):
        assert await node.process(context) == 3
