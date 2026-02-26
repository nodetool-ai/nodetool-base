
import pytest
import sys
from unittest.mock import MagicMock
from typing import AsyncGenerator

# Mock dependencies
sys.modules["nodetool.config.logging_config"] = MagicMock()
sys.modules["nodetool.config.logging_config"].get_logger = MagicMock(return_value=MagicMock())
sys.modules["nodetool.workflows.processing_context"] = MagicMock()
sys.modules["nodetool.workflows.base_node"] = MagicMock()
sys.modules["nodetool.config.environment"] = MagicMock()

# Setup BaseNode mock
from pydantic import BaseModel
class BaseNode(BaseModel):
    # Mock iter_any_input to return fields of the model
    async def iter_any_input(self) -> AsyncGenerator[tuple[str, any], None]:
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            yield field_name, value

sys.modules["nodetool.workflows.base_node"].BaseNode = BaseNode

# Setup ProcessingContext mock
class ProcessingContext:
    def __init__(self, user_id, auth_token):
        pass
sys.modules["nodetool.workflows.processing_context"].ProcessingContext = ProcessingContext

# Now import the node
from nodetool.nodes.nodetool.dictionary import FilterDictByQuery

@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")

@pytest.mark.asyncio
async def test_filter_dict_by_query_legitimate(context):
    """Test FilterDictByQuery with legitimate pandas query syntax."""
    node = FilterDictByQuery(
        condition="age >= 30 and score > 75",
        value={"name": "B", "age": 30, "score": 90}
    )

    results = []
    async for output in node.gen_process(context):
        results.append(output)

    assert len(results) == 1
    assert results[0]["output"]["name"] == "B"

@pytest.mark.asyncio
async def test_filter_dict_by_query_filtered_out(context):
    """Test FilterDictByQuery filtering out non-matching items."""
    node = FilterDictByQuery(
        condition="age >= 30",
        value={"name": "A", "age": 25, "score": 80}
    )

    results = []
    async for output in node.gen_process(context):
        results.append(output)

    assert len(results) == 0

@pytest.mark.asyncio
async def test_filter_dict_by_query_injection_attempt(context):
    """Test that code injection attempts are blocked."""
    # Attempt to access os via injection
    # With the fix, local_dict and global_dict are empty, so @pd or @os should fail
    # creating a NameError or similar which is caught by the except block

    node = FilterDictByQuery(
        condition="@pd.io.common.os.system('echo INJECTION > injection.txt')",
        value={"a": 1}
    )

    results = []
    # This should yield nothing because the query will raise an exception internally
    # which is caught and ignored by the node
    async for output in node.gen_process(context):
        results.append(output)

    assert len(results) == 0

    import os
    assert not os.path.exists("injection.txt"), "Injection vulnerability succeeded!"

@pytest.mark.asyncio
async def test_filter_dict_by_query_builtins_access(context):
    """Test that builtins access is blocked."""
    node = FilterDictByQuery(
        condition="__import__('os').system('echo BUILTINS > builtins.txt')",
        value={"a": 1}
    )

    results = []
    async for output in node.gen_process(context):
        results.append(output)

    assert len(results) == 0

    import os
    assert not os.path.exists("builtins.txt"), "Builtins access vulnerability succeeded!"
