import pytest
from unittest.mock import MagicMock
import sys
import os
from pydantic import BaseModel, ConfigDict

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

@pytest.fixture
def context():
    # Mock ProcessingContext completely for the test
    mock_ctx = MagicMock()
    mock_ctx.user_id = "test"
    mock_ctx.auth_token = "test"
    return mock_ctx

# Depending on whether we can import the real class or not:
try:
    from nodetool.nodes.nodetool.list import Difference
except ImportError:
    pass

# Mock dependencies if they are missing
if "nodetool.workflows.base_node" not in sys.modules:
    class MockBaseNode(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        def __init__(self, **data):
            super().__init__(**data)
        async def process(self, context):
            pass

    mock_workflows = MagicMock()
    mock_workflows.base_node.BaseNode = MockBaseNode
    sys.modules["nodetool.workflows"] = mock_workflows
    sys.modules["nodetool.workflows.base_node"] = mock_workflows.base_node
    sys.modules["nodetool.workflows.processing_context"] = MagicMock()

if "nodetool.metadata.types" not in sys.modules:
    mock_metadata = MagicMock()
    sys.modules["nodetool.metadata"] = mock_metadata
    sys.modules["nodetool.metadata.types"] = mock_metadata.types

# Re-import after mocking
from nodetool.nodes.nodetool.list import Difference


@pytest.mark.asyncio
async def test_difference_integers(context):
    node = Difference(list1=[1, 2, 3], list2=[2, 3, 4])
    result = await node.process(context)
    assert isinstance(result, list)
    assert set(result) == {1}

@pytest.mark.asyncio
async def test_difference_strings(context):
    node = Difference(list1=["a", "b", "c"], list2=["b", "c", "d"])
    result = await node.process(context)
    assert isinstance(result, list)
    assert set(result) == {"a"}

@pytest.mark.asyncio
async def test_difference_empty_inputs(context):
    node = Difference(list1=[], list2=[1, 2])
    result = await node.process(context)
    assert result == []

    node = Difference(list1=[1, 2], list2=[])
    result = await node.process(context)
    assert set(result) == {1, 2}

    node = Difference(list1=[], list2=[])
    result = await node.process(context)
    assert result == []

@pytest.mark.asyncio
async def test_difference_no_overlap(context):
    node = Difference(list1=[1, 2], list2=[3, 4])
    result = await node.process(context)
    assert set(result) == {1, 2}

@pytest.mark.asyncio
async def test_difference_subset(context):
    node = Difference(list1=[1, 2], list2=[1, 2, 3])
    result = await node.process(context)
    assert result == []

@pytest.mark.asyncio
async def test_difference_duplicates(context):
    node = Difference(list1=[1, 1, 2, 3], list2=[3, 4])
    result = await node.process(context)
    assert set(result) == {1, 2}

@pytest.mark.asyncio
async def test_difference_unhashable(context):
    node = Difference(list1=[{"a": 1}], list2=[{"a": 1}])
    with pytest.raises(TypeError):
        await node.process(context)
