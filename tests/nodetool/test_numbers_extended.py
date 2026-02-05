"""Extended tests for numbers nodes to improve coverage."""

import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.api_graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.nodes.nodetool.control import ForEach, Collect
from nodetool.nodes.nodetool.numbers import FilterNumber, FilterNumberRange
from nodetool.nodes.nodetool.output import Output


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


async def run_filter_graph(context, input_list, filter_node_type, filter_data):
    """Run a workflow graph with streaming filter nodes."""
    nodes = [
        APINode(id="source", type=ForEach.get_node_type(), data={"input_list": input_list}),
        APINode(id="filter", type=filter_node_type, data=filter_data),
        APINode(id="collect", type=Collect.get_node_type(), data={}),
        APINode(id="out", type=Output.get_node_type(), data={"name": "result"}),
    ]
    edges = [
        APIEdge(source="source", sourceHandle="output", target="filter", targetHandle="value"),
        APIEdge(source="filter", sourceHandle="output", target="collect", targetHandle="input_item"),
        APIEdge(source="collect", sourceHandle="output", target="out", targetHandle="value"),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)
    req = RunJobRequest(graph=graph)

    result = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            result = msg.value
    return result


# Test FilterNumber with various filter types
@pytest.mark.asyncio
async def test_filter_number_equal_to(context: ProcessingContext):
    """Test FilterNumber with equal_to filter."""
    result = await run_filter_graph(
        context,
        [1, 2, 3, 4, 5],
        FilterNumber.get_node_type(),
        {"filter_type": "equal_to", "compare_value": 3},
    )
    assert result == [3]


@pytest.mark.asyncio
async def test_filter_number_odd(context: ProcessingContext):
    """Test FilterNumber with odd filter."""
    result = await run_filter_graph(
        context,
        [1, 2, 3, 4, 5, 6],
        FilterNumber.get_node_type(),
        {"filter_type": "odd"},
    )
    assert result == [1, 3, 5]


@pytest.mark.asyncio
async def test_filter_number_positive(context: ProcessingContext):
    """Test FilterNumber with positive filter."""
    result = await run_filter_graph(
        context,
        [-3, -1, 0, 2, 5],
        FilterNumber.get_node_type(),
        {"filter_type": "positive"},
    )
    assert result == [2, 5]


@pytest.mark.asyncio
async def test_filter_number_negative(context: ProcessingContext):
    """Test FilterNumber with negative filter."""
    result = await run_filter_graph(
        context,
        [-3, -1, 0, 2, 5],
        FilterNumber.get_node_type(),
        {"filter_type": "negative"},
    )
    assert result == [-3, -1]


@pytest.mark.asyncio
async def test_filter_number_even_with_zero(context: ProcessingContext):
    """Test FilterNumber with even filter including zero."""
    result = await run_filter_graph(
        context,
        [0, 1, 2, 3, 4],
        FilterNumber.get_node_type(),
        {"filter_type": "even"},
    )
    assert result == [0, 2, 4]


@pytest.mark.asyncio
async def test_filter_number_with_floats(context: ProcessingContext):
    """Test FilterNumber with float values."""
    result = await run_filter_graph(
        context,
        [1.5, 2.5, 3.5, 4.5],
        FilterNumber.get_node_type(),
        {"filter_type": "greater_than", "compare_value": 2.0},
    )
    assert result == [2.5, 3.5, 4.5]


@pytest.mark.asyncio
async def test_filter_number_empty_input(context: ProcessingContext):
    """Test FilterNumber with empty input list."""
    result = await run_filter_graph(
        context,
        [],
        FilterNumber.get_node_type(),
        {"filter_type": "greater_than", "compare_value": 0},
    )
    assert result == []


# Test FilterNumberRange with various configurations
@pytest.mark.asyncio
async def test_filter_number_range_all_outside(context: ProcessingContext):
    """Test FilterNumberRange where no values are in range."""
    result = await run_filter_graph(
        context,
        [1, 2, 3],
        FilterNumberRange.get_node_type(),
        {"min_value": 10, "max_value": 20, "inclusive": True},
    )
    assert result == []


@pytest.mark.asyncio
async def test_filter_number_range_all_inside(context: ProcessingContext):
    """Test FilterNumberRange where all values are in range."""
    result = await run_filter_graph(
        context,
        [5, 6, 7, 8, 9],
        FilterNumberRange.get_node_type(),
        {"min_value": 0, "max_value": 10, "inclusive": True},
    )
    assert result == [5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_filter_number_range_boundary_exclusive(context: ProcessingContext):
    """Test FilterNumberRange with exclusive boundaries."""
    result = await run_filter_graph(
        context,
        [1, 2, 3, 4, 5],
        FilterNumberRange.get_node_type(),
        {"min_value": 1, "max_value": 5, "inclusive": False},
    )
    assert result == [2, 3, 4]


@pytest.mark.asyncio
async def test_filter_number_range_boundary_inclusive(context: ProcessingContext):
    """Test FilterNumberRange with inclusive boundaries."""
    result = await run_filter_graph(
        context,
        [1, 2, 3, 4, 5],
        FilterNumberRange.get_node_type(),
        {"min_value": 1, "max_value": 5, "inclusive": True},
    )
    assert result == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_filter_number_range_negative_values(context: ProcessingContext):
    """Test FilterNumberRange with negative values."""
    result = await run_filter_graph(
        context,
        [-10, -5, 0, 5, 10],
        FilterNumberRange.get_node_type(),
        {"min_value": -6, "max_value": 6, "inclusive": True},
    )
    assert result == [-5, 0, 5]


@pytest.mark.asyncio
async def test_filter_number_range_floats(context: ProcessingContext):
    """Test FilterNumberRange with float values."""
    result = await run_filter_graph(
        context,
        [0.5, 1.5, 2.5, 3.5],
        FilterNumberRange.get_node_type(),
        {"min_value": 1.0, "max_value": 3.0, "inclusive": True},
    )
    assert result == [1.5, 2.5]
