"""Extended tests for list nodes to improve coverage."""

import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.nodes.nodetool.control import Collect
from nodetool.nodes.nodetool.list import (
    GenerateSequence,
    Randomize,
    Sort,
    Intersection,
    Difference,
    Chunk,
    Sum,
    Average,
    Minimum,
    Maximum,
    Product,
    Flatten,
    SaveList,
)
from nodetool.nodes.nodetool.output import Output


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# Test Sum node
@pytest.mark.asyncio
async def test_sum_basic(context: ProcessingContext):
    """Test Sum with basic list of numbers."""
    node = Sum(values=[1, 2, 3, 4, 5])
    result = await node.process(context)
    assert result == 15


@pytest.mark.asyncio
async def test_sum_floats(context: ProcessingContext):
    """Test Sum with float values."""
    node = Sum(values=[1.5, 2.5, 3.0])
    result = await node.process(context)
    assert result == 7.0


@pytest.mark.asyncio
async def test_sum_empty(context: ProcessingContext):
    """Test Sum raises error for empty list."""
    node = Sum(values=[])
    with pytest.raises(ValueError, match="Cannot sum empty list"):
        await node.process(context)


@pytest.mark.asyncio
async def test_sum_mixed_int_float(context: ProcessingContext):
    """Test Sum with mixed integers and floats."""
    node = Sum(values=[1, 2.5, 3, 4.5])
    result = await node.process(context)
    assert result == 11.0


# Test Average node
@pytest.mark.asyncio
async def test_average_basic(context: ProcessingContext):
    """Test Average with basic list of numbers."""
    node = Average(values=[2, 4, 6, 8, 10])
    result = await node.process(context)
    assert result == 6.0


@pytest.mark.asyncio
async def test_average_floats(context: ProcessingContext):
    """Test Average with float values."""
    node = Average(values=[1.0, 2.0, 3.0])
    result = await node.process(context)
    assert result == 2.0


@pytest.mark.asyncio
async def test_average_empty(context: ProcessingContext):
    """Test Average raises error for empty list."""
    node = Average(values=[])
    with pytest.raises(ValueError, match="Cannot average empty list"):
        await node.process(context)


# Test Minimum node
@pytest.mark.asyncio
async def test_minimum_basic(context: ProcessingContext):
    """Test Minimum with basic list of numbers."""
    node = Minimum(values=[5, 2, 8, 1, 9])
    result = await node.process(context)
    assert result == 1


@pytest.mark.asyncio
async def test_minimum_negative(context: ProcessingContext):
    """Test Minimum with negative values."""
    node = Minimum(values=[-5, -2, -8, -1])
    result = await node.process(context)
    assert result == -8


@pytest.mark.asyncio
async def test_minimum_empty(context: ProcessingContext):
    """Test Minimum raises error for empty list."""
    node = Minimum(values=[])
    with pytest.raises(ValueError, match="Cannot find minimum of empty list"):
        await node.process(context)


# Test Maximum node
@pytest.mark.asyncio
async def test_maximum_basic(context: ProcessingContext):
    """Test Maximum with basic list of numbers."""
    node = Maximum(values=[5, 2, 8, 1, 9])
    result = await node.process(context)
    assert result == 9


@pytest.mark.asyncio
async def test_maximum_negative(context: ProcessingContext):
    """Test Maximum with negative values."""
    node = Maximum(values=[-5, -2, -8, -1])
    result = await node.process(context)
    assert result == -1


@pytest.mark.asyncio
async def test_maximum_empty(context: ProcessingContext):
    """Test Maximum raises error for empty list."""
    node = Maximum(values=[])
    with pytest.raises(ValueError, match="Cannot find maximum of empty list"):
        await node.process(context)


# Test Product node
@pytest.mark.asyncio
async def test_product_basic(context: ProcessingContext):
    """Test Product with basic list of numbers."""
    node = Product(values=[2, 3, 4])
    result = await node.process(context)
    assert result == 24


@pytest.mark.asyncio
async def test_product_with_zero(context: ProcessingContext):
    """Test Product with zero in list."""
    node = Product(values=[2, 0, 4])
    result = await node.process(context)
    assert result == 0


@pytest.mark.asyncio
async def test_product_single(context: ProcessingContext):
    """Test Product with single element."""
    node = Product(values=[5])
    result = await node.process(context)
    assert result == 5


@pytest.mark.asyncio
async def test_product_empty(context: ProcessingContext):
    """Test Product raises error for empty list."""
    node = Product(values=[])
    with pytest.raises(ValueError, match="Cannot calculate product of empty list"):
        await node.process(context)


# Test Flatten node
@pytest.mark.asyncio
async def test_flatten_simple(context: ProcessingContext):
    """Test Flatten with simple nested list."""
    node = Flatten(values=[[1, 2], [3, 4], [5]])
    result = await node.process(context)
    assert result == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_flatten_deep_nested(context: ProcessingContext):
    """Test Flatten with deeply nested list (unlimited depth)."""
    node = Flatten(values=[[1, [2, [3, [4]]]], [5]], max_depth=-1)
    result = await node.process(context)
    assert result == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_flatten_max_depth_1(context: ProcessingContext):
    """Test Flatten with max_depth=1."""
    node = Flatten(values=[[1, [2, 3]], [4, [5]]], max_depth=1)
    result = await node.process(context)
    assert result == [1, [2, 3], 4, [5]]


@pytest.mark.asyncio
async def test_flatten_max_depth_2(context: ProcessingContext):
    """Test Flatten with max_depth=2."""
    node = Flatten(values=[[[1, 2], [3]], [[4], [5, 6]]], max_depth=2)
    result = await node.process(context)
    assert result == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_flatten_already_flat(context: ProcessingContext):
    """Test Flatten with already flat list."""
    node = Flatten(values=[1, 2, 3, 4, 5])
    result = await node.process(context)
    assert result == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_flatten_empty(context: ProcessingContext):
    """Test Flatten with empty list."""
    node = Flatten(values=[])
    result = await node.process(context)
    assert result == []


@pytest.mark.asyncio
async def test_flatten_mixed_types(context: ProcessingContext):
    """Test Flatten with mixed types."""
    node = Flatten(values=[["a", "b"], ["c"], "d"])
    result = await node.process(context)
    assert result == ["a", "b", "c", "d"]


# Test Intersection node
@pytest.mark.asyncio
async def test_intersection_basic(context: ProcessingContext):
    """Test Intersection with overlapping lists."""
    node = Intersection(list1=[1, 2, 3, 4], list2=[3, 4, 5, 6])
    result = await node.process(context)
    assert set(result) == {3, 4}


@pytest.mark.asyncio
async def test_intersection_no_overlap(context: ProcessingContext):
    """Test Intersection with no overlapping elements."""
    node = Intersection(list1=[1, 2], list2=[3, 4])
    result = await node.process(context)
    assert result == []


@pytest.mark.asyncio
async def test_intersection_complete_overlap(context: ProcessingContext):
    """Test Intersection where one list is subset of other."""
    node = Intersection(list1=[1, 2, 3], list2=[1, 2, 3, 4, 5])
    result = await node.process(context)
    assert set(result) == {1, 2, 3}


# Test Difference node
@pytest.mark.asyncio
async def test_difference_basic(context: ProcessingContext):
    """Test Difference with overlapping lists."""
    node = Difference(list1=[1, 2, 3, 4], list2=[3, 4, 5, 6])
    result = await node.process(context)
    assert set(result) == {1, 2}


@pytest.mark.asyncio
async def test_difference_no_overlap(context: ProcessingContext):
    """Test Difference with no overlapping elements."""
    node = Difference(list1=[1, 2], list2=[3, 4])
    result = await node.process(context)
    assert set(result) == {1, 2}


@pytest.mark.asyncio
async def test_difference_empty_result(context: ProcessingContext):
    """Test Difference where all elements are in second list."""
    node = Difference(list1=[1, 2, 3], list2=[1, 2, 3, 4, 5])
    result = await node.process(context)
    assert result == []


# Test Randomize node
@pytest.mark.asyncio
async def test_randomize_basic(context: ProcessingContext):
    """Test Randomize shuffles the list."""
    node = Randomize(values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = await node.process(context)
    # Should have same elements
    assert sorted(result) == sorted(node.values)
    # Should not modify original list
    assert node.values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.mark.asyncio
async def test_randomize_empty(context: ProcessingContext):
    """Test Randomize with empty list."""
    node = Randomize(values=[])
    result = await node.process(context)
    assert result == []


# Test Sort node
@pytest.mark.asyncio
async def test_sort_ascending(context: ProcessingContext):
    """Test Sort with ascending order."""
    node = Sort(values=[3, 1, 4, 1, 5, 9, 2, 6], order=Sort.SortOrder.ASCENDING)
    result = await node.process(context)
    assert result == [1, 1, 2, 3, 4, 5, 6, 9]


@pytest.mark.asyncio
async def test_sort_descending(context: ProcessingContext):
    """Test Sort with descending order."""
    node = Sort(values=[3, 1, 4, 1, 5, 9, 2, 6], order=Sort.SortOrder.DESCENDING)
    result = await node.process(context)
    assert result == [9, 6, 5, 4, 3, 2, 1, 1]


@pytest.mark.asyncio
async def test_sort_strings(context: ProcessingContext):
    """Test Sort with strings."""
    node = Sort(values=["banana", "apple", "cherry"], order=Sort.SortOrder.ASCENDING)
    result = await node.process(context)
    assert result == ["apple", "banana", "cherry"]


# Test Chunk node
@pytest.mark.asyncio
async def test_chunk_even(context: ProcessingContext):
    """Test Chunk with evenly divisible list."""
    node = Chunk(values=[1, 2, 3, 4, 5, 6], chunk_size=2)
    result = await node.process(context)
    assert result == [[1, 2], [3, 4], [5, 6]]


@pytest.mark.asyncio
async def test_chunk_uneven(context: ProcessingContext):
    """Test Chunk with unevenly divisible list."""
    node = Chunk(values=[1, 2, 3, 4, 5], chunk_size=2)
    result = await node.process(context)
    assert result == [[1, 2], [3, 4], [5]]


@pytest.mark.asyncio
async def test_chunk_larger_than_list(context: ProcessingContext):
    """Test Chunk when chunk_size is larger than list."""
    node = Chunk(values=[1, 2, 3], chunk_size=10)
    result = await node.process(context)
    assert result == [[1, 2, 3]]


@pytest.mark.asyncio
async def test_chunk_size_one(context: ProcessingContext):
    """Test Chunk with chunk_size=1."""
    node = Chunk(values=[1, 2, 3], chunk_size=1)
    result = await node.process(context)
    assert result == [[1], [2], [3]]


# Helper function for streaming tests
async def run_stream_generator_graph(context, generator_node_type, generator_data):
    """Run a workflow graph with streaming generator nodes."""
    nodes = [
        APINode(id="gen", type=generator_node_type, data=generator_data),
        APINode(id="collect", type=Collect.get_node_type(), data={}),
        APINode(id="out", type=Output.get_node_type(), data={"name": "result"}),
    ]
    edges = [
        APIEdge(source="gen", sourceHandle="output", target="collect", targetHandle="input_item"),
        APIEdge(source="collect", sourceHandle="output", target="out", targetHandle="value"),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)
    req = RunJobRequest(graph=graph)

    result = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            result = msg.value
    return result


# Test GenerateSequence streaming node
@pytest.mark.asyncio
async def test_generate_sequence_basic(context: ProcessingContext):
    """Test GenerateSequence with basic range."""
    result = await run_stream_generator_graph(
        context,
        GenerateSequence.get_node_type(),
        {"start": 0, "stop": 5, "step": 1},
    )
    assert result == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_generate_sequence_step(context: ProcessingContext):
    """Test GenerateSequence with step > 1."""
    result = await run_stream_generator_graph(
        context,
        GenerateSequence.get_node_type(),
        {"start": 0, "stop": 10, "step": 2},
    )
    assert result == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_generate_sequence_negative_step(context: ProcessingContext):
    """Test GenerateSequence with negative step."""
    result = await run_stream_generator_graph(
        context,
        GenerateSequence.get_node_type(),
        {"start": 5, "stop": 0, "step": -1},
    )
    assert result == [5, 4, 3, 2, 1]


# Test SaveList node
@pytest.mark.asyncio
async def test_save_list(context: ProcessingContext):
    """Test SaveList creates a text asset."""
    node = SaveList(values=["line1", "line2", "line3"], name="test_output.txt")
    result = await node.process(context)
    # Result should be a TextRef
    assert result is not None
    assert result.uri is not None
