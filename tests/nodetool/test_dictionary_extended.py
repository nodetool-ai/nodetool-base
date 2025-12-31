"""Extended tests for dictionary nodes to improve coverage."""

import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.nodes.nodetool.control import ForEach, Collect
from nodetool.nodes.nodetool.dictionary import (
    ArgMax,
    MakeDictionary,
    ReduceDictionaries,
    FilterDictByValue,
    FilterDictByRange,
    FilterDictByNumber,
    FilterDictRegex,
    FilterDictByQuery,
)
from nodetool.nodes.nodetool.output import Output


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# Test ArgMax node
@pytest.mark.asyncio
async def test_argmax_basic(context: ProcessingContext):
    """Test ArgMax returns the key with highest value."""
    node = ArgMax(scores={"cat": 0.7, "dog": 0.9, "bird": 0.3})
    result = await node.process(context)
    assert result == "dog"


@pytest.mark.asyncio
async def test_argmax_empty_dict(context: ProcessingContext):
    """Test ArgMax raises ValueError for empty dictionary."""
    node = ArgMax(scores={})
    with pytest.raises(ValueError, match="Input dictionary cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_argmax_negative_values(context: ProcessingContext):
    """Test ArgMax with negative values."""
    node = ArgMax(scores={"a": -5, "b": -1, "c": -10})
    result = await node.process(context)
    assert result == "b"


@pytest.mark.asyncio
async def test_argmax_equal_values(context: ProcessingContext):
    """Test ArgMax with equal values returns a consistent result."""
    node = ArgMax(scores={"a": 0.5, "b": 0.5})
    result = await node.process(context)
    # Should return one of the keys with maximum value
    assert result in ["a", "b"]


# Test MakeDictionary node
@pytest.mark.asyncio
async def test_make_dictionary(context: ProcessingContext):
    """Test MakeDictionary creates dictionary from dynamic properties."""
    node = MakeDictionary()
    node._dynamic_properties = {"key1": "value1", "key2": 42}
    result = await node.process(context)
    assert result == {"key1": "value1", "key2": 42}


@pytest.mark.asyncio
async def test_make_dictionary_empty(context: ProcessingContext):
    """Test MakeDictionary with no properties."""
    node = MakeDictionary()
    node._dynamic_properties = {}
    result = await node.process(context)
    assert result == {}


# Test ReduceDictionaries with conflict resolution
@pytest.mark.asyncio
async def test_reduce_dictionaries_first_conflict(context: ProcessingContext):
    """Test ReduceDictionaries with FIRST conflict resolution."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "value": "first"},
            {"id": 1, "value": "second"},  # Duplicate key
        ],
        key_field="id",
        value_field="value",
        conflict_resolution=ReduceDictionaries.ConflictResolution.FIRST,
    )
    result = await node.process(context)
    assert result == {1: "first"}  # First value should win


@pytest.mark.asyncio
async def test_reduce_dictionaries_last_conflict(context: ProcessingContext):
    """Test ReduceDictionaries with LAST conflict resolution."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "value": "first"},
            {"id": 1, "value": "second"},  # Duplicate key
        ],
        key_field="id",
        value_field="value",
        conflict_resolution=ReduceDictionaries.ConflictResolution.LAST,
    )
    result = await node.process(context)
    assert result == {1: "second"}  # Last value should win


@pytest.mark.asyncio
async def test_reduce_dictionaries_error_conflict(context: ProcessingContext):
    """Test ReduceDictionaries with ERROR conflict resolution."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "value": "first"},
            {"id": 1, "value": "second"},  # Duplicate key
        ],
        key_field="id",
        value_field="value",
        conflict_resolution=ReduceDictionaries.ConflictResolution.ERROR,
    )
    with pytest.raises(ValueError, match="Duplicate key found"):
        await node.process(context)


@pytest.mark.asyncio
async def test_reduce_dictionaries_missing_key_field(context: ProcessingContext):
    """Test ReduceDictionaries raises error when key_field is missing."""
    node = ReduceDictionaries(
        dictionaries=[{"other": 1, "value": "test"}],
        key_field="id",
        value_field="value",
    )
    with pytest.raises(ValueError, match="Key field 'id' not found"):
        await node.process(context)


@pytest.mark.asyncio
async def test_reduce_dictionaries_missing_value_field(context: ProcessingContext):
    """Test ReduceDictionaries raises error when value_field is missing."""
    node = ReduceDictionaries(
        dictionaries=[{"id": 1, "other": "test"}],
        key_field="id",
        value_field="value",
    )
    with pytest.raises(ValueError, match="Value field 'value' not found"):
        await node.process(context)


@pytest.mark.asyncio
async def test_reduce_dictionaries_no_value_field(context: ProcessingContext):
    """Test ReduceDictionaries without value_field uses entire dict minus key."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ],
        key_field="id",
        value_field="",  # No value field specified
    )
    result = await node.process(context)
    assert result == {
        1: {"name": "Alice", "age": 30},
        2: {"name": "Bob", "age": 25},
    }


# Helper function for streaming filter tests
async def run_stream_filter_graph(context, input_list, filter_node_type, filter_data):
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


# Test FilterDictByValue streaming node
@pytest.mark.asyncio
async def test_filter_dict_by_value_contains(context: ProcessingContext):
    """Test FilterDictByValue with CONTAINS filter."""
    data = [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@test.org"},
        {"name": "Charlie", "email": "charlie@example.com"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "email", "filter_type": "contains", "criteria": "example"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"Alice", "Charlie"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_starts_with(context: ProcessingContext):
    """Test FilterDictByValue with STARTS_WITH filter."""
    data = [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "administrator"},
        {"name": "Charlie", "role": "user"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "role", "filter_type": "starts_with", "criteria": "admin"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"Alice", "Bob"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_ends_with(context: ProcessingContext):
    """Test FilterDictByValue with ENDS_WITH filter."""
    data = [
        {"name": "report.txt"},
        {"name": "data.csv"},
        {"name": "notes.txt"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "name", "filter_type": "ends_with", "criteria": ".txt"},
    )
    assert len(result) == 2


@pytest.mark.asyncio
async def test_filter_dict_by_value_type_is(context: ProcessingContext):
    """Test FilterDictByValue with TYPE_IS filter."""
    data = [
        {"name": "a", "value": 123},
        {"name": "b", "value": "string"},
        {"name": "c", "value": 456},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "value", "filter_type": "type_is", "criteria": "int"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"a", "c"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_length_greater(context: ProcessingContext):
    """Test FilterDictByValue with LENGTH_GREATER filter."""
    data = [
        {"name": "A", "tags": ["a", "b", "c"]},
        {"name": "B", "tags": ["x"]},
        {"name": "C", "tags": ["p", "q", "r", "s"]},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "tags", "filter_type": "length_greater", "criteria": "2"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_length_less(context: ProcessingContext):
    """Test FilterDictByValue with LENGTH_LESS filter."""
    data = [
        {"name": "A", "text": "short"},
        {"name": "B", "text": "this is a longer text"},
        {"name": "C", "text": "hi"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "text", "filter_type": "length_less", "criteria": "10"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_exact_length(context: ProcessingContext):
    """Test FilterDictByValue with EXACT_LENGTH filter."""
    data = [
        {"name": "A", "code": "AB"},
        {"name": "B", "code": "ABC"},
        {"name": "C", "code": "XY"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "code", "filter_type": "exact_length", "criteria": "2"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


# Test FilterDictByRange streaming node
@pytest.mark.asyncio
async def test_filter_dict_by_range_inclusive(context: ProcessingContext):
    """Test FilterDictByRange with inclusive range."""
    data = [
        {"name": "A", "price": 10},
        {"name": "B", "price": 25},
        {"name": "C", "price": 50},
        {"name": "D", "price": 75},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByRange.get_node_type(),
        {"key": "price", "min_value": 20, "max_value": 60, "inclusive": True},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"B", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_range_exclusive(context: ProcessingContext):
    """Test FilterDictByRange with exclusive range."""
    data = [
        {"name": "A", "score": 1},
        {"name": "B", "score": 2},
        {"name": "C", "score": 3},
        {"name": "D", "score": 4},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByRange.get_node_type(),
        {"key": "score", "min_value": 1, "max_value": 4, "inclusive": False},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"B", "C"}


# Test FilterDictByNumber streaming node
@pytest.mark.asyncio
async def test_filter_dict_by_number_less_than(context: ProcessingContext):
    """Test FilterDictByNumber with LESS_THAN filter."""
    data = [
        {"name": "A", "count": 5},
        {"name": "B", "count": 10},
        {"name": "C", "count": 3},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "count", "filter_type": "less_than", "compare_value": 6},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_number_equal_to(context: ProcessingContext):
    """Test FilterDictByNumber with EQUAL_TO filter."""
    data = [
        {"name": "A", "status": 200},
        {"name": "B", "status": 404},
        {"name": "C", "status": 200},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "status", "filter_type": "equal_to", "compare_value": 200},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_number_even(context: ProcessingContext):
    """Test FilterDictByNumber with EVEN filter."""
    data = [
        {"name": "A", "value": 2},
        {"name": "B", "value": 3},
        {"name": "C", "value": 4},
        {"name": "D", "value": 5},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "value", "filter_type": "even"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_number_odd(context: ProcessingContext):
    """Test FilterDictByNumber with ODD filter."""
    data = [
        {"name": "A", "value": 2},
        {"name": "B", "value": 3},
        {"name": "C", "value": 5},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "value", "filter_type": "odd"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"B", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_number_positive(context: ProcessingContext):
    """Test FilterDictByNumber with POSITIVE filter."""
    data = [
        {"name": "A", "value": 5},
        {"name": "B", "value": -3},
        {"name": "C", "value": 0},
        {"name": "D", "value": 10},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "value", "filter_type": "positive"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "D"}


@pytest.mark.asyncio
async def test_filter_dict_by_number_negative(context: ProcessingContext):
    """Test FilterDictByNumber with NEGATIVE filter."""
    data = [
        {"name": "A", "value": 5},
        {"name": "B", "value": -3},
        {"name": "C", "value": 0},
        {"name": "D", "value": -10},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "value", "filter_type": "negative"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"B", "D"}


# Test FilterDictRegex streaming node
@pytest.mark.asyncio
async def test_filter_dict_regex_search(context: ProcessingContext):
    """Test FilterDictRegex with partial match (search)."""
    data = [
        {"name": "A", "email": "alice@example.com"},
        {"name": "B", "email": "bob@test.org"},
        {"name": "C", "email": "charlie@example.net"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictRegex.get_node_type(),
        {"key": "email", "pattern": r"@example\.", "full_match": False},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_regex_full_match(context: ProcessingContext):
    """Test FilterDictRegex with full match."""
    data = [
        {"name": "A", "code": "AB123"},
        {"name": "B", "code": "XY987"},
        {"name": "C", "code": "AB"},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictRegex.get_node_type(),
        {"key": "code", "pattern": r"[A-Z]{2}\d{3}", "full_match": True},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"A", "B"}


# Test FilterDictByQuery streaming node
@pytest.mark.asyncio
async def test_filter_dict_by_query_simple(context: ProcessingContext):
    """Test FilterDictByQuery with simple condition."""
    data = [
        {"name": "A", "age": 25, "active": True},
        {"name": "B", "age": 30, "active": False},
        {"name": "C", "age": 35, "active": True},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByQuery.get_node_type(),
        {"condition": "age > 28"},
    )
    assert len(result) == 2
    assert {r["name"] for r in result} == {"B", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_query_complex(context: ProcessingContext):
    """Test FilterDictByQuery with complex condition."""
    data = [
        {"name": "A", "age": 25, "score": 80},
        {"name": "B", "age": 30, "score": 90},
        {"name": "C", "age": 35, "score": 70},
        {"name": "D", "age": 22, "score": 95},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByQuery.get_node_type(),
        {"condition": "age < 32 and score > 75"},
    )
    # A (age=25, score=80), B (age=30, score=90), D (age=22, score=95) match the condition
    assert len(result) == 3
    assert {r["name"] for r in result} == {"A", "B", "D"}


@pytest.mark.asyncio
async def test_filter_dict_by_query_empty_condition(context: ProcessingContext):
    """Test FilterDictByQuery with empty condition passes all."""
    data = [
        {"name": "A", "value": 1},
        {"name": "B", "value": 2},
    ]
    result = await run_stream_filter_graph(
        context,
        data,
        FilterDictByQuery.get_node_type(),
        {"condition": ""},
    )
    assert len(result) == 2
