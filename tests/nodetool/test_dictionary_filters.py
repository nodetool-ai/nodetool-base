
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.nodes.nodetool.control import ForEach, Collect
from nodetool.nodes.nodetool.dictionary import (
    FilterDictByValue,
    FilterDictByNumber,
    FilterDictByRange,
    FilterDictRegex,
    FilterDictByQuery,
)
from nodetool.nodes.nodetool.output import Output

@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")

async def run_simple_filter_graph(context, input_list, filter_node_type, filter_data):
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

@pytest.mark.asyncio
async def test_filter_dict_by_value(context):
    data = [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "user"},
        {"name": "Charlie", "role": "user"}
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "role", "filter_type": "equals", "criteria": "admin"}
    )
    assert len(res) == 1
    assert res[0]["name"] == "Alice"

@pytest.mark.asyncio
async def test_filter_dict_by_number(context):
    data = [
        {"name": "A", "score": 10},
        {"name": "B", "score": 5},
        {"name": "C", "score": 15}
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "score", "filter_type": "greater_than", "compare_value": 8.0}
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_range(context):
    """Test FilterDictByRange with inclusive range."""
    data = [
        {"name": "A", "price": 10.0},
        {"name": "B", "price": 50.0},
        {"name": "C", "price": 100.0},
        {"name": "D", "price": 150.0},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByRange.get_node_type(),
        {"key": "price", "min_value": 40.0, "max_value": 120.0, "inclusive": True},
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"B", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_range_exclusive(context):
    """Test FilterDictByRange with exclusive range."""
    data = [
        {"name": "A", "score": 10},
        {"name": "B", "score": 20},
        {"name": "C", "score": 30},
        {"name": "D", "score": 40},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByRange.get_node_type(),
        {"key": "score", "min_value": 10, "max_value": 40, "inclusive": False},
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"B", "C"}


@pytest.mark.asyncio
async def test_filter_dict_regex(context):
    """Test FilterDictRegex with partial match."""
    data = [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@test.org"},
        {"name": "Charlie", "email": "charlie@example.com"},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictRegex.get_node_type(),
        {"key": "email", "pattern": r"@example\.com$", "full_match": False},
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"Alice", "Charlie"}


@pytest.mark.asyncio
async def test_filter_dict_regex_full_match(context):
    """Test FilterDictRegex with full match."""
    data = [
        {"name": "A", "code": "ABC123"},
        {"name": "B", "code": "ABC123XYZ"},
        {"name": "C", "code": "ABC999"},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictRegex.get_node_type(),
        {"key": "code", "pattern": r"ABC\d{3}", "full_match": True},
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"A", "C"}


@pytest.mark.asyncio
async def test_filter_dict_by_query(context):
    """Test FilterDictByQuery with pandas query syntax."""
    data = [
        {"name": "A", "age": 25, "score": 80},
        {"name": "B", "age": 30, "score": 90},
        {"name": "C", "age": 35, "score": 70},
        {"name": "D", "age": 40, "score": 95},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByQuery.get_node_type(),
        {"condition": "age >= 30 and score > 75"},
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"B", "D"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_starts_with(context):
    """Test FilterDictByValue with starts_with filter."""
    data = [
        {"name": "Alice Smith"},
        {"name": "Bob Jones"},
        {"name": "Alex Brown"},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "name", "filter_type": "starts_with", "criteria": "Al"},
    )
    assert len(res) == 2
    names = {x["name"] for x in res}
    assert "Alice Smith" in names
    assert "Alex Brown" in names


@pytest.mark.asyncio
async def test_filter_dict_by_value_ends_with(context):
    """Test FilterDictByValue with ends_with filter."""
    data = [
        {"filename": "report.pdf"},
        {"filename": "data.csv"},
        {"filename": "summary.pdf"},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "filename", "filter_type": "ends_with", "criteria": ".pdf"},
    )
    assert len(res) == 2
    assert {x["filename"] for x in res} == {"report.pdf", "summary.pdf"}


@pytest.mark.asyncio
async def test_filter_dict_by_value_contains(context):
    """Test FilterDictByValue with contains filter."""
    data = [
        {"description": "Important meeting"},
        {"description": "Casual chat"},
        {"description": "Important deadline"},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByValue.get_node_type(),
        {"key": "description", "filter_type": "contains", "criteria": "Important"},
    )
    assert len(res) == 2


@pytest.mark.asyncio
async def test_filter_dict_by_number_less_than(context):
    """Test FilterDictByNumber with less_than filter."""
    data = [
        {"name": "A", "count": 5},
        {"name": "B", "count": 10},
        {"name": "C", "count": 15},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "count", "filter_type": "less_than", "compare_value": 12.0},
    )
    assert len(res) == 2
    assert {x["name"] for x in res} == {"A", "B"}


@pytest.mark.asyncio
async def test_filter_dict_by_number_even(context):
    """Test FilterDictByNumber with even filter."""
    data = [
        {"id": 1, "value": 2},
        {"id": 2, "value": 3},
        {"id": 3, "value": 4},
        {"id": 4, "value": 5},
    ]
    res = await run_simple_filter_graph(
        context,
        data,
        FilterDictByNumber.get_node_type(),
        {"key": "value", "filter_type": "even", "compare_value": 0},
    )
    assert len(res) == 2
    assert {x["id"] for x in res} == {1, 3}
