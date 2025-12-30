
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.nodes.nodetool.control import ForEach, Collect
from nodetool.nodes.nodetool.dictionary import FilterDictByValue, FilterDictByNumber
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
