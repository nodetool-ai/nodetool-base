
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.dsl.nodetool.numbers import FilterNumber, FilterNumberRange
from nodetool.dsl.nodetool.output import ListOutput
from nodetool.dsl.nodetool.control import ForEach, Collect

@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")

@pytest.mark.asyncio
async def test_filter_number_greater_than():
    # Stream: [1, 2, 3, 4, 5] -> Filter(>3) -> [4, 5]
    source = ForEach(input_list=[1, 2, 3, 4, 5])
    
    # We need to construct the filter node
    # Since we are using DSL, we assume DSL matches Node definition
    # But wait, DSL generation might not be instant if I just added the file.
    # The DSL is dynamically generated from nodes? 
    # Usually in this codebase, DSL is available or needs to be used via direct node construction if DSL isn't updated.
    # The tests I saw use `from nodetool.dsl.nodetool...` implies DSL exists.
    # However, I just created `numbers.py`. It likely won't have DSL yet unless the system auto-generates it on the fly or I shouldn't use DSL for this test.
    # I should use APINode and run_workflow manually like in `test_control_nodes.py` to be safe and test the NODE implementation, not the DSL.
    pass

from nodetool.types.graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.nodes.nodetool.control import ForEach, Collect
from nodetool.nodes.nodetool.numbers import FilterNumber, FilterNumberRange
from nodetool.nodes.nodetool.output import ListOutput

async def run_simple_filter_graph(context, input_list, filter_node_type, filter_data):
    nodes = [
        APINode(id="source", type=ForEach.get_node_type(), data={"input_list": input_list}),
        APINode(id="filter", type=filter_node_type, data=filter_data),
        APINode(id="collect", type=Collect.get_node_type(), data={}),
        APINode(id="out", type=ListOutput.get_node_type(), data={"name": "result"}),
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
async def test_filter_number_greater_than(context):
    res = await run_simple_filter_graph(
        context, 
        [1, 2, 3, 4, 5], 
        FilterNumber.get_node_type(), 
        {"filter_type": "greater_than", "compare_value": 3.0}
    )
    assert res == [4, 5]

@pytest.mark.asyncio
async def test_filter_number_less_than(context):
    res = await run_simple_filter_graph(
        context, 
        [1, 2, 3, 4, 5], 
        FilterNumber.get_node_type(), 
        {"filter_type": "less_than", "compare_value": 3.0}
    )
    assert res == [1, 2]

@pytest.mark.asyncio
async def test_filter_number_even(context):
    res = await run_simple_filter_graph(
        context, 
        [1, 2, 3, 4, 5], 
        FilterNumber.get_node_type(), 
        {"filter_type": "even"}
    )
    assert res == [2, 4]

@pytest.mark.asyncio
async def test_filter_number_range(context):
    res = await run_simple_filter_graph(
        context, 
        [1, 2, 3, 4, 5], 
        FilterNumberRange.get_node_type(), 
        {"min_value": 2.0, "max_value": 4.0, "inclusive": True}
    )
    assert res == [2, 3, 4]

@pytest.mark.asyncio
async def test_filter_number_range_exclusive(context):
    res = await run_simple_filter_graph(
        context, 
        [1, 2, 3, 4, 5], 
        FilterNumberRange.get_node_type(), 
        {"min_value": 2.0, "max_value": 4.0, "inclusive": False}
    )
    assert res == [3]
