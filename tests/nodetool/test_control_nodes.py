import os
from nodetool.config.logging_config import configure_logging
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.workflows.types import OutputUpdate
from nodetool.workflows.run_workflow import run_workflow
import pytest
import asyncio
from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.types.graph import Node as APINode, Edge as APIEdge, Graph as APIGraph


# Reuse the generic OutputNode as a sink via API graph
OUTPUT_NODE_TYPE = "nodetool.workflows.base_node.OutputNode"


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_if_routes_true_and_not_false(context: ProcessingContext):
    import nodetool.nodes.nodetool.control

    nodes = [
        APINode(
            id="if",
            type="nodetool.control.If",
            data={"condition": True, "value": "hello"},
        ),
        APINode(id="out_true", type=OUTPUT_NODE_TYPE, data={"name": "true_sink"}),
        APINode(id="out_false", type=OUTPUT_NODE_TYPE, data={"name": "false_sink"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="if",
            sourceHandle="if_true",
            target="out_true",
            targetHandle="value",
        ),
        APIEdge(
            id="e2",
            source="if",
            sourceHandle="if_false",
            target="out_false",
            targetHandle="value",
        ),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(graph=graph)
    context = ProcessingContext(user_id="test", auth_token="test")
    found_true = False

    async for msg in run_workflow(req, context=context):
        print(msg)
        if isinstance(msg, OutputUpdate):
            if msg.node_id == "out_true":
                found_true = True
                assert msg.value == "hello"
            elif msg.node_id == "out_false":
                pytest.fail("False branch should not emit")

    assert found_true, "True branch should emit"


@pytest.mark.asyncio
async def test_if_streams_values_with_static_true_condition(context: ProcessingContext):
    import nodetool.nodes.nodetool.control

    class ValueProducer:
        @staticmethod
        def node(items: list[Any], delay: float = 0.0) -> APINode:
            return APINode(
                id="vprod",
                type="test.control.ValueProducer",
                data={"items": items, "delay": delay},
            )

    # Define a streaming producer inline and register via NODE_BY_TYPE by creating the class
    from nodetool.workflows.base_node import BaseNode

    class _ValueProducerNode(BaseNode):  # type: ignore
        items: list[Any] = []
        delay: float = 0.0

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.ValueProducer"

        @classmethod
        def return_type(cls):
            return {"out": Any}

        async def gen_process(self, _):  # type: ignore
            for x in self.items:
                yield ("out", x)
                if self.delay:
                    await asyncio.sleep(self.delay)

    nodes = [
        ValueProducer.node(["a", "b", "c"], delay=0.01),
        APINode(id="if", type="nodetool.control.If", data={"condition": True}),
        APINode(id="out", type=OUTPUT_NODE_TYPE, data={"name": "passed"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="vprod",
            sourceHandle="out",
            target="if",
            targetHandle="value",
        ),
        APIEdge(
            id="e2",
            source="if",
            sourceHandle="if_true",
            target="out",
            targetHandle="value",
        ),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(graph=graph)
    values = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            values.append(msg.value)

    assert values == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_if_toggles_between_branches_with_streaming_condition_and_values(
    context: ProcessingContext,
):
    from nodetool.workflows.base_node import BaseNode
    import nodetool.nodes.nodetool.control

    class _CondProducer(BaseNode):  # type: ignore
        values: list[bool] = [True, False]
        delays: list[float] = [0.0, 0.02]

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.CondProducer"

        @classmethod
        def return_type(cls):
            return {"out": bool}

        async def gen_process(self, _):  # type: ignore
            for v, d in zip(self.values, self.delays):
                yield ("out", v)
                if d:
                    await asyncio.sleep(d)

    class _ValProducer(BaseNode):  # type: ignore
        values: list[str] = ["A", "B", "C"]
        delay: float = 0.01

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.ValProducer"

        @classmethod
        def return_type(cls):
            return {"out": str}

        async def gen_process(self, _):  # type: ignore
            for v in self.values:
                yield ("out", v)
                if self.delay:
                    await asyncio.sleep(self.delay)

    nodes = [
        APINode(
            id="cond",
            type="test.control.CondProducer",
            data={"values": [True, True, False], "delays": [0.0, 0.01, 0.02]},
        ),
        APINode(
            id="val",
            type="test.control.ValProducer",
            data={"values": ["A", "B", "C"], "delay": 0.01},
        ),
        APINode(id="if", type="nodetool.control.If", data={}),
        APINode(id="out_true", type=OUTPUT_NODE_TYPE, data={"name": "true_sink"}),
        APINode(id="out_false", type=OUTPUT_NODE_TYPE, data={"name": "false_sink"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="cond",
            sourceHandle="out",
            target="if",
            targetHandle="condition",
        ),
        APIEdge(
            id="e2", source="val", sourceHandle="out", target="if", targetHandle="value"
        ),
        APIEdge(
            id="e3",
            source="if",
            sourceHandle="if_true",
            target="out_true",
            targetHandle="value",
        ),
        APIEdge(
            id="e4",
            source="if",
            sourceHandle="if_false",
            target="out_false",
            targetHandle="value",
        ),
    ]
    # Ensure we pair inputs per index to avoid early None routes
    graph = APIGraph(
        nodes=[
            APINode(**{**n.model_dump(), **({"sync_mode": "on_any"} if n.id != "if" else {})}) if isinstance(n, APINode) else n  # type: ignore
            for n in nodes
        ],
        edges=edges,
    )

    req = RunJobRequest(graph=graph)
    # Set sync_mode zip_all for If to pair per-index
    req = RunJobRequest(
        graph=APIGraph(
            nodes=[
                APINode(
                    id=n.id,
                    type=n.type,
                    data=n.data,
                    ui_properties=n.ui_properties,
                    dynamic_properties=n.dynamic_properties,
                    dynamic_outputs=n.dynamic_outputs,
                    sync_mode=("zip_all" if n.id == "if" else n.sync_mode),
                )
                for n in graph.nodes
            ],
            edges=graph.edges,
        )
    )

    true_values = []
    false_values = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate):
            if msg.node_id == "out_true":
                true_values.append(msg.value)
            elif msg.node_id == "out_false":
                false_values.append(msg.value)

    # With the delays configured, expected routing: A->true, B->true, C->false
    assert true_values == ["A", "B"]
    assert false_values == ["C"]


@pytest.mark.asyncio
async def test_reroute_passes_stream_through(context: ProcessingContext):
    from nodetool.workflows.base_node import BaseNode
    import nodetool.nodes.nodetool.control

    class _Stream(BaseNode):  # type: ignore
        values: list[int] = [1, 2, 3]

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.IntStream"

        @classmethod
        def return_type(cls):
            return {"out": int}

        async def gen_process(self, _):  # type: ignore
            for v in self.values:
                yield ("out", v)

    nodes = [
        APINode(id="src", type="test.control.IntStream", data={"values": [1, 2, 3]}),
        APINode(id="reroute", type="nodetool.control.Reroute", data={}),
        APINode(id="out", type=OUTPUT_NODE_TYPE, data={"name": "sink"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="src",
            sourceHandle="out",
            target="reroute",
            targetHandle="input_value",
        ),
        APIEdge(
            id="e2",
            source="reroute",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(graph=graph)
    values = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            values.append(msg.value)

    assert values == [1, 2, 3]


@pytest.mark.asyncio
async def test_collect_node_aggregates_stream(context: ProcessingContext):
    from nodetool.workflows.base_node import BaseNode
    import nodetool.nodes.nodetool.control

    class _Stream(BaseNode):  # type: ignore
        values: list[str] = ["x", "y", "z"]

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.StrStream"

        @classmethod
        def return_type(cls):
            return {"out": str}

        async def gen_process(self, _):  # type: ignore
            for v in self.values:
                yield ("out", v)
                await asyncio.sleep(0.005)

    nodes = [
        APINode(
            id="src", type="test.control.StrStream", data={"values": ["x", "y", "z"]}
        ),
        APINode(id="collect", type="nodetool.control.CollectNode", data={}),
        APINode(id="out", type=OUTPUT_NODE_TYPE, data={"name": "items"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="src",
            sourceHandle="out",
            target="collect",
            targetHandle="input_item",
        ),
        APIEdge(
            id="e2",
            source="collect",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(graph=graph)
    values = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            values.append(msg.value)

    assert values == [["x", "y", "z"]]


@pytest.mark.asyncio
async def test_collect_node_handles_empty_stream(context: ProcessingContext):
    import nodetool.nodes.nodetool.control

    nodes = [
        APINode(id="collect", type="nodetool.control.CollectNode", data={}),
        APINode(id="out", type=OUTPUT_NODE_TYPE, data={"name": "items"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="collect",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(graph=graph)
    values = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            values.append(msg.value)

    assert values == [[]]


@pytest.mark.asyncio
async def test_foreach_emits_last_item_and_index_only_in_current_engine(
    context: ProcessingContext,
):
    import nodetool.nodes.nodetool.control

    # ForEach currently runs in non-streaming mode; multiple emits are collected, last one routed.
    nodes = [
        APINode(
            id="each",
            type="nodetool.control.ForEach",
            data={"input_list": [10, 11, 12]},
        ),
        APINode(id="out_item", type=OUTPUT_NODE_TYPE, data={"name": "item"}),
        APINode(id="out_index", type=OUTPUT_NODE_TYPE, data={"name": "idx"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="each",
            sourceHandle="output",
            target="out_item",
            targetHandle="value",
        ),
        APIEdge(
            id="e2",
            source="each",
            sourceHandle="index",
            target="out_index",
            targetHandle="value",
        ),
    ]
    graph = APIGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(graph=graph)
    item_values = []
    idx_values = []
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate):
            if msg.node_id == "out_item":
                item_values.append(msg.value)
            elif msg.node_id == "out_index":
                idx_values.append(msg.value)

    assert item_values == [12]
    assert idx_values == [2]
