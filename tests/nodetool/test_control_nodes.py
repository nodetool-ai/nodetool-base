from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.types import OutputUpdate
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.types.api_graph import Node as APINode, Edge as APIEdge, Graph as APIGraph
from nodetool.nodes.nodetool.output import Output
from nodetool.nodes.nodetool.control import If, ForEach, Reroute, Collect
from nodetool.nodes.nodetool.list import GenerateSequence
import pytest
import asyncio
from typing import AsyncGenerator, TypedDict

DEFAULT_TIMEOUT_SECONDS = 1.0


class _AsyncTimeoutIterator:
    def __init__(self, agen, timeout: float):
        self._agen = agen
        self._timeout = timeout

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await asyncio.wait_for(self._agen.__anext__(), self._timeout)
        except asyncio.TimeoutError:
            if hasattr(self._agen, "aclose"):
                await self._agen.aclose()  # type: ignore[attr-defined]
            pytest.fail(f"run_workflow did not yield within {self._timeout} seconds")
        except StopAsyncIteration:
            raise


def iter_with_timeout(agen, timeout: float = DEFAULT_TIMEOUT_SECONDS):
    return _AsyncTimeoutIterator(agen.__aiter__(), timeout)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_if_routes_true_and_not_false(context: ProcessingContext):

    nodes = [
        APINode(
            id="if",
            type=If.get_node_type(),
            data={"condition": True, "value": "hello"},
        ),
        APINode(
            id="out_true",
            type=Output.get_node_type(),
            data={"name": "true_sink"},
        ),
        APINode(
            id="out_false",
            type=Output.get_node_type(),
            data={"name": "false_sink"},
        ),
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
    found_false = False

    async for msg in iter_with_timeout(
        run_workflow(req, context=context, use_thread=False)
    ):
        if isinstance(msg, OutputUpdate):
            if msg.node_id == "out_true":
                found_true = True
                assert msg.value == "hello"
            elif msg.node_id == "out_false":
                found_false = True
                # Output nodes emit None by default when they don't receive input
                assert msg.value is None

    assert found_true, "True branch should emit"
    assert found_false, "False branch should emit default value"


@pytest.mark.asyncio
async def test_if_streams_values_with_static_true_condition(context: ProcessingContext):

    nodes = [
        APINode(
            id="gen",
            type=GenerateSequence.get_node_type(),
            data={"start": 0, "stop": 3, "step": 1},
        ),
        APINode(id="if", type=If.get_node_type(), data={"condition": True}),
        APINode(id="out", type=Output.get_node_type(), data={"name": "passed"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="gen",
            sourceHandle="output",
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
    async for msg in iter_with_timeout(
        run_workflow(req, context=context, use_thread=False)
    ):
        if isinstance(msg, OutputUpdate):
            values.append(msg.value)

    assert values == [0, 1, 2]


@pytest.mark.asyncio
async def test_if_toggles_between_branches_with_streaming_condition_and_values(
    context: ProcessingContext,
):

    class _CondProducer(BaseNode):
        values: list[bool] = [True, False]
        delays: list[float] = [0.0, 0.02]

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.CondProducer"

        class OutputType(TypedDict):
            out: bool

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[OutputType, None]:
            for v, d in zip(self.values, self.delays):
                yield {"out": v}
                if d:
                    await asyncio.sleep(d)

    class _ValProducer(BaseNode):  # type: ignore
        values: list[str] = ["A", "B", "C"]
        delay: float = 0.01

        @classmethod
        def get_node_type(cls) -> str:
            return "test.control.ValProducer"

        class OutputType(TypedDict):
            out: str

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[OutputType, None]:
            for v in self.values:
                yield {"out": v}
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
        APINode(id="if", type=If.get_node_type(), data={}),
        APINode(
            id="out_true",
            type=Output.get_node_type(),
            data={"name": "true_sink"},
        ),
        APINode(
            id="out_false",
            type=Output.get_node_type(),
            data={"name": "false_sink"},
        ),
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
    async for msg in iter_with_timeout(run_workflow(req, context=context)):
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
    nodes = [
        APINode(
            id="src",
            type=GenerateSequence.get_node_type(),
            data={"start": 0, "stop": 3, "step": 1},
        ),
        APINode(id="reroute", type=Reroute.get_node_type(), data={}),
        APINode(id="out", type=Output.get_node_type(), data={"name": "sink"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="src",
            sourceHandle="output",
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

    async for msg in iter_with_timeout(
        run_workflow(req, context=context, use_thread=False)
    ):
        if isinstance(msg, OutputUpdate):
            values.append(msg.value)

    assert values == [0, 1, 2]


@pytest.mark.asyncio
async def test_collect_node_aggregates_stream(context: ProcessingContext):

    nodes = [
        APINode(
            id="src",
            type=GenerateSequence.get_node_type(),
            data={"start": 0, "stop": 3, "step": 1},
        ),
        APINode(id="collect", type=Collect.get_node_type(), data={}),
        APINode(id="out", type=Output.get_node_type(), data={"name": "items"}),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="src",
            sourceHandle="output",
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
    async for msg in iter_with_timeout(
        run_workflow(req, context=context, use_thread=False)
    ):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            values.append(msg.value)

    assert values == [[0, 1, 2]]


@pytest.mark.asyncio
async def test_collect_node_handles_empty_stream(context: ProcessingContext):
    nodes = [
        APINode(id="collect", type=Collect.get_node_type(), data={}),
        APINode(id="out", type=Output.get_node_type(), data={"name": "items"}),
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
    async for msg in iter_with_timeout(
        run_workflow(req, context=context, use_thread=False)
    ):
        if isinstance(msg, OutputUpdate) and msg.node_id == "out":
            values.append(msg.value)

    assert values == [[]]


@pytest.mark.asyncio
async def test_foreach_emits_last_item_and_index_only_in_current_engine(
    context: ProcessingContext,
):

    # ForEach currently runs in non-streaming mode; multiple emits are collected, last one routed.
    nodes = [
        APINode(
            id="each",
            type=ForEach.get_node_type(),
            data={"input_list": [10, 11, 12]},
        ),
        APINode(
            id="out_item", type=Output.get_node_type(), data={"name": "item"}
        ),
        APINode(
            id="out_index", type=Output.get_node_type(), data={"name": "idx"}
        ),
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
    outputs = {}
    async for msg in iter_with_timeout(
        run_workflow(req, context=context, use_thread=False)
    ):
        if isinstance(msg, OutputUpdate):
            outputs[msg.node_id] = msg.value

    assert outputs.get("out_item") == 12
    assert outputs.get("out_index") == 2
