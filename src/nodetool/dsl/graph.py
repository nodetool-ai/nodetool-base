from __future__ import annotations

from typing import Any, Dict, Generic, Iterable, TypeVar

from pydantic import BaseModel, ConfigDict

from nodetool.dsl.handles import OutputHandle
from nodetool.workflows.processing_context import ProcessingContext

T = TypeVar("T")


class GraphNode(BaseModel, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __hash__(self) -> int:
        return id(self)

    @classmethod
    def get_node_class(cls):
        raise NotImplementedError

    @property
    def output(self) -> OutputHandle[T]:
        # Multi-output nodes override via OutputsProxy in generated code.
        return OutputHandle(self, "output")


class SingleOutputGraphNode(GraphNode[T]):
    """Graph node with a single output named 'output'."""

    @property
    def output(self) -> OutputHandle[T]:
        return OutputHandle(self, "output")


async def graph_result(*nodes: Iterable[GraphNode[Any]] | GraphNode[Any]) -> Dict[str, Any]:
    """Evaluate one or more output graph nodes and return their results."""

    # Flatten arguments into a list
    flattened: list[GraphNode[Any]] = []
    for node in nodes:
        if isinstance(node, GraphNode):
            flattened.append(node)
        else:
            flattened.extend(list(node))  # type: ignore[arg-type]

    context = ProcessingContext(user_id="graph", auth_token=None)
    cache: Dict[GraphNode[Any], Dict[str, Any]] = {}

    async def resolve_value(value: Any) -> Any:
        if isinstance(value, OutputHandle):
            return await evaluate_node(value.node, value.output_name)
        if isinstance(value, list):
            return [await resolve_value(v) for v in value]
        if isinstance(value, dict):
            return {k: await resolve_value(v) for k, v in value.items()}
        return value

    async def evaluate_node(node: GraphNode[Any], output_name: str = "output") -> Any:
        if node in cache and output_name in cache[node]:
            return cache[node][output_name]

        node_cls = node.get_node_class()
        kwargs: Dict[str, Any] = {}
        for field_name, value in node.__dict__.items():
            if field_name.startswith("_"):
                continue
            kwargs[field_name] = await resolve_value(value)

        instance = node_cls(**kwargs)
        if hasattr(instance, "process"):
            result = await instance.process(context)
        else:  # pragma: no cover - safety net
            result = None

        formatted = (
            instance.result_for_all_outputs(result)
            if hasattr(instance, "result_for_all_outputs")
            else {"output": result}
        )
        cache[node] = formatted
        return formatted.get(output_name)

    results: Dict[str, Any] = {}
    for node in flattened:
        key = getattr(node, "name", None) or node.__class__.__name__
        results[key] = await evaluate_node(node)
    return results

