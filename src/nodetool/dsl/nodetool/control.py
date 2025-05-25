from pydantic import Field
from pydantic_core import PydanticUndefined
from typing import Any
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class CollectorNode(GraphNode):
    """
    Collect items until a "done" event and return them as a list.
    collector, aggregate, list, stream

    Use cases:
    - Gather results from multiple processing steps
    - Collect streaming data into batches
    - Aggregate outputs from parallel operations
    """

    input_item: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="The input item to collect."
    )
    event: types.Event | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Event(type="event", name="", payload={}),
        description="Signal end of stream",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.control.Collector"


class If(GraphNode):
    """
    Conditionally executes one of two branches based on a condition.
    control, flow, condition, logic, else, true, false, switch, toggle, flow-control

    Use cases:
    - Branch workflow based on conditions
    - Handle different cases in data processing
    - Implement decision logic
    """

    condition: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="The condition to evaluate"
    )
    value: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="The value to pass to the next node"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.control.If"


class IteratorNode(GraphNode):
    """
    Iterate over a list and emit each item sequentially.
    iterator, loop, list, sequence

    Use cases:
    - Process each item of a collection in order
    - Drive downstream nodes with individual elements
    """

    input_list: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="The list of items to iterate over."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.control.Iterator"
