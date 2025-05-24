from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class CollectorNode(GraphNode):
    """
    Collects items from the input and yields a list once the "done" event is received.
    This is the opposite of IteratorNode - it collects items into a list rather than
    iterating over a list.

    Use cases:
    1. Gather results from multiple processing steps into a single list
    2. Collect streaming data into batches for batch processing
    3. Aggregate results from parallel operations
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
    Iterates over a list of items and triggers downstream execution for each item
    by yielding them one by one using `gen_process`.
    """

    input_list: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="The list of items to iterate over."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.control.Iterator"
