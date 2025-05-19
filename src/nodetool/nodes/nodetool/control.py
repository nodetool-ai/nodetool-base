from enum import Enum
from typing import Any, AsyncGenerator, Optional, List
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Event


class If(BaseNode):
    """
    Conditionally executes one of two branches based on a condition.
    control, flow, condition, logic, else, true, false, switch, toggle, flow-control

    Use cases:
    - Branch workflow based on conditions
    - Handle different cases in data processing
    - Implement decision logic
    """

    condition: bool = Field(default=False, description="The condition to evaluate")
    value: Any = Field(default=None, description="The value to pass to the next node")

    @classmethod
    def return_type(cls):
        return {"if_true": Any, "if_false": Any}

    async def gen_process(self, context: Any) -> AsyncGenerator[tuple[str, Any], None]:
        if self.condition:
            yield "if_true", self.value
        else:
            yield "if_false", self.value


class IteratorNode(BaseNode):
    """
    Iterates over a list of items and triggers downstream execution for each item
    by yielding them one by one using `gen_process`.
    """

    input_list: list[Any] = Field(
        default_factory=list, description="The list of items to iterate over."
    )

    @classmethod
    def get_title(cls) -> str:
        return "Iterator"

    @classmethod
    def return_type(cls):
        return {"output": Any, "index": int, "event": Event}

    async def gen_process(self, context: Any) -> AsyncGenerator[tuple[str, Any], None]:
        """Iterate over `self.input_list` and yield each item and its index.

        For each item in the `self.input_list`, this generator yields two tuples:
        first, the item itself associated with the 'output' slot, and second,
        the index of the item associated with the 'index' slot.

        Args:
            context: The execution context for the node. (Currently unused).

        Yields:
            Tuples of (slot_name, value), where `slot_name` is either
            'output' or 'index'.
        """
        for index, item in enumerate(self.input_list):
            yield "output", item
            yield "index", index
            yield "event", Event(name="iterator")

        yield "event", Event(name="done")


class CollectorNode(BaseNode):
    """
    Collects items from the input and yields a list once the "done" event is received.
    This is the opposite of IteratorNode - it collects items into a list rather than
    iterating over a list.

    Use cases:
    1. Gather results from multiple processing steps into a single list
    2. Collect streaming data into batches for batch processing
    3. Aggregate results from parallel operations
    """

    input_item: Any = Field(default=None, description="The input item to collect.")
    event: Event = Field(default=Event(), description="Signal end of stream")

    @classmethod
    def get_title(cls) -> str:
        return "Collector"

    @classmethod
    def return_type(cls):
        return {"output": list[Any], "event": Event}

    async def handle_event(self, context: Any, event: Event):
        """Handle incoming events and items.

        Collects items until a "done" event is received, then yields the collected list.

        Args:
            context: The execution context for the node.
            event: The event to handle.

        Yields:
            Tuples of (slot_name, value), where `slot_name` is either
            'output' or 'event'.
        """
        if not hasattr(self, "_collected_items"):
            self._collected_items = []

        if event.name == "done":
            yield "output", self._collected_items
            yield "event", Event(name="done")
            self._collected_items = []
        elif event.name == "iterator":
            if self.input_item is not None:
                self._collected_items.append(self.input_item)
        else:
            raise ValueError(f"Unknown event: {event.name}")
