from typing import Any, AsyncGenerator, Type, TypedDict
from collections import deque
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.io import NodeInputs, NodeOutputs


class If(BaseNode):
    """
    Route values to different branches based on boolean condition.

    Evaluates a condition and emits the value to either if_true or if_false output.
    Supports both static conditions and streaming condition/value pairs. When used
    with streams, routes each value according to paired or latest condition.

    Parameters:
    - condition (required, default=False): Boolean to evaluate, or stream of booleans
    - value (optional, default=()): Value to route, or stream of values

    Returns: Dictionary with "if_true" (value or None) and "if_false" (value or None).
    Only one field is set per emission.

    Typical usage: Implement conditional logic in workflows, handle different data
    cases, or filter streams. Precede with comparison or boolean logic nodes. Follow
    with separate processing paths for true/false cases.

    control, flow, condition, logic, else, true, false, switch, toggle, flow-control
    """

    condition: bool = Field(default=False, description="The condition to evaluate")
    value: Any = Field(default=(), description="The value to pass to the next node")

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:  # type: ignore[override]
        # Treat inbound values/conditions as streams when present
        return True

    class OutputType(TypedDict):
        if_true: Any
        if_false: Any

    async def gen_process(self, context: Any) -> AsyncGenerator[OutputType, None]:
        # Stream-aware implementation: route each incoming value according to the
        # latest condition (static property or streamed condition updates). If no
        # inbound streams arrive, fall back to a single emission using configured properties.

        emitted_any = False

        # Maintain the latest condition and optional pairing queues
        current_condition: bool = bool(self.condition)
        condition_queue: deque[bool] = deque()
        value_queue: deque[Any] = deque()

        async for handle, item in self.iter_any_input():
            if handle == "condition":
                try:
                    current_condition = bool(item)
                except Exception:
                    current_condition = bool(item)
                condition_queue.append(current_condition)
            elif handle == "value":
                value_queue.append(item)

            # While we have values to emit, pair with the next condition if available,
            # otherwise use the latest known condition (static condition behaves like infinite supply).
            while value_queue:
                value = value_queue.popleft()
                cond = (
                    condition_queue.popleft() if condition_queue else current_condition
                )
                emitted_any = True
                if cond:
                    yield {"if_true": value, "if_false": None}
                else:
                    yield {"if_true": None, "if_false": value}

        # Fallback for the case with no inbound streams: emit once using configured properties
        if not emitted_any:
            if current_condition:
                yield {"if_true": self.value, "if_false": None}
            else:
                yield {"if_true": None, "if_false": self.value}


class ForEach(BaseNode):
    """
    Iterate over list and emit each element individually with its index.

    Converts a list into a stream of individual items, allowing downstream nodes
    to process each element separately. Emits items sequentially in order.

    Parameters:
    - input_list (required): List of items to iterate over

    Yields: Dictionary with "output" (current item) and "index" (zero-based position)
    for each list element

    Typical usage: Process list elements individually, drive parallel operations per
    item, or transform collections. Precede with list generation or API nodes that
    return arrays. Follow with item-level processing nodes, then Collect to reaggregate.

    iterator, loop, list, sequence
    """

    input_list: list[Any] = Field(
        default=[], description="The list of items to iterate over."
    )

    @classmethod
    def get_title(cls) -> str:
        return "For Each"

    class OutputType(TypedDict):
        output: Any
        index: int

    async def gen_process(self, context: Any) -> AsyncGenerator[OutputType, None]:
        for index, item in enumerate(self.input_list):
            yield {"output": item, "index": index}


class Collect(BaseNode):
    """
    Accumulate all stream items into a single list.

    Consumes an input stream completely, gathering all emitted items, then returns
    them as a single list. Blocks until the upstream stream completes.

    Parameters:
    - input_item (required): Stream of items to collect (use streaming input)

    Returns: Dictionary with "output" (list of all collected items)

    Typical usage: Aggregate results from ForEach loops, gather parallel processing
    results, or batch streaming data. Follow ForEach or other streaming nodes. Use
    before nodes that expect complete collections rather than individual items.

    collector, aggregate, list, stream
    """

    input_item: Any = Field(default=(), description="The input item to collect.")

    @classmethod
    def get_title(cls) -> str:
        return "Collect"

    class OutputType(TypedDict):
        output: list[Any]

    async def run(self, context: Any, inputs: NodeInputs, outputs: NodeOutputs) -> None:
        collected_items = []
        async for input_item in inputs.stream("input_item"):
            collected_items.append(input_item)
        await outputs.emit("output", collected_items)

    @classmethod
    def is_streaming_input(cls) -> bool:  # type: ignore[override]
        # Consume inbound values as a stream to avoid pre-gather races in actor
        return True


class Reroute(BaseNode):
    """
    Pass value through unchanged to simplify workflow visual layout.

    Acts as a passthrough connector that forwards the input value without
    modification. Useful for organizing complex workflows and routing connections
    more cleanly in the visual editor.

    Parameters:
    - input_value (required): Value to pass through

    Returns: The input value unchanged

    Typical usage: Organize visual layouts, reduce crossing connections, create
    cleaner workflow diagrams, or split one output to multiple consumers. Can be
    placed anywhere to improve readability without affecting data flow.

    reroute, passthrough, organize, tidy, flow, connection, redirect
    """

    input_value: Any = Field(
        default=(), description="Value to pass through unchanged"
    )

    @classmethod
    def get_title(cls) -> str:
        return "Reroute"

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def return_type(cls) -> Type:
        return Any  # type: ignore

    async def run(self, context: Any, inputs: NodeInputs, outputs: NodeOutputs):
        async for input_item in inputs.stream("input_value"):
            await outputs.emit("output", input_item)
