from typing import Any
from collections import deque
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.io import NodeInputs, NodeOutputs


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
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:  # type: ignore[override]
        # Treat inbound values/conditions as streams when present
        return True

    @classmethod
    def return_type(cls):
        return {"if_true": Any, "if_false": Any}

    async def gen_process(self, context: Any) -> Any:
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
                    yield "if_true", value
                else:
                    yield "if_false", value

        # Fallback for the case with no inbound streams: emit once using configured properties
        if not emitted_any:
            if current_condition:
                yield "if_true", self.value
            else:
                yield "if_false", self.value


class ForEach(BaseNode):
    """
    Iterate over a list and emit each item sequentially.
    iterator, loop, list, sequence

    Use cases:
    - Process each item of a collection in order
    - Drive downstream nodes with individual elements
    """

    input_list: list[Any] = Field(
        default=[], description="The list of items to iterate over."
    )

    @classmethod
    def get_title(cls) -> str:
        return "For Each"

    @classmethod
    def return_type(cls):
        return {"output": Any, "index": int}

    async def run(self, context: Any, inputs: NodeInputs, outputs: NodeOutputs):
        for index, item in enumerate(self.input_list):
            await outputs.emit("output", item)
            await outputs.emit("index", index)


class Collect(BaseNode):
    """
    Collect items until the end of the stream and return them as a list.
    collector, aggregate, list, stream

    Use cases:
    - Gather results from multiple processing steps
    - Collect streaming data into batches
    - Aggregate outputs from parallel operations
    """

    input_item: Any = Field(default=None, description="The input item to collect.")

    @classmethod
    def get_title(cls) -> str:
        return "Collect"

    @classmethod
    def return_type(cls):
        return {"output": list[Any]}

    async def run(self, context: Any, inputs: NodeInputs, outputs: NodeOutputs):
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
    Pass data through unchanged for tidier workflow layouts.
    reroute, passthrough, organize, tidy, flow, connection, redirect

    Use cases:
    - Organize complex workflows by routing connections
    - Create cleaner visual layouts
    - Redirect data flow without modification
    """

    input_value: Any = Field(
        default=None, description="Value to pass through unchanged"
    )

    @classmethod
    def get_title(cls) -> str:
        return "Reroute"

    @classmethod
    def return_type(cls):
        return {"output": Any}

    async def process(self, context: Any) -> Any:
        # Return a mapping from output slot name to value as expected by the runner
        return {"output": self.input_value}
