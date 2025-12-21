from enum import Enum
from typing import AsyncGenerator, TypedDict

from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class FilterNumber(BaseNode):
    """
    Filter number stream based on numerical conditions (comparison, parity, sign).

    Processes streaming numbers and emits only those matching the filter criteria.
    Supports comparison filters (>, <, ==), parity filters (even/odd), and sign
    filters (positive/negative). Streaming configuration values can update filter
    dynamically.

    Parameters:
    - value (required): Input number stream or single value
    - filter_type (required, default=greater_than): Type of filter to apply
    - compare_value (optional, default=0): Reference value for comparison filters

    Yields: Dictionary with "output" (number) for each matching value

    Typical usage: Filter numeric streams, extract values meeting criteria, or
    implement numeric predicates. Precede with number-generating nodes. Follow
    with Collect or further stream processing.

    filter, numbers, numeric, stream
    """

    class FilterNumberType(str, Enum):
        GREATER_THAN = "greater_than"
        LESS_THAN = "less_than"
        EQUAL_TO = "equal_to"
        EVEN = "even"
        ODD = "odd"
        POSITIVE = "positive"
        NEGATIVE = "negative"

    value: float = Field(default=0.0, description="Input number stream")
    filter_type: FilterNumberType = Field(
        default=FilterNumberType.GREATER_THAN, description="The type of filter to apply"
    )
    compare_value: float = Field(
        default=0,
        description="The comparison value (for greater_than, less_than, equal_to)",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: float

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        # Track latest config values in case they are streamed
        current_filter_type = self.filter_type
        current_compare_value = self.compare_value

        async for handle, item in self.iter_any_input():
            if handle == "filter_type":
                current_filter_type = item
                continue
            elif handle == "compare_value":
                current_compare_value = item
                continue
            elif handle == "value":
                # Process the number
                num = item
                if not isinstance(num, (int, float)):
                    continue

                matched = False
                if current_filter_type == self.FilterNumberType.GREATER_THAN:
                     if current_compare_value is not None and num > current_compare_value:
                         matched = True
                elif current_filter_type == self.FilterNumberType.LESS_THAN:
                     if current_compare_value is not None and num < current_compare_value:
                         matched = True
                elif current_filter_type == self.FilterNumberType.EQUAL_TO:
                     if current_compare_value is not None and num == current_compare_value:
                         matched = True
                elif current_filter_type == self.FilterNumberType.EVEN:
                     if num % 2 == 0:
                         matched = True
                elif current_filter_type == self.FilterNumberType.ODD:
                     if num % 2 != 0:
                         matched = True
                elif current_filter_type == self.FilterNumberType.POSITIVE:
                     if num > 0:
                         matched = True
                elif current_filter_type == self.FilterNumberType.NEGATIVE:
                     if num < 0:
                         matched = True
                
                if matched:
                    yield {"output": num}


class FilterNumberRange(BaseNode):
    """
    Filters a stream of numbers to find values within a specified range.
    filter, numbers, range, between, stream

    Use cases:
    - Find numbers within a specific range
    - Filter data points within bounds
    - Implement range-based filtering
    """

    value: float = Field(default=0.0, description="Input number stream")
    min_value: float = Field(default=0, description="Minimum value")
    max_value: float = Field(default=0, description="Maximum value")
    inclusive: bool = Field(default=True, description="Inclusive bounds")

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: float

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        current_min = self.min_value
        current_max = self.max_value
        current_inclusive = self.inclusive

        async for handle, item in self.iter_any_input():
            if handle == "min_value":
                current_min = item
                continue
            elif handle == "max_value":
                current_max = item
                continue
            elif handle == "inclusive":
                current_inclusive = item
                continue
            elif handle == "value":
                num = item
                if not isinstance(num, (int, float)):
                    continue
                
                matched = False
                if current_inclusive:
                    if current_min <= num <= current_max:
                        matched = True
                else:
                    if current_min < num < current_max:
                        matched = True
                
                if matched:
                    yield {"output": num}
