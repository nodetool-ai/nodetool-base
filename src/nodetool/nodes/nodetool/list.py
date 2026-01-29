from datetime import datetime
from enum import Enum
from functools import reduce
from io import BytesIO
import random
from pydantic import Field
from nodetool.metadata.types import TextRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from typing import Any, AsyncGenerator, TypedDict

class Length(BaseNode):
    """
    Calculates the length of a list.
    list, count, size

    Use cases:
    - Determine the number of elements in a list
    - Check if a list is empty
    - Validate list size constraints
    """

    values: list[Any] = []

    async def process(self, context: ProcessingContext) -> int:
        return len(self.values)


class ListRange(BaseNode):
    """
    Generates a list of integers within a specified range.
    list, range, sequence, numbers

    Use cases:
    - Create numbered lists
    - Generate index sequences
    - Produce arithmetic progressions
    """

    start: int = 0
    stop: int = 0
    step: int = 1

    async def process(self, context: ProcessingContext) -> list[int]:
        return list(range(self.start, self.stop, self.step))


class GenerateSequence(BaseNode):
    """
    Iterates over a sequence of numbers.
    list, range, sequence, numbers
    """

    start: int = 0
    stop: int = 0
    step: int = 1

    class OutputType(TypedDict):
        output: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        for i in range(self.start, self.stop, self.step):
            yield {"output": i}


class Slice(BaseNode):
    """
    Extracts a subset from a list using start, stop, and step indices.
    list, slice, subset, extract

    Notes:
    - stop=0 means "slice to end" (no upper limit)
    - Negative indices work as in Python (e.g., start=-3 for last 3 items)
    Use cases:
    - Extract a portion of a list
    - Implement pagination
    - Get every nth element
    """
    values: list[Any] = Field(default=[], description="The input list to slice.")
    start: int = Field(default=0, description="Starting index (inclusive). Negative values count from end.")
    stop: int = Field(default=0, description="Ending index (exclusive). 0 means slice to end of list.")
    step: int = Field(default=1, description="Step between elements. Negative for reverse order.")

    async def process(self, context: ProcessingContext) -> list[Any]:
        # Treat stop=0 as "no limit" (slice to end), matching common user expectation
        effective_stop = self.stop if self.stop != 0 else None
        return self.values[self.start : effective_stop : self.step]


class SelectElements(BaseNode):
    """
    Selects specific values from a list using index positions. Stop=0 selects elements until the end of the list.
    list, select, index, extract

    Use cases:
    - Pick specific elements by their positions
    - Rearrange list elements
    - Create a new list from selected indices
    """

    values: list[Any] = Field(default=[])
    indices: list[int] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return [self.values[index] for index in self.indices]


class GetElement(BaseNode):
    """
    Retrieves a single value from a list at a specific index.
    list, get, extract, value

    Use cases:
    - Access a specific element by position
    - Implement array-like indexing
    - Extract the first or last element
    """

    values: list[Any] = Field(default=[])
    index: int = Field(default=0)

    async def process(self, context: ProcessingContext) -> Any:
        return self.values[self.index]


class Append(BaseNode):
    """
    Adds a value to the end of a list.
    list, add, insert, extend

    Use cases:
    - Grow a list dynamically
    - Add new elements to an existing list
    - Implement a stack-like structure
    """

    values: list[Any] = Field(default=[])
    value: Any = Field(default=(), description="The value to append to the list.")

    async def process(self, context: ProcessingContext) -> list[Any]:
        self.values.append(self.value)
        return self.values


class Extend(BaseNode):
    """
    Merges one list into another, extending the original list.
    list, merge, concatenate, combine

    Use cases:
    - Combine multiple lists
    - Add all elements from one list to another
    """

    values: list[Any] = Field(default=[])
    other_values: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        self.values.extend(self.other_values)
        return self.values


class Dedupe(BaseNode):
    """
    Removes duplicate elements from a list, ensuring uniqueness.
    list, unique, distinct, deduplicate

    Use cases:
    - Remove redundant entries
    - Create a set-like structure
    - Ensure list elements are unique
    """

    values: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return list(set(self.values))


class Reverse(BaseNode):
    """
    Inverts the order of elements in a list.
    list, reverse, invert, flip

    Use cases:
    - Reverse the order of a sequence
    """

    values: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return self.values[::(-1)]


class SaveList(BaseNode):
    """
    Saves a list to a text file, placing each element on a new line.
    list, save, file, serialize

    Use cases:
    - Export list data to a file
    - Create a simple text-based database
    - Generate line-separated output
    """

    values: list[Any] = Field(default=[])
    name: str = Field(
        title="Name",
        default="text.txt",
        description="""
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    def required_inputs(self):
        return ["values"]

    async def process(self, context: ProcessingContext) -> TextRef:
        values = "\n".join([str(value) for value in self.values])
        filename = datetime.now().strftime(self.name)
        asset = await context.create_asset(
            name=filename, content_type="text/plain", content=BytesIO(values.encode())
        )
        url = await context.get_asset_url(asset.id)
        return TextRef(uri=url, asset_id=asset.id)


class Randomize(BaseNode):
    """
    Randomly shuffles the elements of a list.
    list, shuffle, random, order

    Use cases:
    - Randomize the order of items in a playlist
    - Implement random sampling without replacement
    - Create randomized data sets for testing
    """

    values: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        shuffled = self.values.copy()
        random.shuffle(shuffled)
        return shuffled


class Sort(BaseNode):
    """
    Sorts the elements of a list in ascending or descending order.
    list, sort, order, arrange

    Use cases:
    - Organize data in a specific order
    - Prepare data for binary search or other algorithms
    - Rank items based on their values
    """

    class SortOrder(str, Enum):
        ASCENDING = "ascending"
        DESCENDING = "descending"

    values: list[Any] = Field(default=[])
    order: SortOrder = SortOrder.ASCENDING

    async def process(self, context: ProcessingContext) -> list[Any]:
        return sorted(self.values, reverse=(self.order == self.SortOrder.DESCENDING))




class Intersection(BaseNode):
    """
    Finds common elements between two lists.
    list, set, intersection, common

    Use cases:
    - Find elements present in both lists
    - Identify shared items between collections
    - Filter for matching elements
    """

    list1: list[Any] = Field(default=[])
    list2: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return list(set(self.list1).intersection(set(self.list2)))


class Union(BaseNode):
    """
    Combines unique elements from two lists.
    list, set, union, combine

    Use cases:
    - Merge lists while removing duplicates
    - Combine collections uniquely
    - Create comprehensive set of items
    """

    list1: list[Any] = Field(default=[])
    list2: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return list(set(self.list1).union(set(self.list2)))


class Difference(BaseNode):
    """
    Finds elements that exist in first list but not in second list.
    list, set, difference, subtract

    Use cases:
    - Find unique elements in one list
    - Remove items present in another list
    - Identify distinct elements
    """

    list1: list[Any] = Field(default=[])
    list2: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return list(set(self.list1).difference(set(self.list2)))


class Chunk(BaseNode):
    """
    Splits a list into smaller chunks of specified size.
    list, chunk, split, group

    Use cases:
    - Batch processing
    - Pagination
    - Creating sublists of fixed size
    """

    values: list[Any] = Field(default=[])
    chunk_size: int = Field(default=1, gt=0)

    async def process(self, context: ProcessingContext) -> list[list[Any]]:
        return [
            self.values[i : i + self.chunk_size]
            for i in range(0, len(self.values), self.chunk_size)
        ]


class Sum(BaseNode):
    """
    Calculates the sum of a list of numbers.
    list, sum, aggregate, math

    Use cases:
    - Calculate total of numeric values
    - Add up all elements in a list
    """

    values: list[float] = Field(default=[])

    async def process(self, context: ProcessingContext) -> float:
        if not self.values:
            raise ValueError("Cannot sum empty list")
        if not all(isinstance(x, (int, float)) for x in self.values):
            raise ValueError("All values must be numbers")
        return sum(self.values)


class Average(BaseNode):
    """
    Calculates the arithmetic mean of a list of numbers.
    list, average, mean, aggregate, math

    Use cases:
    - Find average value
    - Calculate mean of numeric data
    """

    values: list[float] = Field(default=[])

    async def process(self, context: ProcessingContext) -> float:
        if not self.values:
            raise ValueError("Cannot average empty list")
        if not all(isinstance(x, (int, float)) for x in self.values):
            raise ValueError("All values must be numbers")
        return sum(self.values) / len(self.values)


class Minimum(BaseNode):
    """
    Finds the smallest value in a list of numbers.
    list, min, minimum, aggregate, math

    Use cases:
    - Find lowest value
    - Get smallest number in dataset
    """

    values: list[float] = Field(default=[])

    async def process(self, context: ProcessingContext) -> float:
        if not self.values:
            raise ValueError("Cannot find minimum of empty list")
        if not all(isinstance(x, (int, float)) for x in self.values):
            raise ValueError("All values must be numbers")
        return min(self.values)


class Maximum(BaseNode):
    """
    Finds the largest value in a list of numbers.
    list, max, maximum, aggregate, math

    Use cases:
    - Find highest value
    - Get largest number in dataset
    """

    values: list[float] = Field(default=[])

    async def process(self, context: ProcessingContext) -> float:
        if not self.values:
            raise ValueError("Cannot find maximum of empty list")
        if not all(isinstance(x, (int, float)) for x in self.values):
            raise ValueError("All values must be numbers")
        return max(self.values)


class Product(BaseNode):
    """
    Calculates the product of all numbers in a list.
    list, product, multiply, aggregate, math

    Use cases:
    - Multiply all numbers together
    - Calculate compound values
    """

    values: list[float] = Field(default=[])

    async def process(self, context: ProcessingContext) -> float:
        if not self.values:
            raise ValueError("Cannot calculate product of empty list")
        if not all(isinstance(x, (int, float)) for x in self.values):
            raise ValueError("All values must be numbers")
        return reduce(lambda x, y: x * y, self.values)


class Flatten(BaseNode):
    """
    Flattens a nested list structure into a single flat list.
    list, flatten, nested, structure

    Use cases:
    - Convert nested lists into a single flat list
    - Simplify complex list structures
    - Process hierarchical data as a sequence

    Examples:
    [[1, 2], [3, 4]] -> [1, 2, 3, 4]
    [[1, [2, 3]], [4, [5, 6]]] -> [1, 2, 3, 4, 5, 6]
    """

    values: list[Any] = Field(default=[])
    max_depth: int = Field(default=-1, ge=-1)

    def _flatten(self, lst: list[Any], current_depth: int = 0) -> list[Any]:
        result = []
        for item in lst:
            if isinstance(item, list) and (
                self.max_depth == -1 or current_depth < self.max_depth
            ):
                result.extend(self._flatten(item, current_depth + 1))
            else:
                result.append(item)
        return result

    async def process(self, context: ProcessingContext) -> list[Any]:
        if not isinstance(self.values, list):
            raise ValueError("Input must be a list")
        return self._flatten(self.values)
