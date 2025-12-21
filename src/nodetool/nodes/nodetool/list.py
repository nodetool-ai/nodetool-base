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
    Calculate number of elements in a list.

    Returns the count of items in the list. Empty lists return 0.

    Parameters:
    - values (required): List to measure

    Returns: Integer count of elements

    Typical usage: Validate list size, check for empty lists, or count results.
    Follow with comparison, conditional logic, or output nodes.

    list, count, size
    """

    values: list[Any] = []

    async def process(self, context: ProcessingContext) -> int:
        return len(self.values)


class ListRange(BaseNode):
    """
    Generate list of integers from start to stop with optional step.

    Creates sequence of integers using Python range semantics. Stop value is
    exclusive. Negative step produces descending sequences.

    Parameters:
    - start (required): Starting value (inclusive)
    - stop (required): Ending value (exclusive)
    - step (optional, default=1): Increment between values

    Returns: List of integers

    Typical usage: Generate index sequences, create numbered lists, or produce
    arithmetic progressions. Follow with ForEach for iteration or mapping operations.

    list, range, sequence, numbers
    """

    start: int = 0
    stop: int = 0
    step: int = 1

    async def process(self, context: ProcessingContext) -> list[int]:
        return list(range(self.start, self.stop, self.step))


class GenerateSequence(BaseNode):
    """
    Stream integers from start to stop, emitting each value individually.

    Produces integers using range semantics but as a stream rather than complete
    list. Useful for driving iteration without pre-allocating full sequence.

    Parameters:
    - start (required): Starting value (inclusive)
    - stop (required): Ending value (exclusive)
    - step (optional, default=1): Increment between values

    Yields: Dictionary with "output" (integer) for each value in sequence

    Typical usage: Drive downstream processing with sequential numbers, implement
    loops without full list, or generate indices on demand. Follow with nodes that
    consume streaming inputs.

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
    Extract subset of list using start, stop, and step indices.

    Returns portion of list using Python slice semantics. Negative indices count
    from end. Step > 1 skips elements; negative step reverses direction.

    Parameters:
    - values (required): List to slice
    - start (optional, default=0): Starting index (inclusive)
    - stop (optional, default=0): Ending index (exclusive, 0 means end)
    - step (optional, default=1): Index increment

    Returns: New list with selected elements

    Typical usage: Extract list portion, implement pagination, get every nth element,
    or reverse lists. Follow with further list operations or ForEach.

    list, slice, subset, extract
    """

    values: list[Any] = Field(default=[])
    start: int = Field(default=0)
    stop: int = Field(default=0)
    step: int = Field(default=1)

    async def process(self, context: ProcessingContext) -> list[Any]:
        return self.values[self.start : self.stop : self.step]


class SelectElements(BaseNode):
    """
    Create new list by selecting elements at specified index positions.

    Picks elements from input list using provided indices. Indices can be in any
    order and can repeat. Maintains index order in output.

    Parameters:
    - values (required): Source list
    - indices (required): List of integer positions to select

    Returns: New list containing selected elements

    Raises: IndexError if any index is out of range

    Typical usage: Reorder list elements, pick specific items by position, or
    extract multiple non-contiguous elements. Follow with further processing.

    list, select, index, extract
    """

    values: list[Any] = Field(default=[])
    indices: list[int] = Field(default=[])

    async def process(self, context: ProcessingContext) -> list[Any]:
        return [self.values[index] for index in self.indices]


class GetElement(BaseNode):
    """
    Retrieve single element from list at specified index.

    Returns the value at the given position. Negative indices count from end
    (-1 is last element).

    Parameters:
    - values (required): List to access
    - index (required, default=0): Position to retrieve (0-based)

    Returns: Element at index

    Raises: IndexError if index out of range

    Typical usage: Access first/last element, get specific position, or extract
    single result from list. Follow with type-specific processing or output.

    list, get, extract, value
    """

    values: list[Any] = Field(default=[])
    index: int = Field(default=0)

    async def process(self, context: ProcessingContext) -> Any:
        return self.values[self.index]


class Append(BaseNode):
    """
    Add single value to end of list, modifying in place.

    Appends value to the list, modifying the original list object.

    Parameters:
    - values (required): List to extend
    - value (required): Value to add at end

    Returns: Modified list with new element at end

    Typical usage: Build lists incrementally, add computed results, or grow
    collections dynamically. Follow with further list operations or output.

    list, add, insert, extend
    """

    values: list[Any] = Field(default=[])
    value: Any = Field(default=(), description="The value to append to the list.")

    async def process(self, context: ProcessingContext) -> list[Any]:
        self.values.append(self.value)
        return self.values


class Extend(BaseNode):
    """
    Concatenate second list to first list, modifying first list in place.

    Adds all elements from other_values to the end of values list. Original
    values list is modified.

    Parameters:
    - values (required): List to extend
    - other_values (required): List to append to first list

    Returns: Modified first list with all elements from both lists

    Typical usage: Combine multiple lists, merge results from parallel operations,
    or build aggregate collections. Follow with deduplication or sorting if needed.

    list, merge, concatenate, combine
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
