from typing import ClassVar
import numpy as np
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray


class Stack(BaseNode):
    """
    Stack multiple arrays along a specified axis.
    array, stack, concatenate, join, merge, axis

    Use cases:
    - Combine multiple 2D arrays into a 3D array
    - Stack time series data from multiple sources
    - Merge feature vectors for machine learning models
    """

    arrays: list[NPArray] = Field(default=[], description="Arrays to stack")
    axis: int = Field(0, description="The axis to stack along.", ge=0)

    async def process(self, context: ProcessingContext) -> NPArray:
        arrays = [to_numpy(array) for array in self.arrays]
        stacked_array = np.stack(arrays, axis=self.axis)
        return NPArray.from_numpy(stacked_array)


class MatMul(BaseNode):
    """
    Perform matrix multiplication on two input arrays.
    array, matrix, multiplication, linear algebra

    Use cases:
    - Implement linear transformations
    - Calculate dot products of vectors
    - Perform matrix operations in neural networks
    """

    _layout: ClassVar[str] = "small"
    a: NPArray = Field(default=NPArray(), description="First input array")
    b: NPArray = Field(default=NPArray(), description="Second input array")

    async def process(self, context: ProcessingContext) -> NPArray:
        a = to_numpy(self.a)
        b = to_numpy(self.b)
        return NPArray.from_numpy(np.matmul(a, b))


class TransposeArray(BaseNode):
    """
    Transpose the dimensions of the input array.
    array, transpose, reshape, dimensions

    Use cases:
    - Convert row vectors to column vectors
    - Rearrange data for compatibility with other operations
    - Implement certain linear algebra operations
    """

    _layout: ClassVar[str] = "small"
    values: NPArray = Field(default=NPArray(), description="Array to transpose")

    async def process(self, context: ProcessingContext) -> NPArray:
        return NPArray.from_numpy(np.transpose(to_numpy(self.values)))


class SliceArray(BaseNode):
    """
    Extract a slice of an array along a specified axis.
    array, slice, subset, index

    Use cases:
    - Extract specific time periods from time series data
    - Select subset of features from datasets
    - Create sliding windows over sequential data
    """

    values: NPArray = Field(default=NPArray(), description="The input array to slice")
    start: int = Field(default=0, description="Starting index (inclusive)")
    stop: int = Field(default=0, description="Ending index (exclusive)")
    step: int = Field(default=1, description="Step size between elements")
    axis: int = Field(default=0, description="Axis along which to slice")

    async def process(self, context: ProcessingContext) -> NPArray:
        arr = to_numpy(self.values)
        slicing = [slice(None)] * arr.ndim
        slicing[self.axis] = slice(self.start, self.stop, self.step)
        return NPArray.from_numpy(arr[tuple(slicing)])


class IndexArray(BaseNode):
    """
    Select specific indices from an array along a specified axis.
    array, index, select, subset

    Use cases:
    - Extract specific samples from a dataset
    - Select particular features or dimensions
    - Implement batch sampling operations
    """

    values: NPArray = Field(default=NPArray(), description="The input array to index")
    indices: str = Field(
        default="", description="The comma separated indices to select"
    )
    axis: int = Field(default=0, description="Axis along which to index")

    async def process(self, context: ProcessingContext) -> NPArray:
        arr = to_numpy(self.values)
        indices_list = [int(x) for x in self.indices.split(",") if x.strip()]

        # Use numpy's take function instead of direct indexing
        result = np.take(arr, indices_list, axis=self.axis)
        return NPArray.from_numpy(result)


class SplitArray(BaseNode):
    """
    Split an array into multiple sub-arrays along a specified axis.
    array, split, divide, partition

    Use cases:
    - Divide datasets into training/validation splits
    - Create batches from large arrays
    - Separate multi-channel data
    """

    values: NPArray = Field(default=NPArray(), description="The input array to split")
    num_splits: int = Field(
        default=0, description="Number of equal splits to create", gt=0
    )
    axis: int = Field(default=0, description="Axis along which to split")

    async def process(self, context: ProcessingContext) -> list[NPArray]:
        arr = to_numpy(self.values)
        split_arrays = np.array_split(arr, self.num_splits, axis=self.axis)
        return [NPArray.from_numpy(split_arr) for split_arr in split_arrays]
