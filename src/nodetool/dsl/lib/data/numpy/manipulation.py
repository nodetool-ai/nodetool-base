from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class IndexArray(GraphNode):
    """
    Select specific indices from an array along a specified axis.
    array, index, select, subset

    Use cases:
    - Extract specific samples from a dataset
    - Select particular features or dimensions
    - Implement batch sampling operations
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to index",
    )
    indices: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The comma separated indices to select"
    )
    axis: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Axis along which to index"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.manipulation.IndexArray"


class MatMul(GraphNode):
    """
    Perform matrix multiplication on two input arrays.
    array, matrix, multiplication, linear algebra

    Use cases:
    - Implement linear transformations
    - Calculate dot products of vectors
    - Perform matrix operations in neural networks
    """

    a: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="First input array",
    )
    b: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Second input array",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.manipulation.MatMul"


class SliceArray(GraphNode):
    """
    Extract a slice of an array along a specified axis.
    array, slice, subset, index

    Use cases:
    - Extract specific time periods from time series data
    - Select subset of features from datasets
    - Create sliding windows over sequential data
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to slice",
    )
    start: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Starting index (inclusive)"
    )
    stop: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Ending index (exclusive)"
    )
    step: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Step size between elements"
    )
    axis: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Axis along which to slice"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.manipulation.SliceArray"


class SplitArray(GraphNode):
    """
    Split an array into multiple sub-arrays along a specified axis.
    array, split, divide, partition

    Use cases:
    - Divide datasets into training/validation splits
    - Create batches from large arrays
    - Separate multi-channel data
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to split",
    )
    num_splits: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Number of equal splits to create"
    )
    axis: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Axis along which to split"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.manipulation.SplitArray"


class Stack(GraphNode):
    """
    Stack multiple arrays along a specified axis.
    array, stack, concatenate, join, merge, axis

    Use cases:
    - Combine multiple 2D arrays into a 3D array
    - Stack time series data from multiple sources
    - Merge feature vectors for machine learning models
    """

    arrays: list[types.NPArray] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="Arrays to stack"
    )
    axis: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The axis to stack along."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.manipulation.Stack"


class TransposeArray(GraphNode):
    """
    Transpose the dimensions of the input array.
    array, transpose, reshape, dimensions

    Use cases:
    - Convert row vectors to column vectors
    - Rearrange data for compatibility with other operations
    - Implement certain linear algebra operations
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Array to transpose",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.manipulation.TransposeArray"
