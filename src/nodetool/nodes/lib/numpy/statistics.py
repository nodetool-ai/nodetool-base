import numpy as np
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray


class MaxArray(BaseNode):
    """
    Compute the maximum value along a specified axis of a array.
    array, maximum, reduction, statistics

    Use cases:
    - Find peak values in time series data
    - Implement max pooling in neural networks
    - Determine highest scores across multiple categories
    """

    values: NPArray = Field(default=NPArray(), description="Input array")
    axis: int = Field(default=0, description="Axis along which to compute maximum")

    async def process(self, context: ProcessingContext) -> NPArray | float | int:
        res = np.max(to_numpy(self.values), axis=self.axis)
        if res.size == 1:
            return res.item()
        else:
            return NPArray.from_numpy(res)


class MinArray(BaseNode):
    """
    Calculate the minimum value along a specified axis of a array.
    array, minimum, reduction, statistics

    Use cases:
    - Find lowest values in datasets
    - Implement min pooling in neural networks
    - Determine minimum thresholds across categories
    """

    values: NPArray = Field(default=NPArray(), description="Input array")
    axis: int = Field(default=0, description="Axis along which to compute minimum")

    async def process(self, context: ProcessingContext) -> NPArray | float | int:
        res = np.min(to_numpy(self.values), axis=self.axis)
        if res.size == 1:
            return res.item()
        else:
            return NPArray.from_numpy(res)


class MeanArray(BaseNode):
    """
    Compute the mean value along a specified axis of a array.
    array, average, reduction, statistics

    Use cases:
    - Calculate average values in datasets
    - Implement mean pooling in neural networks
    - Compute centroids in clustering algorithms
    """

    values: NPArray = Field(default=NPArray(), description="Input array")
    axis: int = Field(default=0, description="Axis along which to compute mean")

    async def process(self, context: ProcessingContext) -> NPArray | float | int:
        res = np.mean(to_numpy(self.values), axis=self.axis)
        if res.size == 1:
            return res.item()
        else:
            return NPArray.from_numpy(res)


class SumArray(BaseNode):
    """
    Calculate the sum of values along a specified axis of a array.
    array, summation, reduction, statistics

    Use cases:
    - Compute total values across categories
    - Implement sum pooling in neural networks
    - Calculate cumulative metrics in time series data
    """

    values: NPArray = Field(default=NPArray(), description="Input array")
    axis: int = Field(default=0, description="Axis along which to compute sum")

    async def process(self, context: ProcessingContext) -> NPArray | float | int:
        res = np.sum(to_numpy(self.values), axis=self.axis)
        if res.size == 1:
            return res.item()
        else:
            return NPArray.from_numpy(res)


class ArgMaxArray(BaseNode):
    """
    Find indices of maximum values along a specified axis of a array.
    array, argmax, index, maximum

    Use cases:
    - Determine winning classes in classification tasks
    - Find peaks in signal processing
    - Locate best-performing items in datasets
    """

    values: NPArray = Field(default=NPArray(), description="Input array")
    axis: int = Field(default=0, description="Axis along which to find maximum indices")

    async def process(self, context: ProcessingContext) -> NPArray | int:
        res = np.argmax(to_numpy(self.values), axis=self.axis)
        if res.size == 1:
            return res.item()
        else:
            return NPArray.from_numpy(res)


class ArgMinArray(BaseNode):
    """
    Find indices of minimum values along a specified axis of a array.
    array, argmin, index, minimum

    Use cases:
    - Locate lowest-performing items in datasets
    - Find troughs in signal processing
    - Determine least likely classes in classification tasks
    """

    values: NPArray = Field(default=NPArray(), description="Input array")
    axis: int = Field(default=0, description="Axis along which to find minimum indices")

    async def process(self, context: ProcessingContext) -> NPArray | int:
        res = np.argmin(to_numpy(self.values), axis=self.axis)
        if res.size == 1:
            return res.item()
        else:
            return NPArray.from_numpy(res)
