from typing import ClassVar
import numpy as np
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray
from .utils import convert_output, pad_arrays


class PowerArray(BaseNode):
    """
    Raises the base array to the power of the exponent element-wise.
    math, exponentiation, power, pow, **

    Use cases:
    - Calculating compound interest
    - Implementing polynomial functions
    - Applying non-linear transformations to data
    """

    _layout: ClassVar[str] = "small"

    base: float | int | NPArray = Field(title="Base", default=1.0)
    exponent: float | int | NPArray = Field(title="Exponent", default=2.0)

    async def process(self, context: ProcessingContext) -> float | int | NPArray:
        a = to_numpy(self.base)
        b = to_numpy(self.exponent)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            (a, b) = pad_arrays(a, b)
        return await convert_output(context, np.power(a, b))


class SqrtArray(BaseNode):
    """
    Calculates the square root of the input array element-wise.
    math, square root, sqrt, √

    Use cases:
    - Normalizing data
    - Calculating distances in Euclidean space
    - Finding intermediate values in binary search
    """

    _layout: ClassVar[str] = "small"

    values: NPArray = Field(default=NPArray(), description="Input array")

    async def process(self, context: ProcessingContext) -> float | int | NPArray:
        return await convert_output(
            context, np.sqrt(to_numpy(self.values).astype(np.float32))
        )


class ExpArray(BaseNode):
    """
    Calculate the exponential of each element in a array.
    array, exponential, math, activation

    Use cases:
    - Implement exponential activation functions
    - Calculate growth rates in scientific models
    - Transform data for certain statistical analyses
    """

    values: NPArray = Field(default=NPArray(), description="Input array")

    async def process(self, context: ProcessingContext) -> float | int | NPArray:
        return await convert_output(
            context, np.exp(to_numpy(self.values).astype(np.float32))
        )


class LogArray(BaseNode):
    """
    Calculate the natural logarithm of each element in a array.
    array, logarithm, math, transformation

    Use cases:
    - Implement log transformations on data
    - Calculate entropy in information theory
    - Normalize data with large ranges
    """

    values: NPArray = Field(default=NPArray(), description="Input array")

    async def process(self, context: ProcessingContext) -> float | int | NPArray:
        return await convert_output(
            context, np.log(to_numpy(self.values).astype(np.float32))
        )


class AbsArray(BaseNode):
    """
    Compute the absolute value of each element in a array.
    array, absolute, magnitude

    Use cases:
    - Calculate magnitudes of complex numbers
    - Preprocess data for certain algorithms
    - Implement activation functions in neural networks
    """

    values: NPArray = Field(
        default=NPArray(),
        description="The input array to compute the absolute values from.",
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        abs_array = np.abs(to_numpy(self.values))
        if abs_array.size == 1:
            return abs_array.item()
        else:
            return NPArray.from_numpy(abs_array)


class SineArray(BaseNode):
    """
    Computes the sine of input angles in radians.
    math, trigonometry, sine, sin

    Use cases:
    - Calculating vertical components in physics
    - Generating smooth periodic functions
    - Audio signal processing
    """

    _layout: ClassVar[str] = "small"

    angle_rad: float | int | NPArray = Field(title="Angle (Radians)", default=0.0)

    async def process(self, context: ProcessingContext) -> float | NPArray:
        res = np.sin(to_numpy(self.angle_rad))
        return await convert_output(context, res)


class CosineArray(BaseNode):
    """
    Computes the cosine of input angles in radians.
    math, trigonometry, cosine, cos

    Use cases:
    - Calculating horizontal components in physics
    - Creating circular motions
    - Phase calculations in signal processing
    """

    _layout: ClassVar[str] = "small"

    angle_rad: float | int | NPArray = Field(title="Angle (Radians)", default=0.0)

    async def process(self, context: ProcessingContext) -> float | NPArray:
        res = np.cos(to_numpy(self.angle_rad))
        return await convert_output(context, res)
