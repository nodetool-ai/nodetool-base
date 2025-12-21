import math
from enum import Enum
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from typing import ClassVar


class Add(BaseNode):
    """
    Adds two numbers together.
    math, add, plus, +, sum

    Use cases:
    - Perform basic arithmetic operations
    - Calculate totals and sums
    - Combine numerical values
    - Increment counters and scores
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a + self.b


class Subtract(BaseNode):
    """
    Subtracts B from A.
    math, subtract, minus, -, difference

    Use cases:
    - Calculate differences between values
    - Determine remaining amounts
    - Compute offsets and deltas
    - Track decrements and reductions
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a - self.b


class Multiply(BaseNode):
    """
    Multiplies two numbers together.
    math, multiply, product, *, times

    Use cases:
    - Calculate products and totals
    - Scale values by factors
    - Compute areas and volumes
    - Apply multipliers and rates
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True
    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a * self.b


class Divide(BaseNode):
    """
    Divides A by B to calculate the quotient.
    math, divide, division, quotient, /

    Use cases:
    - Calculate averages and ratios
    - Distribute quantities evenly
    - Determine rates and proportions
    - Compute per-unit values
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=1.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a / self.b


class Modulus(BaseNode):
    """
    Computes A modulo B to find the remainder after division.
    math, modulus, modulo, remainder, %

    Use cases:
    - Determine if numbers are even or odd
    - Implement cyclic patterns and rotations
    - Calculate remainders in division
    - Build repeating sequences
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True
    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=1.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a % self.b


class MathFunction(BaseNode):
    """
    Performs a selected unary math operation on an input.
    math, negate, absolute, square, cube, square_root, cube_root, sine, cosine, tangent, arcsine, arccosine, arctangent, log,   -, abs, ^2, ^3, sqrt, cbrt, sin, cos, tan, asin, acos, atan, log
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    class Operation(str, Enum):
        NEGATE = "negate"
        ABSOLUTE = "absolute"
        SQUARE = "square"
        CUBE = "cube"
        SQUARE_ROOT = "square_root"
        CUBE_ROOT = "cube_root"
        SINE = "sine"
        COSINE = "cosine"
        TANGENT = "tangent"
        ARCSINE = "arcsin"
        ARCCOSINE = "arccos"
        ARCTANGENT = "arctan"
        LOG = "log"

    input: int | float = Field(title="Input", default=0.0)
    operation: Operation = Field(
        default=Operation.NEGATE, description="Unary operation to perform"
    )

    async def process(self, context: ProcessingContext) -> int | float:
        if self.operation == self.Operation.NEGATE:
            return -self.input
        elif self.operation == self.Operation.ABSOLUTE:
            return abs(self.input)
        elif self.operation == self.Operation.SQUARE:
            return self.input * self.input
        elif self.operation == self.Operation.CUBE:
            return self.input * self.input * self.input
        elif self.operation == self.Operation.SQUARE_ROOT:
            return math.sqrt(self.input)
        elif self.operation == self.Operation.CUBE_ROOT:
            value = self.input
            # Real cube root for negative numbers as well
            return math.copysign(abs(value) ** (1 / 3), value)
        elif self.operation == self.Operation.SINE:
            return math.sin(self.input)
        elif self.operation == self.Operation.COSINE:
            return math.cos(self.input)
        elif self.operation == self.Operation.TANGENT:
            return math.tan(self.input)
        elif self.operation == self.Operation.ARCSINE:
            return math.asin(self.input)
        elif self.operation == self.Operation.ARCCOSINE:
            return math.acos(self.input)
        elif self.operation == self.Operation.ARCTANGENT:
            return math.atan(self.input)
        elif self.operation == self.Operation.LOG:
            return math.log(self.input)
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


class Sine(BaseNode):
    """
    Computes sine of the given angle in radians.
    math, sine, trig
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    angle_rad: int | float = Field(title="Angle (rad)", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return math.sin(self.angle_rad)


class Cosine(BaseNode):
    """
    Computes cosine of the given angle in radians.
    math, cosine, trig
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    angle_rad: int | float = Field(title="Angle (rad)", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return math.cos(self.angle_rad)


class Power(BaseNode):
    """
    Raises base to the given exponent.
    math, power, exponent, ^
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    base: int | float = Field(title="Base", default=0.0)
    exponent: int | float = Field(title="Exponent", default=1.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return math.pow(self.base, self.exponent)


class Sqrt(BaseNode):
    """
    Computes square root of x.
    math, sqrt, square_root
    """

    _layout: ClassVar[str] = "small"
    _expose_as_tool = True

    x: int | float = Field(title="X", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return math.sqrt(self.x)
