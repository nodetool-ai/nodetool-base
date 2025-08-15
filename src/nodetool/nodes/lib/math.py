import math
from enum import Enum
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


class Add(BaseNode):
    """
    Adds two numbers.
    math, add, plus
    """

    _layout = "small"
    _expose_as_tool = True

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a + self.b


class Subtract(BaseNode):
    """
    Subtracts B from A.
    math, subtract, minus
    """

    _layout = "small"
    _expose_as_tool = True

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a - self.b


class Multiply(BaseNode):
    """
    Multiplies two numbers.
    math, multiply, product
    """

    _layout = "small"
    _expose_as_tool = True
    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a * self.b


class Divide(BaseNode):
    """
    Divides A by B.
    math, divide, division, quotient
    """

    _layout = "small"
    _expose_as_tool = True

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=1.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a / self.b


class Modulus(BaseNode):
    """
    Computes A modulo B.
    math, modulus, modulo, remainder
    """

    _layout = "small"
    _expose_as_tool = True
    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=1.0)

    async def process(self, context: ProcessingContext) -> int | float:
        return self.a % self.b


class MathFunction(BaseNode):
    """
    Performs a selected unary math operation on an input.
    math, negate, absolute, square, cube, square_root, cube_root, sine, cosine, tangent, arcsine, arccosine, arctangent, log
    """

    _layout = "small"
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
