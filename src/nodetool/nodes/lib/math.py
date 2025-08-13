import asyncio
import math
from enum import Enum
from nodetool.agents.agent import Agent
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.workflows.types import Chunk
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


class BinaryOp(BaseNode):
    """Performs a selected binary math operation on two inputs."""

    _layout = "small"

    class Operation(str, Enum):
        ADD = "add"
        SUBTRACT = "subtract"
        MULTIPLY = "multiply"
        DIVIDE = "divide"
        MODULUS = "modulus"

    a: int | float = Field(title="A", default=0.0)
    b: int | float = Field(title="B", default=0.0)
    operation: Operation = Field(
        default=Operation.ADD, description="Binary operation to perform"
    )

    async def process(self, context: ProcessingContext) -> int | float:
        if self.operation == self.Operation.ADD:
            return self.a + self.b
        elif self.operation == self.Operation.SUBTRACT:
            return self.a - self.b
        elif self.operation == self.Operation.MULTIPLY:
            return self.a * self.b
        elif self.operation == self.Operation.DIVIDE:
            return self.a / self.b
        elif self.operation == self.Operation.MODULUS:
            return self.a % self.b
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


class UnaryOp(BaseNode):
    """Performs a selected unary math operation on an input."""

    _layout = "small"

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
