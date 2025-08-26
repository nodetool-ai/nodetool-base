from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Add(GraphNode):
    """
    Adds two numbers.
    math, add, plus, +, sum
    """

    a: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    b: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Add"


class Cosine(GraphNode):
    """
    Computes cosine of the given angle in radians.
    math, cosine, trig
    """

    angle_rad: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Cosine"


class Divide(GraphNode):
    """
    Divides A by B.
    math, divide, division, quotient, /
    """

    a: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    b: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Divide"


import nodetool.nodes.lib.math


class MathFunction(GraphNode):
    """
    Performs a selected unary math operation on an input.
    math, negate, absolute, square, cube, square_root, cube_root, sine, cosine, tangent, arcsine, arccosine, arctangent, log
    """

    Operation: typing.ClassVar[type] = nodetool.nodes.lib.math.MathFunction.Operation
    input: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    operation: nodetool.nodes.lib.math.MathFunction.Operation = Field(
        default=nodetool.nodes.lib.math.MathFunction.Operation.NEGATE,
        description="Unary operation to perform",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.MathFunction"


class Modulus(GraphNode):
    """
    Computes A modulo B.
    math, modulus, modulo, remainder
    """

    a: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    b: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Modulus"


class Multiply(GraphNode):
    """
    Multiplies two numbers.
    math, multiply, product, *
    """

    a: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    b: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Multiply"


class Power(GraphNode):
    """
    Raises base to the given exponent.
    math, power, exponent, ^, **
    """

    base: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    exponent: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Power"


class Sine(GraphNode):
    """
    Computes sine of the given angle in radians.
    math, sine, trig
    """

    angle_rad: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Sine"


class Sqrt(GraphNode):
    """
    Computes square root of x.
    math, sqrt, square_root
    """

    x: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Sqrt"


class Subtract(GraphNode):
    """
    Subtracts B from A.
    math, subtract, minus, -, difference
    """

    a: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    b: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.Subtract"
