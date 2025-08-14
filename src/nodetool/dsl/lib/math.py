from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.math


class BinaryOp(GraphNode):
    """Performs a selected binary math operation on two inputs."""

    Operation: typing.ClassVar[type] = nodetool.nodes.lib.math.BinaryOp.Operation
    a: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    b: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    operation: nodetool.nodes.lib.math.BinaryOp.Operation = Field(
        default=nodetool.nodes.lib.math.BinaryOp.Operation.ADD,
        description="Binary operation to perform",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.BinaryOp"


import nodetool.nodes.lib.math


class UnaryOp(GraphNode):
    """Performs a selected unary math operation on an input."""

    Operation: typing.ClassVar[type] = nodetool.nodes.lib.math.UnaryOp.Operation
    input: int | float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    operation: nodetool.nodes.lib.math.UnaryOp.Operation = Field(
        default=nodetool.nodes.lib.math.UnaryOp.Operation.NEGATE,
        description="Unary operation to perform",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.math.UnaryOp"
