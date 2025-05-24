from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class BinaryOperation(GraphNode):
    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.utils.BinaryOperation"
