from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Compress(GraphNode):
    """
    Compress binary data using the zlib algorithm.
    zlib, compress, deflate, binary

    Use cases:
    - Reduce size of binary data
    - Prepare payloads for transmission
    - Store data in compressed form
    """

    data: bytes | GraphNode | tuple[GraphNode, str] = Field(
        default=b"", description="Data to compress"
    )
    level: int | GraphNode | tuple[GraphNode, str] = Field(
        default=9, description="Compression level"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.zlib.Compress"


class Decompress(GraphNode):
    """
    Decompress zlib-compressed binary data.
    zlib, decompress, inflate, binary

    Use cases:
    - Restore compressed payloads
    - Read previously compressed files
    - Handle zlib streams from external services
    """

    data: bytes | GraphNode | tuple[GraphNode, str] = Field(
        default=b"", description="Data to decompress"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.zlib.Decompress"
