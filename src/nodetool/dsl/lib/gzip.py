from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class GzipCompress(GraphNode):
    """
    Compress bytes using gzip.
    gzip, compress, bytes

    Use cases:
    - Reduce size of binary data
    - Store assets in compressed form
    - Prepare data for network transfer
    """

    data: bytes | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Data to compress"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.gzip.GzipCompress"


class GzipDecompress(GraphNode):
    """
    Decompress gzip data.
    gzip, decompress, bytes

    Use cases:
    - Restore compressed files
    - Read data from gzip archives
    - Process network payloads
    """

    data: bytes | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Gzip data to decompress"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.gzip.GzipDecompress"
