import gzip
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class GzipCompress(BaseNode):
    """
    Compress bytes using gzip.
    gzip, compress, bytes

    Use cases:
    - Reduce size of binary data
    - Store assets in compressed form
    - Prepare data for network transfer
    """

    data: bytes | None = Field(default=None, description="Data to compress")

    _expose_as_tool: bool = True

    async def process(self, context: ProcessingContext) -> bytes:
        if not self.data:
            raise ValueError("data cannot be empty")
        return gzip.compress(self.data)


class GzipDecompress(BaseNode):
    """
    Decompress gzip data.
    gzip, decompress, bytes

    Use cases:
    - Restore compressed files
    - Read data from gzip archives
    - Process network payloads
    """

    data: bytes | None = Field(
        default=None,
        description="Gzip data to decompress",
    )

    _expose_as_tool: bool = True

    async def process(self, context: ProcessingContext) -> bytes:
        if not self.data:
            raise ValueError("data cannot be empty")
        return gzip.decompress(self.data)
