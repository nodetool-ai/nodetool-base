import zlib
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class Compress(BaseNode):
    """
    Compress binary data using the zlib algorithm.
    zlib, compress, deflate, binary

    Use cases:
    - Reduce size of binary data
    - Prepare payloads for transmission
    - Store data in compressed form
    """

    data: bytes = Field(default=b"", description="Data to compress")
    level: int = Field(default=9, ge=0, le=9, description="Compression level")

    async def process(self, context: ProcessingContext) -> bytes:
        return zlib.compress(self.data, self.level)


class Decompress(BaseNode):
    """
    Decompress zlib-compressed binary data.
    zlib, decompress, inflate, binary

    Use cases:
    - Restore compressed payloads
    - Read previously compressed files
    - Handle zlib streams from external services
    """

    data: bytes = Field(default=b"", description="Data to decompress")

    async def process(self, context: ProcessingContext) -> bytes:
        return zlib.decompress(self.data)
