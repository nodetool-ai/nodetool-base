import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib import zlib as zlib_nodes


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_compress_decompress_roundtrip(context):
    original = b"hello world" * 10
    compressed = await zlib_nodes.Compress(data=original).process(context)
    assert isinstance(compressed, bytes)
    decompressed = await zlib_nodes.Decompress(data=compressed).process(context)
    assert decompressed == original
