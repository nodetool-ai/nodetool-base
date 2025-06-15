import gzip
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.gzip import GzipCompress, GzipDecompress


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_gzip_compress(context: ProcessingContext):
    data = b"hello world"
    node = GzipCompress(data=data)
    result = await node.process(context)
    assert gzip.decompress(result) == data


@pytest.mark.asyncio
async def test_gzip_decompress(context: ProcessingContext):
    data = b"compress me"
    compressed = gzip.compress(data)
    node = GzipDecompress(data=compressed)
    result = await node.process(context)
    assert result == data
