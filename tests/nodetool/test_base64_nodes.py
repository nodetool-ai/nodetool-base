import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.base64 import Encode, Decode


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_encode_base64(context: ProcessingContext):
    node = Encode(text="hello")
    result = await node.process(context)
    assert result == "aGVsbG8="


@pytest.mark.asyncio
async def test_decode_base64(context: ProcessingContext):
    node = Decode(data="aGVsbG8=")
    result = await node.process(context)
    assert result == "hello"
