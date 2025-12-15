import pytest
from unittest.mock import AsyncMock, MagicMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.http import GetRequest, PostRequest, PutRequest, DeleteRequest, HeadRequest


class DummyResponse:
    def __init__(self, text: str = "", headers: dict | None = None, encoding: str = "utf-8"):
        self.content = text.encode(encoding)
        self.encoding = encoding
        self.headers = headers or {}


@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=ProcessingContext)
    ctx.http_get = AsyncMock(return_value=DummyResponse("ok-get"))
    ctx.http_post = AsyncMock(return_value=DummyResponse("ok-post"))
    ctx.http_put = AsyncMock(return_value=DummyResponse("ok-put"))
    ctx.http_delete = AsyncMock(return_value=DummyResponse("ok-delete"))
    ctx.http_head = AsyncMock(return_value=DummyResponse(headers={"X": "1"}))
    return ctx


@pytest.mark.asyncio
async def test_http_basic_requests(mock_context: ProcessingContext):
    assert await GetRequest(url="http://x").process(mock_context) == "ok-get"
    assert (
        await PostRequest(url="http://x", data="d").process(mock_context) == "ok-post"
    )
    assert await PutRequest(url="http://x", data="d").process(mock_context) == "ok-put"
    assert await DeleteRequest(url="http://x").process(mock_context) == "ok-delete"


@pytest.mark.asyncio
async def test_http_head(mock_context: ProcessingContext):
    result = await HeadRequest(url="http://x").process(mock_context)
    assert result == {"X": "1"}

