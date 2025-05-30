import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.urllib import (
    ParseURL,
    JoinURL,
    EncodeQueryParams,
    QuoteURL,
    UnquoteURL,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_parse_url(context: ProcessingContext):
    node = ParseURL(url="https://example.com/path?x=1#frag")
    result = await node.process(context)
    assert result["scheme"] == "https"
    assert result["netloc"] == "example.com"
    assert result["path"] == "/path"
    assert result["query"] == "x=1"
    assert result["fragment"] == "frag"


@pytest.mark.asyncio
async def test_join_url(context: ProcessingContext):
    node = JoinURL(base="https://example.com/api/", url="v1")
    result = await node.process(context)
    assert result == "https://example.com/api/v1"


@pytest.mark.asyncio
async def test_encode_query_params(context: ProcessingContext):
    node = EncodeQueryParams(params={"a": "1", "b": "2"})
    result = await node.process(context)
    assert "a=1" in result and "b=2" in result


@pytest.mark.asyncio
async def test_quote_unquote(context: ProcessingContext):
    quoted = await QuoteURL(text="hello world").process(context)
    assert "%20" in quoted
    unquoted = await UnquoteURL(text=quoted).process(context)
    assert unquoted == "hello world"
