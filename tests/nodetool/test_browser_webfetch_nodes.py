import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.browser import WebFetch, DownloadFile


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class DummyResponse:
    def __init__(self, status=200, headers=None, text_value="", bytes_value=b""):
        self.status = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._text_value = text_value
        self._bytes_value = bytes_value

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text_value

    async def read(self):
        return self._bytes_value


class DummySession:
    def __init__(self, response: DummyResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url):
        return self._response


@pytest.mark.asyncio
async def test_webfetch_extracts_and_converts_to_markdown(context: ProcessingContext):
    html = "<html><body><div id='main'>Hello <b>World</b></div></body></html>"
    resp = DummyResponse(text_value=html)

    with patch("nodetool.nodes.lib.browser.aiohttp.ClientSession", return_value=DummySession(resp)):
        node = WebFetch(url="https://x", selector="#main")
        md = await node.process(context)
        assert "Hello" in md and "World" in md


@pytest.mark.asyncio
async def test_download_file_returns_bytes(context: ProcessingContext):
    data = b"abc123"
    resp = DummyResponse(text_value="", bytes_value=data)

    with patch("nodetool.nodes.lib.browser.aiohttp.ClientSession", return_value=DummySession(resp)):
        node = DownloadFile(url="https://x/file.bin")
        out = await node.process(context)
        assert out == data

