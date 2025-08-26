import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.markitdown import ConvertToMarkdown
from nodetool.metadata.types import DocumentRef


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class DummyResult:
    def __init__(self, text_content: str):
        self.text_content = text_content


class DummyMarkItDown:
    def convert(self, uri):
        return DummyResult("dummy markdown")


@pytest.mark.asyncio
async def test_convert_to_markdown_with_uri(context: ProcessingContext):
    with patch("nodetool.nodes.lib.markitdown.MarkItDown", DummyMarkItDown):
        doc = DocumentRef(uri="file:///tmp/file.docx")
        node = ConvertToMarkdown(document=doc)
        out = await node.process(context)
        assert out == "dummy markdown"

