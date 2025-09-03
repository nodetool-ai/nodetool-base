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
    # Create a temporary test file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
        temp_file.write(b"dummy content")
        temp_file_path = temp_file.name

    try:
        with patch("nodetool.nodes.lib.markitdown.MarkItDown", DummyMarkItDown):
            doc = DocumentRef(uri=f"file://{temp_file_path}")
            node = ConvertToMarkdown(document=doc)
            out = await node.process(context)
            assert out.data == "dummy markdown"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
