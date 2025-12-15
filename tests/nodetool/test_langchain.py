import pytest
from nodetool.metadata.types import DocumentRef
from nodetool.nodes.nodetool.document import (
    SplitRecursively,
    SplitMarkdown,
)
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def processing_context():
    # Using test values for required parameters
    return ProcessingContext(user_id="test-user", auth_token="test-token")


class TestSplitRecursively:
    @pytest.mark.asyncio
    async def test_process_splits_text_correctly(self, processing_context):
        doc_ref = DocumentRef(
            uri="test-doc", data="First line\nSecond line\nThird line"
        )
        node = SplitRecursively(
            document=doc_ref,
            chunk_size=5,
            chunk_overlap=0,
            separators=["\n\n", "\n", "."],
        )
        result = []
        async for chunk in node.gen_process(processing_context):
            result.append(chunk)

        assert result == [
            {"text": "First line", "source_id": "test-doc:0", "start_index": 0},
            {
                "text": "\nSecond line",
                "source_id": "test-doc:1",
                "start_index": 10,
            },
            {
                "text": "\nThird line",
                "source_id": "test-doc:2",
                "start_index": 22,
            },
        ]


class TestSplitMarkdown:
    @pytest.mark.asyncio
    async def test_process_splits_markdown_correctly(self, processing_context):
        doc_ref = DocumentRef(
            uri="test-md-doc", data="# Header 1\nContent 1\n## Header 2\nContent 2"
        )
        node = SplitMarkdown(
            document=doc_ref,
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
            strip_headers=True,
        )

        result = []
        async for chunk in node.gen_process(processing_context):
            result.append(chunk)

        # Assert
        assert result == [
            {"text": "Content 1", "source_id": "test-md-doc", "start_index": 0},
            {"text": "Content 2", "source_id": "test-md-doc", "start_index": 0},
        ]