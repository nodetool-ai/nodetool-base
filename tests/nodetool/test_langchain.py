import pytest
from unittest.mock import patch, MagicMock
from nodetool.nodes.lib.langchain import (
    RecursiveTextSplitter,
    MarkdownSplitter,
    SentenceSplitter,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import TextChunk


@pytest.fixture
def processing_context():
    # Using test values for required parameters
    return ProcessingContext(user_id="test-user", auth_token="test-token")


@pytest.fixture
def mock_document():
    return MagicMock(name="Document")


class TestRecursiveTextSplitter:
    @pytest.mark.asyncio
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    @patch("langchain_core.documents.Document")
    async def test_process_splits_text_correctly(
        self, mock_document_class, mock_splitter, processing_context
    ):
        # Setup
        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance

        # Create mock documents with metadata
        mock_docs = [
            MagicMock(page_content="Chunk 1", metadata={"start_index": 0}),
            MagicMock(page_content="Chunk 2", metadata={"start_index": 100}),
        ]
        mock_splitter_instance.split_documents.return_value = mock_docs

        # Create node instance
        from nodetool.metadata.types import DocumentRef

        doc_ref = DocumentRef(uri="test-doc", data="Sample text for testing")
        node = RecursiveTextSplitter(
            document=doc_ref,
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "Chunk 1"
        assert result[0].source_id == "test-doc:0"
        assert result[0].start_index == 0
        assert result[1].text == "Chunk 2"
        assert result[1].source_id == "test-doc:1"
        assert result[1].start_index == 100

        # Verify splitter was created with correct parameters
        mock_splitter.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

        # Verify Document was called with the correct text
        mock_document_class.assert_called_once_with(
            page_content="Sample text for testing"
        )


class TestMarkdownSplitter:
    @pytest.mark.asyncio
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    async def test_process_splits_markdown_correctly(
        self, mock_md_splitter, processing_context
    ):
        # Setup
        mock_md_splitter_instance = MagicMock()
        mock_md_splitter.return_value = mock_md_splitter_instance

        # Create mock documents with metadata
        mock_docs = [
            MagicMock(
                page_content="# Header 1\nContent 1", metadata={"start_index": 0}
            ),
            MagicMock(
                page_content="## Header 2\nContent 2", metadata={"start_index": 50}
            ),
        ]
        mock_md_splitter_instance.split_text.return_value = mock_docs

        # Create node instance
        from nodetool.metadata.types import DocumentRef

        doc_ref = DocumentRef(
            uri="test-md-doc", data="# Header 1\nContent 1\n## Header 2\nContent 2"
        )
        node = MarkdownSplitter(
            document=doc_ref,
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
            strip_headers=True,
        )

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "# Header 1\nContent 1"
        assert result[0].source_id == "test-md-doc"
        assert result[0].start_index == 0
        assert result[1].text == "## Header 2\nContent 2"
        assert result[1].source_id == "test-md-doc"
        assert result[1].start_index == 50

        # Verify splitter was created with correct parameters
        mock_md_splitter.assert_called_once_with(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
            strip_headers=True,
            return_each_line=False,
        )

    @pytest.mark.asyncio
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    @patch("langchain_core.documents.Document")
    async def test_process_with_chunk_size(
        self,
        mock_document_class,
        mock_text_splitter,
        mock_md_splitter,
        processing_context,
    ):
        # Setup
        mock_md_splitter_instance = MagicMock()
        mock_md_splitter.return_value = mock_md_splitter_instance

        mock_text_splitter_instance = MagicMock()
        mock_text_splitter.return_value = mock_text_splitter_instance

        # Create mock documents
        mock_md_docs = [MagicMock(page_content="Content", metadata={})]
        mock_md_splitter_instance.split_text.return_value = mock_md_docs

        mock_split_docs = [
            MagicMock(page_content="Chunk 1", metadata={"start_index": 0}),
            MagicMock(page_content="Chunk 2", metadata={"start_index": 20}),
        ]
        mock_text_splitter_instance.split_documents.return_value = mock_split_docs

        # Create node instance with chunk_size
        from nodetool.metadata.types import DocumentRef

        doc_ref = DocumentRef(uri="test-md-doc", data="# Long markdown content")
        node = MarkdownSplitter(
            document=doc_ref,
            chunk_size=100,
            chunk_overlap=20,
        )

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert len(result) == 2
        assert result[0].text == "Chunk 1"
        assert result[1].text == "Chunk 2"

        # Verify text splitter was created with correct parameters
        mock_text_splitter.assert_called_once_with(chunk_size=100, chunk_overlap=20)


class TestSentenceSplitter:
    @pytest.mark.asyncio
    @patch("langchain_text_splitters.SentenceTransformersTokenTextSplitter")
    @patch("langchain_core.documents.Document")
    async def test_process_splits_sentences_correctly(
        self, mock_document_class, mock_splitter, processing_context
    ):
        # Setup
        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance

        # Create mock documents with metadata
        mock_docs = [
            MagicMock(page_content="Sentence 1.", metadata={"start_index": 0}),
            MagicMock(page_content="Sentence 2.", metadata={"start_index": 12}),
        ]
        mock_splitter_instance.split_documents.return_value = mock_docs

        # Create node instance
        from nodetool.metadata.types import DocumentRef

        doc_ref = DocumentRef(uri="test-sentence-doc", data="Sentence 1. Sentence 2.")
        node = SentenceSplitter(
            document=doc_ref,
            chunk_size=30,
            chunk_overlap=5,
        )

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "Sentence 1."
        assert result[0].source_id == "test-sentence-doc:0"
        assert result[0].start_index == 0
        assert result[1].text == "Sentence 2."
        assert result[1].source_id == "test-sentence-doc:1"
        assert result[1].start_index == 12

        # Verify splitter was created with correct parameters
        mock_splitter.assert_called_once_with(
            chunk_size=30, chunk_overlap=5, add_start_index=True
        )

        # Verify Document was called with the correct text
        mock_document_class.assert_called_once_with(
            page_content="Sentence 1. Sentence 2."
        )
