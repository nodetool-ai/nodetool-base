import pytest
import os
import tempfile
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DocumentRef, DataframeRef
from nodetool.nodes.lib.docx import (
    CreateDocument,
    AddHeading,
    AddParagraph,
    AddPageBreak,
    SetDocumentProperties,
    ParagraphAlignment,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class TestCreateDocument:
    """Tests for CreateDocument node."""

    @pytest.mark.asyncio
    async def test_create_document(self, context: ProcessingContext):
        node = CreateDocument()
        result = await node.process(context)
        assert isinstance(result, DocumentRef)
        assert result.data is not None


class TestAddHeading:
    """Tests for AddHeading node."""

    @pytest.mark.asyncio
    async def test_add_heading(self, context: ProcessingContext):
        # First create a document
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        # Add heading
        heading_node = AddHeading(document=doc, text="Test Heading", level=1)
        result = await heading_node.process(context)

        assert isinstance(result, DocumentRef)
        assert result.data is not None
        # Document should now have content
        paragraphs = list(result.data.paragraphs)
        assert len(paragraphs) >= 1

    @pytest.mark.asyncio
    async def test_add_heading_different_levels(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        for level in [1, 2, 3]:
            heading_node = AddHeading(document=doc, text=f"Heading Level {level}", level=level)
            doc = await heading_node.process(context)

        assert doc.data is not None


class TestAddParagraph:
    """Tests for AddParagraph node."""

    @pytest.mark.asyncio
    async def test_add_simple_paragraph(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        para_node = AddParagraph(document=doc, text="This is a test paragraph.")
        result = await para_node.process(context)

        assert isinstance(result, DocumentRef)
        paragraphs = list(result.data.paragraphs)
        assert len(paragraphs) >= 1

    @pytest.mark.asyncio
    async def test_add_formatted_paragraph(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        para_node = AddParagraph(
            document=doc,
            text="Bold and italic text",
            bold=True,
            italic=True,
            font_size=14,
            alignment=ParagraphAlignment.CENTER,
        )
        result = await para_node.process(context)
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_paragraph_alignments(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        for alignment in ParagraphAlignment:
            para_node = AddParagraph(
                document=doc,
                text=f"{alignment.value} aligned text",
                alignment=alignment,
            )
            doc = await para_node.process(context)

        assert doc.data is not None


class TestAddPageBreak:
    """Tests for AddPageBreak node."""

    @pytest.mark.asyncio
    async def test_add_page_break(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        # Add paragraph
        para_node = AddParagraph(document=doc, text="Page 1 content")
        doc = await para_node.process(context)

        # Add page break
        break_node = AddPageBreak(document=doc)
        result = await break_node.process(context)

        assert result.data is not None


class TestSetDocumentProperties:
    """Tests for SetDocumentProperties node."""

    @pytest.mark.asyncio
    async def test_set_properties(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        props_node = SetDocumentProperties(
            document=doc,
            title="Test Document",
            author="Test Author",
            subject="Test Subject",
            keywords="test, document, keywords",
        )
        result = await props_node.process(context)

        assert result.data is not None
        core_props = result.data.core_properties
        assert core_props.title == "Test Document"
        assert core_props.author == "Test Author"
        assert core_props.subject == "Test Subject"
        assert core_props.keywords == "test, document, keywords"

    @pytest.mark.asyncio
    async def test_partial_properties(self, context: ProcessingContext):
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        # Only set some properties
        props_node = SetDocumentProperties(
            document=doc,
            title="Only Title",
        )
        result = await props_node.process(context)

        assert result.data is not None
        core_props = result.data.core_properties
        assert core_props.title == "Only Title"


class TestDocumentWorkflow:
    """Integration tests for document creation workflows."""

    @pytest.mark.asyncio
    async def test_complete_document_workflow(self, context: ProcessingContext):
        """Test creating a complete document with multiple elements."""
        # Create document
        doc_node = CreateDocument()
        doc = await doc_node.process(context)

        # Set properties
        props_node = SetDocumentProperties(
            document=doc,
            title="Complete Document",
            author="Test Suite",
        )
        doc = await props_node.process(context)

        # Add heading
        heading_node = AddHeading(document=doc, text="Introduction", level=1)
        doc = await heading_node.process(context)

        # Add paragraph
        para_node = AddParagraph(
            document=doc,
            text="This is the introduction paragraph.",
        )
        doc = await para_node.process(context)

        # Add page break
        break_node = AddPageBreak(document=doc)
        doc = await break_node.process(context)

        # Add second section
        heading2_node = AddHeading(document=doc, text="Section 2", level=2)
        doc = await heading2_node.process(context)

        para2_node = AddParagraph(
            document=doc,
            text="This is section 2 content.",
            bold=True,
        )
        doc = await para2_node.process(context)

        # Verify final document
        assert doc.data is not None
        paragraphs = list(doc.data.paragraphs)
        assert len(paragraphs) >= 4
