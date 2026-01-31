import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.pdfplumber import (
    ExtractText,
    GetPageCount,
    ExtractPageMetadata,
)
from nodetool.metadata.types import DocumentRef
import io


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# Minimal valid PDF content for testing
# This is a minimal PDF that should be parseable by pdfplumber
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello PDF) Tj
ET
endstream
endobj
5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000359 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
440
%%EOF"""


@pytest.fixture
def pdf_document():
    """Create a DocumentRef with PDF content."""
    return DocumentRef(data=MINIMAL_PDF)


class TestGetPageCount:
    """Tests for GetPageCount node."""

    @pytest.mark.asyncio
    async def test_get_page_count(self, context: ProcessingContext, pdf_document):
        """Test counting pages in a PDF."""
        node = GetPageCount(pdf=pdf_document)
        result = await node.process(context)
        
        assert result == 1  # Our minimal PDF has 1 page


class TestExtractPageMetadata:
    """Tests for ExtractPageMetadata node."""

    @pytest.mark.asyncio
    async def test_extract_metadata(self, context: ProcessingContext, pdf_document):
        """Test extracting page metadata."""
        node = ExtractPageMetadata(pdf=pdf_document, start_page=0, end_page=1)
        result = await node.process(context)
        
        assert len(result) == 1
        assert result[0]["page_number"] == 1
        assert "width" in result[0]
        assert "height" in result[0]
        # Standard US Letter size
        assert result[0]["width"] == 612.0
        assert result[0]["height"] == 792.0


class TestExtractText:
    """Tests for ExtractText node."""

    @pytest.mark.asyncio
    async def test_extract_text(self, context: ProcessingContext, pdf_document):
        """Test extracting text from PDF."""
        node = ExtractText(pdf=pdf_document, start_page=0, end_page=1)
        result = await node.process(context)
        
        # The minimal PDF contains "Hello PDF" text
        assert "Hello PDF" in result

    @pytest.mark.asyncio
    async def test_extract_text_page_range(self, context: ProcessingContext, pdf_document):
        """Test extracting text from specific page range."""
        # Request pages that don't exist
        node = ExtractText(pdf=pdf_document, start_page=5, end_page=10)
        result = await node.process(context)
        
        # Should return empty since pages don't exist
        assert result == ""
