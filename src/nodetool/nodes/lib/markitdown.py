import os
import tempfile
from markitdown import MarkItDown
from pydantic import Field
from typing import ClassVar
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DocumentRef


class ConvertToMarkdown(BaseNode):
    """
    Converts various document formats to markdown using MarkItDown.
    markdown, convert, document

    Use cases:
    - Convert Word documents to markdown
    - Convert Excel files to markdown tables
    - Convert PowerPoint to markdown content
    """

    document: DocumentRef = Field(
        default=DocumentRef(), description="The document to convert to markdown"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> DocumentRef:
        temp_file = None
        try:
            file_content = await context.download_file(self.document.uri)
            with open(file_content.name, "rb") as f:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(f.read())
                temp_file.flush()
                uri = temp_file.name

            md = MarkItDown()

            # Convert document to markdown
            result = md.convert(uri)

            # Return the markdown text content
            return DocumentRef(type="document", uri=uri, data=result.text_content)
        finally:
            if temp_file:
                temp_file.close()
                os.remove(temp_file.name)
