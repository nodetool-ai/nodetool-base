import html
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class Escape(BaseNode):
    """
    Escape special characters in text into HTML-safe sequences.
    html, escape, entities, convert

    Use cases:
    - Prepare text for inclusion in HTML
    - Prevent cross-site scripting in user content
    - Encode strings for web output
    """

    text: str = Field(default="", description="The text to escape")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        escaped = html.escape(self.text)
        # html.escape uses hex value for single quotes (&#x27;)
        # but tests expect the decimal representation
        return escaped.replace("&#x27;", "&#39;")


class Unescape(BaseNode):
    """
    Convert HTML entities back to normal text.
    html, unescape, entities, decode

    Use cases:
    - Decode HTML-encoded data
    - Process text scraped from the web
    - Convert form submissions to plain text
    """

    text: str = Field(default="", description="The HTML text to unescape")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        return html.unescape(self.text)
