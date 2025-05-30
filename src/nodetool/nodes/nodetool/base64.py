import base64
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class Encode(BaseNode):
    """Encodes text to Base64 format.
    base64, encode, string

    Use cases:
    - Prepare text for transmission
    - Embed data in JSON or HTML
    """

    text: str = Field(default="", description="Text to encode")

    @classmethod
    def get_title(cls):
        return "Encode Base64"

    async def process(self, context: ProcessingContext) -> str:
        return base64.b64encode(self.text.encode()).decode()


class Decode(BaseNode):
    """Decodes Base64 text to plain string.
    base64, decode, string

    Use cases:
    - Read encoded data
    - Extract original text from Base64
    """

    data: str = Field(default="", description="Base64 encoded text")

    @classmethod
    def get_title(cls):
        return "Decode Base64"

    async def process(self, context: ProcessingContext) -> str:
        return base64.b64decode(self.data.encode()).decode()
