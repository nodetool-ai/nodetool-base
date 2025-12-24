import textwrap
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


class Dedent(BaseNode):
    """
    Removes any common leading whitespace from every line in text.
    textwrap, dedent, whitespace
    """

    text: str = Field(default="")

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.dedent(self.text)


class Fill(BaseNode):
    """
    Wraps text to a specified width, returning a formatted string.
    textwrap, fill, wrap
    """

    text: str = Field(default="")
    width: int = Field(default=70)

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.fill(self.text, width=self.width)


class Indent(BaseNode):
    """
    Adds a prefix to the beginning of each line in the text.
    textwrap, indent, prefix
    """

    text: str = Field(default="")
    prefix: str = Field(default="    ")

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.indent(self.text, prefix=self.prefix)


class Shorten(BaseNode):
    """
    Shortens text to fit within a width, using a placeholder if truncated.
    textwrap, shorten, truncate
    """

    text: str = Field(default="")
    width: int = Field(default=70)
    placeholder: str = Field(default="...")

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.shorten(self.text, width=self.width, placeholder=self.placeholder)


class Wrap(BaseNode):
    """
    Wraps text to a specified width, returning a list of lines.
    textwrap, wrap, lines
    """

    text: str = Field(default="")
    width: int = Field(default=70)

    async def process(self, context: ProcessingContext) -> list[str]:
        return textwrap.wrap(self.text, width=self.width)
