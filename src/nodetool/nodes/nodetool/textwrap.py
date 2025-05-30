import textwrap
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


class Fill(BaseNode):
    """
    Wraps text to a specified width, returning a formatted string.
    textwrap, fill, wrap
    """

    text: str = Field(title="Text", default="")
    width: int = Field(title="Width", default=70, ge=1)

    @classmethod
    def get_title(cls):
        return "Fill Text"

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.fill(self.text, width=self.width)


class Wrap(BaseNode):
    """
    Wraps text to a specified width, returning a list of lines.
    textwrap, wrap, lines
    """

    text: str = Field(title="Text", default="")
    width: int = Field(title="Width", default=70, ge=1)

    @classmethod
    def get_title(cls):
        return "Wrap Text"

    async def process(self, context: ProcessingContext) -> list[str]:
        return textwrap.wrap(self.text, width=self.width)


class Shorten(BaseNode):
    """
    Shortens text to fit within a width, using a placeholder if truncated.
    textwrap, shorten, truncate
    """

    text: str = Field(title="Text", default="")
    width: int = Field(title="Width", default=70, ge=1)
    placeholder: str = Field(title="Placeholder", default="...")

    @classmethod
    def get_title(cls):
        return "Shorten Text"

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.shorten(
            self.text, width=self.width, placeholder=self.placeholder
        )


class Indent(BaseNode):
    """
    Adds a prefix to the beginning of each line in the text.
    textwrap, indent, prefix
    """

    text: str = Field(title="Text", default="")
    prefix: str = Field(title="Prefix", default="    ")

    @classmethod
    def get_title(cls):
        return "Indent Text"

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.indent(self.text, prefix=self.prefix)


class Dedent(BaseNode):
    """
    Removes any common leading whitespace from every line in text.
    textwrap, dedent, whitespace
    """

    text: str = Field(title="Text", default="")

    @classmethod
    def get_title(cls):
        return "Dedent Text"

    async def process(self, context: ProcessingContext) -> str:
        return textwrap.dedent(self.text)
