from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Dedent(GraphNode):
    """
    Removes any common leading whitespace from every line in text.
    textwrap, dedent, whitespace
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)

    @classmethod
    def get_node_type(cls):
        return "nodetool.textwrap.Dedent"


class Fill(GraphNode):
    """
    Wraps text to a specified width, returning a formatted string.
    textwrap, fill, wrap
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    width: int | GraphNode | tuple[GraphNode, str] = Field(default=70, description=None)

    @classmethod
    def get_node_type(cls):
        return "nodetool.textwrap.Fill"


class Indent(GraphNode):
    """
    Adds a prefix to the beginning of each line in the text.
    textwrap, indent, prefix
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    prefix: str | GraphNode | tuple[GraphNode, str] = Field(
        default="    ", description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.textwrap.Indent"


class Shorten(GraphNode):
    """
    Shortens text to fit within a width, using a placeholder if truncated.
    textwrap, shorten, truncate
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    width: int | GraphNode | tuple[GraphNode, str] = Field(default=70, description=None)
    placeholder: str | GraphNode | tuple[GraphNode, str] = Field(
        default="...", description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.textwrap.Shorten"


class Wrap(GraphNode):
    """
    Wraps text to a specified width, returning a list of lines.
    textwrap, wrap, lines
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    width: int | GraphNode | tuple[GraphNode, str] = Field(default=70, description=None)

    @classmethod
    def get_node_type(cls):
        return "nodetool.textwrap.Wrap"
