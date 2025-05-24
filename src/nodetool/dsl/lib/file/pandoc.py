from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.file.pandoc
import nodetool.nodes.lib.file.pandoc


class ConvertFile(GraphNode):
    """
    Converts between different document formats using pandoc.
    convert, document, format, pandoc

    Use cases:
    - Convert between various document formats (Markdown, HTML, LaTeX, etc.)
    - Generate documentation in different formats
    - Create publication-ready documents
    """

    InputFormat: typing.ClassVar[type] = nodetool.nodes.lib.file.pandoc.InputFormat
    OutputFormat: typing.ClassVar[type] = nodetool.nodes.lib.file.pandoc.OutputFormat
    input_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Path to the input file",
    )
    input_format: nodetool.nodes.lib.file.pandoc.InputFormat = Field(
        default=nodetool.nodes.lib.file.pandoc.InputFormat.MARKDOWN,
        description="Input format",
    )
    output_format: nodetool.nodes.lib.file.pandoc.OutputFormat = Field(
        default=nodetool.nodes.lib.file.pandoc.OutputFormat.PDF,
        description="Output format",
    )
    extra_args: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="Additional pandoc arguments"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.file.pandoc.ConvertFile"


import nodetool.nodes.lib.file.pandoc
import nodetool.nodes.lib.file.pandoc


class ConvertText(GraphNode):
    """
    Converts text content between different document formats using pandoc.
    convert, text, format, pandoc

    Use cases:
    - Convert text content between various formats (Markdown, HTML, LaTeX, etc.)
    - Transform content without saving to disk
    - Process text snippets in different formats
    """

    InputFormat: typing.ClassVar[type] = nodetool.nodes.lib.file.pandoc.InputFormat
    OutputFormat: typing.ClassVar[type] = nodetool.nodes.lib.file.pandoc.OutputFormat
    content: str | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="Text content to convert"
    )
    input_format: nodetool.nodes.lib.file.pandoc.InputFormat = Field(
        default=nodetool.nodes.lib.file.pandoc.InputFormat(PydanticUndefined),
        description="Input format",
    )
    output_format: nodetool.nodes.lib.file.pandoc.OutputFormat = Field(
        default=nodetool.nodes.lib.file.pandoc.OutputFormat(PydanticUndefined),
        description="Output format",
    )
    extra_args: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="Additional pandoc arguments"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.file.pandoc.ConvertText"
