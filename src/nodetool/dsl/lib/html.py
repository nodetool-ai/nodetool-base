from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Escape(GraphNode):
    """
    Escape special characters in text into HTML-safe sequences.
    html, escape, entities, convert

    Use cases:
    - Prepare text for inclusion in HTML
    - Prevent cross-site scripting in user content
    - Encode strings for web output
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to escape"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.html.Escape"


class Unescape(GraphNode):
    """
    Convert HTML entities back to normal text.
    html, unescape, entities, decode

    Use cases:
    - Decode HTML-encoded data
    - Process text scraped from the web
    - Convert form submissions to plain text
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The HTML text to unescape"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.html.Unescape"
