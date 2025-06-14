from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Decode(GraphNode):
    """Decodes Base64 text to plain string.
    base64, decode, string

    Use cases:
    - Read encoded data
    - Extract original text from Base64
    """

    data: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Base64 encoded text"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.base64.Decode"


class Encode(GraphNode):
    """Encodes text to Base64 format.
    base64, encode, string

    Use cases:
    - Prepare text for transmission
    - Embed data in JSON or HTML
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text to encode"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.base64.Encode"
