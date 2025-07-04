from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class EncodeQueryParams(GraphNode):
    """
    Encode a dictionary of parameters into a query string using
    ``urllib.parse.urlencode``.
    urllib, query, encode, params

    Use cases:
    - Build GET request URLs
    - Serialize data for APIs
    - Convert parameters to query strings
    """

    params: dict[str, str] | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="Parameters to encode"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.urllib.EncodeQueryParams"


class JoinURL(GraphNode):
    """
    Join a base URL with a relative URL using ``urllib.parse.urljoin``.
    urllib, join, url

    Use cases:
    - Build absolute links from relative paths
    - Combine API base with endpoints
    - Resolve resources from a base URL
    """

    base: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Base URL"
    )
    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Relative or absolute URL"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.urllib.JoinURL"


class ParseURL(GraphNode):
    """
    Parse a URL into its components using ``urllib.parse.urlparse``.
    urllib, parse, url

    Use cases:
    - Inspect links for validation
    - Extract host or path information
    - Analyze query parameters
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL to parse"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.urllib.ParseURL"


class QuoteURL(GraphNode):
    """
    Percent-encode a string for safe use in URLs using ``urllib.parse.quote``.
    urllib, quote, encode

    Use cases:
    - Escape spaces or special characters
    - Prepare text for query parameters
    - Encode file names in URLs
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text to quote"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.urllib.QuoteURL"


class UnquoteURL(GraphNode):
    """
    Decode a percent-encoded URL string using ``urllib.parse.unquote``.
    urllib, unquote, decode

    Use cases:
    - Convert encoded URLs to readable form
    - Parse user input from URLs
    - Display unescaped paths
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Encoded text"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.urllib.UnquoteURL"
