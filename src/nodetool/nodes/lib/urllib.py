from urllib.parse import urlparse, urljoin, urlencode, quote, unquote
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class ParseURL(BaseNode):
    """
    Parse a URL into its components using ``urllib.parse.urlparse``.
    urllib, parse, url

    Use cases:
    - Inspect links for validation
    - Extract host or path information
    - Analyze query parameters
    """

    url: str = Field(default="", description="URL to parse")

    @classmethod
    def get_title(cls):
        return "Parse URL"

    async def process(self, context: ProcessingContext) -> dict:
        parsed = urlparse(self.url)
        return {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "username": parsed.username,
            "password": parsed.password,
            "hostname": parsed.hostname,
            "port": parsed.port,
        }


class JoinURL(BaseNode):
    """
    Join a base URL with a relative URL using ``urllib.parse.urljoin``.
    urllib, join, url

    Use cases:
    - Build absolute links from relative paths
    - Combine API base with endpoints
    - Resolve resources from a base URL
    """

    base: str = Field(default="", description="Base URL")
    url: str = Field(default="", description="Relative or absolute URL")

    @classmethod
    def get_title(cls):
        return "Join URL"

    async def process(self, context: ProcessingContext) -> str:
        return urljoin(self.base, self.url)


class EncodeQueryParams(BaseNode):
    """
    Encode a dictionary of parameters into a query string using
    ``urllib.parse.urlencode``.
    urllib, query, encode, params

    Use cases:
    - Build GET request URLs
    - Serialize data for APIs
    - Convert parameters to query strings
    """

    params: dict[str, str] = Field(
        default_factory=dict, description="Parameters to encode"
    )

    @classmethod
    def get_title(cls):
        return "Encode Query Params"

    async def process(self, context: ProcessingContext) -> str:
        return urlencode(self.params, doseq=True)


class QuoteURL(BaseNode):
    """
    Percent-encode a string for safe use in URLs using ``urllib.parse.quote``.
    urllib, quote, encode

    Use cases:
    - Escape spaces or special characters
    - Prepare text for query parameters
    - Encode file names in URLs
    """

    text: str = Field(default="", description="Text to quote")

    @classmethod
    def get_title(cls):
        return "Quote URL"

    async def process(self, context: ProcessingContext) -> str:
        return quote(self.text)


class UnquoteURL(BaseNode):
    """
    Decode a percent-encoded URL string using ``urllib.parse.unquote``.
    urllib, unquote, decode

    Use cases:
    - Convert encoded URLs to readable form
    - Parse user input from URLs
    - Display unescaped paths
    """

    text: str = Field(default="", description="Encoded text")

    @classmethod
    def get_title(cls):
        return "Unquote URL"

    async def process(self, context: ProcessingContext) -> str:
        return unquote(self.text)
