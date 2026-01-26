#!/usr/bin/env python

"""
Brave Search nodes for Nodetool.
Provides nodes for Web Search, News Search, Image Search, and Video Search using the Brave Search API.
"""

from enum import Enum
from typing import Any, ClassVar, TypedDict

from pydantic import Field, BaseModel

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class SafeSearchLevel(str, Enum):
    """Safe search filter level."""

    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class BraveWebResult(BaseModel):
    """A web search result from Brave Search."""

    title: str = ""
    url: str = ""
    description: str = ""
    page_age: str = ""
    language: str = ""
    family_friendly: bool = True


class BraveNewsResult(BaseModel):
    """A news result from Brave Search."""

    title: str = ""
    url: str = ""
    description: str = ""
    age: str = ""
    source: str = ""
    thumbnail_url: str = ""


class BraveImageResult(BaseModel):
    """An image result from Brave Search."""

    title: str = ""
    url: str = ""
    source_url: str = ""
    thumbnail_url: str = ""
    width: int = 0
    height: int = 0


class BraveVideoResult(BaseModel):
    """A video result from Brave Search."""

    title: str = ""
    url: str = ""
    description: str = ""
    age: str = ""
    thumbnail_url: str = ""
    creator: str = ""
    publisher: str = ""
    duration: str = ""


class BraveSearchBase(BaseNode):
    """Base node for Brave Search API requests.

    brave, search, api
    """

    BRAVE_API_BASE = "https://api.search.brave.com/res/v1"

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BraveSearchBase

    async def get_api_key(self, context: ProcessingContext) -> str:
        """Get the Brave Search API key from environment."""
        api_key = await context.get_environment_secret("BRAVE_API_KEY")
        if not api_key:
            raise ValueError(
                "BRAVE_API_KEY is required. Please set the BRAVE_API_KEY environment variable."
            )
        return api_key

    async def make_request(
        self, context: ProcessingContext, endpoint: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a request to the Brave Search API."""
        api_key = await self.get_api_key(context)

        url = f"{self.BRAVE_API_BASE}/{endpoint}"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

        # Filter out None and empty values from params
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        response = await context.http_get(url, headers=headers, params=params)
        return response.json()


class BraveWebSearch(BraveSearchBase):
    """
    Search the web using Brave Search API.
    brave, search, web, query, internet
    """

    query: str = Field(default="", description="Search query or keyword to search for")
    count: int = Field(
        default=10, ge=1, le=100, description="Number of results to return (1-100)"
    )
    offset: int = Field(
        default=0, ge=0, description="Offset for pagination (default: 0)"
    )
    country: str = Field(
        default="",
        description="Country code for search results (e.g., 'us', 'gb', 'de')",
    )
    search_lang: str = Field(
        default="",
        description="Language code for search (e.g., 'en', 'fr', 'de')",
    )
    safesearch: SafeSearchLevel = Field(
        default=SafeSearchLevel.MODERATE,
        description="Safe search filter level",
    )
    freshness: str = Field(
        default="",
        description="Filter by content freshness (e.g., 'pd' for past day, 'pw' for past week, 'pm' for past month, 'py' for past year)",
    )
    text_decorations: bool = Field(
        default=False,
        description="Include text decorations like bold markers in snippets",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[BraveWebResult]:
        if not self.query:
            raise ValueError("Query is required for web search")

        params = {
            "q": self.query,
            "count": self.count,
            "offset": self.offset,
            "country": self.country if self.country else None,
            "search_lang": self.search_lang if self.search_lang else None,
            "safesearch": self.safesearch.value,
            "freshness": self.freshness if self.freshness else None,
            "text_decorations": self.text_decorations,
        }

        result = await self.make_request(context, "web/search", params)

        web_results = result.get("web", {}).get("results", [])

        return [
            BraveWebResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
                page_age=r.get("page_age", ""),
                language=r.get("language", ""),
                family_friendly=r.get("family_friendly", True),
            )
            for r in web_results
        ]


class BraveNewsSearch(BraveSearchBase):
    """
    Search news articles using Brave Search API.
    brave, search, news, articles, headlines
    """

    query: str = Field(
        default="", description="Search query or keyword for news articles"
    )
    count: int = Field(
        default=10, ge=1, le=100, description="Number of results to return (1-100)"
    )
    offset: int = Field(
        default=0, ge=0, description="Offset for pagination (default: 0)"
    )
    country: str = Field(
        default="",
        description="Country code for news results (e.g., 'us', 'gb', 'de')",
    )
    search_lang: str = Field(
        default="",
        description="Language code for search (e.g., 'en', 'fr', 'de')",
    )
    safesearch: SafeSearchLevel = Field(
        default=SafeSearchLevel.MODERATE,
        description="Safe search filter level",
    )
    freshness: str = Field(
        default="",
        description="Filter by content freshness (e.g., 'pd' for past day, 'pw' for past week, 'pm' for past month)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[BraveNewsResult]:
        if not self.query:
            raise ValueError("Query is required for news search")

        params = {
            "q": self.query,
            "count": self.count,
            "offset": self.offset,
            "country": self.country if self.country else None,
            "search_lang": self.search_lang if self.search_lang else None,
            "safesearch": self.safesearch.value,
            "freshness": self.freshness if self.freshness else None,
        }

        result = await self.make_request(context, "news/search", params)

        news_results = result.get("results", [])

        return [
            BraveNewsResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
                age=r.get("age", ""),
                source=(
                    r.get("meta_url", {}).get("netloc", "")
                    if isinstance(r.get("meta_url"), dict)
                    else ""
                ),
                thumbnail_url=(
                    r.get("thumbnail", {}).get("src", "")
                    if isinstance(r.get("thumbnail"), dict)
                    else ""
                ),
            )
            for r in news_results
        ]


class BraveImageSearch(BraveSearchBase):
    """
    Search for images using Brave Search API.
    brave, search, images, visual, pictures
    """

    query: str = Field(default="", description="Search query or keyword for images")
    count: int = Field(
        default=20, ge=1, le=150, description="Number of results to return (1-150)"
    )
    country: str = Field(
        default="",
        description="Country code for image results (e.g., 'us', 'gb', 'de')",
    )
    search_lang: str = Field(
        default="",
        description="Language code for search (e.g., 'en', 'fr', 'de')",
    )
    safesearch: SafeSearchLevel = Field(
        default=SafeSearchLevel.MODERATE,
        description="Safe search filter level",
    )
    spellcheck: bool = Field(
        default=True,
        description="Enable spell check for the query",
    )

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        results: list[BraveImageResult]
        images: list[ImageRef]

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required for image search")

        params = {
            "q": self.query,
            "count": self.count,
            "country": self.country if self.country else None,
            "search_lang": self.search_lang if self.search_lang else None,
            "safesearch": self.safesearch.value,
            "spellcheck": self.spellcheck,
        }

        result = await self.make_request(context, "images/search", params)

        image_results = result.get("results", [])

        brave_results = [
            BraveImageResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                source_url=r.get("source", ""),
                thumbnail_url=(
                    r.get("thumbnail", {}).get("src", "")
                    if isinstance(r.get("thumbnail"), dict)
                    else ""
                ),
                width=(
                    r.get("properties", {}).get("width", 0)
                    if isinstance(r.get("properties"), dict)
                    else 0
                ),
                height=(
                    r.get("properties", {}).get("height", 0)
                    if isinstance(r.get("properties"), dict)
                    else 0
                ),
            )
            for r in image_results
        ]

        image_refs = [
            ImageRef(uri=r.get("url", "")) for r in image_results if r.get("url")
        ]

        return {
            "results": brave_results,
            "images": image_refs,
        }


class BraveVideoSearch(BraveSearchBase):
    """
    Search for videos using Brave Search API.
    brave, search, videos, media, streaming
    """

    query: str = Field(default="", description="Search query or keyword for videos")
    count: int = Field(
        default=10, ge=1, le=50, description="Number of results to return (1-50)"
    )
    offset: int = Field(
        default=0, ge=0, description="Offset for pagination (default: 0)"
    )
    country: str = Field(
        default="",
        description="Country code for video results (e.g., 'us', 'gb', 'de')",
    )
    search_lang: str = Field(
        default="",
        description="Language code for search (e.g., 'en', 'fr', 'de')",
    )
    safesearch: SafeSearchLevel = Field(
        default=SafeSearchLevel.MODERATE,
        description="Safe search filter level",
    )
    freshness: str = Field(
        default="",
        description="Filter by content freshness (e.g., 'pd' for past day, 'pw' for past week, 'pm' for past month)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[BraveVideoResult]:
        if not self.query:
            raise ValueError("Query is required for video search")

        params = {
            "q": self.query,
            "count": self.count,
            "offset": self.offset,
            "country": self.country if self.country else None,
            "search_lang": self.search_lang if self.search_lang else None,
            "safesearch": self.safesearch.value,
            "freshness": self.freshness if self.freshness else None,
        }

        result = await self.make_request(context, "videos/search", params)

        video_results = result.get("results", [])

        return [
            BraveVideoResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
                age=r.get("age", ""),
                thumbnail_url=(
                    r.get("thumbnail", {}).get("src", "")
                    if isinstance(r.get("thumbnail"), dict)
                    else ""
                ),
                creator=r.get("creator", ""),
                publisher=(
                    r.get("meta_url", {}).get("netloc", "")
                    if isinstance(r.get("meta_url"), dict)
                    else ""
                ),
                duration=(
                    r.get("video", {}).get("duration", "")
                    if isinstance(r.get("video"), dict)
                    else ""
                ),
            )
            for r in video_results
        ]
