from datetime import datetime
from typing import AsyncGenerator, ClassVar, TypedDict
import feedparser
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Datetime


class FetchRSSFeed(BaseNode):
    """
    Fetches and parses an RSS feed from a URL.
    rss, feed, network

    Use cases:
    - Monitor news feeds
    - Aggregate content from multiple sources
    - Process blog updates
    """

    url: str = Field(default="", description="URL of the RSS feed to fetch")

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "Fetch RSS Feed"

    class OutputType(TypedDict):
        title: str
        link: str
        published: Datetime
        summary: str
        author: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        feed = feedparser.parse(self.url)

        data = []
        for entry in feed.entries:
            # Use published_parsed instead of manual parsing
            published = datetime.now()  # fallback
            if entry.get("published_parsed"):
                published = datetime(*entry.published_parsed[:6])  # type: ignore

            yield {
                "title": str(entry.get("title", "")),
                "link": str(entry.get("link", "")),
                "published": Datetime.from_datetime(published),
                "summary": str(entry.get("summary", "")),
                "author": str(entry.get("author", "")),
            }


class ExtractFeedMetadata(BaseNode):
    """
    Extracts metadata from an RSS feed.
    rss, metadata, feed

    Use cases:
    - Get feed information
    - Validate feed details
    - Extract feed metadata
    """

    @classmethod
    def get_title(cls):
        return "Extract Feed Metadata"

    url: str = Field(default="", description="URL of the RSS feed")

    async def process(self, context: ProcessingContext) -> dict:
        feed = feedparser.parse(self.url)

        return {
            "title": feed.get("title", ""),
            "description": feed.get("description", ""),
            "link": feed.get("link", ""),
            "language": feed.get("language", ""),
            "updated": feed.get("updated", ""),
            "generator": feed.get("generator", ""),
            "entry_count": len(feed.entries),
        }
