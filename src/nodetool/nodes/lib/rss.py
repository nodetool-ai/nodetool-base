from datetime import datetime
from typing import AsyncGenerator, ClassVar, TypedDict

from pydantic import Field

from nodetool.metadata.types import Datetime
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


def _parse_feed(url: str):
    import feedparser

    return feedparser.parse(url)


class FetchRSSFeed(BaseNode):
    """
    Fetch and parse RSS feed from URL, emitting each entry as a stream item.

    Downloads and parses an RSS/Atom feed, extracting entry metadata including title,
    link, publication date, summary, and author. Emits entries as a stream for
    individual processing.

    Parameters:
    - url (required): RSS/Atom feed URL

    Yields: Dictionary for each entry with "title" (string), "link" (URL string),
    "published" (Datetime), "summary" (content string), and "author" (string)

    Side effects: Network request to fetch feed

    Typical usage: Monitor news feeds, aggregate blog updates, or collect content
    from multiple sources. Follow with filtering, text processing, or summarization
    nodes. Use Collect node to gather all entries.

    rss, feed, network
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
        feed = _parse_feed(self.url)

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
    Extract metadata about an RSS feed itself (not individual entries).

    Fetches and parses the feed to extract high-level information about the feed
    source, including title, description, language, and entry count.

    Parameters:
    - url (required): RSS/Atom feed URL

    Returns: Dictionary with "title", "description", "link", "language", "updated",
    "generator" (all strings), and "entry_count" (integer)

    Side effects: Network request to fetch feed

    Typical usage: Validate feed sources, catalog feeds, or extract feed information
    before processing entries. Precede with URL list or configuration nodes. Follow
    with conditional logic or data aggregation.

    rss, metadata, feed
    """

    @classmethod
    def get_title(cls):
        return "Extract Feed Metadata"

    url: str = Field(default="", description="URL of the RSS feed")

    async def process(self, context: ProcessingContext) -> dict:
        parsed = _parse_feed(self.url)
        feed_info = parsed.feed

        return {
            "title": feed_info.get("title", ""),
            "description": feed_info.get("description", ""),
            "link": feed_info.get("link", ""),
            "language": feed_info.get("language", ""),
            "updated": feed_info.get("updated", ""),
            "generator": feed_info.get("generator", ""),
            "entry_count": len(parsed.entries),
        }
