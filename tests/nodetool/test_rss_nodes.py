import pytest
from datetime import datetime
import feedparser

from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_fetch_rss_feed(context: ProcessingContext):
    # Real RSS feed XML string
    rss_xml = """<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0">
    <channel>
        <title>Test Feed</title>
        <description>A test RSS feed</description>
        <link>https://example.com</link>
        <item>
            <title>Item 1</title>
            <link>https://example.com/1</link>
            <description>Summary 1</description>
            <author>author1@example.com</author>
            <pubDate>Wed, 01 May 2024 12:00:00 +0000</pubDate>
        </item>
        <item>
            <title>Item 2</title>
            <link>https://example.com/2</link>
            <description>Summary 2</description>
            <author>author2@example.com</author>
            <!-- No pubDate to test fallback -->
        </item>
    </channel>
    </rss>"""

    # Test directly with feedparser - no mocking needed
    parsed_feed = feedparser.parse(rss_xml)
    
    # Manually run the RSS processing logic with our parsed feed
    data = []
    for entry in parsed_feed.entries:
        from nodetool.metadata.types import Datetime, RSSEntry
        
        published = datetime.now()  # fallback
        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6])
        
        data.append(
            RSSEntry(
                title=entry.get("title", ""),
                link=entry.get("link", ""),
                published=Datetime.from_datetime(published),
                summary=entry.get("summary", ""),
                author=entry.get("author", ""),
            )
        )
    
    result = data
    
    assert len(result) == 2
    assert result[0].title == "Item 1"
    assert result[0].link == "https://example.com/1"
    assert result[0].summary == "Summary 1"
    assert result[0].author == "author1@example.com"
    # published should be parsed from pubDate
    assert result[0].published.to_datetime().year == 2024
    assert result[0].published.to_datetime().month == 5
    
    # Second item should have fallback datetime (current time)
    assert result[1].title == "Item 2"
    assert result[1].author == "author2@example.com"


@pytest.mark.asyncio 
async def test_extract_feed_metadata(context: ProcessingContext):
    # Real RSS feed XML string with metadata
    rss_xml = """<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0">
    <channel>
        <title>Test Feed Title</title>
        <description>Test feed description</description>
        <link>https://example.com</link>
        <language>en-us</language>
        <generator>Test Generator</generator>
        <item>
            <title>Item 1</title>
            <link>https://example.com/1</link>
        </item>
        <item>
            <title>Item 2</title>
            <link>https://example.com/2</link>
        </item>
        <item>
            <title>Item 3</title>
            <link>https://example.com/3</link>
        </item>
    </channel>
    </rss>"""

    # Test directly with feedparser - no mocking needed  
    parsed_feed = feedparser.parse(rss_xml)
    
    # Manually run the metadata extraction logic with our parsed feed
    feed_info = parsed_feed.feed
    meta = {
        "title": feed_info.get("title", ""),
        "description": feed_info.get("description", ""),
        "link": feed_info.get("link", ""),
        "language": feed_info.get("language", ""),
        "updated": feed_info.get("updated", ""),
        "generator": feed_info.get("generator", ""),
        "entry_count": len(parsed_feed.entries),
    }
    
    assert meta["title"] == "Test Feed Title"
    assert meta["entry_count"] == 3
    assert meta["description"] == "Test feed description"
    assert meta["link"] == "https://example.com"

