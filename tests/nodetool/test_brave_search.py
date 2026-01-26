"""Tests for Brave Search nodes."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.search.brave import (
    BraveSearchBase,
    BraveWebSearch,
    BraveNewsSearch,
    BraveImageSearch,
    BraveVideoSearch,
    SafeSearchLevel,
    BraveWebResult,
    BraveNewsResult,
    BraveImageResult,
    BraveVideoResult,
)


class DummyResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data: dict):
        self._json_data = json_data

    def json(self):
        return self._json_data


@pytest.fixture
def mock_context():
    """Create a mock ProcessingContext for testing."""
    ctx = MagicMock(spec=ProcessingContext)
    ctx.get_environment_secret = AsyncMock(return_value="test-api-key")
    ctx.http_get = AsyncMock(return_value=DummyResponse({}))
    return ctx


@pytest.mark.asyncio
async def test_brave_search_base_visibility():
    """Test that BraveSearchBase is not visible but subclasses are."""
    assert BraveSearchBase.is_visible() is False
    assert BraveWebSearch.is_visible() is True
    assert BraveNewsSearch.is_visible() is True
    assert BraveImageSearch.is_visible() is True
    assert BraveVideoSearch.is_visible() is True


@pytest.mark.asyncio
async def test_brave_web_search_requires_query(mock_context):
    """Test that BraveWebSearch raises error when query is empty."""
    node = BraveWebSearch(query="")
    with pytest.raises(ValueError, match="Query is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_brave_web_search_success(mock_context):
    """Test successful web search."""
    mock_response = {
        "web": {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "description": "A test description",
                    "page_age": "2024-01-01",
                    "language": "en",
                    "family_friendly": True,
                },
                {
                    "title": "Another Result",
                    "url": "https://example2.com",
                    "description": "Another description",
                    "page_age": "2024-01-02",
                    "language": "en",
                    "family_friendly": True,
                },
            ]
        }
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveWebSearch(query="test query", count=10)
    result = await node.process(mock_context)

    assert len(result) == 2
    assert isinstance(result[0], BraveWebResult)
    assert result[0].title == "Test Result"
    assert result[0].url == "https://example.com"
    assert result[0].description == "A test description"


@pytest.mark.asyncio
async def test_brave_web_search_empty_results(mock_context):
    """Test web search with no results."""
    mock_context.http_get = AsyncMock(
        return_value=DummyResponse({"web": {"results": []}})
    )

    node = BraveWebSearch(query="nonexistent query")
    result = await node.process(mock_context)

    assert len(result) == 0


@pytest.mark.asyncio
async def test_brave_web_search_api_key_required(mock_context):
    """Test that API key is required."""
    mock_context.get_environment_secret = AsyncMock(return_value=None)

    node = BraveWebSearch(query="test")
    with pytest.raises(ValueError, match="BRAVE_API_KEY is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_brave_news_search_requires_query(mock_context):
    """Test that BraveNewsSearch raises error when query is empty."""
    node = BraveNewsSearch(query="")
    with pytest.raises(ValueError, match="Query is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_brave_news_search_success(mock_context):
    """Test successful news search."""
    mock_response = {
        "results": [
            {
                "title": "News Article",
                "url": "https://news.example.com",
                "description": "A news description",
                "age": "2 hours ago",
                "meta_url": {"netloc": "news.example.com"},
                "thumbnail": {"src": "https://example.com/thumb.jpg"},
            }
        ]
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveNewsSearch(query="breaking news")
    result = await node.process(mock_context)

    assert len(result) == 1
    assert isinstance(result[0], BraveNewsResult)
    assert result[0].title == "News Article"
    assert result[0].source == "news.example.com"
    assert result[0].thumbnail_url == "https://example.com/thumb.jpg"


@pytest.mark.asyncio
async def test_brave_image_search_requires_query(mock_context):
    """Test that BraveImageSearch raises error when query is empty."""
    node = BraveImageSearch(query="")
    with pytest.raises(ValueError, match="Query is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_brave_image_search_success(mock_context):
    """Test successful image search."""
    mock_response = {
        "results": [
            {
                "title": "Image Title",
                "url": "https://example.com/image.jpg",
                "source": "https://example.com/page",
                "thumbnail": {"src": "https://example.com/thumb.jpg"},
                "properties": {"width": 1920, "height": 1080},
            }
        ]
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveImageSearch(query="cats")
    result = await node.process(mock_context)

    assert "results" in result
    assert "images" in result
    assert len(result["results"]) == 1
    assert len(result["images"]) == 1
    assert isinstance(result["results"][0], BraveImageResult)
    assert result["results"][0].title == "Image Title"
    assert result["results"][0].width == 1920
    assert result["results"][0].height == 1080
    assert result["images"][0].uri == "https://example.com/image.jpg"


@pytest.mark.asyncio
async def test_brave_video_search_requires_query(mock_context):
    """Test that BraveVideoSearch raises error when query is empty."""
    node = BraveVideoSearch(query="")
    with pytest.raises(ValueError, match="Query is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_brave_video_search_success(mock_context):
    """Test successful video search."""
    mock_response = {
        "results": [
            {
                "title": "Video Title",
                "url": "https://youtube.com/watch?v=123",
                "description": "A video description",
                "age": "1 day ago",
                "thumbnail": {"src": "https://example.com/video-thumb.jpg"},
                "creator": "VideoCreator",
                "meta_url": {"netloc": "youtube.com"},
                "video": {"duration": "10:30"},
            }
        ]
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveVideoSearch(query="tutorial")
    result = await node.process(mock_context)

    assert len(result) == 1
    assert isinstance(result[0], BraveVideoResult)
    assert result[0].title == "Video Title"
    assert result[0].creator == "VideoCreator"
    assert result[0].publisher == "youtube.com"
    assert result[0].duration == "10:30"


@pytest.mark.asyncio
async def test_brave_search_safesearch_levels():
    """Test SafeSearchLevel enum values."""
    assert SafeSearchLevel.OFF.value == "off"
    assert SafeSearchLevel.MODERATE.value == "moderate"
    assert SafeSearchLevel.STRICT.value == "strict"


@pytest.mark.asyncio
async def test_brave_web_search_with_parameters(mock_context):
    """Test that parameters are passed correctly to the API."""
    mock_context.http_get = AsyncMock(
        return_value=DummyResponse({"web": {"results": []}})
    )

    node = BraveWebSearch(
        query="test",
        count=20,
        offset=10,
        country="us",
        search_lang="en",
        safesearch=SafeSearchLevel.STRICT,
        freshness="pd",
        text_decorations=True,
    )
    await node.process(mock_context)

    # Verify the http_get was called with correct parameters
    mock_context.http_get.assert_called_once()
    call_args = mock_context.http_get.call_args

    assert call_args[0][0] == "https://api.search.brave.com/res/v1/web/search"
    assert call_args[1]["params"]["q"] == "test"
    assert call_args[1]["params"]["count"] == 20
    assert call_args[1]["params"]["offset"] == 10
    assert call_args[1]["params"]["country"] == "us"
    assert call_args[1]["params"]["search_lang"] == "en"
    assert call_args[1]["params"]["safesearch"] == "strict"
    assert call_args[1]["params"]["freshness"] == "pd"
    assert call_args[1]["params"]["text_decorations"] is True


@pytest.mark.asyncio
async def test_brave_search_filters_empty_params(mock_context):
    """Test that empty parameters are filtered out."""
    mock_context.http_get = AsyncMock(
        return_value=DummyResponse({"web": {"results": []}})
    )

    node = BraveWebSearch(query="test", country="", search_lang="", freshness="")
    await node.process(mock_context)

    call_args = mock_context.http_get.call_args
    params = call_args[1]["params"]

    # Empty params should not be in the params dict
    assert "country" not in params or params.get("country") is None
    assert "search_lang" not in params or params.get("search_lang") is None
    assert "freshness" not in params or params.get("freshness") is None


@pytest.mark.asyncio
async def test_brave_search_headers(mock_context):
    """Test that correct headers are sent."""
    mock_context.http_get = AsyncMock(
        return_value=DummyResponse({"web": {"results": []}})
    )

    node = BraveWebSearch(query="test")
    await node.process(mock_context)

    call_args = mock_context.http_get.call_args
    headers = call_args[1]["headers"]

    assert headers["Accept"] == "application/json"
    assert headers["Accept-Encoding"] == "gzip"
    assert headers["X-Subscription-Token"] == "test-api-key"


@pytest.mark.asyncio
async def test_brave_news_search_missing_fields(mock_context):
    """Test news search handles missing fields gracefully."""
    mock_response = {
        "results": [
            {
                "title": "Minimal News",
                "url": "https://news.example.com",
                # Missing description, age, meta_url, thumbnail
            }
        ]
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveNewsSearch(query="news")
    result = await node.process(mock_context)

    assert len(result) == 1
    assert result[0].title == "Minimal News"
    assert result[0].description == ""
    assert result[0].age == ""
    assert result[0].source == ""
    assert result[0].thumbnail_url == ""


@pytest.mark.asyncio
async def test_brave_image_search_missing_properties(mock_context):
    """Test image search handles missing properties gracefully."""
    mock_response = {
        "results": [
            {
                "title": "Image Without Properties",
                "url": "https://example.com/image.jpg",
                # Missing source, thumbnail, properties
            }
        ]
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveImageSearch(query="images")
    result = await node.process(mock_context)

    assert len(result["results"]) == 1
    assert result["results"][0].width == 0
    assert result["results"][0].height == 0
    assert result["results"][0].thumbnail_url == ""


@pytest.mark.asyncio
async def test_brave_video_search_missing_video_data(mock_context):
    """Test video search handles missing video data gracefully."""
    mock_response = {
        "results": [
            {
                "title": "Video Without Duration",
                "url": "https://youtube.com/watch?v=123",
                # Missing video.duration, creator, etc.
            }
        ]
    }
    mock_context.http_get = AsyncMock(return_value=DummyResponse(mock_response))

    node = BraveVideoSearch(query="videos")
    result = await node.process(mock_context)

    assert len(result) == 1
    assert result[0].duration == ""
    assert result[0].creator == ""
    assert result[0].publisher == ""
