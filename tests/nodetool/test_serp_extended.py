"""
Tests for extended SERP nodes.

Tests the new Google extended and alternative search engine nodes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nodetool.workflows.processing_context import ProcessingContext

# Import the nodes to test
from nodetool.nodes.search.google_extended import (
    GoogleAutocomplete,
    GoogleTrendsInterestOverTime,
    GoogleVideos,
    GoogleFlights,
    GoogleHotels,
    GoogleMapsDirections,
    GoogleFinanceMarkets,
    GooglePatents,
    GooglePlay,
)
from nodetool.nodes.search.alternative_engines import (
    BingSearch,
    DuckDuckGoSearch,
    YouTubeSearch,
    AmazonSearch,
)


@pytest.fixture
def mock_context():
    """Create a mock processing context."""
    ctx = MagicMock(spec=ProcessingContext)
    ctx.get_secret = AsyncMock(return_value="test_api_key")
    return ctx


@pytest.fixture
def mock_provider_response():
    """Create a mock SerpApi provider response."""
    return {
        "search_metadata": {
            "id": "test123",
            "status": "Success",
            "created_at": "2024-01-01",
            "processed_at": "2024-01-01",
        },
        "suggestions": [
            {"value": "test suggestion 1"},
            {"value": "test suggestion 2"},
        ],
    }


# ========== Google Extended Nodes Tests ==========


@pytest.mark.asyncio
async def test_google_autocomplete(mock_context, mock_provider_response):
    """Test GoogleAutocomplete node."""
    node = GoogleAutocomplete(query="test query", language="en", country="us")
    
    # Patch _call_serp_engine directly
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        assert "search_metadata" in result
        mock_call.assert_called_once()
        call_args, call_kwargs = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_autocomplete"
        assert params["q"] == "test query"


@pytest.mark.asyncio
async def test_google_autocomplete_missing_query(mock_context):
    """Test GoogleAutocomplete fails with missing query."""
    node = GoogleAutocomplete(query="", language="en", country="us")
    
    with pytest.raises(ValueError, match="Query is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_google_trends_interest_over_time(mock_context, mock_provider_response):
    """Test GoogleTrendsInterestOverTime node."""
    node = GoogleTrendsInterestOverTime(
        query="python", geo="US", date_range="today 12-m", category=0
    )
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_trends"
        assert params["q"] == "python"
        assert params["data_type"] == "TIMESERIES"


@pytest.mark.asyncio
async def test_google_videos(mock_context, mock_provider_response):
    """Test GoogleVideos node."""
    node = GoogleVideos(
        query="ai tutorial", num_results=10, duration="m", upload_date="w"
    )
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_videos"
        assert params["q"] == "ai tutorial"
        assert "tbs" in params
        assert "dur:m" in params["tbs"]
        assert "qdr:w" in params["tbs"]


@pytest.mark.asyncio
async def test_google_flights(mock_context, mock_provider_response):
    """Test GoogleFlights node."""
    node = GoogleFlights(
        departure="JFK",
        arrival="LAX",
        outbound_date="2024-06-01",
        return_date="2024-06-08",
        adults=2,
        currency="USD",
    )
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_flights"
        assert params["departure_id"] == "JFK"
        assert params["arrival_id"] == "LAX"


@pytest.mark.asyncio
async def test_google_hotels(mock_context, mock_provider_response):
    """Test GoogleHotels node."""
    node = GoogleHotels(
        query="hotels in Paris",
        check_in="2024-07-01",
        check_out="2024-07-05",
        adults=2,
        children=1,
    )
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_hotels"
        assert params["q"] == "hotels in Paris"


@pytest.mark.asyncio
async def test_google_maps_directions(mock_context, mock_provider_response):
    """Test GoogleMapsDirections node."""
    node = GoogleMapsDirections(
        origin="New York, NY",
        destination="Boston, MA",
        mode="driving",
    )
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_maps_directions"
        assert params["origin"] == "New York, NY"
        assert params["destination"] == "Boston, MA"


@pytest.mark.asyncio
async def test_google_finance_markets(mock_context, mock_provider_response):
    """Test GoogleFinanceMarkets node."""
    node = GoogleFinanceMarkets(trend="indexes", language="en")
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_finance_markets"
        assert params["trend"] == "indexes"


@pytest.mark.asyncio
async def test_google_patents(mock_context, mock_provider_response):
    """Test GooglePatents node."""
    node = GooglePatents(query="machine learning", num_results=10)
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_patents"
        assert params["q"] == "machine learning"


@pytest.mark.asyncio
async def test_google_play(mock_context, mock_provider_response):
    """Test GooglePlay node."""
    node = GooglePlay(query="productivity app", store="apps", num_results=10)
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "google_play"
        assert params["q"] == "productivity app"


# ========== Alternative Search Engines Tests ==========


@pytest.mark.asyncio
async def test_bing_search(mock_context, mock_provider_response):
    """Test BingSearch node."""
    node = BingSearch(query="test query", num_results=10)
    
    with patch("nodetool.nodes.search.alternative_engines._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "bing"
        assert params["q"] == "test query"


@pytest.mark.asyncio
async def test_duckduckgo_search(mock_context, mock_provider_response):
    """Test DuckDuckGoSearch node."""
    node = DuckDuckGoSearch(query="privacy search")
    
    with patch("nodetool.nodes.search.alternative_engines._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "duckduckgo"
        assert params["q"] == "privacy search"


@pytest.mark.asyncio
async def test_youtube_search(mock_context, mock_provider_response):
    """Test YouTubeSearch node."""
    node = YouTubeSearch(query="python tutorial", num_results=10)
    
    with patch("nodetool.nodes.search.alternative_engines._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "youtube"
        assert params["search_query"] == "python tutorial"


@pytest.mark.asyncio
async def test_amazon_search(mock_context, mock_provider_response):
    """Test AmazonSearch node."""
    node = AmazonSearch(query="laptop", num_results=10)
    
    with patch("nodetool.nodes.search.alternative_engines._call_serp_engine") as mock_call:
        mock_call.return_value = mock_provider_response
        
        result = await node.process(mock_context)
        
        assert result is not None
        call_args, _ = mock_call.call_args
        engine = call_args[1]
        params = call_args[2]
        assert engine == "amazon"
        assert params["query"] == "laptop"


# ========== Error Handling Tests ==========


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import issue in nodetool-core when testing missing API key")
async def test_missing_api_key(mock_context):
    """Test that nodes fail gracefully when API key is missing."""
    mock_context.get_secret = AsyncMock(return_value=None)
    node = GoogleAutocomplete(query="test")
    
    with pytest.raises(ValueError, match="SERPAPI_API_KEY not found"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_serpapi_error_response(mock_context):
    """Test handling of SerpApi error responses."""
    node = GoogleAutocomplete(query="test")
    
    with patch("nodetool.nodes.search.google_extended._call_serp_engine") as mock_call:
        mock_call.side_effect = ValueError("Invalid API key")
        
        with pytest.raises(ValueError, match="Invalid API key"):
            await node.process(mock_context)
