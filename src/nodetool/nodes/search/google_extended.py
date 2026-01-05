#!/usr/bin/env python

"""
Extended Google SERP nodes for Nodetool.
Provides additional Google search engines and sub-APIs beyond the base set.
"""

from pydantic import Field
from typing import Any, Dict, ClassVar

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


async def _call_serp_engine(
    context: ProcessingContext, 
    engine: str, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Helper function to call a SerpApi engine with the given parameters.
    
    Args:
        context: Processing context
        engine: SerpApi engine name (e.g., 'google_autocomplete')
        params: Additional parameters for the API call
        
    Returns:
        Dict with API response or raises ValueError on error
    """
    from nodetool.agents.serp_providers.serp_api_provider import SerpApiProvider
    
    serpapi_key = await context.get_secret("SERPAPI_API_KEY")
    if not serpapi_key:
        raise ValueError(
            "SERPAPI_API_KEY not found. Please configure your SerpApi credentials."
        )
    
    provider = SerpApiProvider(api_key=serpapi_key)
    async with provider:
        all_params = {"engine": engine, **params}
        result = await provider._make_request(all_params)
        
        if "error" in result and not isinstance(result.get("search_metadata"), dict):
            raise ValueError(result.get("error", "SerpApi request failed"))
            
        serpapi_error_status = result.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result.get("error", f"SerpApi returned an error: {result}"))
    
    return result


class GoogleAutocomplete(BaseNode):
    """
    Get autocomplete suggestions from Google Search.
    google, autocomplete, suggestions, search, query

    Use cases:
    - Generate query suggestions for search interfaces
    - Discover related search terms
    - Build autocomplete features
    - Research popular search queries
    - Expand keyword lists for SEO
    """

    query: str = Field(
        default="", description="Query string to get autocomplete suggestions for"
    )
    language: str = Field(
        default="en", description="Language code (e.g., 'en', 'es', 'fr')"
    )
    country: str = Field(
        default="us", description="Country code (e.g., 'us', 'uk', 'ca')"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "google_autocomplete",
            {
                "q": self.query,
                "hl": self.language,
                "gl": self.country,
            }
        )
        return result


class GoogleTrendsInterestOverTime(BaseNode):
    """
    Get interest over time data from Google Trends.
    google, trends, analytics, time-series, search, interest

    Use cases:
    - Track search interest trends over time
    - Analyze seasonal patterns
    - Compare multiple search terms
    - Monitor brand awareness changes
    - Research topic popularity evolution
    """

    query: str = Field(default="", description="Search term to analyze trends for")
    geo: str = Field(
        default="", description="Geographic location code (e.g., 'US', 'GB', 'world')"
    )
    date_range: str = Field(
        default="today 12-m",
        description="Date range (e.g., 'now 7-d', 'today 12-m', 'all')",
    )
    category: int = Field(default=0, description="Category code (0 for all categories)")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        params = {
            "q": self.query,
            "data_type": "TIMESERIES",
            "hl": "en",
        }
        if self.geo:
            params["geo"] = self.geo
        if self.date_range:
            params["date"] = self.date_range
        if self.category > 0:
            params["cat"] = self.category

        result = await _call_serp_engine(context, "google_trends", params)
        return result


class GoogleTrendsInterestByRegion(BaseNode):
    """
    Get interest by region data from Google Trends.
    google, trends, geography, regional, search, interest

    Use cases:
    - Identify geographic markets with highest interest
    - Plan regional marketing campaigns
    - Understand geographic demand patterns
    - Target content to specific regions
    - Research international expansion opportunities
    """

    query: str = Field(default="", description="Search term to analyze regional trends")
    geo: str = Field(
        default="", description="Geographic scope (e.g., 'US', 'GB', 'world')"
    )
    date_range: str = Field(
        default="today 12-m", description="Date range for analysis"
    )
    category: int = Field(default=0, description="Category code (0 for all categories)")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        params = {
            "q": self.query,
            "data_type": "GEO_MAP",
            "hl": "en",
        }
        if self.geo:
            params["geo"] = self.geo
        if self.date_range:
            params["date"] = self.date_range
        if self.category > 0:
            params["cat"] = self.category

        result = await _call_serp_engine(context, "google_trends", params)
        return result


class GoogleTrendsRelatedQueries(BaseNode):
    """
    Get related queries from Google Trends.
    google, trends, related, queries, search, suggestions

    Use cases:
    - Discover related search terms
    - Expand keyword research
    - Find content ideas
    - Understand user search behavior
    - Identify trending variations
    """

    query: str = Field(default="", description="Search term to find related queries")
    geo: str = Field(default="", description="Geographic location code")
    date_range: str = Field(default="today 12-m", description="Date range")
    category: int = Field(default=0, description="Category code")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        params = {
            "q": self.query,
            "data_type": "RELATED_QUERIES",
            "hl": "en",
        }
        if self.geo:
            params["geo"] = self.geo
        if self.date_range:
            params["date"] = self.date_range
        if self.category > 0:
            params["cat"] = self.category

        result = await _call_serp_engine(context, "google_trends", params)
        return result


class GoogleTrendsRelatedTopics(BaseNode):
    """
    Get related topics from Google Trends.
    google, trends, topics, related, search, analysis

    Use cases:
    - Discover related topic areas
    - Find content themes
    - Understand topic relationships
    - Research adjacent markets
    - Identify expansion opportunities
    """

    query: str = Field(default="", description="Search term to find related topics")
    geo: str = Field(default="", description="Geographic location code")
    date_range: str = Field(default="today 12-m", description="Date range")
    category: int = Field(default=0, description="Category code")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        params = {
            "q": self.query,
            "data_type": "RELATED_TOPICS",
            "hl": "en",
        }
        if self.geo:
            params["geo"] = self.geo
        if self.date_range:
            params["date"] = self.date_range
        if self.category > 0:
            params["cat"] = self.category

        result = await _call_serp_engine(context, "google_trends", params)
        return result


class GoogleTrendsTrendingNow(BaseNode):
    """
    Get currently trending searches from Google Trends.
    google, trends, trending, realtime, hot, searches

    Use cases:
    - Monitor breaking trends
    - Discover viral topics
    - Create timely content
    - Track real-time search interest
    - Identify emerging stories
    """

    geo: str = Field(
        default="US", description="Geographic location code (e.g., 'US', 'GB')"
    )
    frequency: str = Field(
        default="daily",
        description="Frequency: 'daily' for last 24 hours, 'realtime' for recent hours",
    )
    hours: int = Field(
        default=4, description="Time window in hours for realtime trends (1-48)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        params = {
            "frequency": self.frequency,
            "hl": "en",
        }
        if self.geo:
            params["geo"] = self.geo
        if self.frequency == "realtime" and self.hours:
            params["hours"] = self.hours

        result = await _call_serp_engine(context, "google_trends_trending_now", params)
        return result


class GoogleVideos(BaseNode):
    """
    Search Google Videos for video content.
    google, videos, search, video, content, media

    Use cases:
    - Find videos on specific topics
    - Discover video content for research
    - Filter videos by duration and upload date
    - Gather video URLs for analysis
    - Monitor video content trends
    """

    query: str = Field(default="", description="Video search query")
    num_results: int = Field(
        default=10, description="Maximum number of video results to return"
    )
    duration: str = Field(
        default="",
        description="Duration filter: 's' (short <4min), 'm' (medium 4-20min), 'l' (long >20min)",
    )
    upload_date: str = Field(
        default="",
        description="Upload date filter: 'h' (hour), 'd' (day), 'w' (week), 'm' (month), 'y' (year)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        params = {
            "q": self.query,
            "num": self.num_results,
            "hl": "en",
            "gl": "us",
        }
        
        tbs_parts = []
        if self.duration:
            tbs_parts.append(f"dur:{self.duration}")
        if self.upload_date:
            tbs_parts.append(f"qdr:{self.upload_date}")
        if tbs_parts:
            params["tbs"] = ",".join(tbs_parts)

        result = await _call_serp_engine(context, "google_videos", params)
        return result


class GoogleFlights(BaseNode):
    """
    Search Google Flights for flight options and pricing.
    google, flights, travel, aviation, booking, prices

    Use cases:
    - Find flight options between airports
    - Compare flight prices
    - Monitor fare changes
    - Build travel planning tools
    - Research route availability
    """

    departure: str = Field(
        default="", description="Departure airport code (e.g., 'JFK', 'LAX')"
    )
    arrival: str = Field(
        default="", description="Arrival airport code (e.g., 'LHR', 'CDG')"
    )
    outbound_date: str = Field(
        default="", description="Outbound date in YYYY-MM-DD format"
    )
    return_date: str = Field(
        default="", description="Return date in YYYY-MM-DD format (optional for one-way)"
    )
    adults: int = Field(default=1, description="Number of adult passengers")
    currency: str = Field(default="USD", description="Currency code (e.g., 'USD', 'EUR')")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.departure or not self.arrival or not self.outbound_date:
            raise ValueError("Departure, arrival, and outbound_date are required")

        params = {
            "departure_id": self.departure,
            "arrival_id": self.arrival,
            "outbound_date": self.outbound_date,
            "adults": self.adults,
            "hl": "en",
            "gl": "us",
        }
        if self.return_date:
            params["return_date"] = self.return_date
        if self.currency:
            params["currency"] = self.currency

        result = await _call_serp_engine(context, "google_flights", params)
        return result


class GoogleFlightsAutocomplete(BaseNode):
    """
    Get airport autocomplete suggestions from Google Flights.
    google, flights, autocomplete, airports, travel

    Use cases:
    - Build flight search interfaces
    - Find airport codes by city name
    - Discover nearby airports
    - Validate airport selections
    - Provide airport suggestions
    """

    query: str = Field(
        default="", description="Airport or city name to search for"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "google_flights_autocomplete",
            {"q": self.query, "hl": "en"}
        )
        return result


class GoogleHotels(BaseNode):
    """
    Search Google Hotels for accommodation options.
    google, hotels, travel, booking, accommodation, lodging

    Use cases:
    - Find hotels in a location
    - Compare hotel prices
    - Check availability for dates
    - Research accommodation options
    - Build travel booking tools
    """

    query: str = Field(
        default="", description="Hotel name or location to search"
    )
    check_in: str = Field(
        default="", description="Check-in date in YYYY-MM-DD format"
    )
    check_out: str = Field(
        default="", description="Check-out date in YYYY-MM-DD format"
    )
    adults: int = Field(default=2, description="Number of adult guests")
    children: int = Field(default=0, description="Number of children")
    currency: str = Field(default="USD", description="Currency code")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query or not self.check_in or not self.check_out:
            raise ValueError("Query, check_in, and check_out are required")

        params = {
            "q": self.query,
            "check_in_date": self.check_in,
            "check_out_date": self.check_out,
            "adults": self.adults,
            "children": self.children,
            "hl": "en",
            "gl": "us",
        }
        if self.currency:
            params["currency"] = self.currency

        result = await _call_serp_engine(context, "google_hotels", params)
        return result


class GoogleHotelsAutocomplete(BaseNode):
    """
    Get hotel location autocomplete suggestions from Google Hotels.
    google, hotels, autocomplete, locations, travel

    Use cases:
    - Build hotel search interfaces
    - Find hotels by location name
    - Provide location suggestions
    - Validate hotel searches
    - Discover nearby hotels
    """

    query: str = Field(
        default="", description="Hotel or location name to search for"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "google_hotels_autocomplete",
            {"q": self.query, "hl": "en"}
        )
        return result


class GoogleMapsAutocomplete(BaseNode):
    """
    Get place autocomplete suggestions from Google Maps.
    google, maps, autocomplete, places, locations

    Use cases:
    - Build location search interfaces
    - Find places by partial name
    - Provide place suggestions
    - Validate location inputs
    - Discover nearby places
    """

    query: str = Field(
        default="", description="Place or location name to search for"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "google_maps_autocomplete",
            {"q": self.query, "hl": "en", "gl": "us"}
        )
        return result


class GoogleMapsDirections(BaseNode):
    """
    Get directions between two locations from Google Maps.
    google, maps, directions, routing, navigation, travel

    Use cases:
    - Calculate routes between locations
    - Get turn-by-turn directions
    - Estimate travel time and distance
    - Compare different travel modes
    - Plan trips and logistics
    """

    origin: str = Field(
        default="", description="Starting location (address or place name)"
    )
    destination: str = Field(
        default="", description="Destination location (address or place name)"
    )
    mode: str = Field(
        default="driving",
        description="Travel mode: 'driving', 'walking', 'bicycling', 'transit'",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.origin or not self.destination:
            raise ValueError("Origin and destination are required")

        result = await _call_serp_engine(
            context,
            "google_maps_directions",
            {
                "origin": self.origin,
                "destination": self.destination,
                "mode": self.mode,
                "hl": "en",
                "gl": "us",
            }
        )
        return result


class GoogleMapsPlacePhotos(BaseNode):
    """
    Get photos for a specific place from Google Maps.
    google, maps, photos, images, places, locations

    Use cases:
    - Retrieve place imagery
    - Build visual place guides
    - Verify business locations
    - Create location catalogs
    - Enhance place listings
    """

    place_id: str = Field(
        default="", description="Google Maps place data_id"
    )
    num_results: int = Field(
        default=10, description="Maximum number of photos to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.place_id:
            raise ValueError("Place ID (data_id) is required")

        result = await _call_serp_engine(
            context,
            "google_maps_photos",
            {"data_id": self.place_id, "num": self.num_results, "hl": "en"}
        )
        return result


class GoogleMapsPlaceReviews(BaseNode):
    """
    Get reviews for a specific place from Google Maps.
    google, maps, reviews, ratings, places, feedback

    Use cases:
    - Analyze customer feedback
    - Monitor business reputation
    - Research place quality
    - Gather user opinions
    - Track review sentiment
    """

    place_id: str = Field(
        default="", description="Google Maps place data_id"
    )
    num_results: int = Field(
        default=10, description="Maximum number of reviews to return"
    )
    sort_by: str = Field(
        default="newestFirst",
        description="Sort order: 'newestFirst', 'mostRelevant', 'highestRating', 'lowestRating'",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.place_id:
            raise ValueError("Place ID (data_id) is required")

        result = await _call_serp_engine(
            context,
            "google_maps_reviews",
            {
                "data_id": self.place_id,
                "num": self.num_results,
                "hl": "en",
                "sort_by": self.sort_by,
            }
        )
        return result


class GoogleFinanceMarkets(BaseNode):
    """
    Get financial market overview data from Google Finance.
    google, finance, markets, stocks, indices, crypto

    Use cases:
    - Monitor market indices
    - Track currency rates
    - Watch cryptocurrency prices
    - Get market snapshots
    - Research financial trends
    """

    trend: str = Field(
        default="indexes",
        description="Market category: 'indexes', 'most-active', 'gainers', 'losers', 'climate-leaders', 'crypto', 'currencies'",
    )
    language: str = Field(
        default="en", description="Language code"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        result = await _call_serp_engine(
            context,
            "google_finance_markets",
            {"trend": self.trend, "hl": self.language, "gl": "us"}
        )
        return result


class GooglePatents(BaseNode):
    """
    Search Google Patents for patent documents.
    google, patents, intellectual, property, search, innovation

    Use cases:
    - Search for patents by topic
    - Research prior art
    - Monitor patent filings
    - Analyze patent landscapes
    - Track innovation trends
    """

    query: str = Field(
        default="", description="Patent search query"
    )
    num_results: int = Field(
        default=10, description="Maximum number of patent results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "google_patents",
            {"q": self.query, "num": self.num_results, "hl": "en"}
        )
        return result


class GooglePlay(BaseNode):
    """
    Search Google Play Store for apps, books, or movies.
    google, play, apps, store, android, search

    Use cases:
    - Find apps by keyword
    - Research app market
    - Discover popular apps
    - Monitor app releases
    - Compare app features
    """

    query: str = Field(
        default="", description="Search query"
    )
    store: str = Field(
        default="apps", description="Store type: 'apps', 'books', 'movies'"
    )
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "google_play",
            {
                "q": self.query,
                "store": self.store,
                "num": self.num_results,
                "hl": "en",
                "gl": "us",
            }
        )
        return result


class GooglePlayProduct(BaseNode):
    """
    Get details for a specific Google Play product (app, book, movie).
    google, play, product, details, app, information

    Use cases:
    - Get app details and metadata
    - Monitor app updates
    - Research app features
    - Compare app specifications
    - Track app ratings
    """

    product_id: str = Field(
        default="", description="Product ID (e.g., app package name like 'com.example.app')"
    )
    store: str = Field(
        default="apps", description="Store type: 'apps', 'books', 'movies'"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.product_id:
            raise ValueError("Product ID is required")

        result = await _call_serp_engine(
            context,
            "google_play_product",
            {"product_id": self.product_id, "store": self.store, "hl": "en", "gl": "us"}
        )
        return result
