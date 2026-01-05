"""
Extended SerpApi provider with additional engines and sub-APIs.

This module extends the base SerpApiProvider from nodetool-core with additional
search engines and specialized API endpoints that are not part of the core interface.
"""

from typing import Any, Optional
from nodetool.agents.serp_providers.serp_api_provider import SerpApiProvider
from nodetool.agents.serp_providers.serp_providers import ErrorResponse
from nodetool.agents.tools._remove_base64_images import _remove_base64_images


class ExtendedSerpApiProvider(SerpApiProvider):
    """
    Extended SerpApi provider with support for additional engines not in the base interface.
    
    This provider adds methods for:
    - Google Autocomplete, Trends, Videos
    - Google Flights, Hotels
    - Google Maps extended (directions, autocomplete, photos, reviews)
    - Google Finance Markets, Patents, Play Store
    - Alternative engines: Bing, DuckDuckGo, Yahoo
    - E-commerce: YouTube, Amazon, Walmart, eBay
    """

    # ========== Google Autocomplete ==========
    async def search_autocomplete(
        self, q: str, hl: Optional[str] = None, gl: Optional[str] = None
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Autocomplete API - get search suggestions.
        
        Args:
            q: Query string to get autocomplete suggestions for
            hl: Language (defaults to provider's hl)
            gl: Country (defaults to provider's gl)
            
        Returns:
            Dict with suggestions list or error response
        """
        params = {
            "engine": "google_autocomplete",
            "q": q,
            "hl": hl or self.hl,
            "gl": gl or self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
        
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Trends ==========
    async def search_trends(
        self,
        q: str,
        data_type: str = "TIMESERIES",
        geo: Optional[str] = None,
        date: Optional[str] = None,
        category: Optional[int] = None,
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Trends API.
        
        Args:
            q: Search term
            data_type: Type of data - TIMESERIES, GEO_MAP, RELATED_QUERIES, RELATED_TOPICS
            geo: Geographic location code (e.g., 'US', 'GB')
            date: Date range (e.g., 'now 7-d', 'today 12-m')
            category: Category code
            
        Returns:
            Trends data or error response
        """
        params = {
            "engine": "google_trends",
            "q": q,
            "data_type": data_type,
            "hl": self.hl,
        }
        if geo:
            params["geo"] = geo
        if date:
            params["date"] = date
        if category is not None:
            params["cat"] = category

        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_trends_trending_now(
        self,
        geo: Optional[str] = None,
        hours: Optional[int] = None,
        frequency: str = "daily",
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Trends Trending Now API (daily or realtime).
        
        Args:
            geo: Geographic location code
            hours: Time window in hours for realtime trends
            frequency: 'daily' or 'realtime'
            
        Returns:
            Trending searches or error response
        """
        params = {
            "engine": "google_trends_trending_now",
            "frequency": frequency,
            "hl": self.hl,
        }
        if geo:
            params["geo"] = geo
        if hours is not None:
            params["hours"] = hours

        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Videos ==========
    async def search_videos(
        self, q: str, num_results: int = 10, filters: Optional[dict] = None
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Videos API.
        
        Args:
            q: Search query
            num_results: Number of results to return
            filters: Dict with optional 'duration' and 'upload_date' filters
            
        Returns:
            Video results or error response
        """
        params = {
            "engine": "google_videos",
            "q": q,
            "num": num_results,
            "hl": self.hl,
            "gl": self.gl,
        }
        if filters:
            tbs_parts = []
            if "duration" in filters:
                # short (s), medium (m), long (l)
                tbs_parts.append(f"dur:{filters['duration']}")
            if "upload_date" in filters:
                # h (hour), d (day), w (week), m (month), y (year)
                tbs_parts.append(f"qdr:{filters['upload_date']}")
            if tbs_parts:
                params["tbs"] = ",".join(tbs_parts)

        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Flights ==========
    async def search_flights(
        self,
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        currency: Optional[str] = None,
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Flights Results API.
        
        Args:
            departure_id: Airport code for departure (e.g., 'JFK')
            arrival_id: Airport code for arrival (e.g., 'LAX')
            outbound_date: Outbound date (YYYY-MM-DD)
            return_date: Optional return date (YYYY-MM-DD)
            adults: Number of adult passengers
            currency: Currency code (e.g., 'USD')
            
        Returns:
            Flight results or error response
        """
        params = {
            "engine": "google_flights",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "adults": adults,
            "hl": self.hl,
            "gl": self.gl,
        }
        if return_date:
            params["return_date"] = return_date
        if currency:
            params["currency"] = currency

        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_flights_autocomplete(
        self, q: str
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Flights Autocomplete API.
        
        Args:
            q: Airport or city name query
            
        Returns:
            Autocomplete suggestions or error response
        """
        params = {
            "engine": "google_flights_autocomplete",
            "q": q,
            "hl": self.hl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Hotels ==========
    async def search_hotels_autocomplete(
        self, q: str
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Hotels Autocomplete API.
        
        Args:
            q: Hotel or location name query
            
        Returns:
            Autocomplete suggestions or error response
        """
        params = {
            "engine": "google_hotels_autocomplete",
            "q": q,
            "hl": self.hl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_hotels(
        self,
        q: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        children: int = 0,
        currency: Optional[str] = None,
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Hotels Search API.
        
        Args:
            q: Hotel or location query
            check_in_date: Check-in date (YYYY-MM-DD)
            check_out_date: Check-out date (YYYY-MM-DD)
            adults: Number of adults
            children: Number of children
            currency: Currency code
            
        Returns:
            Hotel results or error response
        """
        params = {
            "engine": "google_hotels",
            "q": q,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "children": children,
            "hl": self.hl,
            "gl": self.gl,
        }
        if currency:
            params["currency"] = currency

        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Maps Extended ==========
    async def search_maps_autocomplete(
        self, q: str
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Maps Autocomplete API.
        
        Args:
            q: Place or location query
            
        Returns:
            Autocomplete suggestions or error response
        """
        params = {
            "engine": "google_maps_autocomplete",
            "q": q,
            "hl": self.hl,
            "gl": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_maps_directions(
        self, origin: str, destination: str, mode: str = "driving"
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Maps Directions API.
        
        Args:
            origin: Starting point
            destination: End point
            mode: Travel mode - driving, walking, bicycling, transit
            
        Returns:
            Directions data or error response
        """
        params = {
            "engine": "google_maps_directions",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "hl": self.hl,
            "gl": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_maps_photos(
        self, data_id: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Maps Photos API.
        
        Args:
            data_id: Place data ID
            num_results: Number of photos to return
            
        Returns:
            Photos data or error response
        """
        params = {
            "engine": "google_maps_photos",
            "data_id": data_id,
            "num": num_results,
            "hl": self.hl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_maps_reviews(
        self, data_id: str, num_results: int = 10, sort_by: str = "newestFirst"
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Maps Reviews API.
        
        Args:
            data_id: Place data ID
            num_results: Number of reviews to return
            sort_by: Sort order - newestFirst, mostRelevant, highestRating, lowestRating
            
        Returns:
            Reviews data or error response
        """
        params = {
            "engine": "google_maps_reviews",
            "data_id": data_id,
            "num": num_results,
            "hl": self.hl,
            "sort_by": sort_by,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Finance Markets ==========
    async def search_finance_markets(
        self, trend: str = "indexes", hl: Optional[str] = None
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Finance Markets API.
        
        Args:
            trend: Market trend - indexes, most-active, gainers, losers, climate-leaders, crypto, currencies
            hl: Language (defaults to provider's hl)
            
        Returns:
            Market data or error response
        """
        params = {
            "engine": "google_finance_markets",
            "trend": trend,
            "hl": hl or self.hl,
            "gl": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Patents ==========
    async def search_patents(
        self, q: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Patents Search API.
        
        Args:
            q: Patent search query
            num_results: Number of results to return
            
        Returns:
            Patent results or error response
        """
        params = {
            "engine": "google_patents",
            "q": q,
            "num": num_results,
            "hl": self.hl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Google Play ==========
    async def search_play(
        self, q: str, store: str = "apps", num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Play Search API.
        
        Args:
            q: Search query
            store: Store type - apps, books, movies
            num_results: Number of results to return
            
        Returns:
            Play store results or error response
        """
        params = {
            "engine": "google_play",
            "q": q,
            "store": store,
            "num": num_results,
            "hl": self.hl,
            "gl": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def get_play_product(
        self, product_id: str, store: str = "apps"
    ) -> dict[str, Any] | ErrorResponse:
        """
        Google Play Product Details API.
        
        Args:
            product_id: Product ID (e.g., app package name)
            store: Store type - apps, books, movies
            
        Returns:
            Product details or error response
        """
        params = {
            "engine": "google_play_product",
            "product_id": product_id,
            "store": store,
            "hl": self.hl,
            "gl": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== Alternative Search Engines ==========
    async def search_bing(
        self, q: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Bing Search API.
        
        Args:
            q: Search query
            num_results: Number of results to return
            
        Returns:
            Bing search results or error response
        """
        params = {
            "engine": "bing",
            "q": q,
            "count": num_results,
            "cc": self.gl,
            "mkt": f"{self.hl}-{self.gl}",
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_bing_images(
        self, q: str, num_results: int = 20
    ) -> dict[str, Any] | ErrorResponse:
        """
        Bing Images API.
        
        Args:
            q: Search query
            num_results: Number of results to return
            
        Returns:
            Bing image results or error response
        """
        params = {
            "engine": "bing_images",
            "q": q,
            "count": num_results,
            "cc": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_duckduckgo(
        self, q: str
    ) -> dict[str, Any] | ErrorResponse:
        """
        DuckDuckGo Search API.
        
        Args:
            q: Search query
            
        Returns:
            DuckDuckGo search results or error response
        """
        params = {
            "engine": "duckduckgo",
            "q": q,
            "kl": f"{self.gl}-{self.hl}",
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_yahoo(
        self, q: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Yahoo Search API.
        
        Args:
            q: Search query
            num_results: Number of results to return
            
        Returns:
            Yahoo search results or error response
        """
        params = {
            "engine": "yahoo",
            "p": q,  # Yahoo uses 'p' instead of 'q'
            "b": 1,  # Starting position
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    # ========== E-commerce Engines ==========
    async def search_youtube(
        self, search_query: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        YouTube Search API.
        
        Args:
            search_query: Search query
            num_results: Number of results to return
            
        Returns:
            YouTube search results or error response
        """
        params = {
            "engine": "youtube",
            "search_query": search_query,
            "hl": self.hl,
            "gl": self.gl,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_amazon(
        self, query: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Amazon Search API.
        
        Args:
            query: Product search query
            num_results: Number of results to return
            
        Returns:
            Amazon search results or error response
        """
        params = {
            "engine": "amazon",
            "query": query,
            "amazon_domain": f"amazon.{self.gl if self.gl == 'com' else self.gl}",
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_walmart(
        self, query: str
    ) -> dict[str, Any] | ErrorResponse:
        """
        Walmart Search API.
        
        Args:
            query: Product search query
            
        Returns:
            Walmart search results or error response
        """
        params = {
            "engine": "walmart",
            "query": query,
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)

    async def search_ebay(
        self, query: str
    ) -> dict[str, Any] | ErrorResponse:
        """
        eBay Search API.
        
        Args:
            query: Product search query
            
        Returns:
            eBay search results or error response
        """
        params = {
            "engine": "ebay",
            "_nkw": query,  # eBay uses _nkw parameter
            "ebay_domain": f"ebay.{self.gl if self.gl == 'com' else self.gl}",
        }
        result_data = await self._make_request(params)
        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data
            
        serpapi_error_status = result_data.get("search_metadata", {}).get("status") == "Error"
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(result_data.get("error", f"SerpApi returned an error: {result_data}"))
            
        return _remove_base64_images(result_data)
