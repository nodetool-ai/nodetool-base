#!/usr/bin/env python

"""
Alternative search engine nodes for Nodetool.
Provides nodes for Bing, DuckDuckGo, Yahoo, and e-commerce search engines.
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
        engine: SerpApi engine name
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


# ========== Alternative Search Engines ==========


class BingSearch(BaseNode):
    """
    Search Bing for web results.
    bing, search, microsoft, web, query

    Use cases:
    - Get Bing search results
    - Compare with Google results
    - Access Microsoft's search index
    - Research Bing-specific rankings
    - Build multi-engine search tools
    """

    query: str = Field(default="", description="Search query")
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "bing",
            {"q": self.query, "count": self.num_results, "cc": "us", "mkt": "en-us"}
        )
        return result


class BingImages(BaseNode):
    """
    Search Bing Images for image results.
    bing, images, search, visual, pictures

    Use cases:
    - Find images on Bing
    - Compare with Google Images
    - Access Bing's image index
    - Gather visual content
    - Build image search tools
    """

    query: str = Field(default="", description="Image search query")
    num_results: int = Field(
        default=20, description="Maximum number of image results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "bing_images",
            {"q": self.query, "count": self.num_results, "cc": "us"}
        )
        return result


class DuckDuckGoSearch(BaseNode):
    """
    Search DuckDuckGo for privacy-focused web results.
    duckduckgo, search, privacy, web, query

    Use cases:
    - Get privacy-focused search results
    - Access DuckDuckGo's index
    - Compare with other search engines
    - Research neutral rankings
    - Build privacy-respecting tools
    """

    query: str = Field(default="", description="Search query")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "duckduckgo",
            {"q": self.query, "kl": "us-en"}
        )
        return result


class YahooSearch(BaseNode):
    """
    Search Yahoo for web results.
    yahoo, search, web, query

    Use cases:
    - Get Yahoo search results
    - Compare with other engines
    - Access Yahoo's search index
    - Research Yahoo-specific rankings
    - Build multi-engine search
    """

    query: str = Field(default="", description="Search query")
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "yahoo",
            {"p": self.query, "b": 1}
        )
        return result


# ========== E-commerce Search Engines ==========


class YouTubeSearch(BaseNode):
    """
    Search YouTube for video content.
    youtube, video, search, content, media

    Use cases:
    - Find YouTube videos
    - Discover content creators
    - Research video topics
    - Monitor video trends
    - Build video aggregators
    """

    query: str = Field(default="", description="Video search query")
    num_results: int = Field(
        default=10, description="Maximum number of video results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "youtube",
            {"search_query": self.query, "hl": "en", "gl": "us"}
        )
        return result


class AmazonSearch(BaseNode):
    """
    Search Amazon for product listings.
    amazon, ecommerce, products, search, shopping

    Use cases:
    - Find Amazon products
    - Compare product listings
    - Research product availability
    - Monitor pricing
    - Build price comparison tools
    """

    query: str = Field(default="", description="Product search query")
    num_results: int = Field(
        default=10, description="Maximum number of product results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "amazon",
            {"query": self.query, "amazon_domain": "amazon.com"}
        )
        return result


class WalmartSearch(BaseNode):
    """
    Search Walmart for product listings.
    walmart, ecommerce, products, search, shopping

    Use cases:
    - Find Walmart products
    - Compare with Amazon/other retailers
    - Research product availability
    - Monitor Walmart inventory
    - Build retail comparison tools
    """

    query: str = Field(default="", description="Product search query")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "walmart",
            {"query": self.query}
        )
        return result


class EbaySearch(BaseNode):
    """
    Search eBay for product listings and auctions.
    ebay, ecommerce, products, auction, search

    Use cases:
    - Find eBay listings
    - Research auction prices
    - Monitor product availability
    - Compare marketplace prices
    - Track collectibles
    """

    query: str = Field(default="", description="Product search query")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required")

        result = await _call_serp_engine(
            context,
            "ebay",
            {"_nkw": self.query, "ebay_domain": "ebay.com"}
        )
        return result
