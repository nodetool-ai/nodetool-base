#!/usr/bin/env python

"""
Alternative search engine nodes for Nodetool.
Provides nodes for Bing, DuckDuckGo, Yahoo, and e-commerce search engines.
"""

from pydantic import Field
from typing import Any, Dict, ClassVar

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

# Import the extended provider - it's in the nodetool-base package
try:
    from nodetool.providers.extended_serp_provider import ExtendedSerpApiProvider
except ImportError:
    # Fallback for development
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from providers.extended_serp_provider import ExtendedSerpApiProvider


async def _get_extended_serp_provider(context: ProcessingContext) -> ExtendedSerpApiProvider:
    """Get an instance of the Extended SERPAPIProvider with credentials from context."""
    serpapi_key = await context.get_secret("SERPAPI_API_KEY")
    if not serpapi_key:
        raise ValueError(
            "SERPAPI_API_KEY not found. Please configure your SerpApi credentials."
        )
    return ExtendedSerpApiProvider(api_key=serpapi_key)


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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_bing(q=self.query, num_results=self.num_results)

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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_bing_images(
                q=self.query, num_results=self.num_results
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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_duckduckgo(q=self.query)

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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_yahoo(q=self.query, num_results=self.num_results)

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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_youtube(
                search_query=self.query, num_results=self.num_results
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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_amazon(
                query=self.query, num_results=self.num_results
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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_walmart(query=self.query)

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

        provider = await _get_extended_serp_provider(context)
        async with provider:
            result = await provider.search_ebay(query=self.query)

        return result
