#!/usr/bin/env python

"""
SERP (Search Engine Results Page) nodes for Nodetool.
Provides nodes for Google Search, News, Images, Finance, Jobs, Lens, Maps, and Shopping.
"""

from pydantic import Field
from typing import Any, Dict, ClassVar

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import (
    ImageResult,
    JobResult,
    LocalResult,
    NewsResult,
    OrganicResult,
    ShoppingResult,
    VisualMatchResult,
)
from nodetool.agents.serp_providers.types import (
    GoogleJobsResponse,
    GoogleLensResponse,
    GoogleSearchResponse,
    GoogleNewsResponse,
    GoogleImagesResponse,
    GoogleMapsResponse,
    GoogleShoppingResponse,
)

from nodetool.workflows.processing_context import ProcessingContext

# Import from nodetool-core
from nodetool.agents.tools.serp_tools import (
    _get_configured_serp_provider,
)


class GoogleSearch(BaseNode):
    """
    Search Google to retrieve organic search results.
    google, search, serp, web
    """

    keyword: str = Field(
        default="", description="Search query or keyword to search for"
    )
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[OrganicResult]:
        if not self.keyword:
            raise ValueError("Keyword is required")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result = await provider.search(
                keyword=self.keyword, num_results=self.num_results
            )

            result_data = GoogleSearchResponse(**result)

            return result_data.organic_results or []


class GoogleNews(BaseNode):
    """
    Search Google News to retrieve live news articles.
    google, news, serp, articles
    """

    keyword: str = Field(
        default="", description="Search query or keyword for news articles"
    )
    num_results: int = Field(
        default=10, description="Maximum number of news results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[NewsResult]:
        if not self.keyword:
            raise ValueError("Keyword is required")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_news(
                keyword=self.keyword, num_results=self.num_results
            )

            result_data = GoogleNewsResponse(**result_data)

            return result_data.news_results or []


class GoogleImages(BaseNode):
    """
    Search Google Images to retrieve live image results.
    google, images, serp, visual, reverse, search
    """

    keyword: str = Field(default="", description="Search query or keyword for images")
    image_url: str = Field(
        default="", description="URL of image for reverse image search"
    )
    num_results: int = Field(
        default=20, description="Maximum number of image results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[ImageRef]:
        if not self.keyword and not self.image_url:
            raise ValueError("One of 'keyword' or 'image_url' is required.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_images(
                keyword=self.keyword if self.keyword else None,
                image_url=self.image_url if self.image_url else None,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = GoogleImagesResponse(**result_data)

            return [
                ImageRef(
                    uri=image.original,
                )
                for image in response.images_results
            ]


class GoogleFinance(BaseNode):
    """
    Retrieve financial market data from Google Finance.
    google, finance, stocks, market, serp
    """

    query: str = Field(
        default="", description="Stock symbol or company name to search for"
    )
    window: str = Field(
        default="",
        description="Time window for financial data (e.g., '1d', '5d', '1m', '3m', '6m', '1y', '5y')",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.query:
            return {"error": "Query is required for Google Finance search."}

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_finance(
                query=self.query, window=self.window if self.window else None
            )

        if isinstance(result_data, dict) and "error" in result_data:
            return result_data

        return {"success": True, "results": result_data}


class GoogleJobs(BaseNode):
    """
    Search Google Jobs for job listings.
    google, jobs, employment, careers, serp
    """

    query: str = Field(
        default="", description="Job title, skills, or company name to search for"
    )
    location: str = Field(default="", description="Geographic location for job search")
    num_results: int = Field(
        default=10, description="Maximum number of job results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[JobResult]:
        if not self.query:
            raise ValueError("Query is required for Google Jobs search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_jobs(
                query=self.query,
                location=self.location if self.location else None,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = GoogleJobsResponse(**result_data)

            return response.jobs_results or []


class GoogleLens(BaseNode):
    """
    Search with an image URL using Google Lens to find visual matches and related content.
    google, lens, visual, image, search, serp
    """

    image_url: str = Field(
        default="", description="URL of the image to analyze with Google Lens"
    )
    num_results: int = Field(
        default=10, description="Maximum number of visual search results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext):
        if not self.image_url:
            raise ValueError("Image URL is required for Google Lens search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_lens(
                image_url=self.image_url, num_results=self.num_results
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = GoogleLensResponse(**result_data)

            return {
                "results": response.visual_matches,
                "images": [
                    ImageRef(
                        uri=(
                            image.image
                            if image.image
                            else (image.thumbnail if image.thumbnail else "")
                        ),
                    )
                    for image in response.visual_matches
                ],
            }


class GoogleMaps(BaseNode):
    """
    Search Google Maps for places or get details about a specific place.
    google, maps, places, locations, serp
    """

    query: str = Field(default="", description="Place name, address, or location query")
    num_results: int = Field(
        default=10, description="Maximum number of map results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[LocalResult]:
        if not self.query:
            raise ValueError("Query is required for map search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_maps(
                query=self.query,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            for result in result_data["local_results"]:
                result["place_type"] = result.get("type", "")
                del result["type"]

            response = GoogleMapsResponse(**result_data)

            return response.local_results or []


class GoogleShopping(BaseNode):
    """
    Search Google Shopping for products.
    google, shopping, products, ecommerce, serp
    """

    query: str = Field(
        default="", description="Product name or description to search for"
    )
    country: str = Field(
        default="us",
        description="Country code for shopping search (e.g., 'us', 'uk', 'ca')",
    )
    min_price: int = Field(default=0, description="Minimum price filter for products")
    max_price: int = Field(default=0, description="Maximum price filter for products")
    condition: str = Field(
        default="",
        description="Product condition filter (e.g., 'new', 'used', 'refurbished')",
    )
    sort_by: str = Field(
        default="",
        description="Sort order for results (e.g., 'price_low_to_high', 'price_high_to_low', 'review_score')",
    )
    num_results: int = Field(
        default=10, description="Maximum number of shopping results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[ShoppingResult]:
        if not self.query:
            raise ValueError("Query is required for Google Shopping search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_shopping(
                query=self.query,
                country=self.country if self.country else "us",
                min_price=self.min_price if self.min_price > 0 else None,
                max_price=self.max_price if self.max_price > 0 else None,
                condition=self.condition if self.condition else None,
                sort_by=self.sort_by if self.sort_by else None,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = GoogleShoppingResponse(**result_data)

            return response.shopping_results or []
