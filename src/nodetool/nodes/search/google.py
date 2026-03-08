#!/usr/bin/env python

"""
SERP (Search Engine Results Page) nodes for Nodetool.
Provides nodes for Google Search, News, Images, Finance, Jobs, Lens, Maps, and Shopping.
"""

from pydantic import Field
from typing import Any, Dict, ClassVar, TypedDict

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import (
    JobResult,
    LocalResult,
    NewsResult,
    OrganicResult,
    ShoppingResult,
)
from nodetool.agents.serp_providers.serp_types import (
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


def _format_organic_results(results: list[OrganicResult]) -> str:
    """Format organic search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        lines.append(f"    {r.link}")
        if r.date:
            lines.append(f"    Date: {r.date}")
        lines.append(f"    {r.snippet}")
        lines.append("")
    return "\n".join(lines)


def _format_news_results(results: list[NewsResult]) -> str:
    """Format news search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        lines.append(f"    {r.link}")
        if r.date:
            lines.append(f"    Date: {r.date}")
        lines.append("")
    return "\n".join(lines)


def _format_job_results(results: list[JobResult]) -> str:
    """Format job search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"{r.title}")
        if r.company_name:
            lines.append(f"    Company: {r.company_name}")
        if r.location:
            lines.append(f"    Location: {r.location}")
        if r.via:
            lines.append(f"    Via: {r.via}")
        if r.extensions:
            lines.append(f"    {', '.join(r.extensions)}")
        lines.append("")
    return "\n".join(lines)


def _format_local_results(results: list[LocalResult]) -> str:
    """Format local/maps search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        if r.address:
            lines.append(f"    {r.address}")
        if r.rating is not None:
            rating_str = f"    Rating: {r.rating}"
            if r.reviews is not None:
                rating_str += f" ({r.reviews} reviews)"
            lines.append(rating_str)
        if r.price:
            lines.append(f"    Price: {r.price}")
        if r.open_state:
            lines.append(f"    {r.open_state}")
        lines.append("")
    return "\n".join(lines)


def _format_shopping_results(results: list[ShoppingResult]) -> str:
    """Format shopping search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        if r.price:
            price_str = f"    Price: {r.price}"
            if r.old_price:
                price_str += f" (was {r.old_price})"
            lines.append(price_str)
        if r.source:
            lines.append(f"    Source: {r.source}")
        if r.link:
            lines.append(f"    {r.link}")
        if r.rating is not None:
            lines.append(f"    Rating: {r.rating}")
        lines.append("")
    return "\n".join(lines)


class GoogleSearch(BaseNode):
    """
    Search Google to retrieve organic search results from the web.
    google, search, serp, web, query
    """

    class OutputType(TypedDict):
        results: list[OrganicResult]
        text: str

    keyword: str = Field(
        default="", description="Search query or keyword to search for"
    )
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
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
            results = result_data.organic_results or []

            return {
                "results": results,
                "text": _format_organic_results(results),
            }


class GoogleNews(BaseNode):
    """
    Search Google News to retrieve current news articles and headlines.
    google, news, serp, articles, journalism
    """

    class OutputType(TypedDict):
        results: list[NewsResult]
        text: str

    keyword: str = Field(
        default="", description="Search query or keyword for news articles"
    )
    num_results: int = Field(
        default=10, description="Maximum number of news results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
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
            results = result_data.news_results or []

            return {
                "results": results,
                "text": _format_news_results(results),
            }


class GoogleImages(BaseNode):
    """
    Search Google Images to find visual content or perform reverse image search.
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
    Retrieve financial market data and stock information from Google Finance.
    google, finance, stocks, market, serp, trading
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
    Search Google Jobs for employment opportunities and job listings.
    google, jobs, employment, careers, serp, hiring
    """

    class OutputType(TypedDict):
        results: list[JobResult]
        text: str

    query: str = Field(
        default="", description="Job title, skills, or company name to search for"
    )
    location: str = Field(default="", description="Geographic location for job search")
    num_results: int = Field(
        default=10, description="Maximum number of job results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
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
            results = response.jobs_results or []

            return {
                "results": results,
                "text": _format_job_results(results),
            }


class GoogleLens(BaseNode):
    """
    Analyze images using Google Lens to find visual matches and related content.
    google, lens, visual, image, search, serp, identify
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
    Search Google Maps for places, businesses, and get location details.
    google, maps, places, locations, serp, geography
    """

    class OutputType(TypedDict):
        results: list[LocalResult]
        text: str

    query: str = Field(default="", description="Place name, address, or location query")
    num_results: int = Field(
        default=10, description="Maximum number of map results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
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
            results = response.local_results or []

            return {
                "results": results,
                "text": _format_local_results(results),
            }


class GoogleShopping(BaseNode):
    """
    Search Google Shopping for products with filters and pricing information.
    google, shopping, products, ecommerce, serp, prices
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

    class OutputType(TypedDict):
        results: list[ShoppingResult]
        text: str

    async def process(self, context: ProcessingContext) -> OutputType:
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
            results = response.shopping_results or []

            return {
                "results": results,
                "text": _format_shopping_results(results),
            }
