"""
SERP (Search Engine Results Page) nodes for Nodetool.
Provides nodes for Google Search, News, Images, Finance, Jobs, Lens, Maps, and Shopping.
"""

from pydantic import Field
from typing import Any, ClassVar, TypedDict

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


def _format_results(results: list[dict], fields: list[tuple[str, str | None]]) -> str:
    """Generic formatter for search results.

    fields: list of (key, label) tuples. If label is None, value is printed raw.
    """
    lines = []
    for i, r in enumerate(results):
        pos = r.get("position", i + 1)
        title = r.get("title", "Untitled")
        lines.append(f"[{pos}] {title}")
        for key, label in fields:
            val = r.get(key)
            if val:
                if label:
                    lines.append(f"    {label}: {val}")
                else:
                    lines.append(f"    {val}")
        lines.append("")
    return "\n".join(lines)


class GoogleSearch(BaseNode):
    """
    Search Google to retrieve organic search results from the web.
    google, search, serp, web, query
    """

    class OutputType(TypedDict):
        results: list[dict]
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
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search(keyword=self.keyword, num_results=self.num_results)
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            results = result_data.get("organic_results", [])
            text = _format_results(results, [("link", None), ("date", "Date"), ("snippet", None)])
            return {"results": results, "text": text}


class GoogleNews(BaseNode):
    """
    Search Google News to retrieve current news articles and headlines.
    google, news, serp, articles, journalism
    """

    class OutputType(TypedDict):
        results: list[dict]
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
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_news(keyword=self.keyword, num_results=self.num_results)
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            results = result_data.get("news_results", [])
            text = _format_results(results, [("link", None), ("date", "Date")])
            return {"results": results, "text": text}


class GoogleImages(BaseNode):
    """
    Search Google Images to find visual content or perform reverse image search.
    google, images, serp, visual, reverse, search
    """

    keyword: str = Field(default="", description="Search query or keyword for images")
    image_url: str = Field(default="", description="URL of image for reverse image search")
    num_results: int = Field(default=20, description="Maximum number of image results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> list[ImageRef]:
        if not self.keyword and not self.image_url:
            raise ValueError("One of 'keyword' or 'image_url' is required.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
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

            images = result_data.get("images_results", [])
            return [ImageRef(uri=img.get("original", "")) for img in images if img.get("original")]


class GoogleFinance(BaseNode):
    """
    Retrieve financial market data and stock information from Google Finance.
    google, finance, stocks, market, serp, trading
    """

    query: str = Field(default="", description="Stock symbol or company name to search for")
    window: str = Field(
        default="",
        description="Time window for financial data (e.g., '1d', '5d', '1m', '3m', '6m', '1y', '5y')",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        if not self.query:
            raise ValueError("Query is required for Google Finance search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_finance(
                query=self.query, window=self.window if self.window else None
            )
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])
            return result_data


class GoogleJobs(BaseNode):
    """
    Search Google Jobs for employment opportunities and job listings.
    google, jobs, employment, careers, serp, hiring
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Job title, skills, or company name to search for")
    location: str = Field(default="", description="Geographic location for job search")
    num_results: int = Field(default=10, description="Maximum number of job results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required for Google Jobs search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
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

            results = result_data.get("jobs_results", [])
            text = _format_results(results, [
                ("company_name", "Company"), ("location", "Location"),
                ("via", "Via"), ("extensions", "Details"),
            ])
            return {"results": results, "text": text}


class GoogleLens(BaseNode):
    """
    Analyze images using Google Lens to find visual matches and related content.
    google, lens, visual, image, search, serp, identify
    """

    class OutputType(TypedDict):
        results: list[dict]
        images: list[ImageRef]

    image_url: str = Field(default="", description="URL of the image to analyze with Google Lens")
    num_results: int = Field(default=10, description="Maximum number of visual search results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.image_url:
            raise ValueError("Image URL is required for Google Lens search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_lens(image_url=self.image_url, num_results=self.num_results)
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            matches = result_data.get("visual_matches", [])
            images = [
                ImageRef(uri=m.get("image", m.get("thumbnail", "")))
                for m in matches if m.get("image") or m.get("thumbnail")
            ]
            return {"results": matches, "images": images}


class GoogleMaps(BaseNode):
    """
    Search Google Maps for places, businesses, and get location details.
    google, maps, places, locations, serp, geography
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Place name, address, or location query")
    num_results: int = Field(default=10, description="Maximum number of map results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required for map search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_maps(query=self.query, num_results=self.num_results)
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            results = result_data.get("local_results", [])
            lines = []
            for i, r in enumerate(results):
                pos = r.get("position", i + 1)
                lines.append(f"[{pos}] {r.get('title', 'Untitled')}")
                if r.get("address"):
                    lines.append(f"    {r['address']}")
                if r.get("rating") is not None:
                    rating_str = f"    Rating: {r['rating']}"
                    if r.get("reviews") is not None:
                        rating_str += f" ({r['reviews']} reviews)"
                    lines.append(rating_str)
                if r.get("price"):
                    lines.append(f"    Price: {r['price']}")
                if r.get("open_state"):
                    lines.append(f"    {r['open_state']}")
                lines.append("")
            return {"results": results, "text": "\n".join(lines)}


class GoogleShopping(BaseNode):
    """
    Search Google Shopping for products with filters and pricing information.
    google, shopping, products, ecommerce, serp, prices
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Product name or description to search for")
    country: str = Field(default="us", description="Country code for shopping search (e.g., 'us', 'uk', 'ca')")
    min_price: int = Field(default=0, description="Minimum price filter for products")
    max_price: int = Field(default=0, description="Maximum price filter for products")
    condition: str = Field(default="", description="Product condition filter (e.g., 'new', 'used', 'refurbished')")
    sort_by: str = Field(default="", description="Sort order for results (e.g., 'price_low_to_high', 'price_high_to_low', 'review_score')")
    num_results: int = Field(default=10, description="Maximum number of shopping results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required for Google Shopping search.")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
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

            results = result_data.get("shopping_results", [])
            lines = []
            for i, r in enumerate(results):
                pos = r.get("position", i + 1)
                lines.append(f"[{pos}] {r.get('title', 'Untitled')}")
                if r.get("price"):
                    price_str = f"    Price: {r['price']}"
                    if r.get("old_price"):
                        price_str += f" (was {r['old_price']})"
                    lines.append(price_str)
                if r.get("source"):
                    lines.append(f"    Source: {r['source']}")
                if r.get("link"):
                    lines.append(f"    {r['link']}")
                if r.get("rating") is not None:
                    lines.append(f"    Rating: {r['rating']}")
                lines.append("")
            return {"results": results, "text": "\n".join(lines)}
