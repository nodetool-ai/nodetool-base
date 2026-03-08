"""
Amazon search nodes for Nodetool.
Provides nodes for Amazon product search and product details via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.metadata.types import AmazonResult
from nodetool.workflows.base_node import BaseNode
from nodetool.agents.serp_providers.serp_types import (
    AmazonSearchResponse,
    AmazonProductResponse,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


def _format_amazon_results(results: list[AmazonResult]) -> str:
    """Format Amazon search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        if r.asin:
            lines.append(f"    ASIN: {r.asin}")
        if r.price:
            lines.append(f"    Price: {r.price}")
        if r.rating is not None:
            rating_str = f"    Rating: {r.rating}"
            if r.reviews is not None:
                rating_str += f" ({r.reviews} reviews)"
            lines.append(rating_str)
        if r.link:
            lines.append(f"    {r.link}")
        lines.append("")
    return "\n".join(lines)


class AmazonSearch(BaseNode):
    """
    Search Amazon for products, prices, and reviews.
    amazon, search, products, ecommerce, shopping, prices
    """

    class OutputType(TypedDict):
        results: list[AmazonResult]
        text: str

    query: str = Field(
        default="", description="Product name or description to search for"
    )
    amazon_domain: str = Field(
        default="amazon.com",
        description="Amazon domain (e.g., 'amazon.com', 'amazon.co.uk', 'amazon.de')",
    )
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_amazon(
                query=self.query,
                amazon_domain=self.amazon_domain,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = AmazonSearchResponse(**result_data)
            results = response.organic_results or []

            return {
                "results": results,
                "text": _format_amazon_results(results),
            }


class AmazonProduct(BaseNode):
    """
    Get detailed information about a specific Amazon product by ASIN.
    amazon, product, details, asin, ecommerce, reviews
    """

    product_id: str = Field(
        default="", description="Amazon ASIN (product ID) to look up"
    )
    amazon_domain: str = Field(
        default="amazon.com",
        description="Amazon domain (e.g., 'amazon.com', 'amazon.co.uk', 'amazon.de')",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict:
        if not self.product_id:
            raise ValueError("Product ID (ASIN) is required")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_amazon_product(
                product_id=self.product_id,
                amazon_domain=self.amazon_domain,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = AmazonProductResponse(**result_data)
            return response.product_result
