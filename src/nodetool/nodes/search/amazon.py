"""
Amazon search nodes for Nodetool.
Provides nodes for Amazon product search and product details via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


class AmazonSearch(BaseNode):
    """
    Search Amazon for products, prices, and reviews.
    amazon, search, products, ecommerce, shopping, prices
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Product name or description to search for")
    amazon_domain: str = Field(
        default="amazon.com",
        description="Amazon domain (e.g., 'amazon.com', 'amazon.co.uk', 'amazon.de')",
    )
    num_results: int = Field(default=10, description="Maximum number of results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_amazon(
                query=self.query, amazon_domain=self.amazon_domain, num_results=self.num_results,
            )
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            results = result_data.get("organic_results", [])
            lines = []
            for i, r in enumerate(results):
                pos = r.get("position", i + 1)
                lines.append(f"[{pos}] {r.get('title', 'Untitled')}")
                if r.get("asin"):
                    lines.append(f"    ASIN: {r['asin']}")
                if r.get("price"):
                    lines.append(f"    Price: {r['price']}")
                if r.get("rating") is not None:
                    rating_str = f"    Rating: {r['rating']}"
                    if r.get("reviews") is not None:
                        rating_str += f" ({r['reviews']} reviews)"
                    lines.append(rating_str)
                if r.get("link"):
                    lines.append(f"    {r['link']}")
                lines.append("")
            return {"results": results, "text": "\n".join(lines)}


class AmazonProduct(BaseNode):
    """
    Get detailed information about a specific Amazon product by ASIN.
    amazon, product, details, asin, ecommerce, reviews
    """

    product_id: str = Field(default="", description="Amazon ASIN (product ID) to look up")
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
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_amazon_product(
                product_id=self.product_id, amazon_domain=self.amazon_domain,
            )
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])
            return result_data.get("product_result", result_data)
