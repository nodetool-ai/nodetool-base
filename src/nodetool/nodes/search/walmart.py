"""
Walmart search nodes for Nodetool.
Provides nodes for Walmart product search and product details via SerpAPI.
"""

from pydantic import Field
from typing import Any, TypedDict

from nodetool.nodes.search._base import SerpNode
from nodetool.workflows.processing_context import ProcessingContext


class WalmartSearch(SerpNode):
    """
    Search Walmart for products, prices, and reviews.
    walmart, search, products, ecommerce, shopping, prices
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Product name or description to search for")
    num_results: int = Field(default=10, description="Maximum number of results to return")
    sort_by: str = Field(
        default="best_match",
        description="Sort order for results: 'best_match', 'best_seller', 'price_low', 'price_high', 'rating_high', 'new'",
    )
    min_price: float | None = Field(default=None, description="Minimum price filter")
    max_price: float | None = Field(default=None, description="Maximum price filter")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "query": self.query,
        }
        if self.sort_by and self.sort_by != "best_match":
            params["sort"] = self.sort_by
        if self.min_price is not None:
            params["min_price"] = self.min_price
        if self.max_price is not None:
            params["max_price"] = self.max_price

        result_data = await self._search_raw(context, "walmart", params)

        results = result_data.get("organic_results", [])
        results = results[: self.num_results]

        formatted_lines = []
        for i, r in enumerate(results):
            pos = r.get("position", i + 1)
            title = r.get("title", "Untitled")
            formatted_lines.append(f"[{pos}] {title}")
            if r.get("us_item_id"):
                formatted_lines.append(f"    ID: {r['us_item_id']}")
            offer = r.get("primary_offer", {})
            if isinstance(offer, dict) and offer.get("offer_price"):
                formatted_lines.append(f"    Price: ${offer['offer_price']}")
            if r.get("rating") is not None:
                rating_str = f"    Rating: {r['rating']}"
                if r.get("reviews") is not None:
                    rating_str += f" ({r['reviews']} reviews)"
                formatted_lines.append(rating_str)
            if r.get("seller_name"):
                formatted_lines.append(f"    Seller: {r['seller_name']}")
            if r.get("product_page_url"):
                formatted_lines.append(f"    {r['product_page_url']}")
            formatted_lines.append("")

        return {"results": results, "text": "\n".join(formatted_lines)}


class WalmartProduct(SerpNode):
    """
    Get detailed information about a specific Walmart product by product ID.
    walmart, product, details, ecommerce, reviews, pricing
    """

    product_id: str = Field(
        default="",
        description="Walmart product ID (found in walmart.com/ip/{product_id} URLs)",
    )

    async def process(self, context: ProcessingContext) -> dict:
        if not self.product_id:
            raise ValueError("Product ID is required")

        result_data = await self._search_raw(context, "walmart_product", {
            "product_id": self.product_id,
        })
        return result_data.get("product_result", result_data)
