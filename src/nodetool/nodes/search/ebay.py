"""
eBay search nodes for Nodetool.
Provides nodes for eBay product search and product details via SerpAPI.
"""

from pydantic import Field
from typing import Any, TypedDict

from nodetool.nodes.search._base import SerpNode
from nodetool.workflows.processing_context import ProcessingContext


class EbaySearch(SerpNode):
    """
    Search eBay for products, prices, and deals.
    ebay, search, products, ecommerce, shopping, prices, auction
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Product name or description to search for")
    ebay_domain: str = Field(
        default="ebay.com",
        description="eBay domain (e.g., 'ebay.com', 'ebay.co.uk', 'ebay.de')",
    )
    num_results: int = Field(default=10, description="Maximum number of results to return")
    condition: str = Field(
        default="",
        description="Item condition filter: 'new', 'used', or empty for all",
    )
    min_price: float | None = Field(default=None, description="Minimum price filter")
    max_price: float | None = Field(default=None, description="Maximum price filter")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "_nkw": self.query,
            "ebay_domain": self.ebay_domain,
        }
        if self.condition:
            condition_map = {"new": "1000", "used": "3000"}
            condition_id = condition_map.get(self.condition.lower(), self.condition)
            params["LH_ItemCondition"] = condition_id
        if self.min_price is not None:
            params["_udlo"] = self.min_price
        if self.max_price is not None:
            params["_udhi"] = self.max_price

        result_data = await self._search_raw(context, "ebay", params)

        results = result_data.get("organic_results", [])
        results = results[: self.num_results]

        formatted_lines = []
        for i, r in enumerate(results):
            pos = r.get("position", i + 1)
            title = r.get("title", "Untitled")
            formatted_lines.append(f"[{pos}] {title}")
            if r.get("product_id"):
                formatted_lines.append(f"    ID: {r['product_id']}")
            price = r.get("price")
            if isinstance(price, dict):
                if price.get("raw"):
                    formatted_lines.append(f"    Price: {price['raw']}")
            elif price:
                formatted_lines.append(f"    Price: {price}")
            if r.get("condition"):
                formatted_lines.append(f"    Condition: {r['condition']}")
            if r.get("rating") is not None:
                rating_str = f"    Rating: {r['rating']}"
                if r.get("reviews") is not None:
                    rating_str += f" ({r['reviews']} reviews)"
                formatted_lines.append(rating_str)
            if r.get("shipping"):
                formatted_lines.append(f"    Shipping: {r['shipping']}")
            if r.get("link"):
                formatted_lines.append(f"    {r['link']}")
            formatted_lines.append("")

        return {"results": results, "text": "\n".join(formatted_lines)}


class EbayProduct(SerpNode):
    """
    Get detailed information about a specific eBay product by product ID.
    ebay, product, details, ecommerce, reviews, pricing, auction
    """

    product_id: str = Field(
        default="",
        description="eBay product ID to look up",
    )
    ebay_domain: str = Field(
        default="ebay.com",
        description="eBay domain (e.g., 'ebay.com', 'ebay.co.uk', 'ebay.de')",
    )

    async def process(self, context: ProcessingContext) -> dict:
        if not self.product_id:
            raise ValueError("Product ID is required")

        result_data = await self._search_raw(context, "ebay_product", {
            "product_id": self.product_id,
            "ebay_domain": self.ebay_domain,
        })
        return result_data.get("product_results", result_data)
