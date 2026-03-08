"""
Retail and app store search nodes for Nodetool.
Provides nodes for Home Depot and Apple App Store via SerpAPI.
"""

from pydantic import Field
from typing import Any, TypedDict

from nodetool.nodes.search._base import SerpNode, format_results
from nodetool.workflows.processing_context import ProcessingContext


class HomeDepotSearch(SerpNode):
    """
    Search Home Depot for products, tools, and home improvement items.
    home depot, search, products, hardware, tools, home improvement
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Product name or description to search for")
    num_results: int = Field(default=10, description="Maximum number of results to return")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "q": self.query,
            "ps": min(self.num_results, 48),
        }
        result_data = await self._search_raw(context, "home_depot", params)

        results = result_data.get("products", [])[:self.num_results]
        text = format_results(results, [
            ("brand", "Brand"),
            ("price", "Price"),
            ("rating", "Rating"),
            ("model_number", "Model"),
            ("link", None),
        ])
        return {"results": results, "text": text}


class HomeDepotProduct(SerpNode):
    """
    Get detailed information about a specific Home Depot product by product ID.
    home depot, product, details, home improvement, hardware
    """

    product_id: str = Field(default="", description="Home Depot product ID to look up")

    async def process(self, context: ProcessingContext) -> dict:
        if not self.product_id:
            raise ValueError("Product ID is required")

        params: dict[str, Any] = {
            "product_id": self.product_id,
        }
        result_data = await self._search_raw(context, "home_depot_product", params)

        return result_data.get("product_result", result_data)


class AppleAppStore(SerpNode):
    """
    Search the Apple App Store for apps, games, and utilities.
    apple, app store, search, apps, ios, iphone, ipad, mobile
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="App name or keyword to search for")
    num_results: int = Field(default=10, description="Maximum number of results to return")
    country: str = Field(default="us", description="Two-letter country code (e.g., 'us', 'gb', 'de')")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "term": self.query,
            "country": self.country,
            "num": min(self.num_results, 200),
        }
        result_data = await self._search_raw(context, "apple_app_store", params)

        results = result_data.get("organic_results", [])[:self.num_results]
        text = format_results(results, [
            ("developer", "Developer"),
            ("price", "Price"),
            ("rating", "Rating"),
            ("description", "Description"),
            ("link", None),
        ])
        return {"results": results, "text": text}
