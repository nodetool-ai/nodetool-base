"""
Yelp search nodes for Nodetool.
Provides nodes for searching businesses and reviews via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


class YelpSearch(BaseNode):
    """
    Search Yelp for businesses, restaurants, and local services.
    yelp, search, businesses, restaurants, reviews, local
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Business type or name to search for")
    location: str = Field(default="", description="Location to search in (e.g., 'San Francisco, CA')")
    num_results: int = Field(default=10, description="Maximum number of results to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")
        if not self.location:
            raise ValueError("Location is required for Yelp search")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_yelp(
                query=self.query, location=self.location, num_results=self.num_results,
            )
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            results = result_data.get("organic_results", [])
            lines = []
            for i, r in enumerate(results):
                pos = r.get("position", i + 1)
                lines.append(f"[{pos}] {r.get('title', 'Untitled')}")
                if r.get("rating") is not None:
                    rating_str = f"    Rating: {r['rating']}"
                    if r.get("reviews") is not None:
                        rating_str += f" ({r['reviews']} reviews)"
                    lines.append(rating_str)
                if r.get("price"):
                    lines.append(f"    Price: {r['price']}")
                if r.get("address"):
                    lines.append(f"    Address: {r['address']}")
                categories = r.get("categories", [])
                if categories:
                    if isinstance(categories, list):
                        lines.append(f"    Categories: {', '.join(str(c) for c in categories)}")
                if r.get("link"):
                    lines.append(f"    {r['link']}")
                lines.append("")
            return {"results": results, "text": "\n".join(lines)}
