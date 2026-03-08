"""
Yelp search nodes for Nodetool.
Provides nodes for searching businesses and reviews via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.metadata.types import YelpResult
from nodetool.workflows.base_node import BaseNode
from nodetool.agents.serp_providers.serp_types import YelpSearchResponse
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


def _format_yelp_results(results: list[YelpResult]) -> str:
    """Format Yelp search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        if r.rating is not None:
            rating_str = f"    Rating: {r.rating}"
            if r.reviews is not None:
                rating_str += f" ({r.reviews} reviews)"
            lines.append(rating_str)
        if r.price:
            lines.append(f"    Price: {r.price}")
        if r.address:
            lines.append(f"    Address: {r.address}")
        if r.categories:
            lines.append(f"    Categories: {', '.join(r.categories)}")
        if r.link:
            lines.append(f"    {r.link}")
        lines.append("")
    return "\n".join(lines)


class YelpSearch(BaseNode):
    """
    Search Yelp for businesses, restaurants, and local services.
    yelp, search, businesses, restaurants, reviews, local
    """

    class OutputType(TypedDict):
        results: list[YelpResult]
        text: str

    query: str = Field(
        default="", description="Business type or name to search for"
    )
    location: str = Field(
        default="", description="Location to search in (e.g., 'San Francisco, CA')"
    )
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")
        if not self.location:
            raise ValueError("Location is required for Yelp search")

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(
                error_response.get("error", "Failed to configure SERP provider")
            )
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_yelp(
                query=self.query,
                location=self.location,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = YelpSearchResponse(**result_data)
            results = response.organic_results or []

            return {
                "results": results,
                "text": _format_yelp_results(results),
            }
