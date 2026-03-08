"""
Google Play Store search nodes for Nodetool.
Provides nodes for Google Play app and game search via SerpAPI.
"""

from pydantic import Field
from typing import Any, TypedDict

from nodetool.nodes.search._base import SerpNode
from nodetool.workflows.processing_context import ProcessingContext


class GooglePlaySearch(SerpNode):
    """
    Search the Google Play Store for apps and games.
    google play, search, apps, games, android, mobile, store
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Search query for apps or games")
    store: str = Field(
        default="apps",
        description="Store section to search: 'apps' or 'games'",
    )
    num_results: int = Field(default=10, description="Maximum number of results to return")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "q": self.query,
            "store": self.store,
        }

        result_data = await self._search_raw(context, "google_play", params)

        organic = result_data.get("organic_results", [])
        # organic_results is a list of sections, each with "items"
        results: list[dict] = []
        for section in organic:
            items = section.get("items", [])
            results.extend(items)
        results = results[: self.num_results]

        formatted_lines = []
        for i, r in enumerate(results):
            pos = r.get("position", i + 1)
            title = r.get("title", "Untitled")
            formatted_lines.append(f"[{pos}] {title}")
            if r.get("product_id"):
                formatted_lines.append(f"    ID: {r['product_id']}")
            if r.get("rating") is not None:
                formatted_lines.append(f"    Rating: {r['rating']}")
            if r.get("author"):
                formatted_lines.append(f"    Developer: {r['author']}")
            if r.get("category"):
                formatted_lines.append(f"    Category: {r['category']}")
            if r.get("downloads"):
                formatted_lines.append(f"    Downloads: {r['downloads']}")
            price = r.get("price")
            if price:
                formatted_lines.append(f"    Price: {price}")
            elif r.get("extracted_price") is not None:
                formatted_lines.append(f"    Price: {r['extracted_price']}")
            if r.get("description"):
                desc = r["description"]
                if len(desc) > 150:
                    desc = desc[:147] + "..."
                formatted_lines.append(f"    {desc}")
            if r.get("link"):
                formatted_lines.append(f"    {r['link']}")
            formatted_lines.append("")

        return {"results": results, "text": "\n".join(formatted_lines)}
