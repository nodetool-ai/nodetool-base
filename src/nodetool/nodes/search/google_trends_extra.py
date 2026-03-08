"""
Google Trends Trending Now node for Nodetool.
Retrieves currently trending searches via SerpAPI's google_trends_trending_now engine.
"""

from enum import Enum

from pydantic import Field
from typing import TypedDict

from nodetool.nodes.search._base import SerpNode
from nodetool.workflows.processing_context import ProcessingContext


class TrendingFrequency(str, Enum):
    REALTIME = "realtime"
    DAILY = "daily"


class GoogleTrendingNow(SerpNode):
    """
    Retrieve currently trending searches from Google Trends, including
    real-time and daily trending topics with traffic volume and categories.
    google, trends, trending, viral, popular, realtime, daily, buzz
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    geo: str = Field(
        default="US",
        description="Country code for trending results (e.g., 'US', 'GB', 'DE')",
    )
    category: str = Field(
        default="all",
        description="Trending category to filter by (e.g., 'all', or a category ID)",
    )
    frequency: TrendingFrequency = Field(
        default=TrendingFrequency.REALTIME,
        description="Frequency of trending data: 'realtime' or 'daily'",
    )
    hours: int = Field(
        default=24,
        description="Time range in hours (4, 24, 48, or 168)",
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        params: dict = {
            "geo": self.geo,
            "hours": self.hours,
        }
        if self.category and self.category != "all":
            params["category_id"] = self.category

        result_data = await self._search_raw(
            context, "google_trends_trending_now", params
        )

        # The API returns trending searches under "trending_searches"
        results = result_data.get("trending_searches", [])
        if not results:
            results = result_data.get("realtime_searches", [])
        if not results:
            results = result_data.get("daily_searches", [])

        lines: list[str] = []
        for i, item in enumerate(results):
            query = item.get("query", item.get("title", "Unknown"))
            lines.append(f"[{i + 1}] {query}")

            volume = item.get("search_volume")
            if volume is not None:
                lines.append(f"    Traffic: {volume:,}+ searches")

            increase = item.get("increase_percentage")
            if increase is not None:
                lines.append(f"    Increase: {increase}%")

            active = item.get("active")
            if active is not None:
                lines.append(f"    Active: {'Yes' if active else 'No'}")

            categories = item.get("categories", [])
            if categories:
                cat_names = [c.get("name", "") for c in categories if c.get("name")]
                if cat_names:
                    lines.append(f"    Categories: {', '.join(cat_names)}")

            breakdown = item.get("trend_breakdown", [])
            if breakdown:
                lines.append(f"    Related: {', '.join(breakdown[:5])}")

            lines.append("")

        return {"results": results, "text": "\n".join(lines)}
