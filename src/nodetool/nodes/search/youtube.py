"""
YouTube search nodes for Nodetool.
Provides nodes for searching YouTube videos via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.metadata.types import YouTubeResult
from nodetool.workflows.base_node import BaseNode
from nodetool.agents.serp_providers.serp_types import YouTubeSearchResponse
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


def _format_youtube_results(results: list[YouTubeResult]) -> str:
    """Format YouTube search results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        if r.channel:
            lines.append(f"    Channel: {r.channel}")
        if r.views is not None:
            lines.append(f"    Views: {r.views:,}")
        if r.published_date:
            lines.append(f"    Published: {r.published_date}")
        if r.length:
            lines.append(f"    Duration: {r.length}")
        if r.link:
            lines.append(f"    {r.link}")
        lines.append("")
    return "\n".join(lines)


class YouTubeSearch(BaseNode):
    """
    Search YouTube for videos, channels, and content.
    youtube, search, video, content, streaming, media
    """

    class OutputType(TypedDict):
        results: list[YouTubeResult]
        text: str

    query: str = Field(
        default="", description="Search query for YouTube videos"
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
            result_data = await provider.search_youtube(
                query=self.query,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = YouTubeSearchResponse(**result_data)
            results = response.video_results or []

            return {
                "results": results,
                "text": _format_youtube_results(results),
            }
