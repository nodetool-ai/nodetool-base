"""
YouTube search nodes for Nodetool.
Provides nodes for searching YouTube videos via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


class YouTubeSearch(BaseNode):
    """
    Search YouTube for videos, channels, and content.
    youtube, search, video, content, streaming, media
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Search query for YouTube videos")
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
            result_data = await provider.search_youtube(
                query=self.query, num_results=self.num_results,
            )
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            results = result_data.get("video_results", [])
            lines = []
            for i, r in enumerate(results):
                pos = r.get("position", r.get("position_on_page", i + 1))
                lines.append(f"[{pos}] {r.get('title', 'Untitled')}")
                channel = r.get("channel", "")
                if isinstance(channel, dict):
                    channel = channel.get("name", "")
                if channel:
                    lines.append(f"    Channel: {channel}")
                if r.get("views") is not None:
                    lines.append(f"    Views: {r['views']:,}")
                if r.get("published_date"):
                    lines.append(f"    Published: {r['published_date']}")
                if r.get("length"):
                    lines.append(f"    Duration: {r['length']}")
                if r.get("link"):
                    lines.append(f"    {r['link']}")
                lines.append("")
            return {"results": results, "text": "\n".join(lines)}
