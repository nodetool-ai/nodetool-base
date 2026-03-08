"""
Google Trends search nodes for Nodetool.
Provides nodes for retrieving trend data via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


class GoogleTrends(BaseNode):
    """
    Retrieve Google Trends data showing search interest over time.
    google, trends, popularity, interest, analytics, zeitgeist
    """

    query: str = Field(
        default="", description="Search term to get trends for"
    )
    date: str = Field(
        default="",
        description="Time range (e.g., 'today 12-m' for past year, 'today 3-m', '2024-01-01 2024-12-31')",
    )
    geo: str = Field(
        default="",
        description="Geographic region code (e.g., 'US', 'GB', 'DE')",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict:
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
            result_data = await provider.search_trends(
                query=self.query,
                date=self.date if self.date else None,
                geo=self.geo if self.geo else None,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            return result_data
