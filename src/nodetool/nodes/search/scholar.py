"""
Google Scholar search nodes for Nodetool.
Provides nodes for searching academic papers and citations via SerpAPI.
"""

from pydantic import Field
from typing import ClassVar, TypedDict

from nodetool.metadata.types import ScholarResult
from nodetool.workflows.base_node import BaseNode
from nodetool.agents.serp_providers.serp_types import GoogleScholarResponse
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


def _format_scholar_results(results: list[ScholarResult]) -> str:
    """Format Google Scholar results as readable text."""
    lines = []
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        if r.publication_info:
            lines.append(f"    {r.publication_info}")
        if r.snippet:
            lines.append(f"    {r.snippet}")
        if r.cited_by_count is not None:
            lines.append(f"    Cited by: {r.cited_by_count}")
        if r.link:
            lines.append(f"    {r.link}")
        lines.append("")
    return "\n".join(lines)


class GoogleScholar(BaseNode):
    """
    Search Google Scholar for academic papers, articles, and citations.
    google, scholar, academic, papers, research, citations, science
    """

    class OutputType(TypedDict):
        results: list[ScholarResult]
        text: str

    query: str = Field(
        default="", description="Academic search query"
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
            result_data = await provider.search_scholar(
                query=self.query,
                num_results=self.num_results,
            )

            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])

            response = GoogleScholarResponse(**result_data)
            results = response.organic_results or []

            return {
                "results": results,
                "text": _format_scholar_results(results),
            }
