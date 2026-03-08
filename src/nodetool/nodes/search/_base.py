"""
Shared base class and utilities for all SerpAPI search nodes.
"""

from typing import Any, ClassVar, TypedDict

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider


class ListOutputType(TypedDict):
    """Standard output type for search nodes returning a list of results."""
    results: list[dict]
    text: str


def format_results(results: list[dict], fields: list[tuple[str, str | None]]) -> str:
    """Generic formatter for search results.

    fields: list of (key, label) tuples. If label is None, value is printed raw.
    """
    lines = []
    for i, r in enumerate(results):
        pos = r.get("position", i + 1)
        title = r.get("title", "Untitled")
        lines.append(f"[{pos}] {title}")
        for key, label in fields:
            val = r.get(key)
            if val:
                if label:
                    lines.append(f"    {label}: {val}")
                else:
                    lines.append(f"    {val}")
        lines.append("")
    return "\n".join(lines)


class SerpNode(BaseNode):
    """Base class for SerpAPI-powered search nodes.

    Provides common provider initialization and the search_raw helper.
    Subclasses implement process() and call self._search_raw() instead of
    repeating the 8-line provider boilerplate.
    """

    _expose_as_tool: ClassVar[bool] = True

    async def _search_raw(
        self, context: ProcessingContext, engine: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Initialize provider and execute a raw SerpAPI search.

        Raises ValueError on provider or API errors.
        Returns the full result dict from SerpAPI.
        """
        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            raise ValueError(error_response.get("error", "Failed to configure SERP provider"))
        if not provider_instance:
            raise ValueError("Failed to initialize SERP provider.")

        async with provider_instance as provider:
            result_data = await provider.search_raw(engine, params)
            if isinstance(result_data, dict) and "error" in result_data:
                raise ValueError(result_data["error"])
            return result_data
