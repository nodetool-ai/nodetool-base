"""
International search engine nodes for Nodetool.
Provides nodes for Baidu, Yahoo, and Naver search via SerpAPI.
"""

from pydantic import Field
from typing import TypedDict

from nodetool.nodes.search._base import SerpNode, format_results
from nodetool.workflows.processing_context import ProcessingContext


class BaiduSearch(SerpNode):
    """
    Search Baidu to retrieve organic search results from China's largest search engine.
    baidu, search, serp, web, chinese, china, query
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Search query or keyword to search for")
    num_results: int = Field(
        default=10, description="Maximum number of results to return (max 50)"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(context, "baidu", {
            "q": self.query,
            "rn": self.num_results,
        })

        results = result_data.get("organic_results", [])
        text = format_results(results, [
            ("link", None),
            ("snippet", None),
        ])
        return {"results": results, "text": text}


class YahooSearch(SerpNode):
    """
    Search Yahoo to retrieve organic web search results.
    yahoo, search, serp, web, query
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Search query or keyword to search for")
    num_results: int = Field(
        default=10, description="Maximum number of results to return"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(context, "yahoo", {
            "p": self.query,
        })

        results = result_data.get("organic_results", [])[:self.num_results]
        text = format_results(results, [
            ("link", None),
            ("displayed_link", "URL"),
            ("snippet", None),
        ])
        return {"results": results, "text": text}


class NaverSearch(SerpNode):
    """
    Search Naver to retrieve organic search results from South Korea's largest search engine.
    naver, search, serp, web, korean, korea, query
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Search query or keyword to search for")
    num_results: int = Field(
        default=10, description="Maximum number of results to return (max 100)"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(context, "naver", {
            "query": self.query,
            "num": self.num_results,
        })

        results = result_data.get("organic_results", [])
        text = format_results(results, [
            ("link", None),
            ("snippet", None),
        ])
        return {"results": results, "text": text}
