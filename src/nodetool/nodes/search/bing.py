"""
Bing search nodes for Nodetool.
Provides nodes for Bing web search, news, and images via SerpAPI.
"""

from pydantic import Field
from typing import TypedDict

from nodetool.metadata.types import ImageRef
from nodetool.nodes.search._base import SerpNode, format_results
from nodetool.workflows.processing_context import ProcessingContext


class BingSearch(SerpNode):
    """
    Search Bing to retrieve organic web search results.
    bing, search, serp, web, query
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

        result_data = await self._search_raw(context, "bing", {
            "q": self.query,
            "count": self.num_results,
        })

        results = result_data.get("organic_results", [])
        text = format_results(results, [
            ("link", None),
            ("displayed_link", "URL"),
            ("date", "Date"),
            ("snippet", None),
        ])
        return {"results": results, "text": text}


class BingNews(SerpNode):
    """
    Search Bing News to retrieve current news articles and headlines.
    bing, news, serp, articles, journalism
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(
        default="", description="Search query or keyword for news articles"
    )
    num_results: int = Field(
        default=10, description="Maximum number of news results to return"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(context, "bing_news", {
            "q": self.query,
            "count": self.num_results,
        })

        # Bing News returns results under "organic_results" or "news_results"
        results = result_data.get("organic_results", [])
        if not results:
            results = result_data.get("news_results", [])
        text = format_results(results, [
            ("link", None),
            ("source", "Source"),
            ("date", "Date"),
            ("snippet", None),
        ])
        return {"results": results, "text": text}


class BingImages(SerpNode):
    """
    Search Bing Images to find visual content from across the web.
    bing, images, serp, visual, search
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str
        images: list[ImageRef]

    query: str = Field(default="", description="Search query or keyword for images")
    num_results: int = Field(
        default=20, description="Maximum number of image results to return"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(context, "bing_images", {
            "q": self.query,
            "count": self.num_results,
        })

        results = result_data.get("images_results", [])
        text = format_results(results, [
            ("source", "Source"),
            ("original", None),
            ("size", "Size"),
        ])
        images = [
            ImageRef(uri=img.get("original", ""))
            for img in results
            if img.get("original")
        ]
        return {"results": results, "text": text, "images": images}
