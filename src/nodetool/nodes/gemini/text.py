from enum import Enum
from typing import ClassVar, TypedDict

from nodetool.metadata.types import Provider, Source
from nodetool.providers.gemini_provider import GeminiProvider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class GeminiModel(str, Enum):
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class GroundedSearch(BaseNode):
    """
    Search the web using Google's Gemini API with grounding capabilities.
    google, search, grounded, web, gemini, ai

    This node uses Google's Gemini API to perform web searches and return structured results
    with source information. Requires a Gemini API key.

    Use cases:
    - Research current events and latest information
    - Find reliable sources for fact-checking
    - Gather web-based information with citations
    - Get up-to-date information beyond the model's training data
    """

    _expose_as_tool: ClassVar[bool] = True

    query: str = Field(default="", description="The search query to execute")

    model: GeminiModel = Field(
        default=GeminiModel.GEMINI_2_0_FLASH,
        description="The Gemini model to use for search",
    )

    class OutputType(TypedDict):
        results: list[str]
        sources: list[Source]

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Execute a web search using Gemini API with grounding.

        Returns:
            Dict containing the search results and sources
        """
        from google.genai.types import (
            GenerateContentConfig,
            GoogleSearch,
            Tool as GenAITool,
        )

        if not self.query:
            raise ValueError("Search query is required")

        # Configure Google Search as a tool
        google_search_tool = GenAITool(google_search=GoogleSearch())

        # Generate content with search grounding
        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = await provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]
        response = await client.models.generate_content(
            model=self.model.value,
            contents=self.query,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            ),
        )

        # Extract search results and source information
        results = []
        sources = []

        # Check if we have a valid response with candidates
        if not response or not response.candidates:
            raise ValueError("No response received from Gemini API")

        candidate = response.candidates[0]
        if not candidate or not candidate.content:
            raise ValueError("Invalid response format from Gemini API")

        # Get the main response text
        if candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    results.append(part.text)

        # Extract source information if available
        if (
            candidate.grounding_metadata
            and candidate.grounding_metadata.grounding_chunks
        ):
            # Extract sources from grounding chunks
            chunks = candidate.grounding_metadata.grounding_chunks
            for chunk in chunks:
                if hasattr(chunk, "web") and chunk.web:
                    source = Source(
                        title=chunk.web.title or "",
                        url=chunk.web.uri or "",
                    )
                    if source not in sources and source.url:
                        sources.append(source)

        return {
            "results": results,
            "sources": sources,
        }
