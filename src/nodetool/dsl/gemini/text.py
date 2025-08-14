from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.gemini.text


class GroundedSearch(GraphNode):
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

    GeminiModel: typing.ClassVar[type] = nodetool.nodes.gemini.text.GeminiModel
    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The search query to execute"
    )
    model: nodetool.nodes.gemini.text.GeminiModel = Field(
        default=nodetool.nodes.gemini.text.GeminiModel.GEMINI_2_0_FLASH,
        description="The Gemini model to use for search",
    )

    @classmethod
    def get_node_type(cls):
        return "gemini.text.GroundedSearch"
