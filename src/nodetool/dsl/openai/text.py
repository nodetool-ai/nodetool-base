from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.openai.text


class Embedding(GraphNode):
    """
    Generate vector representations of text for semantic analysis.
    embeddings, similarity, search, clustering, classification

    Uses OpenAI's embedding models to create dense vector representations of text.
    These vectors capture semantic meaning, enabling:
    - Semantic search
    - Text clustering
    - Document classification
    - Recommendation systems
    - Anomaly detection
    - Measuring text similarity and diversity
    """

    EmbeddingModel: typing.ClassVar[type] = (
        nodetool.nodes.openai.text.Embedding.EmbeddingModel
    )
    input: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    model: nodetool.nodes.openai.text.Embedding.EmbeddingModel = Field(
        default=nodetool.nodes.openai.text.Embedding.EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
        description=None,
    )
    chunk_size: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "openai.text.Embedding"


class WebSearch(GraphNode):
    """
    🔍 OpenAI Web Search - Searches the web using OpenAI's web search capabilities.

    This node uses an OpenAI model equipped with web search functionality
    (like gpt-4o with search preview) to answer queries based on current web information.
    Requires an OpenAI API key.
    """

    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The search query to execute."
    )

    @classmethod
    def get_node_type(cls):
        return "openai.text.WebSearch"
