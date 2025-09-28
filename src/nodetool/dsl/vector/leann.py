from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.vector.leann
import nodetool.nodes.vector.leann


class LeannBuilder(GraphNode):
    """
    Create high-performance vector indexes for Retrieval-Augmented Generation (RAG) using LEANN.
    embedding, collection, RAG, index, text, chunk, batch, vector, search, leann

    Embedding Model Selection: Understanding the Trade-offs
    Based on extensive testing with LEANN, embedding models fall into three performance tiers:

    🚀 SMALL MODELS (< 100M parameters)
    - sentence-transformers/all-MiniLM-L6-v2 (22M params)
    🎯 BEST FOR: Prototyping, simple queries, interactive demos, resource-constrained environments

    ⚖️ MEDIUM MODELS (100M-500M parameters)
    - facebook/contriever (110M params)
    - BAAI/bge-base-en-v1.5 (109M params)
    🎯 BEST FOR: Production RAG applications, balanced quality-performance needs

    🏆 LARGE MODELS (500M+ parameters)
    - Qwen/Qwen3-Embedding-0.6B (600M params)
    - intfloat/multilingual-e5-large (560M params)
    🎯 BEST FOR: High-stakes applications, maximum accuracy requirements, production environments

    Index Backend Selection: Optimizing for Scale and Performance

    Choose the right backend based on your dataset size and performance requirements:

    ⚡ HNSW (Hierarchical Navigable Small World)
    🎯 BEST FOR: Datasets < 10M vectors, when memory isn't a bottleneck, default recommendation

    🚀 DISKANN (Disk-based Approximate Nearest Neighbor)
    🎯 BEST FOR: Large datasets (100k+ documents), production environments, when recompute=True
    """

    EmbeddingModel: typing.ClassVar[type] = (
        nodetool.nodes.vector.leann.LeannBuilder.EmbeddingModel
    )
    Backend: typing.ClassVar[type] = nodetool.nodes.vector.leann.LeannBuilder.Backend
    folder: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Output folder where the LEANN index will be stored. Choose a location with sufficient disk space for your dataset.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Unique identifier for this index. Used for organizing and retrieving specific indexes.",
    )
    model: nodetool.nodes.vector.leann.LeannBuilder.EmbeddingModel = Field(
        default=nodetool.nodes.vector.leann.LeannBuilder.EmbeddingModel.sentence_transformers_all_minilm_l6_v2,
        description="Select embedding model based on your quality-speed trade-off. See EmbeddingModel documentation for detailed guidance.",
    )
    backend: nodetool.nodes.vector.leann.LeannBuilder.Backend = Field(
        default=nodetool.nodes.vector.leann.LeannBuilder.Backend.hnsw,
        description="Index backend optimized for your scale. HNSW for most use cases, DiskANN for large datasets. See Backend documentation for details.",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to index"
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the text"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.leann.LeannBuilder"


import nodetool.nodes.vector.leann


class LeannSearcher(GraphNode):
    """
    Search high-performance vector indexes for Retrieval-Augmented Generation (RAG) using LEANN.
    embedding, collection, RAG, index, text, chunk, batch, vector, search, leann
    """

    PruningStrategy: typing.ClassVar[type] = (
        nodetool.nodes.vector.leann.LeannSearcher.PruningStrategy
    )
    folder: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The folder where the index is stored"
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The name of the index"
    )
    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The query to search for"
    )
    top_k: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="The number of results to return"
    )
    complexity: int | GraphNode | tuple[GraphNode, str] = Field(
        default=64, description="The complexity of the search"
    )
    beam_width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="The beam width of the search"
    )
    prune_ratio: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description="The prune ratio of the search"
    )
    recompute_embeddings: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to recompute the embeddings"
    )
    pruning_strategy: nodetool.nodes.vector.leann.LeannSearcher.PruningStrategy = Field(
        default=nodetool.nodes.vector.leann.LeannSearcher.PruningStrategy._global,
        description="The pruning strategy of the search",
    )

    @classmethod
    def get_node_type(cls):
        return "vector.leann.LeannSearcher"
