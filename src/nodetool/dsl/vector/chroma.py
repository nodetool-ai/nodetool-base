from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class CollectionNode(GraphNode):
    """
    Get or create a collection.
    vector, embedding, collection, RAG, get, create, chroma
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The name of the collection to create"
    )
    embedding_model: types.LlamaModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LlamaModel(
            type="llama_model",
            name="",
            repo_id="",
            modified_at="",
            size=0,
            digest="",
            details={},
        ),
        description="Model to use for embedding, search for nomic-embed-text and download it",
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.Collection"


class Count(GraphNode):
    """
    Count the number of documents in a collection.
    vector, embedding, collection, RAG, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to count",
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.Count"


class GetDocuments(GraphNode):
    """
    Get documents from a chroma collection.
    vector, embedding, collection, RAG, retrieve, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to get",
    )
    ids: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The ids of the documents to get"
    )
    limit: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="The limit of the documents to get"
    )
    offset: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The offset of the documents to get"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.GetDocuments"


class HybridSearch(GraphNode):
    """
    Hybrid search combining semantic and keyword-based search for better retrieval. Uses reciprocal rank fusion to combine results from both methods.
    vector, RAG, query, semantic, text, similarity, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to query",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to query"
    )
    n_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="The number of final results to return"
    )
    k_constant: float | GraphNode | tuple[GraphNode, str] = Field(
        default=60.0, description="Constant for reciprocal rank fusion (default: 60.0)"
    )
    min_keyword_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=3, description="Minimum length for keyword tokens"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.HybridSearch"


import nodetool.nodes.vector.chroma


class IndexAggregatedText(GraphNode):
    """
    Index multiple text chunks at once with aggregated embeddings from Ollama.
    vector, embedding, collection, RAG, index, text, chunk, batch, ollama, chroma
    """

    EmbeddingAggregation: typing.ClassVar[type] = (
        nodetool.nodes.vector.chroma.EmbeddingAggregation
    )
    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    document: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The document to index"
    )
    document_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The document ID to associate with the text"
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the text"
    )
    text_chunks: (
        list[nodetool.metadata.types.TextChunk | str]
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=[], description="List of text chunks to index")
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="The context window size to use for the model"
    )
    aggregation: nodetool.nodes.vector.chroma.EmbeddingAggregation = Field(
        default=nodetool.nodes.vector.chroma.EmbeddingAggregation.MEAN,
        description="The aggregation method to use for the embeddings.",
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.IndexAggregatedText"


class IndexEmbedding(GraphNode):
    """
    Index a single embedding vector into a Chroma collection with optional metadata. Creates a searchable entry that can be queried for similarity matching.
    vector, index, embedding, chroma, storage, RAG
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    embedding: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The embedding to index",
    )
    index_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The ID to associate with the embedding"
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the embedding"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.IndexEmbedding"


class IndexImage(GraphNode):
    """
    Index a list of image assets or files.
    vector, embedding, collection, RAG, index, image, batch, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of image assets to index"
    )
    index_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The ID to associate with the image, defaults to the URI of the image",
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the image"
    )
    upsert: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Whether to upsert the images"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.IndexImage"


class IndexString(GraphNode):
    """
    Index a string with a Document ID to a collection.
    vector, embedding, collection, RAG, index, text, string, chroma

    Use cases:
    - Index documents for a vector search
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text content to index"
    )
    document_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Document ID to associate with the text content"
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the text"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.IndexString"


class IndexTextChunk(GraphNode):
    """
    Index a single text chunk.
    vector, embedding, collection, RAG, index, text, chunk, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    document_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The document ID to associate with the text chunk"
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to index"
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the text chunk"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.IndexTextChunk"


class Peek(GraphNode):
    """
    Peek at the documents in a collection.
    vector, embedding, collection, RAG, preview, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to peek",
    )
    limit: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="The limit of the documents to peek"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.Peek"


class QueryImage(GraphNode):
    """
    Query the index for similar images.
    vector, RAG, query, image, search, similarity, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to query",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to query",
    )
    n_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="The number of results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.QueryImage"


class QueryText(GraphNode):
    """
    Query the index for similar text.
    vector, RAG, query, text, search, similarity, chroma
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to query",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to query"
    )
    n_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="The number of results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.QueryText"


class RemoveOverlap(GraphNode):
    """
    Removes overlapping words between consecutive strings in a list. Splits text into words and matches word sequences for more accurate overlap detection.
    vector, RAG, query, text, processing, overlap, deduplication
    """

    documents: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of strings to process for overlap removal"
    )
    min_overlap_words: int | GraphNode | tuple[GraphNode, str] = Field(
        default=2,
        description="Minimum number of words that must overlap to be considered",
    )

    @classmethod
    def get_node_type(cls):
        return "vector.chroma.RemoveOverlap"
