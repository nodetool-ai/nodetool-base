from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.chroma.index


class IndexAggregatedText(GraphNode):
    """
    Index multiple text chunks at once with aggregated embeddings from Ollama.
    chroma, embedding, collection, RAG, index, text, chunk, batch, ollama
    """

    EmbeddingAggregation: typing.ClassVar[type] = (
        nodetool.nodes.chroma.index.EmbeddingAggregation
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
    aggregation: nodetool.nodes.chroma.index.EmbeddingAggregation = Field(
        default=nodetool.nodes.chroma.index.EmbeddingAggregation.MEAN,
        description="The aggregation method to use for the embeddings.",
    )

    @classmethod
    def get_node_type(cls):
        return "chroma.index.IndexAggregatedText"


class IndexEmbedding(GraphNode):
    """
    Index a list of embeddings.
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    embedding: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The embedding to index",
    )
    id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The ID to associate with the embedding"
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the embedding"
    )

    @classmethod
    def get_node_type(cls):
        return "chroma.index.IndexEmbedding"


class IndexImage(GraphNode):
    """
    Index a single image asset.
    chroma, embedding, collection, RAG, index, image
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="Image asset to index",
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the image"
    )

    @classmethod
    def get_node_type(cls):
        return "chroma.index.IndexImage"


class IndexImages(GraphNode):
    """
    Index a list of image assets or files.
    chroma, embedding, collection, RAG, index, image, batch
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    images: list[types.ImageRef] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of image assets to index"
    )
    upsert: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Whether to upsert the images"
    )

    @classmethod
    def get_node_type(cls):
        return "chroma.index.IndexImages"


class IndexString(GraphNode):
    """
    Index a string with a Document ID to a collection.
    chroma, embedding, collection, RAG, index, text, string

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
        return "chroma.index.IndexString"


class IndexTextChunk(GraphNode):
    """
    Index a single text chunk.
    chroma, embedding, collection, RAG, index, text, chunk
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    text_chunk: types.TextChunk | GraphNode | tuple[GraphNode, str] = Field(
        default=types.TextChunk(
            type="text_chunk", text="", source_id="", start_index=0
        ),
        description="Text chunk to index",
    )
    metadata: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The metadata to associate with the text chunk"
    )

    @classmethod
    def get_node_type(cls):
        return "chroma.index.IndexTextChunk"


class IndexTextChunks(GraphNode):
    """
    Index multiple text chunks at once.
    chroma, embedding, collection, RAG, index, text, chunk, batch
    """

    collection: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to index",
    )
    text_chunks: list[types.TextChunk] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of text chunks to index"
    )

    @classmethod
    def get_node_type(cls):
        return "chroma.index.IndexTextChunks"
